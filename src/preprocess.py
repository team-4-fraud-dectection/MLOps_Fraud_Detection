import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation import validate_binary_target, validate_dataframe  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


STRONG_DROP_CANDIDATES = [
    "id_24", "id_25", "id_07", "id_08", "id_21",
    "id_26", "id_22", "id_23", "id_27"
]


class FullPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        top_k_missing: int = 80,
        clip_percentile: float = 0.99,
        cat_min_freq: float = 0.0005,
    ):
        self.top_k_missing = top_k_missing
        self.clip_percentile = clip_percentile
        self.cat_min_freq = cat_min_freq

        self.top_features_ = None
        self.impute_stats_ = None
        self.skew_trans_ = None
        self.cat_manager_ = None
        self.encoder_ = None

    # --------------------- 1. LOAD & MEMORY OPTIMIZE ---------------------
    def _load_and_optimize(self, file_path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(file_path)

        float_cols = df.select_dtypes(include="float64").columns
        int_cols = df.select_dtypes(include="int64").columns

        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].astype("float32")
        if len(int_cols) > 0:
            df[int_cols] = df[int_cols].astype("int32")

        drop_cols = [c for c in STRONG_DROP_CANDIDATES if c in df.columns]
        if drop_cols:
            logging.info("Dropping %d strong-drop candidate columns", len(drop_cols))
            df.drop(columns=drop_cols, inplace=True)

        return df

    # --------------------- 2. MISSING VALUE HANDLING ---------------------
    def _get_top_missing_features(
        self, df: pd.DataFrame, target_col: str = "isFraud"
    ) -> Tuple[List[str], Dict]:
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")

        fraud_mask = df[target_col] == 1
        missing = df.isnull()

        miss_rate = missing.mean()
        miss_fraud = missing[fraud_mask].mean()
        miss_non = missing[~fraud_mask].mean()
        abs_gap_vec = (miss_fraud - miss_non).abs()

        rows = []
        for col in df.columns:
            if col == target_col or miss_rate[col] < 0.001:
                continue

            abs_gap = abs_gap_vec[col]
            miss_rate_col = miss_rate[col]

            is_missing = missing[col]
            fraud_rate_missing = df.loc[is_missing, target_col].mean() if is_missing.any() else 0.0
            fraud_rate_present = df.loc[~is_missing, target_col].mean()
            fraud_rate_gap = abs(fraud_rate_missing - fraud_rate_present)

            if pd.api.types.is_numeric_dtype(df[col]):
                temp = df[[col, target_col]].copy()
                temp[col] = temp[col].fillna(temp[col].median())
                corr = abs(spearmanr(temp[col], temp[target_col], nan_policy="omit")[0])
                corr = 0.0 if pd.isna(corr) else corr
            else:
                contingency = pd.crosstab(df[col].fillna("MISSING"), df[target_col], dropna=False)
                if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                    chi2, _, _, _ = chi2_contingency(contingency)
                    n = df.shape[0]
                    min_dim = min(contingency.shape) - 1
                    corr = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
                else:
                    corr = 0.0

            score = (
                0.35 * abs_gap
                + 0.30 * fraud_rate_gap
                + 0.25 * corr
                + 0.10 * miss_rate_col
            )

            rows.append({"feature": col, "score": score})

        ranking = pd.DataFrame(rows).sort_values("score", ascending=False)
        top_features = ranking.head(self.top_k_missing)["feature"].tolist()

        impute_stats = {}
        for col in top_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                impute_stats[col] = df[col].median()
            else:
                impute_stats[col] = "MISSING"

        return top_features, impute_stats

    def _preprocess_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        indicator_list = []

        for col in self.top_features_:
            if col in df.columns:
                indicator = df[col].isna().astype("int8")
                indicator.name = f"{col}_isna"
                indicator_list.append(indicator)

        if indicator_list:
            df = pd.concat([df, pd.concat(indicator_list, axis=1)], axis=1)

        for col in self.top_features_:
            if col not in df.columns:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(self.impute_stats_.get(col, -999))
            else:
                df[col] = df[col].astype("string").fillna("MISSING").astype("category")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(-999)

        return df

    # --------------------- 3. BASIC NUMERIC NORMALIZATION ---------------------
    class SkewedFeatureTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, log_cols=None, clip_percentile: float = 0.99):
            self.log_cols = log_cols or []
            self.clip_percentile = clip_percentile
            self.clip_thresholds_ = {}

        def fit(self, X, y=None):
            num_cols = X.select_dtypes(include=np.number).columns
            for col in num_cols:
                self.clip_thresholds_[col] = X[col].quantile(self.clip_percentile)
            return self

        def transform(self, X):
            X = X.copy()

            for col, th in self.clip_thresholds_.items():
                if col in X.columns:
                    X[col] = X[col].clip(upper=th)

            for col in self.log_cols:
                if col in X.columns:
                    X[col] = np.log1p(X[col].clip(lower=0))

            return X

    # --------------------- 4. CATEGORICAL NORMALIZATION ---------------------
    class CategoricalLevelManager(BaseEstimator, TransformerMixin):
        def __init__(self, min_freq: float = 0.0005):
            self.min_freq = min_freq
            self.common_labels_ = {}

        def fit(self, X, y=None):
            cat_cols = X.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                freq = X[col].value_counts(normalize=True, dropna=True)
                self.common_labels_[col] = freq[freq >= self.min_freq].index.tolist()
            return self

        def transform(self, X):
            X = X.copy()
            for col, common in self.common_labels_.items():
                if col in X.columns:
                    series = X[col].astype(str)
                    is_null = X[col].isna() | (series == "nan") | (series == "None")
                    X[col] = np.where(
                        is_null,
                        "MISSING",
                        np.where(series.isin(common), series, "OTHER")
                    )
            return X

    # --------------------- 5. BASIC ENCODING ---------------------
    class CategoricalEncoder(BaseEstimator, TransformerMixin):
        def __init__(self):
            self.encoders_ = {}

        def fit(self, X, y=None):
            cat_cols = X.select_dtypes(exclude=np.number).columns
            for col in cat_cols:
                le = LabelEncoder()
                values = X[col].astype(str).fillna("MISSING").tolist()
                values = values + ["RARE"]
                le.fit(values)
                self.encoders_[col] = le
            return self

        def transform(self, X):
            X = X.copy()
            for col, le in self.encoders_.items():
                if col in X.columns:
                    X[col] = X[col].astype(str).map(lambda s: s if s in le.classes_ else "RARE")
                    X[col] = le.transform(X[col])
            return X

    # ====================== MAIN FIT / TRANSFORM ======================
    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, (str, Path)):
            df = self._load_and_optimize(X)
        else:
            df = X.copy()

        # giữ target để tính missing-signal ranking, nhưng không dùng cho encoder logic
        self.top_features_, self.impute_stats_ = self._get_top_missing_features(df)

        df = self._preprocess_missing(df)

        log_cols = [c for c in ["TransactionAmt", "C1", "C2"] if c in df.columns]
        self.skew_trans_ = self.SkewedFeatureTransformer(
            log_cols=log_cols,
            clip_percentile=self.clip_percentile
        )
        df = self.skew_trans_.fit_transform(df)

        feature_df = df.drop(columns=["isFraud"], errors="ignore")

        self.cat_manager_ = self.CategoricalLevelManager(min_freq=self.cat_min_freq)
        feature_df = self.cat_manager_.fit_transform(feature_df)

        self.encoder_ = self.CategoricalEncoder()
        self.encoder_.fit(feature_df)

        logging.info("Preprocessor fitted successfully. Shape after basic preprocess: %s", df.shape)
        return self

    def transform(self, X: pd.DataFrame):
        if isinstance(X, (str, Path)):
            df = self._load_and_optimize(X)
        else:
            df = X.copy()

        target_series = None
        if "isFraud" in df.columns:
            target_series = df["isFraud"].copy()

        df = self._preprocess_missing(df)
        df = self.skew_trans_.transform(df)

        feature_df = df.drop(columns=["isFraud"], errors="ignore")
        feature_df = self.cat_manager_.transform(feature_df)
        feature_df = self.encoder_.transform(feature_df)

        if target_series is not None:
            feature_df["isFraud"] = target_series.values

        return feature_df


def parse_args():
    parser = argparse.ArgumentParser(description="Basic preprocessing for IEEE fraud detection (B-route).")
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/merged_train_data.csv",
        help="Path to merged raw training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed parquet files."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save fitted preprocessor."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio after sorting by TransactionDT."
    )
    parser.add_argument(
        "--top_k_missing",
        type=int,
        default=80,
        help="Top-K missing-signal features to create missing indicators."
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=0.99,
        help="Upper clip percentile for numeric features."
    )
    parser.add_argument(
        "--cat_min_freq",
        type=float,
        default=0.0005,
        help="Minimum relative frequency for keeping categorical levels."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading and preprocessing data from %s", input_path)

    preprocessor = FullPreprocessor(
        top_k_missing=args.top_k_missing,
        clip_percentile=args.clip_percentile,
        cat_min_freq=args.cat_min_freq,
    )
    df_processed = preprocessor.fit_transform(input_path)
    validate_dataframe(
        df_processed,
        dataset_name="preprocessed dataset",
        required_columns=["TransactionDT", "isFraud"],
    )
    validate_binary_target(
        df_processed["isFraud"],
        dataset_name="preprocessed dataset",
        target_name="isFraud",
    )

    if "TransactionDT" not in df_processed.columns:
        raise ValueError("TransactionDT column is required for time-based split.")

    if "isFraud" not in df_processed.columns:
        raise ValueError("isFraud column is required for downstream stages.")

    df_processed = df_processed.sort_values("TransactionDT").reset_index(drop=True)

    split_idx = int(len(df_processed) * args.train_ratio)
    train_df = df_processed.iloc[:split_idx].copy()
    val_df = df_processed.iloc[split_idx:].copy()

    logging.info("Train shape: %s | Val shape: %s", train_df.shape, val_df.shape)
    logging.info("Train fraud rate: %.4f", train_df["isFraud"].mean())
    logging.info("Val fraud rate: %.4f", val_df["isFraud"].mean())

    validate_dataframe(train_df, dataset_name="train split")
    validate_dataframe(val_df, dataset_name="validation split")

    train_out = output_dir / "train_preprocessed.parquet"
    val_out = output_dir / "val_preprocessed.parquet"
    model_out = model_dir / "preprocessor_v1.pkl"

    train_df.to_parquet(train_out, index=False)
    val_df.to_parquet(val_out, index=False)
    joblib.dump(preprocessor, model_out)

    logging.info("Saved train preprocessed data to %s", train_out)
    logging.info("Saved val preprocessed data to %s", val_out)
    logging.info("Saved preprocessor to %s", model_out)


if __name__ == "__main__":
    main()
