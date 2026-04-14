import pandas as pd
import numpy as np
import joblib
import logging
<<<<<<< HEAD
import json
from datetime import datetime
=======
>>>>>>> 42deab75f6c745885fdac8dd40c0b20f5075d102
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

<<<<<<< HEAD
from preprocessing import (
    optimize_memory, drop_useless_features,handle_infinite_and_nan, MissingValueHandler, 
    SkewedFeatureTransformer, CategoricalLevelManager, FrequencyEncoder
)
from feature_engineering import FeatureEngineeringTransformer, FeaturePruner


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FraudMLOpsPipeline:
    def __init__(self, target_col="isFraud", time_col="TransactionDT", output_dir="artifacts"):
        self.target_col = target_col
        self.time_col = time_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).resolve().parent.parent
        self.pipeline = self._build_pipeline()
        
    def _build_pipeline(self):
        return Pipeline([
            ('memory_opt', FunctionTransformer(optimize_memory)),
            ('drop_useless', FunctionTransformer(drop_useless_features)),
            ('missing_handler', MissingValueHandler(target_col=self.target_col)),
            ('skew_trans', SkewedFeatureTransformer(method='yeo-johnson')),
            ('cat_manager', CategoricalLevelManager()),
            ('feature_eng', FeatureEngineeringTransformer()),
            ('final_fillna', FunctionTransformer(handle_infinite_and_nan)),
            ('freq_encoder', FrequencyEncoder()),
            ('pruner', FeaturePruner(target_col=self.target_col, corr_threshold=0.95))
        ])

    def split_data(self, df, train_ratio=0.8):
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()
        
        logger.warning(f"Data Split: Train ({df_train[self.time_col].min()} -> {df_train[self.time_col].max()})")
        logger.warning(f"Data Split: Val   ({df_val[self.time_col].min()} -> {df_val[self.time_col].max()})")
        return df_train, df_val

    def run_train_flow(self, df_raw):
        df_train, df_val = self.split_data(df_raw)
        
        logger.warning("Fitting pipeline on Train set...")
        self.pipeline.fit(df_train)
        
        X_train_clean = self.pipeline.transform(df_train)
        X_val_clean = self.pipeline.transform(df_val)
        
        y_train = X_train_clean[self.target_col]
        y_val = X_val_clean[self.target_col]
        
=======
# Import các transformer
from preprocessing import (
    optimize_memory,
    drop_useless_features,
    handle_infinite_and_nan,
    MissingValueHandler,
    SkewedFeatureTransformer,
    CategoricalLevelManager,
    FrequencyEncoder,
    SemanticSentinelFiller,
)

from _feature_engineering import (
    FeatureEngineeringTransformer,
    TopVSignalTransformer,
    FeaturePruner,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_params() -> dict:
    """Load params.yaml"""
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError("params.yaml not found!")
    with open(params_path, encoding="utf-8") as f:
        return yaml.safe_load(f)   


import yaml   


class FraudMLOpsPipeline:
    def __init__(self):
        self.params = load_params()
        self.preprocess_params = self.params.get("preprocess", {})
        self.features_params = self.params.get("features", {})
        self.target_col = self.features_params.get("target_col", "isFraud")
        self.time_col = "TransactionDT"

        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ('memory_opt', FunctionTransformer(optimize_memory)),
            ('drop_useless', FunctionTransformer(drop_useless_features)),
            ('missing_handler', MissingValueHandler(
                target_col=self.target_col,
                top_k_missing=self.preprocess_params.get("top_k_missing", 80)
            )),
            ('skew_trans', SkewedFeatureTransformer()),
            ('cat_manager', CategoricalLevelManager(
                min_freq=self.preprocess_params.get("cat_min_freq", 0.0005)
            )),
            ('feature_eng', FeatureEngineeringTransformer()),
            ('semantic_sentinel', SemanticSentinelFiller()),
            ('top_v_signals', TopVSignalTransformer()),
            ('final_fillna', FunctionTransformer(handle_infinite_and_nan)),
            ('freq_encoder', FrequencyEncoder()),
            ('pruner', FeaturePruner(
                target_col=self.target_col,
                corr_threshold=0.90
            )),
        ])

    def split_data(self, df: pd.DataFrame):
        df = df.sort_values(self.time_col).reset_index(drop=True)
        split_idx = int(len(df) * self.features_params.get("train_ratio", 0.8))
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()

        logger.info(f"Data Split → Train: {len(df_train):,} rows | Val: {len(df_val):,} rows")
        return df_train, df_val

    def run_train_flow(self, df_raw: pd.DataFrame):
        df_train, df_val = self.split_data(df_raw)

        logger.info("Fitting full pipeline on Train set...")
        self.pipeline.fit(df_train)

        X_train_clean = self.pipeline.transform(df_train)
        X_val_clean = self.pipeline.transform(df_val)

        y_train = X_train_clean[self.target_col]
        y_val = X_val_clean[self.target_col]

>>>>>>> 42deab75f6c745885fdac8dd40c0b20f5075d102
        X_train_final = X_train_clean.drop(columns=[self.target_col])
        X_val_final = X_val_clean.drop(columns=[self.target_col])

        self.save_everything(X_train_final, y_train, X_val_final, y_val)
<<<<<<< HEAD
        
        return X_train_final, y_train, X_val_final, y_val
    
    def save_everything(self, X_train, y_train, X_val, y_val):
        data_dir = self.project_root / "data" / "processed"
        artifact_dir = self.project_root / "artifacts"

        data_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path = artifact_dir / "fraud_pipeline.pkl"
        joblib.dump(self.pipeline, pipeline_path)
        
        x_train_path = data_dir / "X_train.parquet"
        X_train.to_parquet(x_train_path, index=False)
=======
        return X_train_final, y_train, X_val_final, y_val

    def save_everything(self, X_train, y_train, X_val, y_val):
        data_dir = Path("data/processed")
        artifact_dir = Path("artifacts")
        data_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, artifact_dir / "fraud_pipeline.pkl")

        X_train.to_parquet(data_dir / "X_train.parquet", index=False)
>>>>>>> 42deab75f6c745885fdac8dd40c0b20f5075d102
        y_train.to_frame().to_parquet(data_dir / "y_train.parquet", index=False)
        X_val.to_parquet(data_dir / "X_val.parquet", index=False)
        y_val.to_frame().to_parquet(data_dir / "y_val.parquet", index=False)

<<<<<<< HEAD
        logger.warning(f"Pipeline: {pipeline_path.absolute()}")
        logger.warning(f"Data folder: {data_dir.absolute()}")

        metadata = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_count": X_train.shape[1],
            "features": X_train.columns.tolist()
        }
        with open(artifact_dir / "features_info.json", "w") as f:
            json.dump(metadata, f, indent=4)

# ==========================================
if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    data_path = current_file.parent.parent / "data" / "merged_train_data.csv"

    if data_path.exists():
        df = pd.read_csv(data_path)
        ops_pipe = FraudMLOpsPipeline(target_col="isFraud", time_col="TransactionDT")
        X_train, y_train, X_val, y_val = ops_pipe.run_train_flow(df)
=======
        logger.info(f" Pipeline saved: artifacts/fraud_pipeline.pkl")
        logger.info(f" Featured data saved to: data/processed/")


# ==================================================
if __name__ == "__main__":
    data_path = Path("data/merged_train_data.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"File {data_path} is not found")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded raw data: {df.shape}")

    pipeline = FraudMLOpsPipeline()
    X_train, y_train, X_val, y_val = pipeline.run_train_flow(df)
>>>>>>> 42deab75f6c745885fdac8dd40c0b20f5075d102
