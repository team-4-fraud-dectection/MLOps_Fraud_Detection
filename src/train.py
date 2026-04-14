import argparse
import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import mlflow
import mlflow.catboost
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
from catboost import CatBoostClassifier
from mlflow.models import infer_signature

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

RANDOM_STATE = 42
PRIMARY_METRIC = "auprc"
THRESHOLD_METRIC = "f1"


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def sanitize_feature_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def sanitize_feature_columns(X_train: pd.DataFrame, X_val: pd.DataFrame):
    original_cols = list(X_train.columns)
    new_cols = []
    seen = {}

    for col in original_cols:
        new_col = sanitize_feature_name(col)
        if new_col in seen:
            seen[new_col] += 1
            new_col = f"{new_col}_{seen[new_col]}"
        else:
            seen[new_col] = 0
        new_cols.append(new_col)

    mapping = dict(zip(original_cols, new_cols))

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_train.columns = new_cols
    X_val.columns = [mapping[c] for c in X_val.columns]

    return X_train, X_val, mapping


def drop_constant_features(X_train: pd.DataFrame, X_val: pd.DataFrame):
    selector = VarianceThreshold(threshold=0.0)
    selector.fit(X_train)
    kept_cols = X_train.columns[selector.get_support()].tolist()
    dropped_cols = [c for c in X_train.columns if c not in kept_cols]

    X_train_out = X_train[kept_cols].copy()
    X_val_out = X_val[kept_cols].copy()
    return X_train_out, X_val_out, dropped_cols


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train, tune, compare, and track fraud detection models with MLflow."
    )
    parser.add_argument("--x_train_path", type=str, required=True)
    parser.add_argument("--y_train_path", type=str, required=True)
    parser.add_argument("--x_val_path", type=str, required=True)
    parser.add_argument("--y_val_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metrics_path", type=str, required=True)

    parser.add_argument("--enable_tuning", action="store_true")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--tune_models", type=str, default="XGBoost,LightGBM,CatBoost")
    parser.add_argument("--mlflow_experiment", type=str, default="ieee_fraud_detection")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="")
    parser.add_argument("--mlflow_run_name", type=str, default="train_and_tune")
    parser.add_argument("--register_model_name", type=str, default="")
    return parser.parse_args()


def evaluate_predictions(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "auprc": float(average_precision_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError(f"Model {type(model).__name__} has no probability-like output.")


def tune_threshold(y_true, y_proba, metric=THRESHOLD_METRIC):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = -1.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        if metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = thr

    return float(best_threshold), float(best_score)


def build_models():
    models = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=6,
        min_child_samples=50,
        min_split_gain=0.0,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["CatBoost"] = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
    )

    return models


def train_model(name, model, X_train, y_train, X_val, y_val):
    logging.info("Training %s...", name)

    if name == "LightGBM":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    elif name == "XGBoost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif name == "CatBoost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model


def get_best_iteration(name, model):
    if name == "LightGBM":
        return int(getattr(model, "best_iteration_", 0) or 0)
    if name == "XGBoost":
        return int(getattr(model, "best_iteration", 0) or 0)
    if name == "CatBoost":
        try:
            return int(model.get_best_iteration())
        except Exception:
            return 0
    return 0


def log_metrics(prefix: str, metrics: dict):
    mlflow.log_metrics(
        {
            f"{prefix}_auprc": metrics["auprc"],
            f"{prefix}_recall": metrics["recall"],
            f"{prefix}_precision": metrics["precision"],
            f"{prefix}_f1": metrics["f1"],
            f"{prefix}_threshold": metrics["threshold"],
        }
    )


# ---------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------
def suggest_params(trial, model_name: str):
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }

    if model_name == "LightGBM":
        return {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": -1,
        }

    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def build_model_from_params(model_name: str, params: dict):
    if model_name == "XGBoost":
        return xgb.XGBClassifier(**params)
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(**params)
    if model_name == "CatBoost":
        return CatBoostClassifier(**params)
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def run_optuna_tuning(model_name, X_train, y_train, X_val, y_val, n_trials):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is not installed. Install it with: pip install optuna")

    def objective(trial):
        params = suggest_params(trial, model_name)

        with mlflow.start_run(nested=True, run_name=f"{model_name}_trial_{trial.number}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)

            model = build_model_from_params(model_name, params)
            model = train_model(model_name, model, X_train, y_train, X_val, y_val)
            y_val_proba = get_proba(model, X_val)
            metrics = evaluate_predictions(y_val, y_val_proba, threshold=0.5)

            log_metrics("val", metrics)
            best_iter = get_best_iteration(model_name, model)
            if best_iter > 0:
                mlflow.log_metric("best_iteration", best_iter)

            trial.set_user_attr("metrics", metrics)
            return metrics[PRIMARY_METRIC]

    sampler = TPESampler(seed=RANDOM_STATE) if TPESampler else None
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    tracking_uri = args.mlflow_tracking_uri or "file:./mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    model_path = Path(args.model_path)
    metrics_path = Path(args.metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Loading train/validation datasets...")
    X_train = pd.read_parquet(args.x_train_path).astype("float32")
    y_train = pd.read_parquet(args.y_train_path).values.ravel()
    X_val = pd.read_parquet(args.x_val_path).astype("float32")
    y_val = pd.read_parquet(args.y_val_path).values.ravel()

    X_train, X_val, feature_name_mapping = sanitize_feature_columns(X_train, X_val)
    X_train, X_val, dropped_constant_cols = drop_constant_features(X_train, X_val)

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("X_train contains non-finite values (inf or -inf).")
    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("X_val contains non-finite values (inf or -inf).")

    logging.info("Train shape: %s", X_train.shape)
    logging.info("Validation shape: %s", X_val.shape)
    logging.info("Dropped constant features: %d", len(dropped_constant_cols))
    logging.info("Train positive rate: %.4f", y_train.mean())
    logging.info("Validation positive rate: %.4f", y_val.mean())

    models = build_models()
    tuned_model_names = {m.strip() for m in args.tune_models.split(",") if m.strip()}

    all_results = {}
    fitted_models = {}
    best_params_by_model = {}

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.set_tags({
            "pipeline_stage": "train",
            "selection_metric": PRIMARY_METRIC,
            "threshold_tuning_metric": THRESHOLD_METRIC,
        })
        mlflow.log_params({
            "x_train_path": args.x_train_path,
            "y_train_path": args.y_train_path,
            "x_val_path": args.x_val_path,
            "y_val_path": args.y_val_path,
            "train_rows": int(X_train.shape[0]),
            "train_cols": int(X_train.shape[1]),
            "val_rows": int(X_val.shape[0]),
            "val_cols": int(X_val.shape[1]),
            "enable_tuning": args.enable_tuning,
            "n_trials": args.n_trials,
            "mlflow_tracking_uri": tracking_uri,
            "dropped_constant_features": len(dropped_constant_cols),
        })
        if dropped_constant_cols:
            mlflow.log_text("\n".join(dropped_constant_cols), artifact_file="dropped_constant_features.txt")

        for name, base_model in models.items():
            with mlflow.start_run(nested=True, run_name=name):
                mlflow.log_param("model_name", name)

                if args.enable_tuning and name in tuned_model_names:
                    logging.info("Tuning %s with Optuna for %d trials...", name, args.n_trials)
                    best_params, best_auprc = run_optuna_tuning(
                        model_name=name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        n_trials=args.n_trials,
                    )
                    model = build_model_from_params(name, best_params)
                    best_params_by_model[name] = best_params
                    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                    mlflow.log_metric("best_trial_auprc", best_auprc)
                else:
                    model = base_model

                fitted_model = train_model(name, model, X_train, y_train, X_val, y_val)
                fitted_models[name] = fitted_model

                y_val_proba = get_proba(fitted_model, X_val)
                metrics_default = evaluate_predictions(y_val, y_val_proba, threshold=0.5)
                all_results[name] = {"default_threshold_metrics": metrics_default}

                log_metrics("val", metrics_default)
                best_iter = get_best_iteration(name, fitted_model)
                if best_iter > 0:
                    mlflow.log_metric("best_iteration", best_iter)
                mlflow.log_text(
                    json.dumps(metrics_default["classification_report"], indent=2),
                    artifact_file="classification_report.json",
                )

                logging.info(
                    "%s | AUPRC=%.4f | Recall=%.4f | Precision=%.4f | F1=%.4f",
                    name,
                    metrics_default["auprc"],
                    metrics_default["recall"],
                    metrics_default["precision"],
                    metrics_default["f1"],
                )

        ranked = sorted(
            all_results.items(),
            key=lambda x: x[1]["default_threshold_metrics"][PRIMARY_METRIC],
            reverse=True,
        )
        best_model_name = ranked[0][0]
        best_model = fitted_models[best_model_name]
        logging.info("Best model by validation %s: %s", PRIMARY_METRIC.upper(), best_model_name)

        best_model_val_proba = get_proba(best_model, X_val)
        tuned_threshold, _ = tune_threshold(y_val, best_model_val_proba, metric=THRESHOLD_METRIC)
        tuned_metrics = evaluate_predictions(y_val, best_model_val_proba, threshold=tuned_threshold)
        all_results[best_model_name]["tuned_threshold_metrics"] = tuned_metrics

        mlflow.log_param("champion_model_name", best_model_name)
        mlflow.log_metric("champion_default_auprc", all_results[best_model_name]["default_threshold_metrics"]["auprc"])
        mlflow.log_metric("champion_tuned_threshold", tuned_threshold)
        mlflow.log_metric("champion_tuned_f1", tuned_metrics["f1"])
        mlflow.log_metric("champion_tuned_recall", tuned_metrics["recall"])
        mlflow.log_metric("champion_tuned_precision", tuned_metrics["precision"])
        mlflow.log_metric("champion_tuned_auprc", tuned_metrics["auprc"])

        artifact = {
            "model_name": best_model_name,
            "model": best_model,
            "threshold": tuned_threshold,
            "feature_count": int(X_train.shape[1]),
            "feature_names": list(X_train.columns),
            "feature_name_mapping": feature_name_mapping,
            "dropped_constant_features": dropped_constant_cols,
            "best_params_by_model": best_params_by_model,
        }
        joblib.dump(artifact, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="local_artifacts")

        input_example = X_val.head(5)
        y_pred_example = get_proba(best_model, input_example)
        signature = infer_signature(input_example, y_pred_example)

        if best_model_name == "CatBoost":
            mlflow.catboost.log_model(
                cb_model=best_model,
                name="champion_model",
                signature=signature,
                input_example=input_example,
            )
        elif best_model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                name="champion_model",
                signature=signature,
                input_example=input_example,
            )
        elif best_model_name == "LightGBM":
            mlflow.lightgbm.log_model(
                lgb_model=best_model,
                name="champion_model",
                signature=signature,
                input_example=input_example,
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="champion_model",
                signature=signature,
                input_example=input_example,
            )

        if args.register_model_name:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/champion_model"
            mlflow.register_model(model_uri=model_uri, name=args.register_model_name)

        output = {
            "training_setup": {
                "train_data": args.x_train_path,
                "val_data": args.x_val_path,
                "selection_metric": PRIMARY_METRIC,
                "threshold_tuning_metric": THRESHOLD_METRIC,
                "feature_count": int(X_train.shape[1]),
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "enable_tuning": args.enable_tuning,
                "n_trials": args.n_trials,
                "tuned_models": sorted(list(tuned_model_names)),
                "dropped_constant_features": len(dropped_constant_cols),
            },
            "best_model_name": best_model_name,
            "best_model_default_metrics": all_results[best_model_name]["default_threshold_metrics"],
            "best_model_tuned_metrics": tuned_metrics,
            "best_params_by_model": best_params_by_model,
            "all_results": all_results,
        }

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        mlflow.log_artifact(str(metrics_path), artifact_path="local_artifacts")
        mlflow.log_text(json.dumps(feature_name_mapping, indent=2), artifact_file="feature_name_mapping.json")

    logging.info("Saved champion model to %s", model_path)
    logging.info("Saved training metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
