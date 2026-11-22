"""Automated downstream model training and evaluation for engineered features.

Trains baseline LightGBM / XGBoost models on feature matrices and measures
cross-validated ROC-AUC (classification) or RMSE (regression) gain.

Provides:
- train_lightgbm_cv_roc_auc: LightGBM cross-validated ROC-AUC for binary classification
- train_lightgbm_cv_rmse: LightGBM cross-validated RMSE for regression
- train_xgboost_cv_roc_auc: XGBoost cross-validated ROC-AUC for binary classification
- train_xgboost_cv_rmse: XGBoost cross-validated RMSE for regression
- compute_feature_importance_shap: SHAP value computation for a trained model
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# LightGBM
import lightgbm as lgb

# XGBoost
import xgboost as xgb

# SHAP
import shap

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helper: safe metric computation
# ---------------------------------------------------------------------------

def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute ROC AUC, returning 0.5 (random) if only one class present."""
    try:
        return float(roc_auc_score(y_true, y_prob))
    except (ValueError, Exception):
        return 0.5


def _safe_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute RMSE from true and predicted values."""
    try:
        mse = np.mean((y_true - y_pred) ** 2)
        val = float(mse)
        return math.sqrt(val) if math.isfinite(val) and val >= 0 else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# LightGBM cross-validated ROC-AUC (binary classification)
# ---------------------------------------------------------------------------

def train_lightgbm_cv_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
    num_boost_round: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """LightGBM cross-validated ROC-AUC for binary classification.

    Returns dict with "roc_auc" mean and "roc_auc_std" standard deviation.
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": seed,
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: List[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dvalid],
            )

            # Predict probability on validation set
            y_prob = model.predict(X_valid)
            score = _safe_roc_auc(y_valid, y_prob)
            scores.append(score)

    mean_auc = float(np.mean(scores)) if scores else 0.5
    std_auc = float(np.std(scores)) if len(scores) > 1 else 0.0
    return {"roc_auc": mean_auc, "roc_auc_std": std_auc}


# ---------------------------------------------------------------------------
# LightGBM cross-validated RMSE (regression)
# ---------------------------------------------------------------------------

def train_lightgbm_cv_rmse(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
    num_boost_round: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """LightGBM cross-validated RMSE for regression.

    Returns dict with "rmse" mean and "rmse_std" standard deviation.
    """
    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": seed,
        }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: List[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, valid_idx in kf.split(X):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            dtrain = lgb.Dataset(X_train, label=y_train)
            dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dvalid],
            )

            y_pred = model.predict(X_valid)
            score = _safe_rmse(y_valid, y_pred)
            scores.append(score)

    mean_rmse = float(np.mean(scores)) if scores else float("nan")
    std_rmse = float(np.std(scores)) if len(scores) > 1 else 0.0
    return {"rmse": mean_rmse, "rmse_std": std_rmse}


# ---------------------------------------------------------------------------
# XGBoost cross-validated ROC-AUC (binary classification)
# ---------------------------------------------------------------------------

def train_xgboost_cv_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
    num_boost_round: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """XGBoost cross-validated ROC-AUC for binary classification.

    Returns dict with "roc_auc" mean and "roc_auc_std" standard deviation.
    """
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
        }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: List[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, valid_idx in skf.split(X, y):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
            )

            # Predict probability
            y_prob = model.predict(dvalid)
            score = _safe_roc_auc(y_valid, y_prob)
            scores.append(score)

    mean_auc = float(np.mean(scores)) if scores else 0.5
    std_auc = float(np.std(scores)) if len(scores) > 1 else 0.0
    return {"roc_auc": mean_auc, "roc_auc_std": std_auc}


# ---------------------------------------------------------------------------
# XGBoost cross-validated RMSE (regression)
# ---------------------------------------------------------------------------

def train_xgboost_cv_rmse(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 42,
    num_boost_round: int = 100,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """XGBoost cross-validated RMSE for regression.

    Returns dict with "rmse" mean and "rmse_std" standard deviation.
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": seed,
        }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores: List[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, valid_idx in kf.split(X):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dvalid = xgb.DMatrix(X_valid, label=y_valid)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, "train"), (dvalid, "valid")],
            )

            y_pred = model.predict(dvalid)
            score = _safe_rmse(y_valid, y_pred)
            scores.append(score)

    mean_rmse = float(np.mean(scores)) if scores else float("nan")
    std_rmse = float(np.std(scores)) if len(scores) > 1 else 0.0
    return {"rmse": mean_rmse, "rmse_std": std_rmse}


# ---------------------------------------------------------------------------
# SHAP feature importance computation
# ---------------------------------------------------------------------------

def compute_feature_importance_shap(
    model: Any,
    X: np.ndarray,
    *,
    feature_names: Optional[List[str]] = None,
    model_type: str = "auto",
    max_display: int = 20,
) -> Dict[str, float]:
    """Compute SHAP-based feature importance values.

    Returns dict mapping feature name -> mean |SHAP value|.
    """
    try:
        explainer = shap.Explainer(model, X, feature_names=feature_names, model_type=model_type)
        shap_values = explainer(X)

        # Compute mean absolute SHAP value per feature
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]

        importance_dict: Dict[str, float] = {}
        for i, fname in enumerate(feature_names):
            if i < shap_values.shape[1]:
                sv = shap_values.values
                if sv.ndim == 2:
                    av = np.mean(np.abs(sv), axis=0)
                else:
                    av = np.abs(sv)
                importance_dict[fname] = float(np.mean(av))
            else:
                importance_dict[fname] = 0.0

        # Sort by importance descending and limit display
        sorted_items = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        return {k: v for k, v in sorted_items[:max_display]}

    except Exception:
        # Fallback: return zero importances
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        return {fname: 0.0 for fname in feature_names}


# ---------------------------------------------------------------------------
# Public API: run full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "classification",
    model_type: str = "lightgbm",
    n_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a full evaluation pipeline and return metrics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    task : {"classification", "regression"}
        Problem type.
    model_type : {"lightgbm", "xgboost"}
        Which model to train.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    Dict with keys depending on task:
      - classification: {"roc_auc": float, "roc_auc_std": float, "shap_importance": dict}
      - regression: {"rmse": float, "rmse_std": float, "shap_importance": dict}
    """
    result: Dict[str, Any] = {}

    if task == "classification":
        if model_type == "lightgbm":
            metrics = train_lightgbm_cv_roc_auc(
                X, y, n_folds=n_folds, seed=seed
            )
            result["roc_auc"] = metrics["roc_auc"]
            result["roc_auc_std"] = metrics["roc_auc_std"]
        elif model_type == "xgboost":
            metrics = train_xgboost_cv_roc_auc(
                X, y, n_folds=n_folds, seed=seed
            )
            result["roc_auc"] = metrics["roc_auc"]
            result["roc_auc_std"] = metrics["roc_auc_std"]

        # SHAP importance using the last trained model (re-train on full data for SHAP)
        if model_type == "lightgbm":
            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": seed,
            }
            dtrain = lgb.Dataset(X, label=y)
            final_model = lgb.train(params, dtrain, num_boost_round=100)
            shap_imp = compute_feature_importance_shap(
                final_model, X, feature_names=None, max_display=20
            )
        elif model_type == "xgboost":
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": seed,
            }
            dtrain = xgb.DMatrix(X, label=y)
            final_model = xgb.train(params, dtrain, num_boost_round=100)
            shap_imp = compute_feature_importance_shap(
                final_model, X, feature_names=None, max_display=20
            )
        result["shap_importance"] = shap_imp

    elif task == "regression":
        if model_type == "lightgbm":
            metrics = train_lightgbm_cv_rmse(
                X, y, n_folds=n_folds, seed=seed
            )
            result["rmse"] = metrics["rmse"]
            result["rmse_std"] = metrics["rmse_std"]
        elif model_type == "xgboost":
            metrics = train_xgboost_cv_rmse(
                X, y, n_folds=n_folds, seed=seed
            )
            result["rmse"] = metrics["rmse"]
            result["rmse_std"] = metrics["rmse_std"]

        # SHAP importance
        if model_type == "lightgbm":
            params = {
                "objective": "regression",
                "metric": "rmse",
                "boosting_type": "gbdt",
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": seed,
            }
            dtrain = lgb.Dataset(X, label=y)
            final_model = lgb.train(params, dtrain, num_boost_round=100)
            shap_imp = compute_feature_importance_shap(
                final_model, X, feature_names=None, max_display=20
            )
        elif model_type == "xgboost":
            params = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "max_depth": 6,
                "eta": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "seed": seed,
            }
            dtrain = xgb.DMatrix(X, label=y)
            final_model = xgb.train(params, dtrain, num_boost_round=100)
            shap_imp = compute_feature_importance_shap(
                final_model, X, feature_names=None, max_display=20
            )
        result["shap_importance"] = shap_imp

    return result
