"""Targeted tests for automated hyperparameter tuning module."""

import numpy as np

from backend.app.services.hyperparameter_tuning import (
    train_lightgbm_cv_roc_auc,
    train_lightgbm_cv_rmse,
    train_xgboost_cv_roc_auc,
    train_xgboost_cv_rmse,
    compute_feature_importance_shap,
    run_evaluation_pipeline,
)


def test_lightgbm_cv_roc_auc():
    """LightGBM CV ROC-AUC on synthetic binary dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (rng.randn(200) + 3 * X[:, 0] + 2 * X[:, 1] > 0).astype(int)

    result = train_lightgbm_cv_roc_auc(X, y, n_folds=3, seed=42, num_boost_round=50)
    assert 0.5 < result["roc_auc"] <= 1.0, f"ROC AUC too low: {result['roc_auc']}"
    assert result["roc_auc_std"] >= 0.0
    print(f"  lightgbm_cv_roc_auc: {result}")


def test_lightgbm_cv_rmse():
    """LightGBM CV RMSE on synthetic regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.randn(200) * 0.5

    result = train_lightgbm_cv_rmse(X, y, n_folds=3, seed=42, num_boost_round=50)
    assert not np.isnan(result["rmse"]), f"RMSE is NaN: {result['rmse']}"
    assert result["rmse"] > 0, f"RMSE should be positive: {result['rmse']}"
    assert result["rmse_std"] >= 0.0
    print(f"  lightgbm_cv_rmse: {result}")


def test_xgboost_cv_roc_auc():
    """XGBoost CV ROC-AUC on synthetic binary dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (rng.randn(200) + 3 * X[:, 0] + 2 * X[:, 1] > 0).astype(int)

    result = train_xgboost_cv_roc_auc(X, y, n_folds=3, seed=42, num_boost_round=50)
    assert 0.5 < result["roc_auc"] <= 1.0, f"ROC AUC too low: {result['roc_auc']}"
    assert result["roc_auc_std"] >= 0.0
    print(f"  xgboost_cv_roc_auc: {result}")


def test_xgboost_cv_rmse():
    """XGBoost CV RMSE on synthetic regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.randn(200) * 0.5

    result = train_xgboost_cv_rmse(X, y, n_folds=3, seed=42, num_boost_round=50)
    assert not np.isnan(result["rmse"]), f"RMSE is NaN: {result['rmse']}"
    assert result["rmse"] > 0, f"RMSE should be positive: {result['rmse']}"
    assert result["rmse_std"] >= 0.0
    print(f"  xgboost_cv_rmse: {result}")


def test_compute_shap_importance():
    """SHAP feature importance computation."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (rng.randn(100) + 2 * X[:, 0] + X[:, 1] > 0).astype(int)

    # Train a LightGBM model
    import lightgbm as lgb
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 31,
        "random_state": 42,
    }
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=50)

    shap_imp = compute_feature_importance_shap(
        model, X, feature_names=[f"f{i}" for i in range(4)], max_display=4
    )
    assert len(shap_imp) == 4, f"Expected 4 features, got {len(shap_imp)}"
    # All values should be non-negative
    for k, v in shap_imp.items():
        assert v >= 0, f"SHAP importance should be non-negative, got {v} for {k}"
    print(f"  shap_importance: {shap_imp}")


def test_run_evaluation_pipeline_classification():
    """Full pipeline: LightGBM classification."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = (rng.randn(200) + 3 * X[:, 0] + 2 * X[:, 1] > 0).astype(int)

    result = run_evaluation_pipeline(
        X, y, task="classification", model_type="lightgbm", n_folds=3, seed=42
    )
    assert "roc_auc" in result, "Missing roc_auc in result"
    assert 0.5 < result["roc_auc"] <= 1.0, f"ROC AUC out of range: {result['roc_auc']}"
    assert "shap_importance" in result, "Missing shap_importance in result"
    assert len(result["shap_importance"]) > 0, "SHAP importance should not be empty"
    print(f"  run_evaluation_pipeline (cls, lgb): {result}")


def test_run_evaluation_pipeline_regression():
    """Full pipeline: LightGBM regression."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + rng.randn(200) * 0.5

    result = run_evaluation_pipeline(
        X, y, task="regression", model_type="lightgbm", n_folds=3, seed=42
    )
    assert "rmse" in result, "Missing rmse in result"
    assert not np.isnan(result["rmse"]), f"RMSE is NaN: {result['rmse']}"
    assert result["rmse"] > 0, f"RMSE should be positive: {result['rmse']}"
    assert "shap_importance" in result, "Missing shap_importance in result"
    print(f"  run_evaluation_pipeline (reg, lgb): {result}")


if __name__ == "__main__":
    test_lightgbm_cv_roc_auc()
    test_lightgbm_cv_rmse()
    test_xgboost_cv_roc_auc()
    test_xgboost_cv_rmse()
    test_compute_shap_importance()
    test_run_evaluation_pipeline_classification()
    test_run_evaluation_pipeline_regression()
    print("\nAll targeted tests passed!")
