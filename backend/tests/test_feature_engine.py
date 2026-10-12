"""Tests for dataset profiling, deterministic feature engineering, and statistical evaluations."""
import pytest
from backend.app.services.feature_engine import (
    evaluate_candidates,
    generate_deterministic_candidates,
    parse_csv_data,
    profile_dataset,
)
from backend.app.services.importance import (
    mean,
    pearson_correlation,
    spearman_correlation,
    std_dev,
)


@pytest.fixture
def sample_tabular_data():
    csv_text = """age,income,credit_score,debt_to_income,loan_approved
25,40000,650,0.30,1
40,80000,720,0.25,1
20,25000,580,0.45,0
35,60000,680,0.35,1
55,100000,780,0.20,1
22,30000,600,0.40,0
45,85000,730,0.22,1
30,50000,640,0.32,0
"""
    cols, rows = parse_csv_data(csv_text)
    return cols, rows


def test_dataset_profiling(sample_tabular_data):
    cols, rows = sample_tabular_data
    profile = profile_dataset(cols, rows, target_column="loan_approved")

    assert profile.row_count == 8
    assert profile.col_count == 5
    assert "income" in profile.numeric_columns
    assert "age" in profile.numeric_columns
    assert profile.problem_type == "classification"
    assert profile.missing_counts["income"] == 0
    assert profile.stats["income"]["min"] == 25000.0
    assert profile.stats["income"]["max"] == 100000.0


def test_deterministic_candidate_generation(sample_tabular_data):
    cols, rows = sample_tabular_data
    profile = profile_dataset(cols, rows, target_column="loan_approved")
    candidates = generate_deterministic_candidates(profile, target_column="loan_approved")

    assert len(candidates) >= 5
    names = [c["name"] for c in candidates]
    # Verify non-linear, scaling, and interaction features are created
    assert any("log1p" in n for n in names)
    assert any("zscore" in n for n in names)
    assert any("ratio" in n or "prod" in n for n in names)
    for c in candidates:
        assert "loan_approved" not in c["input_columns"], "Target column must not be used as input"


def test_pearson_and_spearman_correlations():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert pytest.approx(pearson_correlation(x, y), 0.001) == 1.0
    assert pytest.approx(spearman_correlation(x, y), 0.001) == 1.0

    y_inv = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert pytest.approx(pearson_correlation(x, y_inv), 0.001) == -1.0


def test_evaluation_verdicts_and_leakage(sample_tabular_data):
    cols, rows = sample_tabular_data
    profile = profile_dataset(cols, rows, target_column="loan_approved")
    candidates = generate_deterministic_candidates(profile, target_column="loan_approved")

    # Inject an intentional leaking candidate to verify safety pruning
    leaking_candidate = {
        "name": "leak_candidate",
        "transform_type": "identity",
        "expression": "df['loan_approved']",
        "input_columns": ["loan_approved"],
        "hypothesis": "Intentional leakage test",
    }
    test_candidates = candidates[:4] + [leaking_candidate]

    evals = evaluate_candidates(
        candidates=test_candidates,
        rows=rows,
        target_column="loan_approved",
    )

    leak_eval = next(e for e in evals if e["feature_name"] == "leak_candidate")
    assert leak_eval["verdict"] == "drop"
    assert "leakage" in leak_eval["reason"].lower()


def test_multicollinearity_detection(sample_tabular_data):
    cols, rows = sample_tabular_data
    c1 = {
        "name": "base_income",
        "transform_type": "standard_scale",
        "expression": "df['income']",
        "input_columns": ["income"],
        "hypothesis": "Base income scale",
    }
    c2 = {
        "name": "duplicate_income",
        "transform_type": "standard_scale",
        "expression": "df['income']",
        "input_columns": ["income"],
        "hypothesis": "Duplicate identical transform",
    }

    evals = evaluate_candidates([c1, c2], rows=rows, target_column="loan_approved", collinearity_threshold=0.9)
    assert evals[0]["verdict"] in ("keep", "drop")
    # Second identical feature must be flagged as redundant collinearity and dropped
    assert evals[1]["verdict"] == "drop"
    assert "collinear" in evals[1]["reason"].lower() or "redundant" in evals[1]["reason"].lower()
