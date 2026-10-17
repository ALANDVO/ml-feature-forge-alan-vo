"""Regression tests for production pipeline code generation.

These guard against generated ``build_features`` code referencing undefined
symbols (e.g. the historical ``category_mean_map`` NameError for target-encoded
features) and verify that every deterministic transform type emits runnable code.
"""
import io

import pytest
from backend.app.services.feature_engine import (
    compute_feature_values,
    generate_deterministic_candidates,
    parse_csv_data,
    profile_dataset,
)
from backend.app.services.pipeline_generator import generate_python_pipeline_code

try:
    import pandas as pd  # noqa: F401
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment dependent
    _PANDAS_AVAILABLE = False


CATEGORICAL_CSV = """age,income,credit_score,loan_amount,debt_to_income,months_employed,city_tier,approved
28,45000,680,12000,0.26,36,tier_1,1
45,85000,720,25000,0.29,120,tier_1,1
22,28000,590,8000,0.35,12,tier_3,0
35,62000,640,18000,0.31,48,tier_2,1
50,110000,790,40000,0.22,180,tier_1,1
24,31000,610,10000,0.40,18,tier_2,0
41,74000,690,22000,0.28,84,tier_1,1
31,52000,630,15000,0.33,30,tier_3,0
"""


def test_target_encode_pipeline_expression_is_self_contained():
    """Target-encoded features must emit runnable code without undefined symbols."""
    cols, rows = parse_csv_data(CATEGORICAL_CSV)
    profile = profile_dataset(cols, rows, target_column="approved")
    candidates = generate_deterministic_candidates(profile, target_column="approved")

    te_candidates = [c for c in candidates if c["transform_type"] == "target_encode"]
    assert te_candidates, "expected at least one target_encode candidate"
    te = te_candidates[0]

    # The historical expression referenced an undefined category_mean_map.
    assert "category_mean_map" not in te["expression"]

    code = generate_python_pipeline_code(
        [te], target_column="approved", dataset_name="credit_sample.csv"
    )
    # Must be syntactically valid Python.
    compile(code, "<pipeline>", "exec")
    # Must not reintroduce the undefined symbol in emitted source.
    assert "category_mean_map" not in code

    if not _PANDAS_AVAILABLE:
        pytest.skip("pandas not installed; skipping execution check")

    df = pd.read_csv(io.StringIO(CATEGORICAL_CSV))
    namespace = {}
    exec(code, namespace)
    out = namespace["build_features"](df)
    assert list(out.columns) == [te["name"]]

    # Generated values must match the deterministic offline engine exactly.
    offline = compute_feature_values(te, rows, target_column="approved")
    generated = out[te["name"]].tolist()
    assert offline == pytest.approx(generated)


def test_all_transform_types_produce_compilable_pipeline():
    """Every deterministic transform must compile; all together must run."""
    cols, rows = parse_csv_data(CATEGORICAL_CSV)
    profile = profile_dataset(cols, rows, target_column="approved")
    candidates = generate_deterministic_candidates(profile, target_column="approved")

    seen_types = {c["transform_type"] for c in candidates}
    expected_types = {
        "log1p", "standard_scale", "quantile_bin",
        "ratio", "product",
        "frequency_encode", "target_encode",
    }
    assert expected_types.issubset(seen_types), f"missing transforms: {expected_types - seen_types}"

    # Each individual transform must compile.
    for c in candidates:
        code = generate_python_pipeline_code([c], target_column="approved", dataset_name="t")
        compile(code, "<pipeline>", "exec")

    if not _PANDAS_AVAILABLE:
        pytest.skip("pandas not installed; skipping execution check")

    df = pd.read_csv(io.StringIO(CATEGORICAL_CSV))
    # A combined pipeline with every transform must execute end-to-end.
    code = generate_python_pipeline_code(
        candidates, target_column="approved", dataset_name="credit_sample.csv"
    )
    namespace = {}
    exec(code, namespace)
    out = namespace["build_features"](df)
    assert out.shape[0] == len(rows)
    assert set(out.columns) == {c["name"] for c in candidates}


def test_datetime_transform_types_produce_compilable_pipeline():
    """Cyclical and weekend transforms over datetime columns must compile and run."""
    csv_text = (
        "date,amount,income,category,target\n"
        "2023-01-05,100,50000,a,1\n"
        "2023-06-15,200,80000,b,0\n"
        "2023-12-20,150,45000,a,1\n"
        "2023-03-10,300,90000,c,0\n"
        "2023-11-25,250,70000,b,1\n"
    )
    cols, rows = parse_csv_data(csv_text)
    profile = profile_dataset(cols, rows, target_column="target")
    candidates = generate_deterministic_candidates(profile, target_column="target")

    dt_types = {c["transform_type"] for c in candidates if c["transform_type"] in ("cyclical_sin", "weekend_indicator")}
    assert dt_types == {"cyclical_sin", "weekend_indicator"}, "datetime transforms not generated"

    for c in candidates:
        code = generate_python_pipeline_code([c], target_column="target", dataset_name="t")
        compile(code, "<pipeline>", "exec")

    if not _PANDAS_AVAILABLE:
        pytest.skip("pandas not installed; skipping execution check")

    df = pd.read_csv(io.StringIO(csv_text))
    code = generate_python_pipeline_code(candidates, target_column="target", dataset_name="t")
    namespace = {}
    exec(code, namespace)
    out = namespace["build_features"](df)
    assert out.shape[0] == len(rows)