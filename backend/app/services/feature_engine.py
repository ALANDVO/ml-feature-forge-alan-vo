"""Core domain engine for dataset profiling, deterministic feature engineering, and evaluation."""
import csv
import io
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from backend.app.models.schemas import DatasetProfile, FeatureCandidate, FeatureEvaluation
from backend.app.services.importance import (
    cyclical_cos,
    cyclical_sin,
    encode_quantile_bin,
    compute_quantiles,
    mean,
    pearson_correlation,
    spearman_correlation,
    std_dev,
)


def is_number(val: Any) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def is_date(val: Any) -> bool:
    if not isinstance(val, str) or len(val) < 8:
        return False
    formats = ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d/%m/%Y")
    for fmt in formats:
        try:
            datetime.strptime(val[:19], fmt)
            return True
        except ValueError:
            pass
    return False


def parse_csv_data(csv_text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    f = io.StringIO(csv_text.strip())
    reader = csv.DictReader(f)
    columns = reader.fieldnames or []
    rows = [r for r in reader]
    return list(columns), rows


def profile_dataset(columns: List[str], rows: List[Dict[str, Any]], target_column: str) -> DatasetProfile:
    row_count = len(rows)
    dtypes: Dict[str, str] = {}
    missing_counts: Dict[str, int] = {}
    missing_rates: Dict[str, float] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    datetime_cols: List[str] = []
    stats: Dict[str, Dict[str, Any]] = {}

    for col in columns:
        vals = [r.get(col) for r in rows]
        non_nulls = [v for v in vals if v is not None and str(v).strip() not in ("", "null", "None", "NA", "nan")]
        missing = row_count - len(non_nulls)
        missing_counts[col] = missing
        missing_rates[col] = round(missing / row_count, 4) if row_count > 0 else 0.0

        if not non_nulls:
            dtypes[col] = "text"
            stats[col] = {"type": "empty"}
            continue

        sample = non_nulls[:100]
        if all(is_number(v) for v in sample):
            dtypes[col] = "numeric"
            numeric_cols.append(col)
            num_vals = [float(v) for v in non_nulls]
            m = mean(num_vals)
            s = std_dev(num_vals, m)
            stats[col] = {
                "min": round(min(num_vals), 4),
                "max": round(max(num_vals), 4),
                "mean": round(m, 4),
                "std": round(s, 4),
                "distinct_count": len(set(num_vals)),
            }
        elif all(is_date(str(v)) for v in sample):
            dtypes[col] = "datetime"
            datetime_cols.append(col)
            stats[col] = {"distinct_count": len(set(non_nulls))}
        else:
            unique_ratio = len(set(non_nulls)) / len(non_nulls) if non_nulls else 1.0
            if unique_ratio < 0.4 or len(set(non_nulls)) <= 15:
                dtypes[col] = "categorical"
                categorical_cols.append(col)
                counts: Dict[str, int] = {}
                for v in non_nulls:
                    sv = str(v)
                    counts[sv] = counts.get(sv, 0) + 1
                top_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                stats[col] = {
                    "distinct_count": len(counts),
                    "top_categories": [c[0] for c in top_cats],
                }
            else:
                dtypes[col] = "text"
                stats[col] = {"distinct_count": len(set(non_nulls))}

    # Detect problem type based on target column
    target_type = dtypes.get(target_column, "numeric")
    if target_type == "numeric":
        t_vals = [float(r[target_column]) for r in rows if r.get(target_column) and is_number(r[target_column])]
        unique_targets = len(set(t_vals))
        problem_type = "classification" if unique_targets <= 10 else "regression"
    else:
        problem_type = "classification"

    return DatasetProfile(
        row_count=row_count,
        col_count=len(columns),
        columns=columns,
        dtypes=dtypes,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        missing_counts=missing_counts,
        missing_rates=missing_rates,
        stats=stats,
        target_column=target_column,
        problem_type=problem_type,
    )


def generate_deterministic_candidates(
    profile: DatasetProfile,
    target_column: str,
    include_interactions: bool = True,
    include_encodings: bool = True,
    include_nonlinear: bool = True,
    include_datetime: bool = True,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    feature_numeric = [c for c in profile.numeric_columns if c != target_column]
    feature_categorical = [c for c in profile.categorical_columns if c != target_column]
    feature_datetime = [c for c in profile.datetime_columns if c != target_column]

    # 1. Non-linear transformations on numeric features
    if include_nonlinear:
        for col in feature_numeric:
            stats = profile.stats.get(col, {})
            c_min = stats.get("min", 0.0)
            if c_min >= 0:
                candidates.append({
                    "name": f"log1p_{col}",
                    "transform_type": "log1p",
                    "expression": f"np.log1p(df['{col}'].clip(lower=0))",
                    "input_columns": [col],
                    "hypothesis": f"Compress skewness in {col} to linearize proportional variance with {target_column}.",
                    "expected_signal": "moderate",
                    "risk": "none",
                })
            candidates.append({
                "name": f"zscore_{col}",
                "transform_type": "standard_scale",
                "expression": f"(df['{col}'] - {stats.get('mean', 0.0)}) / {stats.get('std', 1.0) or 1.0}",
                "input_columns": [col],
                "hypothesis": f"Normalize variance and scale of {col} for numerical stability across training batches.",
                "expected_signal": "moderate",
                "risk": "none",
            })
            candidates.append({
                "name": f"qbin_{col}",
                "transform_type": "quantile_bin",
                "expression": f"pd.qcut(df['{col}'], q=5, labels=False, duplicates='drop')",
                "input_columns": [col],
                "hypothesis": f"Partition {col} into 5 quantile tiers to capture non-monotonic step relationships.",
                "expected_signal": "moderate",
                "risk": "none",
            })

    # 2. Pairwise Interaction features
    if include_interactions and len(feature_numeric) >= 2:
        pairs = []
        for i in range(min(4, len(feature_numeric))):
            for j in range(i + 1, min(4, len(feature_numeric))):
                pairs.append((feature_numeric[i], feature_numeric[j]))

        for c1, c2 in pairs[:6]:
            candidates.append({
                "name": f"ratio_{c1}_per_{c2}",
                "transform_type": "ratio",
                "expression": f"df['{c1}'] / (df['{c2}'].abs() + 1e-5)",
                "input_columns": [c1, c2],
                "hypothesis": f"Relative density ratio of {c1} per unit of {c2} reveals compounding scale leverage.",
                "expected_signal": "strong",
                "risk": "none",
            })
            candidates.append({
                "name": f"prod_{c1}_x_{c2}",
                "transform_type": "product",
                "expression": f"df['{c1}'] * df['{c2}']",
                "input_columns": [c1, c2],
                "hypothesis": f"Multiplicative synergy interaction capturing conditional effects between {c1} and {c2}.",
                "expected_signal": "moderate",
                "risk": "none",
            })

    # 3. Categorical encodings
    if include_encodings:
        for cat in feature_categorical[:4]:
            candidates.append({
                "name": f"freq_{cat}",
                "transform_type": "frequency_encode",
                "expression": f"df['{cat}'].map(df['{cat}'].value_counts(normalize=True))",
                "input_columns": [cat],
                "hypothesis": f"Prior occurrence frequency of category values in {cat} acts as rarity proxy.",
                "expected_signal": "moderate",
                "risk": "none",
            })
            candidates.append({
                "name": f"target_mean_{cat}",
                "transform_type": "target_encode",
                "expression": f"df.groupby('{cat}')['{target_column}'].transform(lambda s: (s.sum() + 5.0 * df['{target_column}'].mean()) / (len(s) + 5.0))",
                "input_columns": [cat],
                "hypothesis": f"Bayesian smoothed mean expected target probability per category level in {cat}.",
                "expected_signal": "strong",
                "risk": "overfit",
            })

    # 4. Datetime features
    if include_datetime:
        for dt_col in feature_datetime[:2]:
            candidates.append({
                "name": f"cyclical_sin_month_{dt_col}",
                "transform_type": "cyclical_sin",
                "expression": f"np.sin(2 * np.pi * pd.to_datetime(df['{dt_col}']).dt.month / 12.0)",
                "input_columns": [dt_col],
                "hypothesis": f"Continuous cyclical sin encoding of annual seasonality from {dt_col}.",
                "expected_signal": "moderate",
                "risk": "none",
            })
            candidates.append({
                "name": f"is_weekend_{dt_col}",
                "transform_type": "weekend_indicator",
                "expression": f"pd.to_datetime(df['{dt_col}']).dt.dayofweek.isin([5, 6]).astype(int)",
                "input_columns": [dt_col],
                "hypothesis": f"Weekly behavioral shift between weekday routines and weekend dynamics in {dt_col}.",
                "expected_signal": "moderate",
                "risk": "none",
            })

    return candidates


def compute_feature_values(
    candidate: Dict[str, Any],
    rows: List[Dict[str, Any]],
    target_column: str,
) -> List[float]:
    """Execute feature transform deterministically on in-memory dataset rows."""
    t_type = candidate.get("transform_type", "")
    inputs = candidate.get("input_columns", [])
    output: List[float] = []

    if t_type == "log1p" and inputs:
        c = inputs[0]
        for r in rows:
            v = float(r[c]) if r.get(c) and is_number(r[c]) else 0.0
            output.append(math.log1p(max(0.0, v)))
    elif t_type == "standard_scale" and inputs:
        c = inputs[0]
        vals = [float(r[c]) if r.get(c) and is_number(r[c]) else 0.0 for r in rows]
        m = mean(vals)
        s = std_dev(vals, m) or 1.0
        output = [(v - m) / s for v in vals]
    elif t_type == "quantile_bin" and inputs:
        c = inputs[0]
        vals = [float(r[c]) if r.get(c) and is_number(r[c]) else 0.0 for r in rows]
        quantiles = compute_quantiles(vals, num_bins=5)
        output = [float(encode_quantile_bin(v, quantiles)) for v in vals]
    elif t_type == "ratio" and len(inputs) >= 2:
        c1, c2 = inputs[0], inputs[1]
        for r in rows:
            v1 = float(r[c1]) if r.get(c1) and is_number(r[c1]) else 0.0
            v2 = float(r[c2]) if r.get(c2) and is_number(r[c2]) else 0.0
            output.append(v1 / (abs(v2) + 1e-5))
    elif t_type == "product" and len(inputs) >= 2:
        c1, c2 = inputs[0], inputs[1]
        for r in rows:
            v1 = float(r[c1]) if r.get(c1) and is_number(r[c1]) else 0.0
            v2 = float(r[c2]) if r.get(c2) and is_number(r[c2]) else 0.0
            output.append(v1 * v2)
    elif t_type == "frequency_encode" and inputs:
        c = inputs[0]
        counts: Dict[str, int] = {}
        for r in rows:
            val = str(r.get(c, ""))
            counts[val] = counts.get(val, 0) + 1
        total = len(rows) or 1
        output = [counts.get(str(r.get(c, "")), 0) / total for r in rows]
    elif t_type == "target_encode" and inputs:
        c = inputs[0]
        target_sums: Dict[str, float] = {}
        target_counts: Dict[str, int] = {}
        all_targets: List[float] = []
        for r in rows:
            t = float(r[target_column]) if r.get(target_column) and is_number(r[target_column]) else 0.0
            cat = str(r.get(c, ""))
            all_targets.append(t)
            target_sums[cat] = target_sums.get(cat, 0.0) + t
            target_counts[cat] = target_counts.get(cat, 0) + 1
        global_mean = mean(all_targets)
        # Empirical Bayes Laplace smoothing
        smoothing = 5.0
        for r in rows:
            cat = str(r.get(c, ""))
            cnt = target_counts.get(cat, 0)
            sm = target_sums.get(cat, 0.0)
            smoothed = (sm + smoothing * global_mean) / (cnt + smoothing)
            output.append(smoothed)
    elif t_type in ("cyclical_sin", "weekend_indicator") and inputs:
        c = inputs[0]
        for r in rows:
            raw = str(r.get(c, ""))
            try:
                dt = datetime.strptime(raw[:10], "%Y-%m-%d")
                if t_type == "cyclical_sin":
                    output.append(cyclical_sin(dt.month, 12.0))
                else:
                    output.append(1.0 if dt.weekday() in (5, 6) else 0.0)
            except Exception:
                output.append(0.0)
    else:
        # Generic fallback
        if inputs:
            c = inputs[0]
            output = [float(r[c]) if r.get(c) and is_number(r[c]) else 0.0 for r in rows]
        else:
            output = [0.0] * len(rows)

    return output


def evaluate_candidates(
    candidates: List[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    target_column: str,
    collinearity_threshold: float = 0.85,
    min_importance_threshold: float = 0.02,
) -> List[Dict[str, Any]]:
    """Evaluate importance, leakage, and collinearity with deterministic verdicts."""
    target_values: List[float] = [
        float(r[target_column]) if r.get(target_column) and is_number(r[target_column]) else 0.0
        for r in rows
    ]

    computed_features: Dict[str, List[float]] = {}
    correlations: Dict[str, float] = {}

    for c in candidates:
        name = c["name"]
        vals = compute_feature_values(c, rows, target_column)
        computed_features[name] = vals
        corr = pearson_correlation(vals, target_values)
        correlations[name] = round(corr, 4)

    evaluations: List[Dict[str, Any]] = []
    feature_names = [c["name"] for c in candidates]

    for i, c in enumerate(candidates):
        name = c["name"]
        vals = computed_features[name]
        target_corr = correlations[name]
        importance = abs(target_corr)

        # 1. Leakage detection
        leakage = "none"
        if target_column in c.get("input_columns", []):
            leakage = "high_target_inclusion"
        elif abs(target_corr) >= 0.999 and len(vals) > 5:
            leakage = "high_target_redundancy"

        # 2. Multicollinearity detection against preceding features
        collinear_with: List[str] = []
        for prev_name in feature_names[:i]:
            prev_vals = computed_features[prev_name]
            p_corr = pearson_correlation(vals, prev_vals)
            if abs(p_corr) >= collinearity_threshold:
                collinear_with.append(prev_name)

        # 3. Variance check
        feat_std = std_dev(vals)

        # 4. Formulate verdict
        if leakage != "none":
            verdict = "drop"
            reason = f"Data leakage detected: {leakage}. Direct target reflection risks severe training distortion."
        elif feat_std < 1e-6:
            verdict = "drop"
            reason = "Zero variance constant column. Offers zero discriminatory information."
        elif collinear_with:
            verdict = "drop"
            reason = f"Redundant multicollinearity (> {collinearity_threshold}) with {', '.join(collinear_with)}."
        elif importance < min_importance_threshold:
            verdict = "drop"
            reason = f"Negligible target correlation (|r| = {round(importance, 4)} < {min_importance_threshold})."
        else:
            verdict = "keep"
            reason = f"Solid signal (|r| = {round(importance, 4)}), low collinearity risk, and clean transform mechanism."

        evaluations.append({
            "feature_name": name,
            "importance_score": round(importance, 4),
            "correlation_target": target_corr,
            "leakage_risk": leakage,
            "collinearity_with": collinear_with,
            "verdict": verdict,
            "reason": reason,
        })

    return evaluations
