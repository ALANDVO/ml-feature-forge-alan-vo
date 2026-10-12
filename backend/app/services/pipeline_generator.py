"""Pipeline code generation and execution validation."""
from typing import Any, Dict, List, Tuple


def generate_python_pipeline_code(
    selected_features: List[Dict[str, Any]],
    target_column: str,
    dataset_name: str,
) -> str:
    lines = [
        "#!/usr/bin/env python3",
        f'"""Production Feature Engineering Pipeline for {dataset_name}.',
        f'Generated automatically by ML Feature Forge (Author: Alan Vo).',
        f'Target column (excluded from output): {target_column}',
        f'Number of active engineered features: {len(selected_features)}',
        '"""',
        "import numpy as np",
        "import pandas as pd",
        "",
        "",
        "def build_features(df: pd.DataFrame) -> pd.DataFrame:",
        '    """Transform input raw DataFrame into engineered feature matrix."""',
        "    out = pd.DataFrame(index=df.index)",
        "",
    ]

    for feat in selected_features:
        name = feat["name"]
        expr = feat["expression"]
        hypo = feat.get("hypothesis", "")
        t_type = feat.get("transform_type", "")
        lines.append(f"    # Feature: {name} ({t_type})")
        if hypo:
            lines.append(f"    # Hypothesis: {hypo[:90]}")
        lines.append(f"    out['{name}'] = {expr}")
        lines.append("")

    lines.extend([
        "    # Fill any numerical residual nulls",
        "    out = out.fillna(0.0)",
        "    return out",
        "",
        "",
        "if __name__ == '__main__':",
        "    print('Feature pipeline compiled successfully.')",
        "",
    ])

    return "\n".join(lines)


def generate_pipeline_json_spec(
    selected_features: List[Dict[str, Any]],
    target_column: str,
    dataset_name: str,
) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "target_column": target_column,
        "feature_count": len(selected_features),
        "features": [
            {
                "name": f["name"],
                "transform_type": f.get("transform_type", "custom"),
                "input_columns": f.get("input_columns", []),
                "expression": f["expression"],
                "hypothesis": f.get("hypothesis", ""),
                "expected_signal": f.get("expected_signal", "moderate"),
            }
            for f in selected_features
        ],
    }
