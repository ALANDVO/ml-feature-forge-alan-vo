#!/usr/bin/env python3
"""ML Feature Forge CLI entrypoint.

Profiles tabular datasets, proposes candidate features, evaluates signal importance,
and generates production feature pipelines.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Ensure repository root is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.feature_engine import (
    evaluate_candidates,
    generate_deterministic_candidates,
    parse_csv_data,
    profile_dataset,
)
from backend.app.services.pipeline_generator import generate_python_pipeline_code


def propose_cmd(args: argparse.Namespace) -> List[Dict[str, Any]]:
    with open(args.data, "r") as f:
        content = f.read()
    cols, rows = parse_csv_data(content)
    if args.target not in cols:
        print(f"Error: Target column '{args.target}' not found in dataset columns: {cols}")
        return []

    profile = profile_dataset(cols, rows, args.target)
    candidates = generate_deterministic_candidates(
        profile=profile,
        target_column=args.target,
        include_interactions=True,
        include_encodings=True,
        include_nonlinear=True,
        include_datetime=True,
    )
    selected = candidates[: args.count]

    print(f"\n============================================================")
    print(f"PROPOSED CANDIDATE FEATURES ({len(selected)} total)")
    print(f"============================================================")
    for i, c in enumerate(selected, 1):
        print(f"  {i}. {c['name']} ({c['transform_type']})")
        print(f"     Inputs:     {', '.join(c.get('input_columns', []))}")
        print(f"     Expression: {c['expression']}")
        print(f"     Hypothesis: {c['hypothesis'][:100]}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(selected, f, indent=2)
        print(f"Candidate features saved to {args.output}")
    return selected


def evaluate_cmd(args: argparse.Namespace) -> List[Dict[str, Any]]:
    with open(args.data, "r") as f:
        content = f.read()
    cols, rows = parse_csv_data(content)
    if not args.features:
        print("Error: Missing --features argument pointing to candidate JSON file.")
        return []

    with open(args.features, "r") as f:
        candidates = json.load(f)

    evals = evaluate_candidates(
        candidates=candidates,
        rows=rows,
        target_column=args.target,
        collinearity_threshold=0.85,
        min_importance_threshold=0.02,
    )

    print(f"\n============================================================")
    print(f"FEATURE EVALUATION RESULTS ({len(evals)} features)")
    print(f"============================================================")
    for e in evals:
        icon = "[KEEP]" if e["verdict"] == "keep" else "[DROP]"
        print(f"  {icon} {e['feature_name']:<30} Signal={e['importance_score']:<6} {e['reason'][:75]}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(evals, f, indent=2)
        print(f"Evaluation results saved to {args.output}")
    return evals


def forge_cmd(args: argparse.Namespace) -> None:
    with open(args.data, "r") as f:
        content = f.read()
    cols, rows = parse_csv_data(content)
    profile = profile_dataset(cols, rows, args.target)

    candidates = generate_deterministic_candidates(
        profile=profile,
        target_column=args.target,
        include_interactions=True,
        include_encodings=True,
        include_nonlinear=True,
        include_datetime=True,
    )

    evals = evaluate_candidates(
        candidates=candidates,
        rows=rows,
        target_column=args.target,
        collinearity_threshold=0.85,
        min_importance_threshold=0.02,
    )

    kept_names = {e["feature_name"] for e in evals if e["verdict"] == "keep"}
    kept_features = [c for c in candidates if c["name"] in kept_names] or candidates[:5]

    code = generate_python_pipeline_code(
        selected_features=kept_features,
        target_column=args.target,
        dataset_name=os.path.basename(args.data),
    )

    print(f"\n============================================================")
    print(f"GENERATED PIPELINE CODE ({len(kept_features)} features)")
    print(f"============================================================")
    print(code)

    if args.output:
        with open(args.output, "w") as f:
            f.write(code)
        print(f"Pipeline saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ml-feature-forge",
        description="ML feature engineering toolkit for tabular datasets",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("propose", help="Propose candidate features")
    pr.add_argument("--data", required=True, help="Path to input CSV dataset")
    pr.add_argument("--target", required=True, help="Target column name")
    pr.add_argument("--count", type=int, default=12, help="Number of features to propose")
    pr.add_argument("--output", default=None, help="Output JSON path")
    pr.set_defaults(fn=propose_cmd)

    ev = sub.add_parser("evaluate", help="Evaluate feature importance and multicollinearity")
    ev.add_argument("--data", required=True, help="Path to input CSV dataset")
    ev.add_argument("--target", required=True, help="Target column name")
    ev.add_argument("--features", required=True, help="JSON file containing candidate features")
    ev.add_argument("--output", default=None, help="Output JSON path")
    ev.set_defaults(fn=evaluate_cmd)

    fg = sub.add_parser("forge", help="Full pipeline: propose, evaluate, and generate code")
    fg.add_argument("--data", required=True, help="Path to input CSV dataset")
    fg.add_argument("--target", required=True, help="Target column name")
    fg.add_argument("--output", default=None, help="Output python file path")
    fg.set_defaults(fn=forge_cmd)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
