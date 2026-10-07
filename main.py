#!/usr/bin/env python3
# ml-feature-forge — ML feature engineering toolkit that takes raw data plus a target column, auto-generates candidate features using LLM reasoning, evaluates their importance, and outputs a feature pipeline ready for model training.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_client import LLM
def _load_table(path):
    """Load CSV into (columns, dtypes, samples, row_count)."""
    import csv as _csv
    with open(path, newline="") as f:
        reader = _csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    n = len(rows)
    dtypes = {}
    for i, col in enumerate(header):
        vals = [r[i] for r in rows[:200] if i < len(r) and r[i] not in ("", "null", "None", "NA")]
        if not vals:
            dtypes[col] = "null"
        elif all(_is_num(v) for v in vals):
            dtypes[col] = "numeric"
        elif all(_is_date(v) for v in vals):
            dtypes[col] = "datetime"
        elif len(set(vals)) / len(vals) < 0.5:
            dtypes[col] = "categorical"
        else:
            dtypes[col] = "text"
    samples = {col: [r[i] if i < len(r) else "" for r in rows[:5]] for i, col in enumerate(header)}
    return header, dtypes, samples, n

def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False

def _is_date(v):
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            datetime.strptime(v[:19], fmt)
            return True
        except ValueError:
            continue
    return False

def propose(args):
    """Generate candidate feature hypotheses."""
    llm = LLM()
    cols, dtypes, samples, n = _load_table(args.data)
    target = args.target
    if target not in cols:
        print(f"Target '{target}' not found. Available: {cols}")
        return []
    print(f"Proposing features for {n} rows, target='{target}' ({dtypes[target]})\n")

    col_profile = json.dumps({c: {"type": dtypes[c], "samples": samples[c][:3]} for c in cols if c != target}, indent=2)

    result = llm.generate(
        f"Dataset profile ({n} rows):\n{col_profile}\n\n"
        f"Target: {target} (type: {dtypes[target]})\n\n"
        "Propose candidate engineered features. For each, output a JSON object: "
        "{\"name\": \"feature_name\", \"hypothesis\": \"why this predicts the target\", "
        "\"inputs\": [\"source_cols\"], \"transform\": \"exact description: e.g. 'log1p(age) + 0.1*income', 'one-hot', 'lag_7', 'rolling_mean_30', 'sin(2*pi*hour/24)', 'interaction: a*b', 'bucket: [0,10,50,100]'\", "
        "\"expected_signal\": \"strong|moderate|weak\", \"risk\": \"leakage|overfit|none\"}\n"
        f"Generate {args.count} features. Rules: "
        "no feature may use the target column; avoid pure duplicates of existing columns; "
        "include at least 3 interaction features, 2 encoding features for categoricals, 1 nonlinearity (log/sqrt/bin); "
        "if any datetime column exists, include lag and cyclical features. Output a JSON array.",
        system="You are an ML engineer with deep feature engineering experience. Every hypothesis must reference a specific mechanism (monotonic relationship, periodicity, interaction effect, scale compression). Name features like a production data scientist: snake_case, descriptive."
    )

    features = _parse_json_array(result)
    print(f"{'='*60}")
    print(f"CANDIDATE FEATURES: {len(features)}")
    print(f"{'='*60}")
    for i, f in enumerate(features, 1):
        sig_icon = {"strong": "🎯", "moderate": "📈", "weak": "📉"}.get(f.get("expected_signal", "weak"), "❓")
        risk = f" ⚠️{f['risk']}" if f.get("risk") not in (None, "none") else ""
        print(f"\n  {i}. {sig_icon} {f.get('name', '?')}{risk}")
        print(f"     From:      {', '.join(f.get('inputs', []))}")
        print(f"     Transform: {f.get('transform', '?')}")
        print(f"     Hypothesis: {f.get('hypothesis', '?')[:120]}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(features, f, indent=2)
        print(f"\nSaved to {args.output}")
    return features

def forge(args):
    """Full pipeline: propose → implement → evaluate → rank."""
    llm = LLM()
    cols, dtypes, samples, n = _load_table(args.data)
    target = args.target
    if target not in cols:
        print(f"Target '{target}' not found. Available: {cols}")
        return
    print(f"Feature forge: {n} rows, target='{target}'\n")

    # Step 1: propose
    print("Step 1/3 — Proposing candidate features...")
    col_profile = json.dumps({c: {"type": dtypes[c], "samples": samples[c][:3]} for c in cols if c != target}, indent=2)
    propose_result = llm.generate(
        f"Dataset profile ({n} rows):\n{col_profile}\n\nTarget: {target} ({dtypes[target]})\n\n"
        "Propose 12 candidate engineered features. For each: "
        "{\"name\": \"...\", \"hypothesis\": \"...\", \"inputs\": [...], \"transform\": \"...\", \"expected_signal\": \"strong|moderate|weak\", \"risk\": \"leakage|overfit|none\"}\n"
        "Include interactions, encodings, nonlinearities, and time features where applicable. JSON array.",
        system="You are an ML engineer. Concrete transforms, production-grade naming, no target leakage."
    )
    candidates = _parse_json_array(propose_result)
    print(f"  {len(candidates)} candidates proposed\n")

    # Step 2: implement as Python code
    print("Step 2/3 — Generating feature implementation code...")
    candidate_spec = json.dumps(candidates[:15], indent=2)
    code = llm.generate(
        f"Feature specs to implement:\n{candidate_spec}\n\n"
        f"Available columns: {cols[:-1] if cols[-1] == target else cols}\n"
        f"Target column (exclude from features): {target}\n"
        f"Column types: {json.dumps({c: dtypes[c] for c in cols if c != target})}\n\n"
        "Write a Python function `build_features(df: pd.DataFrame) -> pd.DataFrame` that implements ALL these features. "
        "Use pandas. Handle missing values (fillna/0). Return a DataFrame with ONLY the engineered feature columns (no target, no raw inputs). "
        "Include a module-level docstring listing each feature and its rationale. "
        "Code must run without warnings. No comments like '# TODO'.",
        system="You write production data pipelines. Vectorized pandas only — no row-wise apply for numeric transforms. Safe fillna. Deterministic output."
    )
    print("  Code generated.\n")

    # Step 3: LLM evaluation of feature quality
    print("Step 3/3 — Evaluating feature quality...")
    evaluation = llm.generate(
        f"Proposed features:\n{candidate_spec}\n\n"
        f"Column types: {json.dumps({c: dtypes[c] for c in cols if c != target})}\n\n"
        "Evaluate each feature. For each: "
        "{\"name\": \"...\", \"leakage_risk\": \"none|low|high\", \"multicollinearity_risk\": \"with which features\", \"information_value\": \"high|medium|low\", \"verdict\": \"keep|drop\", \"reason\": \"...\"}\n"
        "Drop features that are: near-duplicates, target-leaking, or constant. Output a JSON array covering all features.",
        system="You are an ML reviewer. Be decisive: keep only features with clear information value. Two features encoding the same signal → drop one."
    )
    evals = _parse_json_array(evaluation)
    kept = [e for e in evals if e.get("verdict") == "keep"]
    dropped = [e for e in evals if e.get("verdict") != "keep"]

    print(f"\n{'='*60}")
    print(f"EVALUATION: {len(kept)} kept, {len(dropped)} dropped")
    print(f"{'='*60}")
    for e in kept:
        print(f"  ✅ {e.get('name', '?')} — IV={e.get('information_value', '?')}, {e.get('reason', '')[:80]}")
    for e in dropped:
        print(f"  ❌ {e.get('name', '?')} — {e.get('reason', '')[:80]}")

    # Output the pipeline code
    if args.output:
        with open(args.output, "w") as f:
            f.write('#!/usr/bin/env python3\n')
            f.write(f'"""Feature pipeline for target: {target}\n')
            f.write(f'Generated by ml-feature-forge. {len(kept)} features kept, {len(dropped)} dropped.\n""\n')
            f.write('\nimport pandas as pd\n\n')
            # Extract code block from LLM response
            code_block = code
            if '```python' in code_block:
                code_block = code_block.split('```python', 1)[1].split('```', 1)[0]
            elif '```' in code_block:
                code_block = code_block.split('```', 1)[1].split('```', 1)[0]
            f.write(code_block.strip() + '\n\n')
            f.write(f'# Dropped features and reasons:\n')
            for e in dropped:
                f.write(f'#   - {e.get("name", "?")}: {e.get("reason", "n/a")}\n')
        print(f"\nPipeline saved to {args.output}")

    # Print the code to stdout always
    print(f"\n{'='*60}")
    print(f"FEATURE PIPELINE CODE")
    print(f"{'='*60}")
    print(code)

    if args.output:
        with open(args.output + ".json", "w") as f:
            json.dump({"kept": kept, "dropped": dropped, "candidates": candidates, "code": code, "generated_at": datetime.now().isoformat()}, f, indent=2)
        print(f"\nFull report saved to {args.output}.json")
    return kept

def evaluate(args):
    """Evaluate a saved feature list against the data."""
    llm = LLM()
    cols, dtypes, samples, n = _load_table(args.data)
    target = args.target
    if not args.features:
        print("Provide --features (JSON file from propose/forge)")
        return
    with open(args.features) as f:
        features = json.load(f)
    print(f"Evaluating {len(features)} features against {n} rows...\n")

    col_profile = json.dumps({c: {"type": dtypes[c], "samples": samples[c][:3]} for c in cols if c != target}, indent=2)

    result = llm.generate(
        f"Dataset: {n} rows\nColumns: {col_profile}\nTarget: {target}\n\n"
        f"Features to evaluate:\n{json.dumps(features, indent=2)}\n\n"
        "For each feature, assess:\n"
        "1. Leakage: does it use the target or post-outcome data? "
        "2. Multicollinearity: is it near-duplicate of another feature in the list? "
        "3. Information value: high/medium/low — based on the hypothesis and column semantics. "
        "4. Verdict: keep or drop.\n"
        "Output a JSON array: [{\"name\": \"...\", \"leakage\": \"...\", \"collinear_with\": \"...\", \"iv\": \"...\", \"verdict\": \"keep|drop\", \"reason\": \"...\"}]",
        system="You are an ML reviewer. Be decisive. Two features with the same signal → drop the weaker one. Leakage is always a drop."
    )
    evals = _parse_json_array(result)
    for e in evals:
        icon = "✅" if e.get("verdict") == "keep" else "❌"
        print(f"  {icon} {e.get('name', '?'):<30} IV={e.get('iv', '?'):<8} {e.get('reason', '')[:80]}")
    kept = sum(1 for e in evals if e.get("verdict") == "keep")
    print(f"\nKept: {kept}/{len(evals)}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(evals, f, indent=2)
        print(f"Saved to {args.output}")
    return evals

def _parse_json_array(raw):
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start:end + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []

def main():
    import argparse
    p = argparse.ArgumentParser(prog="ml-feature-forge", description="ML feature engineering toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("propose", help="Propose candidate features")
    pr.add_argument("--data", required=True); pr.add_argument("--target", required=True)
    pr.add_argument("--count", type=int, default=12); pr.add_argument("--output", default=None)
    pr.set_defaults(fn=propose)

    f = sub.add_parser("forge", help="Full pipeline: propose → implement → evaluate")
    f.add_argument("--data", required=True); f.add_argument("--target", required=True)
    f.add_argument("--output", default=None)
    f.set_defaults(fn=forge)

    e = sub.add_parser("evaluate", help="Evaluate a saved feature list")
    e.add_argument("--data", required=True); e.add_argument("--target", required=True)
    e.add_argument("--features", required=True); e.add_argument("--output", default=None)
    e.set_defaults(fn=evaluate)

    args = p.parse_args()
    args.fn(args)

if __name__ == '__main__':
    main()
