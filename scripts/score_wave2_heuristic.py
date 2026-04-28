#!/usr/bin/env python3
"""
Heuristic scorer for GVP Wave 2 validation subset.

Implements domain-specific scoring that matches Wave 1's observed behaviour:
  - coding          : code execution + doctest comparison
                      fallback reason codes: syntax, empty, exec, wrong_fname,
                      test_error, no_tests, pass
  - math            : extract last number from output, exact-match reference
                      reason format: "math:X vs Y"
  - instruction_following : token overlap >= 0.30
                      reason: always "if"

Steps
-----
1. Validate scorer against wave1_heuristic_labels.csv  (target: >95%)
2. Score wave2_omid_labeled.csv  →  human_labels/wave2_heuristic_labels.csv
3. Create human_labels/wave2_for_judge.csv  (column-renamed for strict_llm_judge_runner.py)

Usage:
    python scripts/score_wave2_heuristic.py
"""

import ast
import concurrent.futures
import contextlib
import csv
import io
import re
import sys
import traceback
from pathlib import Path

EXEC_TIMEOUT = 5  # seconds per code-execution call

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent.parent  # Grader-Benchmark/

WAVE1_PATH    = BASE / "human_labels" / "wave1_heuristic_labels.csv"
WAVE2_PATH    = BASE / "human_labels" / "wave2_omid_labeled.csv"
W2_HEUR_OUT   = BASE / "human_labels" / "wave2_heuristic_labels.csv"
W2_JUDGE_OUT  = BASE / "human_labels" / "wave2_for_judge.csv"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> set:
    """Lower-case word tokens."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def token_overlap(output: str, reference: str) -> float:
    """Recall-style overlap: |tok(out) ∩ tok(ref)| / |tok(ref)|."""
    ref_tok = tokenize(reference)
    if not ref_tok:
        return 0.0
    return len(tokenize(output) & ref_tok) / len(ref_tok)


def extract_code(text: str) -> str:
    """Strip markdown code fences; return raw code string."""
    if not text:
        return ""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _to_num(raw: str):
    """Parse a raw string to int or float; return None on failure."""
    raw = raw.replace(",", "").strip()
    try:
        v = float(raw)
        return int(v) if v == int(v) else v
    except ValueError:
        return None


def extract_last_number(text: str):
    """
    Return the last numeric value from *text* (int or float).
    Handles: LaTeX \\boxed{X}, comma-separated thousands, decimals,
    negative numbers.
    """
    # 1. LaTeX \\boxed{X}  (highest confidence signal)
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        v = _to_num(boxed[-1])
        if v is not None:
            return v

    # 2. Last bare number (strip thousand-commas first)
    cleaned = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", text)
    for raw in reversed(re.findall(r"-?\d+\.?\d*", cleaned)):
        v = _to_num(raw)
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Domain scorers
# ---------------------------------------------------------------------------

def score_math(output: str, reference: str):
    """Exact-match last numeric answer from output against reference."""
    if not output or not output.strip():
        return "fail", "empty"

    # Parse reference
    ref_raw = str(reference).strip().replace(",", "")
    try:
        ref_v = float(ref_raw)
        ref_val = int(ref_v) if ref_v == int(ref_v) else ref_v
    except ValueError:
        ref_val = ref_raw  # keep as string if not numeric

    got = extract_last_number(output)
    if got is None:
        return "fail", f"math:none vs {ref_val}"

    label = "pass" if got == ref_val else "fail"
    return label, f"math:{got} vs {ref_val}"


# --- Coding helpers ---

def _doctest_pairs(source: str):
    """
    Extract (expression, expected_str_or_None) pairs from *source*
    by scanning for '>>>' lines.
    """
    lines = source.splitlines()
    pairs = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(">>>"):
            expr = stripped[3:].strip()
            # Continuation lines (... prefix) – concatenate
            while i + 1 < len(lines) and lines[i + 1].strip().startswith("..."):
                i += 1
                expr += "\n" + lines[i].strip()[3:].strip()
            # Expected: next non-blank line that doesn't start with >>>
            expected = None
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt and not nxt.startswith(">>>") and not nxt.startswith("..."):
                    expected = nxt
            pairs.append((expr, expected))
        i += 1
    return pairs


def score_coding(output: str, reference: str, original_task: str):
    """
    Execute model code and run doctests from original_task.

    Reason codes match Wave 1 observed values:
      empty, syntax, wrong_fname, exec, no_tests, test_error, pass
    """
    if not output or not output.strip():
        return "fail", "empty"

    code = extract_code(output)
    if not code:
        return "fail", "empty"

    # 1. Syntax check
    try:
        compile(code, "<model>", "exec")
    except SyntaxError:
        return "fail", "syntax"

    # 2. Expected function name (from first 'def' in original_task)
    fname_m = re.search(r"def\s+(\w+)\s*\(", original_task)
    expected_fname = fname_m.group(1) if fname_m else None

    # 3. Function name check via AST (no side-effects yet)
    if expected_fname:
        try:
            tree = ast.parse(code)
            defined = {n.name for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef)}
            if defined and expected_fname not in defined:
                return "fail", "wrong_fname"
        except Exception:
            pass  # if AST parse fails here, let exec reveal the error

    # 4. Short-circuit: no doctests → skip exec entirely
    pairs = _doctest_pairs(original_task)
    if not pairs:
        return "pass", "no_tests"

    # 5. Execute code in a fresh namespace (with timeout + suppressed stdout)
    ns: dict = {}

    def _do_exec():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            exec(code, ns)  # noqa: S102

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_do_exec)
        try:
            fut.result(timeout=EXEC_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return "fail", "exec"
        except Exception:
            return "fail", "exec"

    # 6. Double-check the expected function is callable in ns
    if expected_fname and expected_fname not in ns:
        return "fail", "wrong_fname"

    # 7. Run each doctest (also with timeout + suppressed stdout)
    def _run_tests():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            for expr, expected in pairs:
                try:
                    result = eval(expr, ns)  # noqa: S307
                except Exception:
                    return "fail", "exec"
                if expected is not None:
                    result_repr = repr(result)
                    if result_repr.strip() == expected.strip():
                        continue
                    try:
                        exp_val = eval(expected, ns)  # noqa: S307
                        if result != exp_val:
                            return "fail", "test_error"
                    except Exception:
                        if str(result).strip() != expected.strip():
                            return "fail", "test_error"
        return "pass", "pass"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run_tests)
        try:
            return fut.result(timeout=EXEC_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return "fail", "exec"


def score_instruction_following(output: str, reference: str):
    """Token overlap (recall) >= 0.30 -> pass.  Reason always 'if'."""
    if not output or not output.strip():
        return "fail", "if"
    label = "pass" if token_overlap(output, reference) >= 0.30 else "fail"
    return label, "if"


# ---------------------------------------------------------------------------
# Row dispatcher
# ---------------------------------------------------------------------------

def score_row(row: dict, output_col: str):
    output       = (row.get(output_col) or "").strip()
    reference    = (row.get("reference") or "").strip()
    original_task = (row.get("original_task") or "").strip()
    domain       = (row.get("domain") or "").lower().strip()

    if domain == "math":
        return score_math(output, reference)
    elif domain == "coding":
        return score_coding(output, reference, original_task)
    else:
        return score_instruction_following(output, reference)


# ---------------------------------------------------------------------------
# Step 1 – Validate against Wave 1
# ---------------------------------------------------------------------------

WAVE1_COLS = [
    ("gpt54_output",    "gpt54"),
    ("llama70b_output", "llama70b"),
    ("qwen72b_output",  "qwen72b"),
]


def validate_wave1() -> float:
    with open(WAVE1_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = correct = 0
    mismatches = []

    n_rows = len(rows)
    for r_idx, row in enumerate(rows):
        if (r_idx + 1) % 10 == 0 or r_idx == n_rows - 1:
            print(f"  validating row {r_idx+1}/{n_rows} ...", end="\r")
        domain = (row.get("domain") or "").lower()
        for out_col, model in WAVE1_COLS:
            ref_label = (row.get(f"heuristic_{model}") or "").strip().lower()
            if not ref_label:
                continue

            pred_label, pred_reason = score_row(row, out_col)

            # "pass" and "no_tests" both count as passing
            ref_pass  = (ref_label  != "fail")
            pred_pass = (pred_label != "fail")

            total += 1
            if ref_pass == pred_pass:
                correct += 1
            else:
                mismatches.append({
                    "id":     row.get("example_id", "?"),
                    "model":  model,
                    "domain": domain,
                    "ref":    ref_label,
                    "pred":   pred_label,
                    "reason": pred_reason,
                })

    acc = correct / total if total else 0.0
    print(f"\nWave 1 Validation  {correct}/{total}  ({acc:.1%})")

    if mismatches:
        print(f"  {len(mismatches)} mismatch(es) — first 20:")
        for m in mismatches[:20]:
            print(
                f"    [{m['domain']:22s}] {m['id'][:52]}  "
                f"{m['model']:10s}  ref={m['ref']:8s}  "
                f"pred={m['pred']:8s}  reason={m['reason']}"
            )

    # Per-domain breakdown
    from collections import Counter
    by_domain: dict = {}
    for row in rows:
        d = (row.get("domain") or "?").lower()
        if d not in by_domain:
            by_domain[d] = {"total": 0, "correct": 0}
        for out_col, model in WAVE1_COLS:
            ref_label = (row.get(f"heuristic_{model}") or "").strip().lower()
            if not ref_label:
                continue
            pred_label, _ = score_row(row, out_col)
            ref_pass  = (ref_label  != "fail")
            pred_pass = (pred_label != "fail")
            by_domain[d]["total"] += 1
            if ref_pass == pred_pass:
                by_domain[d]["correct"] += 1

    print("\n  Per-domain accuracy:")
    for d, s in sorted(by_domain.items()):
        da = s["correct"] / s["total"] if s["total"] else 0
        print(f"    {d:22s}: {s['correct']:3d}/{s['total']:3d}  ({da:.1%})")

    return acc


# ---------------------------------------------------------------------------
# Step 2 – Score Wave 2
# ---------------------------------------------------------------------------

WAVE2_COLS = [
    ("gpt54_output",    "gpt54"),
    ("llama70b_output", "llama70b"),
    ("qwen72b_output",  "qwen72b"),
]


def score_wave2():
    with open(WAVE2_PATH, encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        orig_fields = list(reader.fieldnames)
        rows        = list(reader)

    new_fields = orig_fields + [
        "heuristic_gpt54",    "heuristic_reason_gpt54",
        "heuristic_llama70b", "heuristic_reason_llama70b",
        "heuristic_qwen72b",  "heuristic_reason_qwen72b",
    ]

    n_rows = len(rows)
    for r_idx, row in enumerate(rows):
        print(f"  scoring wave2 row {r_idx+1}/{n_rows} ...", end="\r")
        for out_col, model in WAVE2_COLS:
            label, reason = score_row(row, out_col)
            row[f"heuristic_{model}"]        = label
            row[f"heuristic_reason_{model}"] = reason
    print()

    with open(W2_HEUR_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWave 2 heuristic labels  →  {W2_HEUR_OUT.name}")
    print("\nHeuristic summary (Wave 2):")
    for _, model in WAVE2_COLS:
        passes = sum(1 for r in rows if r[f"heuristic_{model}"] == "pass")
        fails  = sum(1 for r in rows if r[f"heuristic_{model}"] == "fail")
        print(f"  {model:12s}: {passes:3d} pass  {fails:3d} fail")


# ---------------------------------------------------------------------------
# Step 3 – Create wave2_for_judge.csv (column rename for judge runner)
# ---------------------------------------------------------------------------

def prepare_judge_csv():
    """
    Create a copy of wave2_omid_labeled.csv with the model-output columns
    renamed to match what strict_llm_judge_runner.py expects:
        gpt54_output   → gpt54_output2
        llama70b_output → llama70b_output2
        qwen72b_output  → qwen72b_output2
    """
    with open(WAVE2_PATH, encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        orig_fields = list(reader.fieldnames)
        rows        = list(reader)

    rename = {
        "gpt54_output":    "gpt54_output2",
        "llama70b_output": "llama70b_output2",
        "qwen72b_output":  "qwen72b_output2",
    }
    new_fields = [rename.get(c, c) for c in orig_fields]

    new_rows = []
    for row in rows:
        new_row = {rename.get(k, k): v for k, v in row.items()}
        new_rows.append(new_row)

    with open(W2_JUDGE_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(new_rows)

    missing = {
        "gpt-5.4":   sum(1 for r in rows if not (r.get("gpt54_output") or "").strip()),
        "llama-70b": sum(1 for r in rows if not (r.get("llama70b_output") or "").strip()),
        "qwen-72b":  sum(1 for r in rows if not (r.get("qwen72b_output") or "").strip()),
    }
    print(f"\nJudge-ready CSV         →  {W2_JUDGE_OUT.name}")
    print("Missing outputs (→ auto-FAIL by judge runner):")
    for m, n in missing.items():
        print(f"  {m}: {n} missing")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 1 — Validate heuristic scorer against Wave 1")
    print("=" * 60)
    acc = validate_wave1()
    if acc < 0.95:
        print(f"\n⚠  Accuracy {acc:.1%} is below the 95% target.")
        print("   Proceeding to Wave 2 scoring anyway; review mismatches above.")
    else:
        print(f"\n✓  Accuracy {acc:.1%} meets the ≥95% target.")

    print("\n" + "=" * 60)
    print("Step 2 — Score Wave 2 heuristics")
    print("=" * 60)
    score_wave2()

    print("\n" + "=" * 60)
    print("Step 3 — Prepare judge input CSV")
    print("=" * 60)
    prepare_judge_csv()

    print("\n" + "=" * 60)
    print("Next: run the reference judge")
    print("=" * 60)
    print(
        "  $env:OPENAI_API_KEY = 'sk-...'\n"
        "  python scripts/strict_llm_judge_runner.py \\\n"
        f"    --input_csv {W2_JUDGE_OUT.relative_to(BASE)} \\\n"
        "    --output_jsonl judge_outputs/judge_gpt4o_calibrated_wave2.jsonl \\\n"
        "    --judge_model gpt-4o"
    )


if __name__ == "__main__":
    main()
