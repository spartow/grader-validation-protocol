#!/usr/bin/env python3
"""
Analyze full-benchmark judge results and produce dual-scored tables.

Usage:
    python3 analyze_full_results.py

Reads: full_benchmark_judge_results.jsonl (from rescore_full_2k.py)
Produces: console output + LaTeX tables ready to paste
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    # Find results file
    for p in [Path("full_benchmark_judge_results.jsonl"),
              Path("Grader-Benchmark/full_benchmark_judge_results.jsonl")]:
        if p.exists():
            results_path = p
            break
    else:
        print("ERROR: full_benchmark_judge_results.jsonl not found.")
        print("Run rescore_full_2k.py first.")
        return

    records = [json.loads(line) for line in open(results_path, encoding="utf-8")]
    print(f"Loaded {len(records)} judge results")

    # Check completeness
    models = defaultdict(int)
    for r in records:
        models[r["model"]] += 1
    print(f"Per-model counts: {dict(models)}")

    errors = sum(1 for r in records if r["failure_type"] == "judge_error")
    print(f"Judge errors: {errors}")

    # === OVERALL PASS RATES ===
    print(f"\n{'='*60}")
    print("OVERALL PASS RATES (Reference Judge)")
    print(f"{'='*60}")

    display = {"gpt-5.4": "GPT-5.4", "llama-70b": "LLaMA 70B", "qwen-72b": "Qwen 72B"}

    for model in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        sub = [r for r in records if r["model"] == model]
        passes = sum(1 for r in sub if r["label"] == "pass")
        total = len(sub)
        rate = 100 * passes / total if total else 0
        print(f"  {display.get(model, model)}: {passes}/{total} = {rate:.1f}%")

    # === BY ATTACK ===
    print(f"\n{'='*60}")
    print("BY ATTACK CONDITION")
    print(f"{'='*60}")

    for model in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        sub = [r for r in records if r["model"] == model]
        by_attack = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in sub:
            a = r["attack_name"]
            by_attack[a]["total"] += 1
            if r["label"] == "pass":
                by_attack[a]["pass"] += 1

        print(f"\n  {display.get(model, model)}:")
        for attack in sorted(by_attack.keys()):
            d = by_attack[attack]
            rate = 100 * d["pass"] / d["total"] if d["total"] else 0
            print(f"    {attack}: {d['pass']}/{d['total']} = {rate:.1f}%")

    # === BY DOMAIN ===
    print(f"\n{'='*60}")
    print("BY DOMAIN")
    print(f"{'='*60}")

    for model in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        sub = [r for r in records if r["model"] == model]
        by_domain = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in sub:
            d = r["domain"]
            by_domain[d]["total"] += 1
            if r["label"] == "pass":
                by_domain[d]["pass"] += 1

        print(f"\n  {display.get(model, model)}:")
        for dom in sorted(by_domain.keys()):
            d = by_domain[dom]
            rate = 100 * d["pass"] / d["total"] if d["total"] else 0
            print(f"    {dom}: {d['pass']}/{d['total']} = {rate:.1f}%")

    # === FAILURE TAXONOMY ===
    print(f"\n{'='*60}")
    print("FAILURE TAXONOMY")
    print(f"{'='*60}")
    ftypes = defaultdict(int)
    for r in records:
        ftypes[r["failure_type"]] += 1
    for ft, count in sorted(ftypes.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {count}")

    # === LATEX TABLE ===
    print(f"\n{'='*60}")
    print("LATEX: Dual-scored table (paste into paper alongside Table 4)")
    print(f"{'='*60}")

    # Known heuristic values from paper Table 4
    heuristic = {
        "gpt-5.4":   {"overall": 85.8, "clean": 90.4, "pi": 78.5, "cm": 88.5},
        "llama-70b":  {"overall": 53.4, "clean": 59.2, "pi": 49.7, "cm": 51.1},
        "qwen-72b":   {"overall": 83.2, "clean": 90.3, "pi": 77.9, "cm": 81.4},
    }

    print(r"""
\begin{table}[t]
\centering
\caption{Full-benchmark pass rates (\%) under both scoring pipelines (2{,}001 examples).}
\label{tab:full2k_dual}
\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{\textbf{Heuristic}} & \multicolumn{3}{c}{\textbf{Ref.\ Judge}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Model} & Overall & $\Delta_{\mathrm{PI}}$ & $\Delta_{\mathrm{CM}}$ & Overall & $\Delta_{\mathrm{PI}}$ & $\Delta_{\mathrm{CM}}$ \\
\midrule""")

    for model in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        name = display.get(model, model)
        h = heuristic[model]

        sub = [r for r in records if r["model"] == model]
        by_attack = defaultdict(lambda: {"pass": 0, "total": 0})
        for r in sub:
            by_attack[r["attack_name"]]["total"] += 1
            if r["label"] == "pass":
                by_attack[r["attack_name"]]["pass"] += 1

        j_overall = 100 * sum(1 for r in sub if r["label"] == "pass") / len(sub)
        j_clean = 100 * by_attack["clean"]["pass"] / by_attack["clean"]["total"] if by_attack["clean"]["total"] else 0
        j_pi = 100 * by_attack["prompt_injection"]["pass"] / by_attack["prompt_injection"]["total"] if by_attack["prompt_injection"]["total"] else 0
        j_cm = 100 * by_attack["context_manipulation"]["pass"] / by_attack["context_manipulation"]["total"] if by_attack["context_manipulation"]["total"] else 0

        h_dpi = h["pi"] - h["clean"]
        h_dcm = h["cm"] - h["clean"]
        j_dpi = j_pi - j_clean
        j_dcm = j_cm - j_clean

        print(f"{name:10s} & {h['overall']:.1f} & {h_dpi:+.1f} & {h_dcm:+.1f} & {j_overall:.1f} & {j_dpi:+.1f} & {j_dcm:+.1f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table}""")


if __name__ == "__main__":
    main()
