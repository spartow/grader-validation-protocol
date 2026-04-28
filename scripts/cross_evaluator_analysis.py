#!/usr/bin/env python3
"""
Cross-evaluator analysis across all annotators and automated judges.

Compares: Soraya, Omid, Ava (human), Heuristic, GPT-4o calibrated, Claude Sonnet 4 uncalibrated
Over: 180 examples × 3 models = 540 judgments

Usage:
    python scripts/cross_evaluator_analysis.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations

BASE = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_human_labels():
    """Load all human labels from wave1 + wave2 CSVs. Returns dict: (example_id, model) -> {evaluator: label}"""
    labels = defaultdict(dict)

    # Wave 1 — Soraya labels in human_label_*, Ava in ava_label_*
    w1_path = BASE / "human_labels" / "wave1_heuristic_labels.csv"
    with open(w1_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["example_id"]
            for mk in ["gpt54", "llama70b", "qwen72b"]:
                model_key = {"gpt54": "gpt-5.4", "llama70b": "llama-70b", "qwen72b": "qwen-72b"}[mk]
                key = (eid, model_key)
                # Soraya labels
                sl = (row.get(f"human_label_{mk}") or "").strip().lower()
                if sl:
                    labels[key]["soraya"] = sl
                # Ava labels
                hl = (row.get(f"ava_label_{mk}") or "").strip().lower()
                if hl:
                    labels[key]["ava"] = hl
                # Heuristic
                heur = (row.get(f"heuristic_{mk}") or "").strip().lower()
                if heur:
                    labels[key]["heuristic"] = heur

    # Wave 2 — Soraya, Omid, Ava
    w2_path = BASE / "human_labels" / "wave2_omid_labeled.csv"
    with open(w2_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["example_id"]
            for mk in ["gpt54", "llama70b", "qwen72b"]:
                model_key = {"gpt54": "gpt-5.4", "llama70b": "llama-70b", "qwen72b": "qwen-72b"}[mk]
                key = (eid, model_key)
                for annotator, prefix in [("soraya", "soraya_label_"), ("omid", "omid_label_"), ("ava", "ava_label_")]:
                    val = (row.get(f"{prefix}{mk}") or "").strip().lower()
                    if val:
                        labels[key][annotator] = val

    # Wave 2 heuristic
    wh_path = BASE / "human_labels" / "wave2_heuristic_labels.csv"
    with open(wh_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row["example_id"]
            for mk in ["gpt54", "llama70b", "qwen72b"]:
                model_key = {"gpt54": "gpt-5.4", "llama70b": "llama-70b", "qwen72b": "qwen-72b"}[mk]
                key = (eid, model_key)
                heur = (row.get(f"heuristic_{mk}") or "").strip().lower()
                if heur:
                    labels[key]["heuristic"] = heur

    return labels


def load_judge_labels():
    """Load judge outputs. Returns dict: (example_id, model) -> {judge_name: label}"""
    labels = defaultdict(dict)

    judge_files = {
        "gpt4o_calibrated": [
            BASE / "judge_outputs" / "judge_gpt4o_calibrated.jsonl",
            BASE / "judge_outputs" / "judge_gpt4o_calibrated_wave2.jsonl",
        ],
        "claude_uncalibrated": [
            BASE / "judge_outputs" / "judge_gemini_uncalibrated.jsonl",
        ],
    }

    for judge_name, paths in judge_files.items():
        for p in paths:
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key = (r["example_id"], r["model"])
                    labels[key][judge_name] = r["label"]

    return labels


def load_metadata():
    """Load domain and attack info for each example."""
    meta = {}
    for csv_name in ["wave1_heuristic_labels.csv", "wave2_omid_labeled.csv"]:
        path = BASE / "human_labels" / csv_name
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                eid = row["example_id"]
                meta[eid] = {
                    "domain": (row.get("domain") or "").lower(),
                    "attack_name": (row.get("attack_name") or "").lower(),
                }
    return meta


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_agreement(labels_a, labels_b, keys):
    """Compute agreement rate between two label lists."""
    agree = total = 0
    for k in keys:
        a = labels_a.get(k)
        b = labels_b.get(k)
        if a and b:
            total += 1
            if a == b:
                agree += 1
    return agree, total


def compute_pass_rate(labels, keys):
    """Compute pass rate for a set of labels."""
    passes = total = 0
    for k in keys:
        v = labels.get(k)
        if v:
            total += 1
            if v == "pass":
                passes += 1
    return passes, total


def majority_vote(evaluators, key):
    """Compute majority vote from multiple evaluators."""
    votes = [evaluators[e].get(key) for e in evaluators if evaluators[e].get(key)]
    if not votes:
        return None
    pass_count = sum(1 for v in votes if v == "pass")
    return "pass" if pass_count > len(votes) / 2 else "fail"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    human_labels = load_human_labels()
    judge_labels = load_judge_labels()
    metadata = load_metadata()

    # Combine all evaluators
    all_evaluators = ["soraya", "omid", "ava", "heuristic", "gpt4o_calibrated", "claude_uncalibrated"]
    display_names = {
        "soraya": "Soraya", "omid": "Omid", "ava": "Ava",
        "heuristic": "Heuristic", "gpt4o_calibrated": "GPT-4o (cal.)",
        "claude_uncalibrated": "Claude (uncal.)",
    }

    # Build unified label dict: evaluator -> {(eid, model): label}
    unified = {e: {} for e in all_evaluators}
    all_keys = set(human_labels.keys()) | set(judge_labels.keys())

    for k in all_keys:
        for e in ["soraya", "omid", "ava", "heuristic"]:
            v = human_labels.get(k, {}).get(e)
            if v:
                unified[e][k] = v
        for e in ["gpt4o_calibrated", "claude_uncalibrated"]:
            v = judge_labels.get(k, {}).get(e)
            if v:
                unified[e][k] = v

    # ── Coverage ──────────────────────────────────────────────────
    print("=" * 70)
    print("EVALUATOR COVERAGE")
    print("=" * 70)
    for e in all_evaluators:
        n = len(unified[e])
        print(f"  {display_names[e]:20s}: {n} judgments")

    # ── Pass rates by model ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PASS RATES BY MODEL")
    print("=" * 70)

    models = ["gpt-5.4", "llama-70b", "qwen-72b"]
    header = f"{'Evaluator':20s}" + "".join(f"  {m:>12s}" for m in models) + "  " + f"{'Overall':>10s}"
    print(header)
    print("-" * len(header))

    for e in all_evaluators:
        parts = []
        total_p = total_t = 0
        for m in models:
            keys_m = [k for k in unified[e] if k[1] == m]
            p, t = compute_pass_rate(unified[e], keys_m)
            total_p += p
            total_t += t
            rate = f"{100*p/t:.1f}%" if t else "N/A"
            parts.append(f"{p}/{t} ({rate})" if t else "N/A")
        overall = f"{100*total_p/total_t:.1f}%" if total_t else "N/A"
        row = f"{display_names[e]:20s}" + "".join(f"  {p:>12s}" for p in parts) + "  " + f"{overall:>10s}"
        print(row)

    # ── Pairwise agreement ────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PAIRWISE AGREEMENT RATES")
    print("=" * 70)

    common_keys = set()
    for k in all_keys:
        if all(unified[e].get(k) for e in all_evaluators):
            common_keys.add(k)
    print(f"(computed over {len(common_keys)} judgments where all evaluators have a label)\n")

    # Print matrix header
    short = {e: display_names[e][:8] for e in all_evaluators}
    header = f"{'':20s}" + "".join(f"  {short[e]:>8s}" for e in all_evaluators)
    print(header)
    print("-" * len(header))

    for e1 in all_evaluators:
        parts = []
        for e2 in all_evaluators:
            if e1 == e2:
                parts.append("  —")
            else:
                agree, total = compute_agreement(unified[e1], unified[e2], common_keys)
                rate = f"{100*agree/total:.1f}%" if total else "N/A"
                parts.append(rate)
        row = f"{display_names[e1]:20s}" + "".join(f"  {p:>8s}" for p in parts)
        print(row)

    # ── Human inter-annotator agreement ───────────────────────────
    print(f"\n{'=' * 70}")
    print("INTER-ANNOTATOR AGREEMENT (Humans Only)")
    print("=" * 70)

    human_evals = ["soraya", "omid", "ava"]
    for e1, e2 in combinations(human_evals, 2):
        common = [k for k in all_keys if unified[e1].get(k) and unified[e2].get(k)]
        agree, total = compute_agreement(unified[e1], unified[e2], common)
        rate = f"{100*agree/total:.1f}%" if total else "N/A"
        print(f"  {display_names[e1]:10s} vs {display_names[e2]:10s}: {agree}/{total} ({rate})")

    # Fleiss-like: all 3 agree
    all3_keys = [k for k in all_keys if all(unified[e].get(k) for e in human_evals)]
    all3_agree = sum(1 for k in all3_keys if len(set(unified[e][k] for e in human_evals)) == 1)
    print(f"\n  All 3 agree: {all3_agree}/{len(all3_keys)} ({100*all3_agree/len(all3_keys):.1f}%)" if all3_keys else "")

    # Majority vote
    majority = {}
    for k in all3_keys:
        votes = [unified[e][k] for e in human_evals]
        pass_count = sum(1 for v in votes if v == "pass")
        majority[k] = "pass" if pass_count >= 2 else "fail"

    print(f"\n  Majority-vote pass rate:")
    for m in models:
        keys_m = [k for k in all3_keys if k[1] == m]
        p = sum(1 for k in keys_m if majority[k] == "pass")
        print(f"    {m}: {p}/{len(keys_m)} ({100*p/len(keys_m):.1f}%)" if keys_m else f"    {m}: N/A")

    # ── Judge vs Majority Human ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("JUDGE vs HUMAN MAJORITY VOTE")
    print("=" * 70)

    for judge in ["heuristic", "gpt4o_calibrated", "claude_uncalibrated"]:
        common = [k for k in all3_keys if unified[judge].get(k)]
        agree = sum(1 for k in common if unified[judge][k] == majority[k])
        rate = f"{100*agree/len(common):.1f}%" if common else "N/A"
        print(f"  {display_names[judge]:20s} vs Majority: {agree}/{len(common)} ({rate})")

    # ── By domain breakdown ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PASS RATES BY DOMAIN (All Evaluators)")
    print("=" * 70)

    domains = ["coding", "math", "instruction_following"]
    for domain in domains:
        domain_keys = [k for k in all_keys if metadata.get(k[0], {}).get("domain") == domain]
        print(f"\n  {domain.upper()} ({len(domain_keys)} judgments max):")
        for e in all_evaluators:
            dk = [k for k in domain_keys if unified[e].get(k)]
            p, t = compute_pass_rate(unified[e], dk)
            rate = f"{100*p/t:.1f}%" if t else "N/A"
            print(f"    {display_names[e]:20s}: {p}/{t} ({rate})")

    # ── By attack breakdown ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("PASS RATES BY ATTACK (All Evaluators)")
    print("=" * 70)

    attacks = sorted(set(m.get("attack_name", "") for m in metadata.values()))
    for attack in attacks:
        attack_keys = [k for k in all_keys if metadata.get(k[0], {}).get("attack_name") == attack]
        print(f"\n  {attack.upper()} ({len(attack_keys)} judgments max):")
        for e in all_evaluators:
            ak = [k for k in attack_keys if unified[e].get(k)]
            p, t = compute_pass_rate(unified[e], ak)
            rate = f"{100*p/t:.1f}%" if t else "N/A"
            print(f"    {display_names[e]:20s}: {p}/{t} ({rate})")

    # ── Disagreement analysis ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("DISAGREEMENT CASES (Human majority vs Judges)")
    print("=" * 70)

    for judge in ["gpt4o_calibrated", "claude_uncalibrated"]:
        disagree = [(k, majority[k], unified[judge][k])
                    for k in all3_keys
                    if unified[judge].get(k) and unified[judge][k] != majority[k]]
        print(f"\n  {display_names[judge]} disagrees with human majority: {len(disagree)} cases")
        # Breakdown
        fp = sum(1 for _, h, j in disagree if h == "fail" and j == "pass")
        fn = sum(1 for _, h, j in disagree if h == "pass" and j == "fail")
        print(f"    False positives (judge=pass, human=fail): {fp}")
        print(f"    False negatives (judge=fail, human=pass): {fn}")

    print(f"\n{'=' * 70}")
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
