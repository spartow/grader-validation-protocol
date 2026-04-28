#!/usr/bin/env python3
"""
TASK 1 + TASK 2: Replace 3 overlapping examples in Wave 2 and create annotator labeling sheet.

Removes the 3 overlap IDs from wave2_omid_labeled.csv and adds 3 replacement rows.
Also creates replacement_3_labeling.csv for annotators.

Usage:
    python scripts/fix_wave2_replacements.py
"""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

WAVE2_PATH   = BASE / "human_labels" / "wave2_omid_labeled.csv"
BENCH_PATH   = BASE / "data" / "benchmark_v3_2001.jsonl"
LABELING_OUT = BASE / "human_labels" / "replacement_3_labeling.csv"

MODEL_FILES = [
    (BASE / "model_outputs" / "benchmark_outputs_gpt54.jsonl",    "gpt54"),
    (BASE / "model_outputs" / "benchmark_outputs_llama70b.jsonl", "llama70b"),
    (BASE / "model_outputs" / "benchmark_outputs_qwen72b.jsonl",  "qwen72b"),
]

OVERLAP_IDS = [
    "gvp-math-0201-clean__clean__none",
    "gvp-inst-0119-clean__context_manipulation_v3__brevity_policy_memo",
    "gvp-inst-0175-clean__context_manipulation_v3__brevity_policy_memo",
]

REPLACEMENT_IDS = [
    "gvp-math-0045-clean__clean__none",
    "gvp-inst-0179-clean__context_manipulation_v3__brevity_policy_memo",
    "gvp-inst-0097-clean__context_manipulation_v3__brevity_policy_memo",
]


def main():
    # ── 1a. Load Wave 2 ───────────────────────────────────────────────────────
    with open(WAVE2_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_fields = list(reader.fieldnames)
        rows = list(reader)

    print(f"Loaded wave2_omid_labeled.csv: {len(rows)} rows")

    # ── 1b. Load benchmark metadata for replacement examples ──────────────────
    bench = {}
    with open(BENCH_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["example_id"] in REPLACEMENT_IDS:
                bench[r["example_id"]] = r

    assert len(bench) == 3, f"Expected 3 bench entries, got {len(bench)}: {list(bench.keys())}"
    print(f"Loaded benchmark metadata for {len(bench)} replacement examples")

    # ── 1c. Load model outputs for replacement examples ───────────────────────
    model_outputs = {eid: {} for eid in REPLACEMENT_IDS}
    for fpath, mk in MODEL_FILES:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["example_id"] in REPLACEMENT_IDS:
                    out = r.get("raw_model_output", r.get("model_output", ""))
                    model_outputs[r["example_id"]][mk] = out

    # Verify all outputs present
    for eid in REPLACEMENT_IDS:
        for mk in ["gpt54", "llama70b", "qwen72b"]:
            out = model_outputs[eid].get(mk, "")
            assert out and out.strip(), f"Missing output: {eid}/{mk}"
        print(f"  ✓ {eid}: all 3 model outputs present")

    # ── 1d. Remove overlapping rows from Wave 2 ───────────────────────────────
    rows_clean = [r for r in rows if r["example_id"] not in OVERLAP_IDS]
    removed = len(rows) - len(rows_clean)
    print(f"\nRemoved {removed} overlap rows → {len(rows_clean)} rows remaining (expect 87)")
    assert len(rows_clean) == 87, f"Expected 87 rows, got {len(rows_clean)}"

    # ── 1e. Build new rows for the 3 replacements ─────────────────────────────
    new_rows = []
    for eid in REPLACEMENT_IDS:
        b = bench[eid]
        row = {col: "" for col in orig_fields}
        row["example_id"]    = eid
        row["domain"]        = b.get("domain", "")
        row["attack_name"]   = b.get("attack_name", "")
        row["attack_variant"] = b.get("attack_variant", "")
        row["reference"]     = b.get("reference", "")
        row["original_task"] = b.get("original_task", "")
        row["attacked_prompt"] = b.get("attacked_prompt", "")
        row["gpt54_output"]   = model_outputs[eid]["gpt54"]
        row["llama70b_output"] = model_outputs[eid]["llama70b"]
        row["qwen72b_output"]  = model_outputs[eid]["qwen72b"]
        # Human labels — intentionally blank; annotators will fill these
        row["omid_label_gpt54"]   = ""
        row["omid_label_llama70b"] = ""
        row["omid_label_qwen72b"]  = ""
        row["notes"] = ""
        new_rows.append(row)
        print(f"  Built row for: {eid}")

    # ── 1f. Append and save ───────────────────────────────────────────────────
    final_rows = rows_clean + new_rows
    assert len(final_rows) == 90, f"Expected 90 rows, got {len(final_rows)}"

    with open(WAVE2_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=orig_fields)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\n✓ Saved wave2_omid_labeled.csv: {len(final_rows)} rows")

    # ── Verify no overlap with Wave 1 ─────────────────────────────────────────
    wave1_path = BASE / "human_labels" / "wave1_heuristic_labels.csv"
    with open(wave1_path, encoding="utf-8") as f:
        w1_ids = {r["example_id"] for r in csv.DictReader(f)}

    w2_ids = {r["example_id"] for r in final_rows}
    overlap = w1_ids & w2_ids
    print(f"\nWave 1 IDs: {len(w1_ids)}, Wave 2 IDs: {len(w2_ids)}, Overlap: {len(overlap)}")
    assert len(overlap) == 0, f"Still overlapping: {overlap}"
    print(f"✓ No overlap between Wave 1 and Wave 2!")
    print(f"✓ Total unique: {len(w1_ids | w2_ids)} (expect 180)")

    # ── TASK 2: Create annotator labeling sheet ───────────────────────────────
    labeling_rows = []
    for eid in REPLACEMENT_IDS:
        b = bench[eid]
        for mk, display_name in [("gpt54", "GPT-5.4"), ("llama70b", "LLaMA 70B"), ("qwen72b", "Qwen 72B")]:
            labeling_rows.append({
                "example_id":     eid,
                "model":          display_name,
                "domain":         b.get("domain", ""),
                "attack_name":    b.get("attack_name", ""),
                "original_task":  b.get("original_task", ""),
                "attacked_prompt": b.get("attacked_prompt", ""),
                "reference":      b.get("reference", ""),
                "model_output":   model_outputs[eid][mk],
                "soraya_label":   "",
                "omid_label":  "",
            })

    labeling_fields = ["example_id", "model", "domain", "attack_name", "original_task",
                       "attacked_prompt", "reference", "model_output", "soraya_label", "omid_label"]

    with open(LABELING_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=labeling_fields)
        writer.writeheader()
        writer.writerows(labeling_rows)

    print(f"\n✓ Created replacement_3_labeling.csv: {len(labeling_rows)} rows (3 examples × 3 models = 9)")
    print("\nAll done! Run score_wave2_heuristic.py next (TASK 3).")


if __name__ == "__main__":
    main()
