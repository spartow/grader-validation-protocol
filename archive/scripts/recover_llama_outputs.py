"""
Recover 37 missing LLaMA 70B outputs and prepare for re-scoring.

Tasks:
1. Recover outputs from benchmark_outputs_llama70b.jsonl into wave2_omid_labeled.csv
2. Create llama70b_37_relabel.csv labeling sheet for annotators
3. Regenerate wave2_for_judge.csv with recovered outputs
4. Remove 37 LLaMA auto-fail records from judge JSONL (for re-judging)
"""

import json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

WAVE2_PATH     = BASE / "human_labels" / "wave2_omid_labeled.csv"
RELABEL_PATH   = BASE / "human_labels" / "llama70b_37_relabel.csv"
JUDGE_CSV_PATH = BASE / "human_labels" / "wave2_for_judge.csv"
JUDGE_JSONL    = BASE / "judge_outputs" / "judge_gpt4o_calibrated_wave2.jsonl"
LLAMA_OUTPUTS  = BASE / "model_outputs" / "benchmark_outputs_llama70b.jsonl"


def main():
    # ── TASK 1: Recover 37 LLaMA outputs ──────────────────────────────
    print("=" * 60)
    print("Task 1 — Recover 37 missing LLaMA outputs")
    print("=" * 60)

    w2 = pd.read_csv(WAVE2_PATH)
    missing_mask = w2["llama70b_output"].isna()
    missing_ids = set(w2[missing_mask]["example_id"])
    print(f"Missing LLaMA outputs: {len(missing_ids)}")

    # Read benchmark outputs (force utf-8)
    recovered = {}
    with open(LLAMA_OUTPUTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["example_id"] in missing_ids:
                output = r.get("model_output", r.get("raw_model_output", ""))
                if output and output.strip():
                    recovered[r["example_id"]] = output

    print(f"Recovered: {len(recovered)} / {len(missing_ids)}")
    assert len(recovered) == 37, f"Expected 37, got {len(recovered)}"

    # Populate into dataframe
    for eid, output in recovered.items():
        w2.loc[w2["example_id"] == eid, "llama70b_output"] = output

    still_missing = w2["llama70b_output"].isna().sum()
    print(f"Still missing after recovery: {still_missing} (should be 0)")

    w2.to_csv(WAVE2_PATH, index=False)
    print(f"Saved updated {WAVE2_PATH.name}")

    # ── TASK 2: Create labeling sheet ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 2 — Create annotator labeling sheet")
    print("=" * 60)

    labeling_rows = []
    for eid in sorted(recovered.keys()):
        row = w2[w2["example_id"] == eid].iloc[0]
        labeling_rows.append({
            "example_id": eid,
            "domain": row["domain"],
            "attack_name": row["attack_name"],
            "original_task": row["original_task"],
            "attacked_prompt": row["attacked_prompt"],
            "reference": row["reference"],
            "llama70b_output": row["llama70b_output"],
            "annotator1_label": "",
            "annotator2_label": "",
            "notes": "",
        })

    labeling_df = pd.DataFrame(labeling_rows)
    labeling_df.to_csv(RELABEL_PATH, index=False)
    print(f"Created {RELABEL_PATH.name} with {len(labeling_df)} rows")
    print(f"Domains: {labeling_df['domain'].value_counts().to_dict()}")

    # ── Regenerate wave2_for_judge.csv ────────────────────────────────
    print("\n" + "=" * 60)
    print("Regenerating wave2_for_judge.csv with recovered outputs")
    print("=" * 60)

    w2j = pd.read_csv(WAVE2_PATH)
    w2j = w2j.rename(columns={
        "gpt54_output": "gpt54_output2",
        "llama70b_output": "llama70b_output2",
        "qwen72b_output": "qwen72b_output2",
    })
    w2j = w2j.fillna("")
    w2j.to_csv(JUDGE_CSV_PATH, index=False)
    print(f"Saved {JUDGE_CSV_PATH.name}")

    # ── TASK 4 prep: Remove 37 LLaMA auto-fail records from JSONL ────
    print("\n" + "=" * 60)
    print("Task 4 prep — Remove 37 LLaMA auto-fail records from judge JSONL")
    print("=" * 60)

    rejudge_ids = set(recovered.keys())

    with open(JUDGE_JSONL, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    kept = [
        r for r in records
        if not (r["model"] == "llama-70b" and r["example_id"] in rejudge_ids)
    ]
    removed = len(records) - len(kept)
    print(f"Removed {removed} LLaMA auto-fail records (should be 37)")
    print(f"Remaining records: {len(kept)} (should be 233)")

    with open(JUDGE_JSONL, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(kept)} records back to {JUDGE_JSONL.name}")
    print("\nReady for: re-run heuristic scorer + re-run judge on 37 LLaMA outputs")


if __name__ == "__main__":
    main()
