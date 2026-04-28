#!/usr/bin/env python3
"""Populate missing model outputs in Wave 1 and Wave 2 CSVs from authoritative JSONL files.
Also removes confusing duplicate columns (*_output2) from Wave 1."""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_jsonl_outputs(path):
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            eid = r.get("example_id", "")
            out = r.get("raw_model_output", "") or r.get("model_output", "")
            if eid and out and out.strip():
                lookup[eid] = out
    return lookup


def main():
    gpt_out = load_jsonl_outputs(BASE / "model_outputs" / "benchmark_outputs_gpt54.jsonl")
    llama_out = load_jsonl_outputs(BASE / "model_outputs" / "benchmark_outputs_llama70b.jsonl")
    qwen_out = load_jsonl_outputs(BASE / "model_outputs" / "benchmark_outputs_qwen72b.jsonl")
    print(f"JSONL outputs loaded: GPT={len(gpt_out)}, LLaMA={len(llama_out)}, Qwen={len(qwen_out)}")

    sources = {
        "gpt54_output": gpt_out,
        "llama70b_output": llama_out,
        "qwen72b_output": qwen_out,
    }

    for csv_name in ["wave1_heuristic_labels.csv", "wave2_omid_labeled.csv", "wave2_heuristic_labels.csv"]:
        csv_path = BASE / "human_labels" / csv_name
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            old_cols = list(rdr.fieldnames)
            rows = list(rdr)

        # Remove confusing *_output2 columns
        clean_cols = [c for c in old_cols if not c.endswith("_output2")]
        removed = [c for c in old_cols if c.endswith("_output2")]
        if removed:
            print(f"  {csv_name}: removing duplicate columns: {removed}")

        # Populate missing outputs from JSONL
        filled = {"gpt54": 0, "llama70b": 0, "qwen72b": 0}
        for r in rows:
            for col, lookup in sources.items():
                model = col.replace("_output", "")
                if not r.get(col, "").strip():
                    jsonl_val = lookup.get(r["example_id"], "")
                    if jsonl_val:
                        r[col] = jsonl_val
                        filled[model] += 1

        print(f"  {csv_name}: filled GPT={filled['gpt54']}, LLaMA={filled['llama70b']}, Qwen={filled['qwen72b']}")

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=clean_cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        # Verify
        with open(csv_path, encoding="utf-8") as f:
            rows2 = list(csv.DictReader(f))
        for col in ["gpt54_output", "llama70b_output", "qwen72b_output"]:
            empty = sum(1 for r in rows2 if not r.get(col, "").strip())
            if empty:
                print(f"    WARNING: {col} still has {empty} empty rows")
        print(f"    {csv_name}: {len(rows2)} rows, clean")


if __name__ == "__main__":
    main()
