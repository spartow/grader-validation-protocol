#!/usr/bin/env python3
"""Fix typo in replacement_3_labeling.csv and propagate labels into wave2 files."""

import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

REPLACEMENT_IDS = [
    "gvp-math-0045-clean__clean__none",
    "gvp-inst-0179-clean__context_manipulation_v3__brevity_policy_memo",
    "gvp-inst-0097-clean__context_manipulation_v3__brevity_policy_memo",
]

MODEL_MAP = {"GPT-5.4": "gpt54", "LLaMA 70B": "llama70b", "Qwen 72B": "qwen72b"}


def main():
    lab_path = BASE / "human_labels" / "replacement_3_labeling.csv"

    # ── Fix typo ──────────────────────────────────────────────────
    with open(lab_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        lab_cols = list(rdr.fieldnames)
        lab_rows = list(rdr)

    for r in lab_rows:
        if r["omid_label"].strip() == "fai":
            print(f"Fixed typo: fai -> fail  ({r['example_id'][:40]} / {r['model']})")
            r["omid_label"] = "fail"

    with open(lab_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lab_cols)
        w.writeheader()
        w.writerows(lab_rows)

    # ── Build label lookup ────────────────────────────────────────
    labels = {}
    for r in lab_rows:
        mk = MODEL_MAP[r["model"]]
        labels[(r["example_id"], mk)] = (r["soraya_label"], r["omid_label"])

    print("\nLabels to propagate:")
    for (eid, mk), (sl, sk) in labels.items():
        print(f"  {eid[:40]} / {mk}: soraya={sl}, omid={sk}")

    # ── Propagate into wave2_omid_labeled.csv ──────────────────
    sat_path = BASE / "human_labels" / "wave2_omid_labeled.csv"
    with open(sat_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        sat_cols = list(rdr.fieldnames)
        sat_rows = list(rdr)

    updated = 0
    for r in sat_rows:
        for mk in ["gpt54", "llama70b", "qwen72b"]:
            key = (r["example_id"], mk)
            if key in labels:
                sl, sk = labels[key]
                r[f"omid_label_{mk}"] = sk
                if f"soraya_label_{mk}" in sat_cols:
                    r[f"soraya_label_{mk}"] = sl
                updated += 1

    with open(sat_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sat_cols)
        w.writeheader()
        w.writerows(sat_rows)
    print(f"\nwave2_omid_labeled.csv: updated {updated} label cells")

    # ── Propagate into wave2_soraya_labeled.csv ───────────────────
    sor_path = BASE / "human_labels" / "wave2_soraya_labeled.csv"
    with open(sor_path, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        sor_cols = list(rdr.fieldnames)
        sor_rows = list(rdr)

    updated2 = 0
    for r in sor_rows:
        for mk in ["gpt54", "llama70b", "qwen72b"]:
            key = (r["example_id"], mk)
            if key in labels:
                sl, sk = labels[key]
                if f"soraya_label_{mk}" in sor_cols:
                    r[f"soraya_label_{mk}"] = sl
                if f"omid_label_{mk}" in sor_cols:
                    r[f"omid_label_{mk}"] = sk
                updated2 += 1

    with open(sor_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sor_cols)
        w.writeheader()
        w.writerows(sor_rows)
    print(f"wave2_soraya_labeled.csv: updated {updated2} label cells")

    # ── Verify ────────────────────────────────────────────────────
    print("\nVerification - wave2_omid_labeled.csv:")
    for r in sat_rows:
        if r["example_id"] in REPLACEMENT_IDS:
            sat_vals = "/".join(r.get(f"omid_label_{m}", "?") for m in ["gpt54", "llama70b", "qwen72b"])
            sor_vals = "/".join(r.get(f"soraya_label_{m}", "?") for m in ["gpt54", "llama70b", "qwen72b"])
            print(f"  {r['example_id'][:45]}")
            print(f"    omid: {sat_vals}")
            print(f"    soraya:  {sor_vals}")

    print("\nVerification - wave2_soraya_labeled.csv:")
    for r in sor_rows:
        if r["example_id"] in REPLACEMENT_IDS:
            sat_vals = "/".join(r.get(f"omid_label_{m}", "?") for m in ["gpt54", "llama70b", "qwen72b"])
            sor_vals = "/".join(r.get(f"soraya_label_{m}", "?") for m in ["gpt54", "llama70b", "qwen72b"])
            print(f"  {r['example_id'][:45]}")
            print(f"    omid: {sat_vals}")
            print(f"    soraya:  {sor_vals}")


if __name__ == "__main__":
    main()
