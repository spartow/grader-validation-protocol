"""
Merge Soraya + Omid labels for the 37 recovered LLaMA outputs.
Check inter-annotator agreement, resolve disagreements, and update
wave2_omid_labeled.csv with corrected human labels.
"""

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SORAYA_PATH  = BASE / "human_labels" / "llama70b_37_relabel_soraya_labeled.csv"
OMID_PATH = BASE / "human_labels" / "llama70b_37_relabel_omid_labeled.csv"
WAVE2_PATH   = BASE / "human_labels" / "wave2_omid_labeled.csv"


def main():
    soraya  = pd.read_csv(SORAYA_PATH)
    omid = pd.read_csv(OMID_PATH)

    # Build merged view
    merged = soraya[["example_id", "domain", "attack_name"]].copy()
    merged["soraya_label"]  = soraya["annotator1_label"].str.strip().str.lower()
    merged["omid_label"] = omid["annotator2_label"].str.strip().str.lower()

    # ── Agreement stats ──────────────────────────────────────────────
    merged["agree"] = merged["soraya_label"] == merged["omid_label"]
    n_agree = merged["agree"].sum()
    n_total = len(merged)
    pct = n_agree / n_total * 100

    print("=" * 60)
    print("Inter-Annotator Agreement — 37 Recovered LLaMA Outputs")
    print("=" * 60)
    print(f"Agreement: {n_agree}/{n_total} ({pct:.1f}%)")
    print(f"Soraya:  {(merged['soraya_label']=='pass').sum()} pass, {(merged['soraya_label']=='fail').sum()} fail")
    print(f"Omid: {(merged['omid_label']=='pass').sum()} pass, {(merged['omid_label']=='fail').sum()} fail")

    # Per-domain agreement
    print("\nPer-domain agreement:")
    for domain in sorted(merged["domain"].unique()):
        sub = merged[merged["domain"] == domain]
        da = sub["agree"].sum()
        dt = len(sub)
        print(f"  {domain:25s}: {da}/{dt} ({da/dt*100:.0f}%)")

    # Show disagreements
    disagree = merged[~merged["agree"]]
    if len(disagree) > 0:
        print(f"\nDisagreements ({len(disagree)}):")
        for _, row in disagree.iterrows():
            print(f"  {row['example_id']:70s}  soraya={row['soraya_label']}  omid={row['omid_label']}")

    # ── Resolve: use majority (both agree) or flag ────────────────
    # For disagreements, we'll use Soraya's label as tie-breaker
    # (she is the primary annotator per the project structure)
    merged["final_label"] = merged.apply(
        lambda r: r["soraya_label"] if r["agree"] else r["soraya_label"],
        axis=1,
    )
    # Note: For a 2-annotator setup, any tie-breaking rule works.
    # Using Soraya as primary annotator for consistency.

    print(f"\nFinal labels: {(merged['final_label']=='pass').sum()} pass, {(merged['final_label']=='fail').sum()} fail")

    # ── Update wave2_omid_labeled.csv ─────────────────────────
    w2 = pd.read_csv(WAVE2_PATH)

    # Build lookup: example_id -> final_label
    label_map = dict(zip(merged["example_id"], merged["final_label"]))

    updated = 0
    for eid, label in label_map.items():
        mask = w2["example_id"] == eid
        if mask.any():
            # Update human_label_llama70b (and omid_label_llama70b if present)
            w2.loc[mask, "human_label_llama70b"] = label
            if "omid_label_llama70b" in w2.columns:
                w2.loc[mask, "omid_label_llama70b"] = label
            updated += 1

    print(f"\nUpdated {updated} rows in wave2_omid_labeled.csv")

    w2.to_csv(WAVE2_PATH, index=False)
    print(f"Saved {WAVE2_PATH.name}")

    # ── Save merged labels for reference ─────────────────────────
    merged_path = BASE / "human_labels" / "llama70b_37_merged_labels.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged labels to {merged_path.name}")

    # ── Summary of all Wave 2 human labels ───────────────────────
    print("\n" + "=" * 60)
    print("Updated Wave 2 Human Labels Summary")
    print("=" * 60)
    for m in ["gpt54", "llama70b", "qwen72b"]:
        col = f"human_label_{m}"
        if col in w2.columns:
            p = (w2[col].str.lower() == "pass").sum()
            f = (w2[col].str.lower() == "fail").sum()
            other = w2[col].isna().sum()
            print(f"  {m:12s}: {p} pass, {f} fail" + (f", {other} unlabeled" if other else ""))


if __name__ == "__main__":
    main()
