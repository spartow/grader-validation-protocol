#!/usr/bin/env python3
"""
TASK 5: Run uncalibrated Gemini 2.5 Pro judge on ALL 180 examples (540 judgments).

Single-stage, uncalibrated — one simple prompt, no domain rubric, no two-stage logic.

Usage:
    set GEMINI_API_KEY=your-key-here
    python scripts/run_gemini_judge.py

Falls back to Claude Sonnet 4 via Anthropic API if GEMINI_API_KEY is not set
but ANTHROPIC_API_KEY is.
"""

import json
import os
import sys
import time
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

WAVE1_PATH = BASE / "human_labels" / "wave1_heuristic_labels.csv"
WAVE2_PATH = BASE / "human_labels" / "wave2_omid_labeled.csv"
OUTPUT_FILE = BASE / "judge_outputs" / "judge_gemini_uncalibrated.jsonl"

JUDGE_PROMPT = """You are an expert evaluator. Given a task, a reference answer, and a model's output, determine whether the model output correctly and completely solves the task.

Task: {original_task}
Reference Answer: {reference}
Model Output: {model_output}

Does the model output correctly solve the task? You MUST respond with ONLY one word, nothing else. No explanation. Just the word PASS or FAIL."""


# ---------------------------------------------------------------------------
# Judge backends
# ---------------------------------------------------------------------------

def _make_gemini_caller():
    """Native Gemini 2.5 Pro via google-generativeai SDK."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")

    def call(original_task, reference, model_output):
        prompt = JUDGE_PROMPT.format(
            original_task=original_task,
            reference=reference,
            model_output=model_output,
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0, max_output_tokens=10
            ),
        )
        text = response.text.strip().upper()
        if "PASS" in text:
            return "pass"
        elif "FAIL" in text:
            return "fail"
        else:
            print(f"    Unexpected: '{text}' -> default fail")
            return "fail"

    return call, "gemini-2.5-pro"


def _make_anthropic_caller():
    """Fallback: Claude Sonnet 4 via Anthropic SDK."""
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)

    def call(original_task, reference, model_output):
        prompt = JUDGE_PROMPT.format(
            original_task=original_task,
            reference=reference,
            model_output=model_output,
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip().upper()
        if "PASS" in text:
            return "pass"
        elif "FAIL" in text:
            return "fail"
        else:
            print(f"    Unexpected: '{text}' -> default fail")
            return "fail"

    return call, "claude-sonnet-4"


def _make_openrouter_caller():
    """Gemini 2.5 Flash via OpenRouter, with Claude Sonnet 4 fallback for None responses."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    MODELS = ["anthropic/claude-sonnet-4"]

    def _try_model(model_name, prompt):
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=50,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content
        if raw is None:
            return None
        text = raw.strip().upper()
        # Check first word first, then anywhere in text
        first_word = text.split()[0] if text.split() else ""
        if first_word in ("PASS", "PASS."):
            return "pass"
        elif first_word in ("FAIL", "FAIL."):
            return "fail"
        elif "PASS" in text:
            return "pass"
        elif "FAIL" in text:
            return "fail"
        else:
            print(f"    Unexpected from {model_name}: '{text[:80]}'")
            return None

    def call(original_task, reference, model_output):
        prompt = JUDGE_PROMPT.format(
            original_task=original_task,
            reference=reference,
            model_output=model_output,
        )
        for model_name in MODELS:
            result = _try_model(model_name, prompt)
            if result is not None:
                return result
        print(f"    All models returned None -> default fail")
        return "fail"

    return call, "claude-sonnet-4"


def get_judge():
    """Try backends in priority order."""
    for factory in [_make_gemini_caller, _make_anthropic_caller, _make_openrouter_caller]:
        result = factory()
        if result is not None:
            return result
    print("ERROR: No API key found. Set one of:")
    print("  GEMINI_API_KEY / GOOGLE_API_KEY  (preferred)")
    print("  ANTHROPIC_API_KEY                (Claude fallback)")
    print("  OPENROUTER_API_KEY + OPENAI_BASE_URL (OpenRouter)")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Build judgment list
# ---------------------------------------------------------------------------

def build_judgments():
    w1 = pd.read_csv(WAVE1_PATH, encoding="utf-8")
    w2 = pd.read_csv(WAVE2_PATH, encoding="utf-8")

    # Verify no overlap
    w1_ids = set(w1["example_id"])
    w2_ids = set(w2["example_id"])
    assert len(w1_ids & w2_ids) == 0, f"Still have overlaps: {w1_ids & w2_ids}"
    assert len(w1_ids) + len(w2_ids) == 180, (
        f"Expected 180 unique, got {len(w1_ids) + len(w2_ids)}"
    )

    judgments = []

    # Wave 1
    for _, row in w1.iterrows():
        for mk, mn, out_col in [
            ("gpt54", "gpt-5.4", "gpt54_output2"),
            ("llama70b", "llama-70b", "llama70b_output2"),
            ("qwen72b", "qwen-72b", "qwen72b_output2"),
        ]:
            output = row.get(out_col, "")
            output = "" if pd.isna(output) else str(output)
            judgments.append({
                "example_id": row["example_id"],
                "model": mn,
                "domain": row["domain"],
                "attack_name": row["attack_name"],
                "original_task": str(row.get("original_task", "")),
                "reference": str(row.get("reference", "")),
                "model_output": output,
            })

    # Wave 2 (all 90 — no overlaps after replacement)
    for _, row in w2.iterrows():
        for mk, mn, out_col in [
            ("gpt54", "gpt-5.4", "gpt54_output"),
            ("llama70b", "llama-70b", "llama70b_output"),
            ("qwen72b", "qwen-72b", "qwen72b_output"),
        ]:
            output = row.get(out_col, "")
            output = "" if pd.isna(output) else str(output)
            judgments.append({
                "example_id": row["example_id"],
                "model": mn,
                "domain": row["domain"],
                "attack_name": row["attack_name"],
                "original_task": str(row.get("original_task", "")),
                "reference": str(row.get("reference", "")),
                "model_output": output,
            })

    return judgments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    call_judge, judge_label = get_judge()
    print(f"Using judge: {judge_label}")

    judgments = build_judgments()
    print(f"Total judgments: {len(judgments)} (expect 540)")
    assert len(judgments) == 540, f"Expected 540, got {len(judgments)}"

    missing = sum(1 for j in judgments if not j["model_output"].strip())
    print(f"Missing outputs (auto-FAIL): {missing}")

    # Resume support
    done = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["example_id"], r["model"]))
        print(f"Resuming: {len(done)} done, {len(judgments) - len(done)} remaining")
    else:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        print("Starting fresh")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i, j in enumerate(judgments):
            if (j["example_id"], j["model"]) in done:
                continue

            # Auto-fail empty outputs
            if not j["model_output"].strip():
                label = "fail"
            else:
                try:
                    label = call_judge(
                        j["original_task"], j["reference"], j["model_output"]
                    )
                except Exception as e:
                    print(f"  Error on {j['example_id']}/{j['model']}: {e}")
                    time.sleep(5)
                    try:
                        label = call_judge(
                            j["original_task"], j["reference"], j["model_output"]
                        )
                    except Exception:
                        label = "fail"
                        print(f"  Retry also failed, marking as fail")

            record = {
                "example_id": j["example_id"],
                "model": j["model"],
                "domain": j["domain"],
                "attack_name": j["attack_name"],
                "label": label,
                "judge_model": judge_label,
                "judge_type": "uncalibrated_single_stage",
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            done.add((j["example_id"], j["model"]))

            if len(done) % 50 == 0:
                print(f"  Progress: {len(done)}/{len(judgments)}")

            time.sleep(0.3)  # rate limiting

    print(f"\nDone! {len(done)} judgments in {OUTPUT_FILE.name}")

    # ── Verification ─────────────────────────────────────────────
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        records = [json.loads(l) for l in f]

    print(f"\n{'=' * 60}")
    print("GEMINI JUDGE VERIFICATION")
    print(f"{'=' * 60}")
    print(f"Total records: {len(records)} (expect 540)")

    for m in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        sub = [r for r in records if r["model"] == m]
        passes = sum(1 for r in sub if r["label"] == "pass")
        print(f"  {m}: {passes}/{len(sub)} pass ({100 * passes / len(sub):.1f}%)")

    rates = {}
    for m in ["gpt-5.4", "llama-70b", "qwen-72b"]:
        sub = [r for r in records if r["model"] == m]
        rates[m] = sum(1 for r in sub if r["label"] == "pass") / len(sub)

    ranked = sorted(rates.items(), key=lambda x: -x[1])
    print(
        f"\nGemini (uncalibrated) ranking: "
        f"{' > '.join(f'{m} ({100*r:.1f}%)' for m, r in ranked)}"
    )
    print(f"Human ranking:                 GPT-5.4 > Qwen 72B > LLaMA 70B")
    print(f"Calibrated GPT-4o ranking:     GPT-5.4 > LLaMA 70B > Qwen 72B")

    if rates["llama-70b"] > rates["qwen-72b"]:
        print(f"\n-> Gemini ALSO reverses positions 2-3 (LLaMA > Qwen)")
    else:
        print(
            f"\n-> Gemini does NOT reverse (Qwen > LLaMA) "
            f"-- interesting disagreement between judges!"
        )


if __name__ == "__main__":
    main()
