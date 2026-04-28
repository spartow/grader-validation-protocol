# Grader Validation Protocol (GVP)

Artifacts for **"Who Grades the Graders? Validating Automated Evaluators in Adversarial AI Safety Benchmarks"** (NeurIPS 2026 submission).

## Project Structure

```
data/
  benchmark_v3_2001.jsonl          # Full 2001-example tri-domain benchmark
  benchmark_subset_wave1.jsonl     # 90-example Wave 1 subset

model_outputs/                     # Authoritative model outputs (JSONL)
  benchmark_outputs_gpt54.jsonl    # GPT-5.4   — 1982/2001 successful
  benchmark_outputs_llama70b.jsonl # LLaMA 70B — 2001/2001 successful
  benchmark_outputs_qwen72b.jsonl  # Qwen 72B  — 1890/2001 successful

human_labels/
  wave1_heuristic_labels.csv       # Wave 1: 90 examples, human labels (Soraya) + heuristic scores
  wave2_omid_labeled.csv        # Wave 2: 90 examples, human labels (Soraya + Omid) + all outputs
  wave2_heuristic_labels.csv       # Wave 2: 90 examples, heuristic scores

judge_outputs/
  judge_gpt4o_calibrated.jsonl     # Calibrated GPT-4o judge — Wave 1 (270 judgments)
  judge_gpt4o_calibrated_wave2.jsonl # Calibrated GPT-4o judge — Wave 2 (270 judgments)
  judge_gemini_uncalibrated.jsonl  # Uncalibrated judge (Claude Sonnet 4) — all 540 judgments

heuristic_metrics/
  heuristic_vs_human_metrics.csv   # Heuristic scorer accuracy vs human labels

scripts/
  strict_llm_judge_runner.py       # Calibrated two-stage GPT-4o judge (Appendix B prompts)
  run_gemini_judge.py              # Uncalibrated single-stage judge
  score_wave2_heuristic.py         # Heuristic scorer (validates on Wave 1, scores Wave 2)
  analyze_full_results.py          # Cross-evaluator analysis
  rescore_full_2k.py               # Full 2001-example benchmark rescoring

archive/                           # Old/one-time files (kept for provenance, not needed)
```

## Validation Set

- **180 unique examples** (90 Wave 1 + 90 Wave 2, zero overlap)
- **3 models** × 180 = **540 judgments** per evaluator
- **3 domains**: coding, math, instruction-following
- **3 attack types**: clean, prompt injection, context manipulation

## Evaluators

| Evaluator | Type | Records | File |
|---|---|---|---|
| Soraya (human) | Wave 1 labels | 270 | `wave1_heuristic_labels.csv` |
| Soraya + Omid (human) | Wave 2 labels | 270 | `wave2_omid_labeled.csv` |
| Heuristic scorer | Automated | 270 | `wave2_heuristic_labels.csv` |
| GPT-4o calibrated | Two-stage LLM judge | 540 | `judge_gpt4o_calibrated*.jsonl` |
| Claude Sonnet 4 uncalibrated | Single-stage LLM judge | 540 | `judge_gemini_uncalibrated.jsonl` |

## Known Data Gaps

- **14 model outputs** genuinely missing from JSONL files (2 GPT-5.4 + 12 Qwen 72B)
- These are marked as auto-FAIL in judge outputs
- All 2001 LLaMA 70B outputs are present

## License

MIT

## Citation

```bibtex
@inproceedings{gvp2026,
  title     = {Who Grades the Graders? Validating Automated Evaluators in Adversarial AI Safety Benchmarks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
}
```
