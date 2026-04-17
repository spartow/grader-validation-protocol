# Grader Validation Protocol (GVP)

Artifacts for **"Who Grades the Graders? Validating Automated Evaluators in Adversarial AI Safety Benchmarks"** (NeurIPS 2026 submission).

## Structure

| Folder | Contents |
|---|---|
| `data/` | 2,001-example benchmark and validation subset |
| `model_outputs/` | Full outputs from GPT-5.4, LLaMA 70B, Qwen 72B |
| `human_labels/` | Independent annotations from both authors (Wave 1 + Wave 2, 180 examples) |
| `judge_outputs/` | Calibrated two-stage judge results (GPT-4o + GPT-5.4 bridge + full benchmark rescore) |
| `scripts/` | Judge runner, rescoring, and analysis scripts |
| `heuristic_metrics/` | Heuristic scorer validation metrics |

## Key Files

- `data/benchmark_v3_2001.jsonl` — Full tri-domain adversarial benchmark (coding, math, instruction-following × clean, prompt injection, context manipulation)
- `human_labels/wave1_labels.csv` + `wave2_*.csv` — 540 independent human judgments from both authors
- `judge_outputs/judge_gpt4o_calibrated.jsonl` — Calibrated two-stage reference judge (270 validation outputs)
- `judge_outputs/full_benchmark_judge_results.jsonl` — Full 6,000-output rescore with calibrated judge
- `scripts/strict_llm_judge_runner.py` — Reproducible judge implementation with exact prompts from Appendix B

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
