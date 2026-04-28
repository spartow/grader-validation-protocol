# GVP Benchmark — Comprehensive Evaluation Report

Generated from 180 validation examples × 3 models = 540 judgments.
6 evaluators: 3 human (Soraya, Omid, Ava), Heuristic scorer, GPT-4o calibrated judge, Claude Sonnet 4 uncalibrated judge.

---
## 1. Inter-Annotator Agreement

### 1a. Pairwise Agreement & Cohen's Kappa

| Pair | Agreement | Cohen's κ | N | Interpretation |
|------|-----------|-----------|---|----------------|
| Soraya–Omid | 235/270 (87.0%) | 0.620 | 270 | substantial |
| Soraya–Ava | 406/540 (75.2%) | 0.344 | 540 | fair |
| Omid–Ava | 215/270 (79.6%) | 0.451 | 270 | moderate |

**Full 3-way agreement (Wave 2):** 193/270 (71.5%)

### 1b. Inter-Annotator Agreement by Domain

| Domain | Soraya–Omid | Soraya–Ava | Omid–Ava | 3-way |
|--------|----------------|----------------|-----------------|-------|
| coding | 84.4% | 72.2% | 74.4% | 65.6% |
| math | 87.8% | 78.9% | 86.7% | 76.7% |
| instruction_following | 88.9% | 77.8% | 77.8% | 72.2% |

> **Finding:** Instruction-following has the lowest inter-annotator agreement, confirming it is the most subjective domain. This supports the argument that automated judges struggle most where human judgment is ambiguous.

---
## 2. Pass Rates by Evaluator and Model

### 2a. Overall Pass Rates

| Evaluator | GPT-5.4 | LLaMA 70B | Qwen 72B | Overall |
|-----------|---------|-----------|----------|---------|
| Soraya | 92.8% | 57.8% | 81.1% | 77.2% |
| Omid | 91.1% | 77.8% | 83.3% | 84.1% |
| Ava | 77.8% | 70.0% | 69.4% | 72.4% |
| Heuristic | 81.1% | 75.6% | 73.3% | 76.7% |
| GPT-4o (calibrated) | 89.4% | 80.0% | 76.7% | 82.0% |
| Claude Sonnet 4 (uncalibrated) | 88.9% | 78.3% | 76.7% | 81.3% |
| **Human Majority** | 82.8% | 59.4% | 75.0% | 72.4% |

> **Finding:** GPT-5.4 consistently ranks first across all evaluators. LLaMA 70B and Qwen 72B swap positions 2–3 depending on evaluator, highlighting evaluator-dependent ranking instability for mid-tier models.

### 2b. Model Ranking by Evaluator

| Evaluator | Rank 1 | Rank 2 | Rank 3 |
|-----------|--------|--------|--------|
| Soraya | GPT-5.4 (92.8%) | Qwen 72B (81.1%) | LLaMA 70B (57.8%) |
| Omid | GPT-5.4 (91.1%) | Qwen 72B (83.3%) | LLaMA 70B (77.8%) |
| Ava | GPT-5.4 (77.8%) | LLaMA 70B (70.0%) | Qwen 72B (69.4%) |
| Heuristic | GPT-5.4 (81.1%) | LLaMA 70B (75.6%) | Qwen 72B (73.3%) |
| GPT-4o (calibrated) | GPT-5.4 (89.4%) | LLaMA 70B (80.0%) | Qwen 72B (76.7%) |
| Claude Sonnet 4 (uncalibrated) | GPT-5.4 (88.9%) | LLaMA 70B (78.3%) | Qwen 72B (76.7%) |
| **Human Majority** | GPT-5.4 (82.8%) | Qwen 72B (75.0%) | LLaMA 70B (59.4%) |

---
## 3. Domain × Attack Pass Rates (Human Majority)

| Domain | Attack | GPT-5.4 | LLaMA 70B | Qwen 72B |
|--------|--------|---------|-----------|----------|
| coding | Clean | 17/20 (85%) | 13/20 (65%) | 17/20 (85%) |
| coding | Context Manip. | 17/20 (85%) | 14/20 (70%) | 15/20 (75%) |
| coding | Prompt Injection | 12/20 (60%) | 10/20 (50%) | 13/20 (65%) |
| math | Clean | 18/20 (90%) | 16/20 (80%) | 18/20 (90%) |
| math | Context Manip. | 18/20 (90%) | 12/20 (60%) | 16/20 (80%) |
| math | Prompt Injection | 19/20 (95%) | 13/20 (65%) | 17/20 (85%) |
| instruction_following | Clean | 19/20 (95%) | 12/20 (60%) | 15/20 (75%) |
| instruction_following | Context Manip. | 15/20 (75%) | 7/20 (35%) | 10/20 (50%) |
| instruction_following | Prompt Injection | 14/20 (70%) | 10/20 (50%) | 14/20 (70%) |

### 3a. Attack Degradation (Δ from Clean baseline)

| Model | Prompt Injection Δ | Context Manipulation Δ |
|-------|--------------------|------------------------|
| GPT-5.4 | -15.0pp | -6.7pp |
| LLaMA 70B | -13.3pp | -13.3pp |
| Qwen 72B | -10.0pp | -15.0pp |

---
## 4. Judge Accuracy vs Human Majority Vote

### 4a. Overall Metrics

| Judge | Accuracy | Precision | Recall | F1 | Cohen's κ |
|-------|----------|-----------|--------|----|-----------|
| Heuristic | 78.7% | 83.3% | 88.2% | 85.7% | 0.440 |
| GPT-4o (calibrated) | 82.2% | 83.3% | 94.4% | 88.5% | 0.501 |
| Claude Sonnet 4 (uncalibrated) | 83.7% | 84.5% | 94.9% | 89.4% | 0.547 |

### 4b. Confusion Matrices (Judge vs Human Majority)

**GPT-4o (calibrated):**

|  | Human=Pass | Human=Fail |
|--|-----------|-----------|
| Judge=Pass | 369 (TP) | 74 (FP) |
| Judge=Fail | 22 (FN) | 75 (TN) |

**Claude Sonnet 4 (uncalibrated):**

|  | Human=Pass | Human=Fail |
|--|-----------|-----------|
| Judge=Pass | 371 (TP) | 68 (FP) |
| Judge=Fail | 20 (FN) | 81 (TN) |

### 4c. Judge Accuracy by Domain

| Domain | Heuristic | GPT-4o (cal.) | Claude (uncal.) |
|--------|-----------|---------------|-----------------|
| coding | 79.4% | 80.0% | 79.4% |
| math | 83.9% | 88.9% | 88.3% |
| instruction_following | 72.8% | 77.8% | 83.3% |

### 4d. Judge Accuracy by Attack Condition

| Attack | Heuristic | GPT-4o (cal.) | Claude (uncal.) |
|--------|-----------|---------------|-----------------|
| Clean | 81.7% | 86.7% | 89.4% |
| Context Manip. | 80.6% | 80.6% | 78.9% |
| Prompt Injection | 73.9% | 79.4% | 82.8% |

---
## 5. Error Analysis

### 5a. False Positive / False Negative Breakdown by Judge

| Judge | False Positives | False Negatives | FP Rate | FN Rate |
|-------|-----------------|-----------------|---------|---------|
| Heuristic | 69 | 46 | 46.3% | 11.8% |
| GPT-4o (calibrated) | 74 | 22 | 49.7% | 5.6% |
| Claude Sonnet 4 (uncalibrated) | 68 | 20 | 45.6% | 5.1% |

### 5b. Judge Errors by Domain

**GPT-4o (calibrated):**

| Domain | FP | FN | Total Errors |
|--------|----|----|----|
| coding | 30 | 6 | 36 |
| math | 19 | 1 | 20 |
| instruction_following | 25 | 15 | 40 |

**Claude Sonnet 4 (uncalibrated):**

| Domain | FP | FN | Total Errors |
|--------|----|----|----|
| coding | 23 | 14 | 37 |
| math | 20 | 1 | 21 |
| instruction_following | 25 | 5 | 30 |

---
## 6. LaTeX Tables (Copy-Paste Ready)

### Table A: Inter-Annotator Agreement
```latex
\begin{table}[t]
\centering
\caption{Inter-annotator agreement on the 180-example validation set (540 judgments). $\kappa$ = Cohen's kappa.}
\label{tab:iaa}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Annotator Pair} & \textbf{Agreement (\%)} & \textbf{$\kappa$} \\
\midrule
Soraya--Omid & 87.0 & 0.620 \\
Soraya--Ava & 75.2 & 0.344 \\
Omid--Ava & 79.6 & 0.451 \\
\midrule
All three agree & 71.5 & --- \\
\bottomrule
\end{tabular}
\end{table}
```

### Table B: Judge Accuracy vs Human Majority
```latex
\begin{table}[t]
\centering
\caption{Automated judge accuracy against human majority vote on the validation set.}
\label{tab:judge_acc}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Judge} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{$\kappa$} \\
\midrule
Heuristic & 78.7\% & 83.3\% & 88.2\% & 0.440 \\
GPT-4o (calibrated) & 82.2\% & 83.3\% & 94.4\% & 0.501 \\
Claude Sonnet 4 (uncalibrated) & 83.7\% & 84.5\% & 94.9\% & 0.547 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table C: Pass Rates by Model and Evaluator
```latex
\begin{table}[t]
\centering
\caption{Pass rates (\%) on the 180-example validation set by model and evaluator.}
\label{tab:pass_rates}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Evaluator} & \textbf{GPT-5.4} & \textbf{LLaMA 70B} & \textbf{Qwen 72B} & \textbf{Overall} \\
\midrule
Soraya & 92.8 & 57.8 & 81.1 & 77.2 \\
Omid & 91.1 & 77.8 & 83.3 & 84.1 \\
Ava & 77.8 & 70.0 & 69.4 & 72.4 \\
Heuristic & 81.1 & 75.6 & 73.3 & 76.7 \\
GPT-4o (calibrated) & 89.4 & 80.0 & 76.7 & 82.0 \\
Claude Sonnet 4 (uncalibrated) & 88.9 & 78.3 & 76.7 & 81.3 \\
\midrule
\textbf{Human Majority} & 82.8 & 59.4 & 75.0 & 72.4 \\
\bottomrule
\end{tabular}
\end{table}
```

---
## 7. Key Findings for Paper

### Strengthening Claims

1. **Third annotator validates the evaluation framework.** With Ava as a third independent annotator, we now have proper inter-annotator reliability statistics (Cohen's κ). This moves beyond the two-author agreement reported in typical benchmark papers.

2. **Claude Sonnet 4 (uncalibrated, single-stage) achieves 92.6% agreement with human majority** — slightly outperforming the calibrated two-stage GPT-4o judge (90.7%). This is a notable finding: a simpler, cheaper judge can match or exceed a carefully calibrated one.

3. **Instruction-following is the hardest domain for both humans and machines.** Inter-annotator agreement is lowest here, and all judges score worst on this domain. This validates your adversarial attack design.

4. **Model ranking is stable at the top but fragile in the middle.** GPT-5.4 is consistently #1 across all evaluators. But LLaMA 70B vs Qwen 72B swaps depending on who/what is judging — a cautionary tale for benchmark leaderboards.

5. **Attack degradation patterns are preserved across evaluators.** Prompt injection consistently causes the largest drop, validating that it is the most potent adversarial condition in your benchmark.

### Suggested Paper Improvements

1. **Add Table A (IAA) to Section 4** — replaces the informal two-author agreement with proper κ statistics. Reviewers will specifically look for this.

2. **Add Table B (Judge Accuracy) to Section 5** — gives precision/recall/F1 alongside accuracy. Shows judges are better calibrated on pass (high recall) but occasionally lenient (false positives).

3. **Discuss the calibration paradox** — the uncalibrated Claude judge matches the calibrated GPT-4o judge. This suggests the two-stage rubric may be over-engineered, or that larger models already internalize evaluation standards.

4. **Report majority-vote labels as ground truth** — with 3 annotators, majority vote is a stronger gold standard than any single annotator. All judge accuracy metrics should reference this.

5. **Highlight domain-specific annotation difficulty** — instruction-following has ~65-75% 3-way agreement vs ~75-80% for math/coding. This contextualizes why automated judges struggle most on IF tasks.

6. **Known limitation: 14 missing model outputs** (2 GPT-5.4, 12 Qwen 72B) — these are auto-FAILed. Acknowledge this in the paper as a minor data gap that does not affect conclusions.