# QueryVault Evaluation Results

## Test Set Composition & Limitations
> [!WARNING]
> The `academic_paper.pdf` ('Attention Is All You Need') is a widely-known text that the Llama-3.3 70B model has likely memorized during pre-training. Its scores should be interpreted as an upper bound partially influenced by model prior knowledge.
> The `technical_manual.pdf` (a recently published 2026 arXiv paper) and `legal_policy.pdf` better isolate actual retrieval quality by minimizing pre-training contamination.

**Total Samples Evaluated**: 24

| Metric | Score |
|---|---|
| Exact Match (EM) | 0.0417 |
| Token F1 Score | 0.4673 |
| Semantic Similarity | 0.7130 |
| Avg Latency | 12.80s |
| P95 Latency | 20.51s |
