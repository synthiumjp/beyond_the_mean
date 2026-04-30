# Beyond the Mean: Within-Model Reliable Change Detection for LLM Evaluation

**Author:** Jon-Paul Cacioli (ORCID: 0009-0000-7054-2014)

**Pre-registration:** [OSF: osf.io/3dnsa](https://osf.io/3dnsa)

**Status:** Data collected. Analysis complete. Manuscript in preparation.

## Overview

This study adapts the Reliable Change Index (RCI; Jacobson & Truax, 1991) from clinical psychology to item-level model-version comparison on MMLU-Pro. Two within-family successor comparisons are tested: Llama 3 to 3.1 (minor update) and Qwen 2.5 to 3 (generational update), with K=10 stochastic samples per item at T=0.7 across 2,000 items (80,000 total trials).

## Key Finding

Across the full benchmark, most items show no reliable change (79% for Llama, 72% for Qwen). Among the analysable subset where stochastic variation occurs, change is predominantly bidirectional with large effect sizes (median |Δp| = 0.50 to 0.90). Full-benchmark churn rates are 21% (minor update) and 28% (generational update). The aggregate accuracy gain is the net residual of opposing item-level movements. Domain-level decomposition reveals that different domains gain and lose across model families.

## Repository Structure

```
beyond_the_mean/
├── data/                          # Trial-level data (JSONL)
│   ├── trials_llama3-8b.jsonl
│   ├── trials_llama3_1-8b.jsonl
│   ├── trials_qwen2_5-7b.jsonl
│   ├── trials_qwen3-8b.jsonl
│   ├── llama3-8b_H.jsonl         # Study 3 greedy baseline (Llama 3)
│   └── llama3.1-8b_H.jsonl       # Study 3 greedy baseline (Llama 3.1)
├── results/                       # Analysis outputs
│   └── btm_results.json
├── run_btm_inference.py           # Data collection script
├── run_btm_analysis.py            # Pre-registered analysis script
├── run_btm_supplementary.py       # Supplementary analyses
├── run_btm_effectsize.py          # Effect size and difficulty-bin analysis
├── local_config.py                # Model paths (not committed)
├── .gitignore
└── README.md
```

## Hardware

AMD Radeon RX 7900 GRE (16 GB VRAM), Q5_K_M quantisation, llama-cpp-python, Vulkan backend.

## Connected Work

- SDT calibration: [arXiv:2603.14893](https://arxiv.org/abs/2603.14893), [arXiv:2603.25112](https://arxiv.org/abs/2603.25112)
- Validity screening: [arXiv:2604.17714](https://arxiv.org/abs/2604.17714), [arXiv:2604.17716](https://arxiv.org/abs/2604.17716)
- Sandbagging: [arXiv:2604.25249](https://arxiv.org/abs/2604.25249)
- Quantisation: [arXiv:2604.08976](https://arxiv.org/abs/2604.08976)
- Metacognitive atlas: Submitted to NeurIPS 2026 E&D

## License

MIT
