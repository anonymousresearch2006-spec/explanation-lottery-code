# The Explanation Lottery: Cross-Model Feature Attribution Disagreement

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Status:** Experimental Suite Complete — Optimised Roadmap 001  
> **Reproducibility:** Fully automated via `reproduce.py`

---

## Overview

Do machine learning models that agree on predictions also agree on explanations?

This project investigates the **Explanation Lottery** — a phenomenon where models produce identical predictions but contradictory feature attribution explanations. The study performs large-scale cross-model explanation comparisons using SHAP across multiple datasets and model families.

The repository provides a fully reproducible experimental pipeline, statistical analysis framework, and verification suite supporting the results reported in the manuscript.

---

## Research Summary

We conduct **93,510 pairwise SHAP comparisons** across **24 datasets** spanning 16 domains.

### Main Findings

- **35.4%** of prediction-agreeing model pairs show substantial explanation disagreement (Spearman ρ < 0.5).
- Disagreement is primarily driven by **inductive bias differences** rather than stochastic noise.
- The effect persists across tabular and image domains.
- We introduce a reliability score for cross-model explanation stability.

---

## Key Quantitative Results

| Metric | Value | Interpretation |
|---|---|---|
| Lottery Rate (ρ = 0.5) | **35.4%** | High instability in consensus models |
| Mean Spearman ρ | 0.57 ± 0.30 | Large variability in explanation agreement |
| MNIST Lottery Rate | 32.2% | Effect generalizes beyond tabular data |
| Reliability Score Threshold | ≥ 0.7 | Stable explanation region |
| Geometric Divergence | 51.7° | High attribution vector divergence |
| Cohen's d | 1.41 | Large effect size |

---

## Repository Structure

The project is organized for modular experimentation and full reproducibility.

```
scripts/              # Experimental pipeline and analysis modules
data/                 # Dataset download utilities (no raw data stored)
results/              # Generated outputs (summary results + figures)
docs/                 # Technical documentation
reproduce.py          # Main execution entry point
requirements.txt      # Environment specification
```

### scripts/

- **01_foundations/** — Initial discovery experiments.
- **02_extension/** — Scaled dataset evaluation and cross-family analysis.
- **03_rigor_and_proofs/** — Statistical verification and theoretical analysis.
- **04_official_roadmap/** — Core optimized experiments used in the manuscript.
- **05_infrastructure/** — Execution utilities and orchestration tools.

### data/

Datasets are **not stored in the repository**.  
They are automatically downloaded from OpenML for reproducibility.

### results/

- `00_publication_findings/` — Final aggregated results used in the paper.
- `01_visualizations/` — Publication-ready figures.

All other intermediate outputs are generated automatically and excluded from version control.

---

## Reproducibility

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

```bash
python data/download_data.py
```

### 3. Run full experimental pipeline

```bash
python reproduce.py
```

This executes the complete experimental roadmap and generates results in:

```
results/00_publication_findings/
```

For detailed instructions see:

```
docs/run_guide.md
```

---

## Case Study: High-Stakes Decision Making

Analysis of the COMPAS recidivism dataset shows that different model architectures attribute the same prediction to different primary factors:

- **XGBoost:** Prior Convictions as dominant driver.
- **Logistic Regression:** Age as dominant driver.

This discrepancy illustrates a potential reliability gap in explanation-based decision systems.

---

## Paper–Code Traceability

All reported results are directly traceable to specific scripts.  
See:

```
docs/paper_code_mapping.md
```

---

## License

MIT License.
