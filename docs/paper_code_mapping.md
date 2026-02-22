# Paper–Code Traceability Map

This document maps all key manuscript results to their generating scripts.
It ensures full reproducibility and verification of reported findings.

---

## Manuscript Results → Script Mapping

| Manuscript Component | Key Finding | Generation Script |
|---|---|---|
| Abstract & Section 5.1 | 35.4% Lottery Rate (ρ < 0.5) | [12_figures_visualization.py](../scripts/04_official_roadmap/12_figures_visualization.py) |
| Figure 1 | Lottery rate bar chart by threshold | [12_figures_visualization.py](../scripts/04_official_roadmap/12_figures_visualization.py) |
| Figure 2 & Table 3 | Tree-Tree vs Tree-Linear distributions | [12_figures_visualization.py](../scripts/04_official_roadmap/12_figures_visualization.py) |
| Section 5.8 (Synthetic) | Ground Truth (87% recovery) | [04_synthetic_ground_truth.py](../scripts/04_official_roadmap/04_synthetic_ground_truth.py) |
| Section 5.5 (COMPAS) | XGBoost vs LR feature attribution | [16_compas_legal.py](../scripts/04_official_roadmap/16_compas_legal.py) |
| Section 5.7 (MNIST) | 32.2% Lottery Rate | [10_mnist_experiment.py](../scripts/04_official_roadmap/10_mnist_experiment.py) |
| Section 6.1 (Reliability) | Cross-model calibration (ρ = 0.70) | [11_reliability_score.py](../scripts/04_official_roadmap/11_reliability_score.py) |
| Table 4 | 24 datasets across 16 domains | [15_extra_datasets.py](../scripts/04_official_roadmap/15_extra_datasets.py) |

---

## Technical Verification

- Spearman ρ computation verified in `11_reliability_score.py`
- All scripts derive results from a shared data foundation (`combined_results.csv`)
- Codebase functions as an executable mirror of the manuscript.