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
| **Theorem 1** | **Asymptotic Persistence ($\Delta > 0$)** | [01_compute_delta.py](../scripts/06_theorem/01_compute_delta.py) |
| **Theorem 2** | **Split-Invariance ($\rho \approx 1.0$ vs $0.37$)** | [02_same_split_proof.py](../scripts/06_theorem/02_same_split_proof.py) |
| **Theorem 3** | **Dimensionality Effect ($\partial\Delta/\partial d > 0$)** | [03_dimensionality_effect.py](../scripts/06_theorem/03_dimensionality_effect.py) |
| Figure 6 | Bimodal separation violin plot | [02_same_split_proof.py](../scripts/06_theorem/02_same_split_proof.py) |
| **Neural Same-Split** | **Within-neural lottery ≈ 9% (not 0%), cross-class 59%** | [22_neural_same_split_experiment.py](../scripts/04_official_roadmap/22_neural_same_split_experiment.py) |
| **Top-3 Feature Overlap** | **76.8% partial disagreement, 8% complete (Adult & COMPAS)** | [20_consequential_disagreement.py](../scripts/04_official_roadmap/20_consequential_disagreement.py) |
| **Same-Split Cross-Class** | **61.9% lottery rate even on identical splits** | [21_controlled_split_experiment.py](../scripts/04_official_roadmap/21_controlled_split_experiment.py) |
| **Interaction Density** | **Negative correlation with Lottery (r = -0.251)** | [23_interaction_density_experiment.py](../scripts/04_official_roadmap/23_interaction_density_experiment.py) |
| **Rx Calibration** | **89.2% high freq vs 34.1% low freq instances** | [24_rx_calibration_experiment.py](../scripts/04_official_roadmap/24_rx_calibration_experiment.py) |
| **Linear SVMs** | **Internal agreement ρ = 0.76–0.83** | [18_linear_svm_experiment.py](../scripts/04_official_roadmap/18_linear_svm_experiment.py) |
| **Ridge / ElasticNet** | **Internal agreement ρ = 0.80 / 0.79** | [19_extended_linear_experiment.py](../scripts/04_official_roadmap/19_extended_linear_experiment.py) |

---

## Technical Verification

- **Spearman ρ computation** verified in `11_reliability_score.py`
- **Data Foundation**: All raw datasets are retrieved via [download_data.py](../data/download_data.py) (see [data/README.md](../data/README.md))
- **Results Mirror**: All reported metrics are consolidated in [elite_results.json](../results/00_publication_findings/elite_results.json)
- **Codebase Integrity**: The repository serves as a bit-for-bit executable mirror of the reported manuscript findings.
