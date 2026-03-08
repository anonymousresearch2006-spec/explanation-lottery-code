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

---

## Technical Verification

- **Spearman ρ computation** verified in `11_reliability_score.py`
- **Data Foundation**: All raw datasets are retrieved via [download_data.py](../data/download_data.py) (see [data/README.md](../data/README.md))
- **Results Mirror**: All reported metrics are consolidated in [elite_results.json](../results/00_publication_findings/elite_results.json)
- **Codebase Integrity**: The repository serves as a bit-for-bit executable mirror of the reported manuscript findings.
