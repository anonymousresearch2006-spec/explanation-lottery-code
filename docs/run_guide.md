# Execution and Reproducibility Guide

## Overview

This guide provides complete instructions for reproducing the experimental pipeline and results for the **Explanation Lottery** research project.

The repository is designed to support fully automated experiment execution with minimal manual configuration. Following the steps below allows any researcher to reconstruct the experimental environment, execute all experiments, and regenerate reported findings.

The reproducibility pipeline performs:

* environment verification,
* dataset acquisition,
* experiment orchestration,
* result generation, and
* output consolidation.

---

## 1. System Requirements

### Python Environment

* Python **3.8 or higher**
* Recommended: Python 3.9–3.11

### Operating Systems

* Windows
* macOS
* Linux

The pipeline is designed to be cross-platform compatible.

---

### Recommended Setup (Best Practice)

Use a virtual environment to isolate dependencies.

#### Create Virtual Environment

```
python -m venv venv
```

#### Activate Environment

**Windows**

```
venv\Scripts\activate
```

**macOS / Linux**

```
source venv/bin/activate
```

---

## 2. Dependency Installation

Install the required scientific computing stack:

```
pip install -r requirements.txt
```

This installs all dependencies required for:

* model training
* explanation computation
* statistical analysis
* visualization
* experiment orchestration

---

## 3. Dataset Environment Setup

The repository does **not include raw datasets** to ensure repository size control and maintain reproducibility standards.

Datasets must be downloaded automatically using the provided retrieval script.

### Download Required Datasets

Run from the project root:

```
python data/download_data.py
```

### What the Script Does

* downloads required datasets from OpenML
* creates the local data directory structure
* caches datasets for reuse
* skips datasets that already exist
* ensures consistent dataset naming
* prepares data for the experimental pipeline

This step must be completed before running experiments.

---

## 4. Full Experimental Reproduction

To execute the complete experimental suite and regenerate all reported findings:

```
python reproduce.py
```

---

### Execution Pipeline

The reproducibility entry point performs the following steps:

1. Verifies required dependencies and environment configuration.
2. Validates dataset availability.
3. Executes the official experimental roadmap (17 experiments).
4. Generates intermediate outputs and logs.
5. Aggregates results into standardized output files.

The full pipeline may require significant computation time depending on system resources.

---

## 5. Running Individual Experiments

Experiments can be executed independently for targeted analysis or debugging.

Example:

```
python scripts/04_official_roadmap/01_novelty_surprise.py
```

This allows:

* incremental execution,
* result inspection,
* experiment validation, and
* targeted reproduction.

All scripts assume execution from the project root directory.

---

## 6. Output Structure and Verification

After execution, results are stored in the `/results/` directory.

### Final Aggregated Results

```
results/00_publication_findings/
```

Contains consolidated metrics and summary statistics used in the main analysis.

---

### Visualizations

```
results/01_visualizations/
```

Contains publication-quality figures and plots.

---

### Raw Experiment Outputs

```
results/02_experiment_raw/
```

Contains dataset-level intermediate outputs.

---

### Execution Logs

```
results/04_system_logs/
```

Contains detailed runtime logs useful for debugging and experiment tracing.

---

## 7. Expected Execution Workflow

The typical reproducibility workflow follows this sequence:

1. Install dependencies.
2. Download datasets.
3. Run the reproducibility pipeline.
4. Inspect generated outputs.

This structured workflow ensures consistent results across environments.

---

## 8. Troubleshooting

### Missing Dependencies

If execution fails due to missing packages:

```
pip install -r requirements.txt
```

---

### Missing Dataset Errors

Reinitialize the dataset cache:

```
python data/download_data.py
```

---

### Execution Errors

Inspect detailed logs:

```
results/04_system_logs/
```

Logs contain error traces and diagnostic information.

---

### Windows Encoding Issues

Some terminals may display encoding artifacts for statistical symbols.

To fix:

```
python scripts/05_infrastructure/_final_fixer.py
```

---

## 9. Reproducibility Commitment

The repository is designed to ensure that:

* experiments execute deterministically,
* results can be independently verified,
* outputs are consistently generated across systems, and
* the full experimental pipeline can be reconstructed from scratch.

All reported findings are generated directly through the provided execution pipeline.

---

## Further Reading
*   **[Research Methodology Overview](./method_overview.md)**
*   **[Project Structure and Component Map](./project_structure.md)**
*   **[Repository Overview](../README.md)**
