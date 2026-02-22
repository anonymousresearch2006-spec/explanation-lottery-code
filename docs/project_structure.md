# Project Structure & Component Map

## Overview

This document provides a detailed description of the repository architecture and its organization across the research lifecycle. The project is structured to ensure:

* full experimental reproducibility,
* modular development,
* clear separation of concerns,
* traceable research progression, and
* reviewer-friendly navigation.

Each component of the repository is organized according to its functional role in the experimental pipeline, from data acquisition through model training, analysis, and result generation.

The architecture follows standard machine learning research practices for scalable experimentation and reproducible evaluation.

---

## Repository Organization Principles

The repository design follows several core principles:

### Modular Design

Each directory represents a distinct stage of the research pipeline. Components can be executed independently or through the main reproducibility entry point.

### Lifecycle Traceability

Scripts are organized according to research development phases, allowing reviewers to trace the evolution of experimental findings.

### Reproducibility by Construction

All outputs, datasets, and model artifacts are generated through deterministic procedures and centralized storage.

### Separation of Inputs, Computation, and Outputs

Data, execution logic, and experimental results are maintained in separate directories to avoid coupling and ensure clarity.

---

## Repository Layout

The project is structured into four primary root directories.

```
project_root/
 ├── scripts/
 ├── results/
 ├── data/
 └── docs/
```

Each directory corresponds to a specific stage of the experimental workflow.

---

## `/scripts/` — Research Engine

This directory contains all executable research code. Scripts are organized by research phase to reflect the methodological progression of the project.

### `01_foundations/`

Initial exploratory experiments establishing baseline behavior and identifying the presence of the explanation disagreement phenomenon.

Typical tasks include:

* baseline model comparisons,
* preliminary SHAP analysis,
* early lottery rate estimation.

---

### `02_extension/`

Large-scale experimentation extending the initial findings.

Includes:

* evaluation across 20+ datasets,
* cross-family model comparisons,
* robustness testing,
* expanded analysis pipelines.

This stage validates the generality of the observed phenomenon.

---

### `03_rigor_and_proofs/`

Formal statistical validation and theoretical investigation.

Includes:

* statistical significance testing,
* stochasticity control,
* mixed-effects modeling,
* formal verification experiments,
* theoretical and PAC-style bounds.

This phase ensures scientific validity and robustness of conclusions.

---

### `04_official_roadmap/`

Contains the finalized experimental pipeline used to generate the primary results reported in the study.

* 17 optimized experiments
* standardized evaluation procedures
* publication-aligned execution order
* reproducible figure generation

This directory represents the canonical experimental workflow.

---

### `05_infrastructure/`

Execution and environment orchestration utilities.

Includes:

* cross-platform execution scripts,
* pipeline runners,
* system setup tools,
* compatibility utilities,
* experiment orchestration logic.

These scripts coordinate execution across all research phases.

---

## `/results/` — Experimental Outputs

This directory stores all generated artifacts produced by the experimental pipeline.
Outputs are centralized to ensure consistent result management and reproducibility.

### `00_publication_findings/`

Final aggregated results supporting the primary scientific claims.

Includes:

* processed datasets,
* summary statistics,
* CSV/JSON result tables.

---

### `01_visualizations/`

Publication-quality visual outputs.

Includes:

* high-resolution figures,
* plots and diagrams,
* visual summaries of findings.

---

### `02_experiment_raw/`

Granular per-dataset outputs generated during experimentation.

Includes intermediate metrics and dataset-level results used for analysis.

---

### `03_model_weights/`

Serialized model checkpoints enabling experiment replication without retraining.

---

### `04_system_logs/`

Execution logs and diagnostic traces providing full experiment traceability.

---

### `05_model_metadata/`

Model-specific configuration data and algorithm metadata.

---

## `/data/` — Dataset Management

This directory manages all dataset assets required for experimentation.

Datasets are obtained automatically through a reproducible retrieval pipeline and stored locally.

### Dataset Management Strategy

* Automated download via `data/download_data.py`
* Local caching of OpenML datasets
* Deterministic dataset naming convention

### Storage Convention

Datasets follow an `ID_context` naming scheme:

```
<ID>_<dataset_name>
```

Example:

```
31_credit
```

This provides consistent dataset referencing across experiments.

The data directory is treated as a reproducible environment dependency rather than a version-controlled asset.

---

## `/docs/` — Technical Documentation

This directory contains supporting documentation explaining the methodology, execution workflow, and repository organization.

### [method_overview.md](./method_overview.md)

Describes the research methodology and experimental pipeline.

### [project_structure.md](./project_structure.md)

Explains repository architecture and component interactions.

### [run_guide.md](./run_guide.md)

Provides comprehensive execution and setup instructions.

### [paper_code_mapping.md](./paper_code_mapping.md)

Explicit mapping of manuscript results to their corresponding research scripts.

Documentation is designed to help reviewers and researchers understand the experimental workflow and reproduce results.

---

## Key Root-Level Files

### `reproduce.py`

Single entry-point script that reconstructs the complete experimental pipeline from scratch.
This script verifies dependencies, prepares the environment, and executes the official experiment roadmap.

---

### `requirements.txt`

Defines the Python environment and dependency specifications required to execute all experiments.

---

### `README.md`

Provides a high-level research summary, results, and quick-start guide.

---

## Execution Flow Summary

The typical workflow follows this sequence:

1. Download datasets using the data pipeline.
2. Execute the reproducibility entry point.
3. Run official experiments.
4. Generate results and visualizations.
5. Store outputs in the results directory.

This structured workflow ensures consistent experiment execution across environments.

---

## Reproducibility Commitment

The repository is designed so that any researcher can:

* reconstruct the experimental environment,
* execute the full pipeline,
* reproduce all reported findings, and
* inspect intermediate outputs.

All components are organized to support transparent and verifiable scientific experimentation.
