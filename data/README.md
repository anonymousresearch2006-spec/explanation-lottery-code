# Explanation Lottery — Data Repository

## Overview

This directory manages all dataset assets required to reproduce the experiments for the **Explanation Lottery** research project.

To ensure:

* anonymous peer review (e.g., TMLR double-blind policy),
* repository size control, and
* full reproducibility,

raw datasets are **not stored in the GitHub repository**. Instead, datasets are automatically downloaded and cached locally using a reproducible data acquisition pipeline.

This design follows standard machine learning research practices for reproducible experimentation.

---

## Dataset Inventory

The project uses **20 publicly available datasets** obtained from the OpenML platform.

These datasets span multiple application domains to evaluate explanation stability across heterogeneous contexts.

### Finance

* Credit risk prediction
* Banknote authentication

### Healthcare

* Breast Cancer Wisconsin
* Diabetes datasets
* Medical diagnostic classification tasks

### Software Engineering

* PC3 and PC4 (NASA defect prediction datasets)

### Security

* Spambase
* Phishing detection datasets

### Additional Domains

Additional datasets cover general classification and tabular learning tasks to ensure diversity in model behavior and explanation outcomes.

All datasets are obtained directly from:

**OpenML:** https://www.openml.org/

---

## Why Datasets Are Not Stored in GitHub

Datasets are intentionally excluded from version control for the following reasons:

* Prevent repository size inflation
* Ensure reproducible environment creation
* Avoid distributing cached data artifacts
* Maintain anonymous submission requirements
* Follow machine learning reproducibility best practices

The repository instead provides an automated data retrieval pipeline.

---

## Directory Behavior

After cloning the repository, this directory initially contains only:

```
data/
 ├── README.md
 └── download_data.py
```

After running the download script, the following structure will be created locally:

```
data/
 ├── openml_cache/
 │    ├── <dataset_id_name>/
 │    └── ...
```

This cache exists only on the local machine and is ignored by Git.

---

## How to Reproduce the Data Environment

### Step 1 — Clone Repository

```
git clone <repository_url>
cd <repository>
```

### Step 2 — Download All Required Datasets

Run from the project root:

```
python data/download_data.py
```

This script will:

* download all required datasets from OpenML
* cache datasets locally
* verify dataset integrity
* skip datasets that already exist
* automatically create required directory structure

---

## Storage and Naming Convention

Datasets are stored using a reproducible naming scheme:

```
data/openml_cache/<dataset_id>_<dataset_name>/
```

Example:

```
data/openml_cache/1510_cancer/
```

This ensures deterministic dataset references and reproducible experiment environments.

---

## Git and Version Control Strategy

This repository uses .gitignore rules to ensure:

* dataset files remain local
* large files are never uploaded to GitHub
* repository state remains consistent across environments

Ignored paths include:

```
data/*
!data/README.md
```

This means:

* dataset contents exist locally
* only documentation is tracked in GitHub
* only the download script is tracked in GitHub

---

## Reproducibility Guarantee

The data pipeline is designed so that:

* any reviewer can recreate the full dataset environment
* experiments run consistently across machines
* no manual dataset setup is required

All experiments assume data exists in the data/ directory.

If datasets are missing, the reproducibility pipeline will prompt the user to run the download script.

---

## Troubleshooting

### Missing Dataset Errors

If experiments fail due to missing data, run:

```
python data/download_data.py
```

### OpenML Connection Issues

Ensure internet connectivity and retry.

---

## Notes

This repository is provided as part of an anonymous research submission.
All dataset sources are public and accessible through OpenML.
