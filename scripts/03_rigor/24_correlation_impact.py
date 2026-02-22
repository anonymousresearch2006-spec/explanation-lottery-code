"""
24_correlation_impact.py
COROLLARY 1 TEST: Feature Correlation vs. Explanation Lottery
Goal: Prove that datasets with higher internal feature correlation have higher lottery rates.
This provides empirical support for the "Correlation Sensitivity" corollary.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from pathlib import Path
import warnings
import json
import os

warnings.filterwarnings('ignore')

print("=" * 70)
print("COROLLARY 1: FEATURE CORRELATION vs. LOTTERY RATE")
print("=" * 70)

RESULTS_DIR = Path("results/rigor")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 25 datasets from our diversity study
DATASETS = [
    (44, "spambase"), (37, "diabetes"), (1462, "banknote"), (1510, "wdbc"),
    (31, "credit-g"), (1461, "bank-marketing"), (1590, "adult"), (1480, "ilpd"),
    (53, "heart-statlog"), (1063, "kc2"), (1068, "pc1"), (1169, "airlines"),
    (40975, "car"), (40994, "climate-simulation"), (1464, "blood-transfusion"),
    (1467, "cnae-9"), (40983, "wilt"), (1489, "phoneme"), (1494, "qsar-biodeg"),
    (1504, "steel-plates-fault"), (1547, "eeg-eye-state"), (40499, "texture"),
    (40668, "connect-4"), (40670, "dna"), (329, "hayes-roth")
]

correlation_stats = []

# Load the observed lottery rates from previous diversity results
# If not available, we can estimate them or use the combined results
DIVERSITY_RESULTS_PATH = Path("results/dataset_diversity/dataset_diversity_results.csv")
if DIVERSITY_RESULTS_PATH.exists():
    div_df = pd.read_csv(DIVERSITY_RESULTS_PATH)
    # Average Rho per dataset
    avg_rho_per_dataset = div_df.groupby('dataset')['mean_rho'].mean().to_dict()
else:
    print("Warning: Diversity results not found. Using dummy rates for code validation.")
    avg_rho_per_dataset = {}

for ds_id, ds_name in DATASETS:
    print(f"Analyzing correlation for: {ds_name}...", end=" ")
    try:
        data = fetch_openml(data_id=ds_id, as_frame=True, parser='auto')
        X = data.data
        
        # Handle non-numeric for correlation
        X_num = X.select_dtypes(include=[np.number])
        if X_num.shape[1] < 2:
            print("SKIPPED (Too few numeric features)")
            continue
            
        # Calculate mean absolute pairwise correlation
        corr_matrix = X_num.corr().abs()
        # Take upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_abs_corr = upper.stack().mean()
        
        # Lottery Rate Proxy: (1 - Avg Rho)
        rho = avg_rho_per_dataset.get(ds_name, np.nan)
        
        correlation_stats.append({
            'dataset': ds_name,
            'mean_feature_corr': float(mean_abs_corr),
            'observed_rho': float(rho),
            'lottery_proxy': 1.0 - float(rho) if not np.isnan(rho) else np.nan
        })
        print(f"OK (Corr: {mean_abs_corr:.3f})")
        
    except Exception as e:
        print(f"FAILED: {str(e)[:50]}")

df_corr = pd.DataFrame(correlation_stats)
df_corr = df_corr.dropna()

if not df_corr.empty and len(df_corr) > 5:
    # Regress Correlation vs Lottery Proxy
    r_val, p_val = stats.pearsonr(df_corr['mean_feature_corr'], df_corr['lottery_proxy'])
    
    print("\n" + "=" * 70)
    print("COROLLARY 1 STATISTICAL RESULT")
    print("=" * 70)
    print(f"Pearson Correlation (Feature Correlation vs. Lottery Rate): r = {r_val:.3f}")
    print(f"p-value: {p_val:.4f}")
    
    if r_val > 0.3:
        print("\nINSIGHT: AS PREDICTED, high feature correlation EXACERBATES the lottery effect.")
    else:
        print("\nINSIGHT: Feature correlation has a weak or non-linear impact on the lottery.")

    # Save results
    df_corr.to_csv(RESULTS_DIR / "correlation_impact_results.csv", index=False)
    summary = {
        'correlation_coefficient': float(r_val),
        'p_value': float(p_val),
        'n_datasets': len(df_corr)
    }
    with open(RESULTS_DIR / "correlation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

print("\nDONE.")
