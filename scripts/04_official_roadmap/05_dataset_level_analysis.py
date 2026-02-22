"""
=============================================================================
05_OPTIMISED_001: DATASET-LEVEL ANALYSIS
=============================================================================
Tier A -- Item 5 | Impact: 4/5 | Effort: 1 day

Goal: Deeper dataset-level analysis showing what drives disagreement.
- Lottery rate per dataset
- Correlation with dimensionality
- Domain differences
- Variation explanation

Output: results/optimised_001/05_dataset_level_analysis/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
import json
import os

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '05_dataset_level_analysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 05: DATASET-LEVEL ANALYSIS")
print("=" * 70)

# Load data
df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
print(f"\nLoaded {len(df):,} comparisons from {df['dataset_id'].nunique()} datasets")

# =============================================================================
# ANALYSIS 1: PER-DATASET LOTTERY RATES
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: PER-DATASET LOTTERY RATES")
print("=" * 70)

dataset_stats = []
for dataset_id in sorted(df['dataset_id'].unique()):
    d = df[df['dataset_id'] == dataset_id]
    
    lottery_rate = (d['spearman'] < 0.5).mean() * 100
    mean_rho = d['spearman'].mean()
    std_rho = d['spearman'].std()
    n_comparisons = len(d)
    n_features = d['n_features'].iloc[0] if 'n_features' in d.columns else None
    
    dataset_stats.append({
        'dataset_id': int(dataset_id),
        'n_comparisons': int(n_comparisons),
        'n_features': int(n_features) if n_features is not None else None,
        'mean_rho': float(mean_rho),
        'std_rho': float(std_rho),
        'lottery_rate': float(lottery_rate),
        'median_rho': float(d['spearman'].median()),
        'min_rho': float(d['spearman'].min()),
        'max_rho': float(d['spearman'].max()),
        'q25_rho': float(d['spearman'].quantile(0.25)),
        'q75_rho': float(d['spearman'].quantile(0.75))
    })

ds_df = pd.DataFrame(dataset_stats)

print(f"\n  {'Dataset ID':<12} {'N':<8} {'Features':<10} {'Mean rho':<10} {'Lottery %':<10}")
print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

for _, row in ds_df.sort_values('lottery_rate', ascending=False).iterrows():
    feat_str = f"{int(row['n_features'])}" if pd.notna(row['n_features']) else "N/A"
    print(f"  {int(row['dataset_id']):<12} {int(row['n_comparisons']):<8} {feat_str:<10} {row['mean_rho']:.3f}     {row['lottery_rate']:.1f}%")

# Summary statistics
print(f"\n  Dataset-level summary:")
print(f"    Mean lottery rate: {ds_df['lottery_rate'].mean():.1f}%")
print(f"    Min lottery rate:  {ds_df['lottery_rate'].min():.1f}%")
print(f"    Max lottery rate:  {ds_df['lottery_rate'].max():.1f}%")
print(f"    Std of rates:      {ds_df['lottery_rate'].std():.1f}%")

# =============================================================================
# ANALYSIS 2: DIMENSIONALITY CORRELATION
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: DIMENSIONALITY vs AGREEMENT")
print("=" * 70)

dim_results = {}

if 'n_features' in df.columns:
    valid_ds = ds_df[ds_df['n_features'].notna()]
    
    # Correlation: n_features vs agreement
    corr_rho, corr_p = stats.spearmanr(valid_ds['n_features'], valid_ds['mean_rho'])
    print(f"\n  Spearman correlation (n_features vs mean rho): r = {corr_rho:.3f}, p = {corr_p:.4f}")
    
    # Correlation: n_features vs lottery rate
    corr_lottery, corr_p2 = stats.spearmanr(valid_ds['n_features'], valid_ds['lottery_rate'])
    print(f"  Spearman correlation (n_features vs lottery rate): r = {corr_lottery:.3f}, p = {corr_p2:.4f}")
    
    # Bin by dimensionality
    dim_bins = pd.cut(valid_ds['n_features'], bins=[0, 10, 20, 50, 200, 10000], 
                      labels=['1-10', '11-20', '21-50', '51-200', '200+'])
    
    print(f"\n  By dimensionality bin:")
    for bin_label in ['1-10', '11-20', '21-50', '51-200', '200+']:
        bin_data = valid_ds[dim_bins == bin_label]
        if len(bin_data) > 0:
            print(f"    {bin_label}: mean rho = {bin_data['mean_rho'].mean():.3f}, "
                  f"lottery = {bin_data['lottery_rate'].mean():.1f}%, "
                  f"n_datasets = {len(bin_data)}")
    
    dim_results = {
        'features_vs_agreement_corr': float(corr_rho),
        'features_vs_agreement_p': float(corr_p),
        'features_vs_lottery_corr': float(corr_lottery),
        'features_vs_lottery_p': float(corr_p2)
    }
    
    if corr_rho < 0:
        print(f"\n  * FINDING: Higher dimensionality -> LOWER agreement (r = {corr_rho:.3f})")
        print(f"    This supports the curse of dimensionality for explanations")
    else:
        print(f"\n  * FINDING: No clear dimensionality effect (r = {corr_rho:.3f})")

# =============================================================================
# ANALYSIS 3: DOMAIN DIFFERENCES
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: DOMAIN DIFFERENCES")
print("=" * 70)

# Domain mapping
DOMAIN_MAP = {
    31: 'Finance', 37: 'Healthcare', 44: 'Security', 1462: 'Finance',
    1489: 'Telecom', 1494: 'Chemistry', 1510: 'Healthcare', 4534: 'Security',
    40536: 'Social', 40975: 'Consumer', 41027: 'Game', 23512: 'Physics',
    1063: 'Software', 40668: 'Game', 40670: 'Biology', 1504: 'Manufacturing',
    1547: 'Neuroscience', 40499: 'Image'
}

df['domain'] = df['dataset_id'].map(lambda x: DOMAIN_MAP.get(x, 'Other'))
domain_analysis = df.groupby('domain').agg(
    mean_rho=('spearman', 'mean'),
    std_rho=('spearman', 'std'),
    lottery_rate=('spearman', lambda x: (x < 0.5).mean() * 100),
    n_comparisons=('spearman', 'count')
).round(3)

print(f"\n  {'Domain':<15} {'Mean rho':<10} {'Lottery %':<12} {'N'}")
print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*8}")
for domain, row in domain_analysis.sort_values('lottery_rate', ascending=False).iterrows():
    print(f"  {domain:<15} {row['mean_rho']:.3f}     {row['lottery_rate']:.1f}%       {int(row['n_comparisons'])}")

# =============================================================================
# ANALYSIS 4: MULTIVARIATE REGRESSION -- WHAT DRIVES DISAGREEMENT?
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: DRIVERS OF EXPLANATION DISAGREEMENT")
print("=" * 70)

# Available predictors
predictors = []
predictor_names = []

if 'n_features' in df.columns:
    predictors.append(df['n_features'])
    predictor_names.append('n_features')

if 'n_samples' in df.columns:
    predictors.append(df['n_samples'])
    predictor_names.append('n_samples')

# Model pair type
tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
df['is_cross_class'] = df.apply(
    lambda r: int((r['model_a'] in tree_models) != (r['model_b'] in tree_models)), axis=1)
predictors.append(df['is_cross_class'])
predictor_names.append('is_cross_class')

print(f"\n  Predictors: {predictor_names}")

# Individual effects
print(f"\n  Individual predictor effects on agreement (rho):")
driver_effects = {}
for name, pred in zip(predictor_names, predictors):
    corr, p_val = stats.spearmanr(pred.dropna(), df['spearman'].loc[pred.dropna().index])
    driver_effects[name] = {'correlation': float(corr), 'p_value': float(p_val)}
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"    {name}: r = {corr:.3f}, p = {p_val:.2e} {sig}")

# Rank drivers
print(f"\n  DRIVER RANKING (by absolute correlation):")
sorted_drivers = sorted(driver_effects.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
for rank, (name, effect) in enumerate(sorted_drivers, 1):
    print(f"    {rank}. {name}: |r| = {abs(effect['correlation']):.3f}")

# =============================================================================
# ANALYSIS 5: DATASET VARIABILITY ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: BETWEEN vs WITHIN DATASET VARIABILITY")
print("=" * 70)

# Between-dataset variability
between_var = ds_df['mean_rho'].var()
# Within-dataset variability (average)
within_var = ds_df['std_rho'].apply(lambda x: x**2).mean()

total_var = df['spearman'].var()

print(f"\n  Total variance:           {total_var:.4f}")
print(f"  Between-dataset variance: {between_var:.4f} ({between_var/total_var*100:.1f}%)")
print(f"  Within-dataset variance:  {within_var:.4f} ({within_var/total_var*100:.1f}%)")
print(f"\n  * {'Dataset choice' if between_var > within_var else 'Instance-level factors'} "
      f"explain more variance in explanation agreement")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'per_dataset': dataset_stats,
    'dimensionality': dim_results,
    'domain_analysis': domain_analysis.to_dict(),
    'drivers': driver_effects,
    'variability': {
        'total_variance': float(total_var),
        'between_dataset': float(between_var),
        'within_dataset': float(within_var),
        'between_pct': float(between_var/total_var*100),
        'within_pct': float(within_var/total_var*100)
    },
    'summary': {
        'mean_lottery_rate': float(ds_df['lottery_rate'].mean()),
        'min_lottery_rate': float(ds_df['lottery_rate'].min()),
        'max_lottery_rate': float(ds_df['lottery_rate'].max()),
        'n_datasets': int(len(ds_df))
    }
}

output_file = os.path.join(OUTPUT_DIR, '05_dataset_level_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

# CSV of per-dataset results
ds_df.to_csv(os.path.join(OUTPUT_DIR, '05_dataset_level_table.csv'), index=False)
print(f"  Saved: {os.path.join(OUTPUT_DIR, '05_dataset_level_table.csv')}")

print("\n" + "=" * 70)
print("EXPERIMENT 05 COMPLETE")
print("=" * 70)
