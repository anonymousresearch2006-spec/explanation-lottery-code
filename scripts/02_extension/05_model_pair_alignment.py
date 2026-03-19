"""
Fix: Extract Tree-Tree vs Tree-Linear comparison
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import os

print("="*60)
print("EXTRACTING TREE vs LINEAR COMPARISON")
print("="*60)

# Load data
df = pd.read_csv('results/combined_results.csv')
print(f"\nLoaded {len(df):,} rows")
print(f"Columns: {list(df.columns)}")

# Check for model pair info
pair_cols = [c for c in df.columns if 'model' in c.lower() or 'pair' in c.lower()]
print(f"\nModel-related columns: {pair_cols}")

# Try to identify model pairs
if 'model1' in df.columns and 'model2' in df.columns:
    print("\nFound model1 and model2 columns")
    
    # Define model types
    tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    
    def get_pair_type(row):
        m1 = str(row['model1']).lower()
        m2 = str(row['model2']).lower()
        
        m1_tree = any(t in m1 for t in tree_models)
        m2_tree = any(t in m2 for t in tree_models)
        m1_lr = 'logistic' in m1 or 'lr' in m1
        m2_lr = 'logistic' in m2 or 'lr' in m2
        
        if m1_tree and m2_tree:
            return 'Tree-Tree'
        elif (m1_tree and m2_lr) or (m1_lr and m2_tree):
            return 'Tree-Linear'
        elif m1_lr and m2_lr:
            return 'Linear-Linear'
        else:
            return 'Unknown'
    
    df['pair_type'] = df.apply(get_pair_type, axis=1)
    
elif 'model_pair' in df.columns:
    print("\nFound model_pair column")
    print(f"Sample values: {df['model_pair'].head(10).tolist()}")
    
    tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    
    def get_pair_type_from_string(pair_str):
        pair_str = str(pair_str).lower()
        
        # Count tree models in the pair
        tree_count = sum(1 for t in tree_models if t in pair_str)
        has_lr = 'logistic' in pair_str or '_lr' in pair_str
        
        if tree_count == 2:
            return 'Tree-Tree'
        elif tree_count == 1 and has_lr:
            return 'Tree-Linear'
        elif has_lr and tree_count == 0:
            return 'Linear-Linear'
        elif tree_count >= 1:
            return 'Tree-Tree'  # Two different tree models
        else:
            return 'Unknown'
    
    df['pair_type'] = df['model_pair'].apply(get_pair_type_from_string)

else:
    # Try to infer from column names
    print("\nNo explicit model columns found. Checking data structure...")
    print(f"First row: {df.iloc[0].to_dict()}")
    
    # If we have columns like 'xgboost_lightgbm_spearman', extract
    spearman_cols = [c for c in df.columns if 'spearman' in c.lower()]
    print(f"Spearman columns: {spearman_cols}")

# Analyze pair types
if 'pair_type' in df.columns:
    print("\n" + "="*60)
    print("MODEL PAIR ANALYSIS")
    print("="*60)
    
    pair_counts = df['pair_type'].value_counts()
    print(f"\nPair type distribution:")
    for pt, count in pair_counts.items():
        print(f"  {pt}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Calculate statistics
    print("\n" + "-"*60)
    print("SPEARMAN BY PAIR TYPE:")
    print("-"*60)
    
    for pair_type in df['pair_type'].unique():
        subset = df[df['pair_type'] == pair_type]['spearman']
        print(f"\n{pair_type}:")
        print(f"  Mean: {subset.mean():.4f}")
        print(f"  Std:  {subset.std():.4f}")
        print(f"  N:    {len(subset):,}")
    
    # Statistical test
    if 'Tree-Tree' in df['pair_type'].values and 'Tree-Linear' in df['pair_type'].values:
        tree_tree = df[df['pair_type'] == 'Tree-Tree']['spearman']
        tree_linear = df[df['pair_type'] == 'Tree-Linear']['spearman']
        
        stat, p_value = mannwhitneyu(tree_tree, tree_linear, alternative='greater')
        
        print("\n" + "="*60)
        print("STATISTICAL TEST: Tree-Tree vs Tree-Linear")
        print("="*60)
        print(f"Tree-Tree mean:    {tree_tree.mean():.4f}")
        print(f"Tree-Linear mean:  {tree_linear.mean():.4f}")
        print(f"Difference:        {tree_tree.mean() - tree_linear.mean():.4f}")
        print(f"Mann-Whitney U:    {stat:,.0f}")
        print(f"p-value:           {'<0.001' if p_value < 0.001 else f'{p_value:.4f}'}")
        print(f"Significant:       {'YES' if p_value < 0.05 else 'NO'}")
        
        # Effect size
        pooled_std = np.sqrt((tree_tree.std()**2 + tree_linear.std()**2) / 2)
        cohens_d = (tree_tree.mean() - tree_linear.mean()) / pooled_std
        print(f"Cohen's d:         {cohens_d:.3f}")
        
        # Save key finding
        key_finding = {
            'tree_tree_mean': float(tree_tree.mean()),
            'tree_tree_std': float(tree_tree.std()),
            'tree_linear_mean': float(tree_linear.mean()),
            'tree_linear_std': float(tree_linear.std()),
            'difference': float(tree_tree.mean() - tree_linear.mean()),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': bool(p_value < 0.05)
        }
        
        import json
        with open('results/upgrade/tree_vs_linear.json', 'w') as f:
            json.dump(key_finding, f, indent=2)
        print("\nSaved: tree_vs_linear.json")

else:
    print("\nCould not classify pair types. Checking unique model pairs...")
    
    # Manual inspection
    if 'model_pair' in df.columns:
        unique_pairs = df['model_pair'].unique()
        print(f"\nUnique model pairs ({len(unique_pairs)}):")
        for p in unique_pairs[:15]:
            subset = df[df['model_pair'] == p]['spearman']
            print(f"  {p}: ρ = {subset.mean():.3f} ± {subset.std():.3f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
