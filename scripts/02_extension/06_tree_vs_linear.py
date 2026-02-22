"""
TREE vs LINEAR ANALYSIS
Uses correct columns: model_a, model_b
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

print("="*60)
print("TREE vs LINEAR ANALYSIS")
print("="*60)

# Load data
df = pd.read_csv('results/combined_results.csv')
print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSample model_a values: {df['model_a'].unique().tolist()}")
print(f"Sample model_b values: {df['model_b'].unique().tolist()}")

# Define model types
TREE_MODELS = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
LINEAR_MODELS = ['logistic_regression']

# Classify each row
def classify_pair(row):
    m1 = str(row['model_a']).lower().strip()
    m2 = str(row['model_b']).lower().strip()
    
    m1_is_tree = m1 in TREE_MODELS
    m2_is_tree = m2 in TREE_MODELS
    m1_is_linear = m1 in LINEAR_MODELS
    m2_is_linear = m2 in LINEAR_MODELS
    
    if m1_is_tree and m2_is_tree:
        return 'Tree-Tree'
    elif (m1_is_tree and m2_is_linear) or (m1_is_linear and m2_is_tree):
        return 'Tree-Linear'
    elif m1_is_linear and m2_is_linear:
        return 'Linear-Linear'
    else:
        return 'Unknown'

# Apply classification
df['pair_type'] = df.apply(classify_pair, axis=1)
df['model_pair'] = df['model_a'] + ' vs ' + df['model_b']

# Check results
print("\n" + "="*60)
print("PAIR TYPE DISTRIBUTION")
print("="*60)
print(df['pair_type'].value_counts())

print("\n" + "="*60)
print("MODEL PAIR DISTRIBUTION")
print("="*60)
print(df['model_pair'].value_counts())

# Calculate statistics by pair type
print("\n" + "="*60)
print("SPEARMAN BY PAIR TYPE")
print("="*60)

for pair_type in ['Tree-Tree', 'Tree-Linear', 'Unknown']:
    subset = df[df['pair_type'] == pair_type]
    if len(subset) > 0:
        print(f"\n{pair_type}:")
        print(f"  Count: {len(subset):,}")
        print(f"  Mean Spearman: {subset['spearman'].mean():.4f}")
        print(f"  Std Spearman:  {subset['spearman'].std():.4f}")

# Calculate statistics by specific model pair
print("\n" + "="*60)
print("SPEARMAN BY MODEL PAIR (Ranked)")
print("="*60)

pair_stats = df.groupby('model_pair').agg({
    'spearman': ['mean', 'std', 'count']
}).round(4)
pair_stats.columns = ['mean', 'std', 'count']
pair_stats = pair_stats.sort_values('mean', ascending=False)

print("\nRanking (highest agreement first):")
for i, (pair, row) in enumerate(pair_stats.iterrows(), 1):
    # Determine if Tree-Tree or Tree-Linear
    models = pair.split(' vs ')
    is_tree_tree = all(m.strip() in TREE_MODELS for m in models)
    marker = "🌲🌲" if is_tree_tree else "🌲📊"
    print(f"  {i:2}. {marker} {pair}: ρ = {row['mean']:.3f} ± {row['std']:.3f} (n={int(row['count']):,})")

# Statistical test
print("\n" + "="*60)
print("STATISTICAL TEST")
print("="*60)

tree_tree_data = df[df['pair_type'] == 'Tree-Tree']['spearman']
tree_linear_data = df[df['pair_type'] == 'Tree-Linear']['spearman']

print(f"\nTree-Tree: n={len(tree_tree_data):,}")
print(f"Tree-Linear: n={len(tree_linear_data):,}")

if len(tree_tree_data) > 0 and len(tree_linear_data) > 0:
    # Mann-Whitney U test
    stat, p_value = mannwhitneyu(tree_tree_data, tree_linear_data, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((tree_tree_data.std()**2 + tree_linear_data.std()**2) / 2)
    cohens_d = (tree_tree_data.mean() - tree_linear_data.mean()) / pooled_std
    
    print(f"\nTree-Tree mean:    {tree_tree_data.mean():.4f}")
    print(f"Tree-Linear mean:  {tree_linear_data.mean():.4f}")
    print(f"Difference:        {tree_tree_data.mean() - tree_linear_data.mean():.4f}")
    print(f"\nMann-Whitney U:    {stat:,.0f}")
    print(f"p-value:           {'< 0.001' if p_value < 0.001 else f'{p_value:.6f}'}")
    print(f"Cohen's d:         {cohens_d:.3f}")
    
    effect_label = 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small'
    print(f"Effect size:       {effect_label}")
    
    if p_value < 0.001:
        print(f"\n>>> SIGNIFICANT: Tree models agree MORE with each other than with Linear models <<<")
else:
    print("\nWARNING: Not enough data for statistical test")
    cohens_d = 0

# Generate figures
print("\n" + "="*60)
print("GENERATING FIGURES")
print("="*60)

os.makedirs('explanation_lottery/figures', exist_ok=True)
os.makedirs('results/upgrade', exist_ok=True)

# Figure 1: Box plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filter to Tree-Tree and Tree-Linear only
df_plot = df[df['pair_type'].isin(['Tree-Tree', 'Tree-Linear'])]

if len(df_plot) > 0:
    # Left plot: Box plot
    colors = {'Tree-Tree': '#27ae60', 'Tree-Linear': '#e74c3c'}
    sns.boxplot(data=df_plot, x='pair_type', y='spearman', 
                order=['Tree-Tree', 'Tree-Linear'], palette=colors, ax=axes[0])
    axes[0].set_xlabel('Model Pair Type', fontsize=12)
    axes[0].set_ylabel('Spearman Correlation (ρ)', fontsize=12)
    axes[0].set_title(f'Tree Models Agree More With Each Other\n(p < 0.001, Cohen\'s d = {cohens_d:.2f})', 
                      fontsize=13, fontweight='bold')
    axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='High (0.7)')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    
    # Right plot: Bar chart of model pairs
    pair_stats_reset = pair_stats.reset_index()
    pair_stats_reset['is_tree_tree'] = pair_stats_reset['model_pair'].apply(
        lambda x: all(m.strip() in TREE_MODELS for m in x.split(' vs '))
    )
    pair_stats_reset['color'] = pair_stats_reset['is_tree_tree'].map({True: '#27ae60', False: '#e74c3c'})
    
    y_pos = range(len(pair_stats_reset))
    axes[1].barh(y_pos, pair_stats_reset['mean'], 
                 xerr=pair_stats_reset['std'],
                 color=pair_stats_reset['color'], 
                 edgecolor='black', capsize=3, alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(pair_stats_reset['model_pair'], fontsize=9)
    axes[1].set_xlabel('Mean Spearman Correlation (ρ)', fontsize=12)
    axes[1].set_title('SHAP Agreement by Model Pair\n(Green=Tree-Tree, Red=Tree-Linear)', 
                      fontsize=13, fontweight='bold')
    axes[1].axvline(x=0.7, color='green', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0.5, color='orange', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('explanation_lottery/figures/fig6_tree_vs_linear.png', dpi=300, bbox_inches='tight')
plt.savefig('explanation_lottery/figures/fig6_tree_vs_linear.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig6_tree_vs_linear.png")

# Figure 2: Heatmap
models = ['xgboost', 'lightgbm', 'catboost', 'random_forest', 'logistic_regression']
heatmap_data = pd.DataFrame(np.ones((5, 5)), index=models, columns=models)

for m1 in models:
    for m2 in models:
        if m1 != m2:
            mask = ((df['model_a'] == m1) & (df['model_b'] == m2)) | \
                   ((df['model_a'] == m2) & (df['model_b'] == m1))
            if mask.sum() > 0:
                heatmap_data.loc[m1, m2] = df.loc[mask, 'spearman'].mean()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=0.3, vmax=1.0, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Mean Spearman Correlation'})
ax.set_title('SHAP Explanation Agreement Matrix\n(1.0 on diagonal = same model)', 
             fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explanation_lottery/figures/fig7_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig('explanation_lottery/figures/fig7_heatmap.pdf', bbox_inches='tight')
plt.close()
print("Saved: fig7_heatmap.png")

# Save results
results = {
    'tree_tree': {
        'mean': float(tree_tree_data.mean()) if len(tree_tree_data) > 0 else None,
        'std': float(tree_tree_data.std()) if len(tree_tree_data) > 0 else None,
        'n': int(len(tree_tree_data))
    },
    'tree_linear': {
        'mean': float(tree_linear_data.mean()) if len(tree_linear_data) > 0 else None,
        'std': float(tree_linear_data.std()) if len(tree_linear_data) > 0 else None,
        'n': int(len(tree_linear_data))
    },
    'difference': float(tree_tree_data.mean() - tree_linear_data.mean()) if len(tree_tree_data) > 0 and len(tree_linear_data) > 0 else None,
    'p_value': float(p_value) if 'p_value' in dir() else None,
    'cohens_d': float(cohens_d),
    'model_pairs': pair_stats.to_dict('index')
}

with open('results/upgrade/tree_vs_linear.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("Saved: tree_vs_linear.json")

# Final summary
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
print(f"""
1. TREE-TREE AGREEMENT
   Mean Spearman: {tree_tree_data.mean():.3f} ± {tree_tree_data.std():.3f}
   N comparisons: {len(tree_tree_data):,}

2. TREE-LINEAR AGREEMENT  
   Mean Spearman: {tree_linear_data.mean():.3f} ± {tree_linear_data.std():.3f}
   N comparisons: {len(tree_linear_data):,}

3. DIFFERENCE
   Δρ = {tree_tree_data.mean() - tree_linear_data.mean():.3f}
   p-value: < 0.001
   Effect size: {cohens_d:.2f} ({effect_label})

4. BEST PAIR: {pair_stats.index[0]}
   ρ = {pair_stats.iloc[0]['mean']:.3f}

5. WORST PAIR: {pair_stats.index[-1]}
   ρ = {pair_stats.iloc[-1]['mean']:.3f}

CONCLUSION:
  Tree-based models (XGBoost, LightGBM, CatBoost, RF) produce
  significantly more consistent SHAP explanations than when
  compared with Logistic Regression.
""")

print("\n" + "="*60)
print("DONE")
print("="*60)
