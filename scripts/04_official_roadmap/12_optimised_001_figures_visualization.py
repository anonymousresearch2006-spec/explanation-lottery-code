"""
=============================================================================
12_OPTIMISED_001: IMPROVE FIGURES + VISUALIZATION
=============================================================================
Tier B -- Item 12 | Impact: 3/5 | Effort: 1 day

Goal: Generate improved, publication-quality figures.
- Dimensionality scatter plot
- Model agreement heatmap
- Lottery rate per dataset bar chart
- Improved distribution plots

Output: results/optimised_001/12_figures_visualization/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '12_figures_visualization')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 12: FIGURES & VISUALIZATION")
print("=" * 70)

# Load data
df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
print(f"\nLoaded {len(df):,} comparisons")

# Classify pairs
tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
df['pair_type'] = df.apply(
    lambda r: 'Tree-Tree' if (r['model_a'] in tree_models and r['model_b'] in tree_models) 
    else 'Tree-Linear', axis=1)

# =============================================================================
# FIGURE 1: LOTTERY RATE PER DATASET
# =============================================================================
print("\n[Fig 1] Lottery rate per dataset...")

ds_stats = df.groupby('dataset_id').agg(
    lottery_rate=('spearman', lambda x: (x < 0.5).mean() * 100),
    mean_rho=('spearman', 'mean'),
    n_features=('n_features', 'first') if 'n_features' in df.columns else ('spearman', 'count')
).reset_index().sort_values('lottery_rate', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#e74c3c' if r > 50 else '#f39c12' if r > 30 else '#2ecc71' for r in ds_stats['lottery_rate']]
bars = ax.barh(range(len(ds_stats)), ds_stats['lottery_rate'], color=colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(ds_stats)))
ax.set_yticklabels([f"Dataset {int(d)}" for d in ds_stats['dataset_id']], fontsize=8)
ax.set_xlabel('Lottery Rate (%)')
ax.set_title('Explanation Lottery Rate by Dataset (tau = 0.5)')
ax.axvline(x=50, color='black', linestyle='--', alpha=0.5, label='50% threshold')
ax.legend()
ax.set_xlim(0, 100)

# Add value labels
for bar, rate in zip(bars, ds_stats['lottery_rate']):
    if rate > 5:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{rate:.0f}%', va='center', fontsize=7)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig_lottery_per_dataset.png')
plt.savefig(fig_path)
plt.close()
print(f"  Saved: {fig_path}")

# =============================================================================
# FIGURE 2: DIMENSIONALITY vs AGREEMENT
# =============================================================================
print("[Fig 2] Dimensionality vs agreement scatter...")

if 'n_features' in df.columns:
    ds_dim = df.groupby('dataset_id').agg(
        mean_rho=('spearman', 'mean'),
        n_features=('n_features', 'first'),
        lottery_rate=('spearman', lambda x: (x < 0.5).mean() * 100),
        n_comparisons=('spearman', 'count')
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(ds_dim['n_features'], ds_dim['mean_rho'], 
                        c=ds_dim['lottery_rate'], cmap='RdYlGn_r',
                        s=np.clip(ds_dim['n_comparisons'] / 10, 20, 200),
                        alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Trend line
    valid = ds_dim[ds_dim['n_features'].notna()]
    if len(valid) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(np.log(valid['n_features'] + 1), valid['mean_rho'])
        x_line = np.linspace(valid['n_features'].min(), valid['n_features'].max(), 100)
        y_line = slope * np.log(x_line + 1) + intercept
        ax.plot(x_line, y_line, 'r--', alpha=0.7, label=f'Log trend (r={r_val:.2f})')
    
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Mean Explanation Agreement (rho)')
    ax.set_title('Dimensionality vs Explanation Agreement')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='tau = 0.5')
    ax.set_xscale('log')
    plt.colorbar(scatter, ax=ax, label='Lottery Rate (%)')
    ax.legend()
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig_dimensionality_scatter.png')
    plt.savefig(fig_path)
    plt.close()
    print(f"  Saved: {fig_path}")

# =============================================================================
# FIGURE 3: MODEL PAIR AGREEMENT HEATMAP
# =============================================================================
print("[Fig 3] Model agreement heatmap...")

all_models = sorted(set(df['model_a'].unique().tolist() + df['model_b'].unique().tolist()))

heatmap_data = np.zeros((len(all_models), len(all_models)))
for i, m_a in enumerate(all_models):
    for j, m_b in enumerate(all_models):
        if i == j:
            heatmap_data[i, j] = 1.0
        else:
            pair_df = df[((df['model_a'] == m_a) & (df['model_b'] == m_b)) |
                        ((df['model_a'] == m_b) & (df['model_b'] == m_a))]
            if len(pair_df) > 0:
                heatmap_data[i, j] = pair_df['spearman'].mean()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

# Labels
short_names = [m.replace('logistic_regression', 'LogReg').replace('random_forest', 'RF')
               .replace('xgboost', 'XGB').replace('lightgbm', 'LGBM').replace('catboost', 'CatB')
               for m in all_models]
ax.set_xticks(range(len(all_models)))
ax.set_yticks(range(len(all_models)))
ax.set_xticklabels(short_names, rotation=45, ha='right')
ax.set_yticklabels(short_names)

# Annotate
for i in range(len(all_models)):
    for j in range(len(all_models)):
        text_color = 'white' if heatmap_data[i, j] < 0.4 else 'black'
        ax.text(j, i, f'{heatmap_data[i,j]:.2f}', ha='center', va='center', 
                fontsize=9, color=text_color, fontweight='bold')

ax.set_title('Mean Explanation Agreement Between Model Pairs')
plt.colorbar(im, ax=ax, label='Mean Spearman rho')
plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig_model_heatmap.png')
plt.savefig(fig_path)
plt.close()
print(f"  Saved: {fig_path}")

# =============================================================================
# FIGURE 4: AGREEMENT DISTRIBUTIONS (Tree-Tree vs Tree-Linear)
# =============================================================================
print("[Fig 4] Agreement distributions...")

fig, ax = plt.subplots(figsize=(8, 5))

tt_data = df[df['pair_type'] == 'Tree-Tree']['spearman']
tl_data = df[df['pair_type'] == 'Tree-Linear']['spearman']

bins = np.linspace(-1, 1, 50)
ax.hist(tt_data, bins=bins, alpha=0.6, color='#3498db', density=True, label=f'Tree-Tree (mu={tt_data.mean():.2f})')
ax.hist(tl_data, bins=bins, alpha=0.6, color='#e74c3c', density=True, label=f'Tree-Linear (mu={tl_data.mean():.2f})')

ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='tau = 0.5')
ax.axvline(x=tt_data.mean(), color='#3498db', linestyle=':', alpha=0.7)
ax.axvline(x=tl_data.mean(), color='#e74c3c', linestyle=':', alpha=0.7)

ax.set_xlabel('Spearman Rank Correlation (rho)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Explanation Agreement by Model Pair Type')
ax.legend(loc='upper left')

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig_agreement_distributions.png')
plt.savefig(fig_path)
plt.close()
print(f"  Saved: {fig_path}")

# =============================================================================
# FIGURE 5: THRESHOLD SENSITIVITY
# =============================================================================
print("[Fig 5] Threshold sensitivity...")

tau_values = np.arange(0.1, 0.95, 0.02)
rates_overall = [(df['spearman'] < t).mean() * 100 for t in tau_values]
rates_tt = [(tt_data < t).mean() * 100 for t in tau_values]
rates_tl = [(tl_data < t).mean() * 100 for t in tau_values]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(tau_values, rates_overall, 'k-', linewidth=2, label='All pairs')
ax.plot(tau_values, rates_tt, 'b--', linewidth=1.5, label='Tree-Tree')
ax.plot(tau_values, rates_tl, 'r--', linewidth=1.5, label='Tree-Linear')
ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7, label='tau = 0.5 (chosen)')
ax.axhline(y=50, color='gray', linestyle=':', alpha=0.3)

ax.set_xlabel('Threshold tau')
ax.set_ylabel('Lottery Rate (%)')
ax.set_title('Threshold Sensitivity Analysis')
ax.legend()
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, 'fig_threshold_sensitivity.png')
plt.savefig(fig_path)
plt.close()
print(f"  Saved: {fig_path}")

# =============================================================================
# SAVE SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'figures_generated': [
        'fig_lottery_per_dataset.png',
        'fig_dimensionality_scatter.png',
        'fig_model_heatmap.png',
        'fig_agreement_distributions.png',
        'fig_threshold_sensitivity.png'
    ],
    'output_dir': OUTPUT_DIR,
    'style': 'Publication-quality (300 DPI)'
}

output_file = os.path.join(OUTPUT_DIR, '12_figures_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 12 COMPLETE -- 5 figures generated")
print("=" * 70)
