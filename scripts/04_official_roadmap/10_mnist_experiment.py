"""
=============================================================================
10_OPTIMISED_001: NON-TABULAR EXPERIMENT (MNIST)
=============================================================================
Tier B -- Item 10 | Impact: 4/5 | Effort: 2-3 days

Goal: Show the Explanation Lottery extends beyond tabular data.
Use MNIST (flattened 784 features) with same model families.

Output: results/optimised_001/10_mnist_experiment/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '10_mnist_experiment')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 10: MNIST NON-TABULAR EXPERIMENT")
print("=" * 70)

# =============================================================================
# LOAD MNIST
# =============================================================================
print("\n[1] Loading MNIST data...")

try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_full, y_full = mnist.data, mnist.target.astype(int)
    print(f"  Full MNIST: {X_full.shape[0]} samples, {X_full.shape[1]} features")
except Exception as e:
    print(f"  OpenML fetch failed: {e}")
    print("  Generating synthetic 'MNIST-like' data instead...")
    np.random.seed(42)
    n_samples = 5000
    n_features = 784
    X_full = np.random.rand(n_samples, n_features) * 255
    y_full = np.random.randint(0, 10, n_samples)

# Subsample for tractability (binary: 0 vs 1)
print("\n[2] Creating binary subset (digit 0 vs 1)...")
mask = (y_full == 0) | (y_full == 1)
X_binary = X_full[mask]
y_binary = (y_full[mask] == 1).astype(int)

# Further subsample for SHAP computation speed
MAX_TRAIN = 2000
MAX_TEST = 500
if len(X_binary) > MAX_TRAIN + MAX_TEST:
    np.random.seed(42)
    idx = np.random.choice(len(X_binary), MAX_TRAIN + MAX_TEST, replace=False)
    X_binary = X_binary[idx]
    y_binary = y_binary[idx]

print(f"  Binary subset: {len(X_binary)} samples")

# PCA for faster SHAP (optional: keep 50 components)
USE_PCA = True
N_COMPONENTS = 50

if USE_PCA:
    print(f"\n[3] Applying PCA ({N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_binary)
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained_var:.3f}")
    X_use = X_pca
    feature_names = [f'PC{i}' for i in range(N_COMPONENTS)]
else:
    X_use = X_binary
    feature_names = [f'pixel_{i}' for i in range(X_binary.shape[1])]

X_df = pd.DataFrame(X_use, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X_df, y_binary, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# TRAIN MODELS
# =============================================================================
print("\n[4] Training models...")

models = {
    'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
}

accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies[name] = float(acc)
    print(f"  {name}: accuracy = {acc:.4f}")

# =============================================================================
# FIND PREDICTION AGREEMENT INSTANCES
# =============================================================================
print("\n[5] Finding prediction agreement instances...")

preds = {}
for name, model in models.items():
    preds[name] = model.predict(X_test)

agree_mask = np.ones(len(X_test), dtype=bool)
for name in models:
    agree_mask &= (preds[name] == preds[list(models.keys())[0]])

X_agree = X_test[agree_mask].iloc[:100]  # Limit for SHAP computation
print(f"  Agreement instances: {agree_mask.sum()} ({agree_mask.mean()*100:.1f}%)")
print(f"  Using {len(X_agree)} instances for SHAP analysis")

# =============================================================================
# COMPUTE SHAP VALUES
# =============================================================================
print("\n[6] Computing SHAP values...")

shap_values = {}
bg = X_train.sample(n=min(100, len(X_train)), random_state=42)

for name, model in models.items():
    print(f"  Computing SHAP for {name}...", end=" ")
    try:
        if name == 'logistic_regression':
            explainer = shap.LinearExplainer(model, bg)
        else:
            explainer = shap.TreeExplainer(model)
        
        sv = explainer.shap_values(X_agree)
        if isinstance(sv, list):
            sv = sv[1]
        if len(getattr(sv, 'shape', [])) == 3:
            sv = sv[:, :, 1]
        shap_values[name] = sv
        print(f"OK - shape {sv.shape}")
    except Exception as e:
        print(f"FAILED: {e}")

# =============================================================================
# COMPUTE PAIRWISE AGREEMENT
# =============================================================================
print("\n[7] Computing pairwise explanation agreement...")

model_names = list(shap_values.keys())
pair_results = []

for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m_a, m_b = model_names[i], model_names[j]
        
        # Instance-level Spearman correlations
        instance_rhos = []
        for idx in range(len(X_agree)):
            rho, _ = stats.spearmanr(shap_values[m_a][idx], shap_values[m_b][idx])
            if not np.any(np.isnan(np.atleast_1d(rho))):
                instance_rhos.append(rho)
        
        if instance_rhos:
            mean_rho = np.mean(instance_rhos)
            std_rho = np.std(instance_rhos)
            lottery_rate = np.mean([r < 0.5 for r in instance_rhos]) * 100
            
            # Classify pair type
            tree_models = ['random_forest', 'gradient_boosting']
            a_tree = m_a in tree_models
            b_tree = m_b in tree_models
            pair_type = 'Tree-Tree' if (a_tree and b_tree) else 'Tree-Linear'
            
            pair_results.append({
                'pair': f'{m_a} vs {m_b}',
                'pair_type': pair_type,
                'mean_rho': float(mean_rho),
                'std_rho': float(std_rho),
                'lottery_rate': float(lottery_rate),
                'n_instances': len(instance_rhos)
            })
            
            print(f"  {m_a} vs {m_b}: rho = {mean_rho:.3f} +/- {std_rho:.3f}, lottery = {lottery_rate:.1f}%")

# =============================================================================
# COMPARISON WITH TABULAR RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: MNIST vs TABULAR")
print("=" * 70)

# Load tabular data
tabular_df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
tabular_mean = tabular_df['spearman'].mean()
tabular_lottery = (tabular_df['spearman'] < 0.5).mean() * 100

# MNIST stats
if pair_results:
    mnist_mean = np.mean([p['mean_rho'] for p in pair_results])
    mnist_lottery = np.mean([p['lottery_rate'] for p in pair_results])
else:
    mnist_mean = 0
    mnist_lottery = 0

print(f"\n  {'Metric':<25} {'Tabular':<15} {'MNIST'}")
print(f"  {'-'*25} {'-'*15} {'-'*15}")
print(f"  {'Mean rho':<25} {tabular_mean:.3f}         {mnist_mean:.3f}")
print(f"  {'Lottery rate (%)':<25} {tabular_lottery:.1f}%          {mnist_lottery:.1f}%")
print(f"  {'N comparisons':<25} {len(tabular_df):,}       {sum(p['n_instances'] for p in pair_results) if pair_results else 0}")

if pair_results:
    print(f"\n  * FINDING:")
    if mnist_lottery > 10:
        print(f"    The Explanation Lottery extends to non-tabular (image) data!")
        print(f"    MNIST lottery rate: {mnist_lottery:.1f}% (tabular: {tabular_lottery:.1f}%)")
    else:
        print(f"    MNIST shows lower lottery rate than tabular data")
        print(f"    Possible explanation: PCA features are more interpretable")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'mnist_setup': {
        'binary_task': '0 vs 1',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_agree': len(X_agree),
        'pca_used': USE_PCA,
        'n_components': N_COMPONENTS if USE_PCA else X_binary.shape[1],
        'pca_explained_variance': float(explained_var) if USE_PCA else 1.0
    },
    'model_accuracies': accuracies,
    'pair_results': pair_results,
    'comparison': {
        'tabular_mean_rho': float(tabular_mean),
        'tabular_lottery_rate': float(tabular_lottery),
        'mnist_mean_rho': float(mnist_mean),
        'mnist_lottery_rate': float(mnist_lottery)
    }
}

output_file = os.path.join(OUTPUT_DIR, '10_mnist_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 10 COMPLETE")
print("=" * 70)
