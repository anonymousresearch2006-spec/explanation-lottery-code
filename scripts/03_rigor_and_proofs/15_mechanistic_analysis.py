"""
15_mechanistic_analysis.py
Q1 UPGRADE: Deep mechanistic analysis of WHY explanations diverge
"""

import numpy as np
import pandas as pd
from scipy import stats
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

print("=" * 70)
print("Q1 UPGRADE: MECHANISTIC ANALYSIS")
print("Why do explanations diverge despite identical predictions?")
print("=" * 70)

RESULTS_DIR = Path("results/mechanistic")
FIGURES_DIR = Path("explanation_lottery/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create dataset with known structure
np.random.seed(42)
n_samples = 1000
# Features: 5 informative, 5 redundant, 5 noise
X_informative = np.random.randn(n_samples, 5)
X_redundant = X_informative[:, :5] + 0.1 * np.random.randn(n_samples, 5)  # Correlated
X_noise = np.random.randn(n_samples, 5)
X = np.hstack([X_informative, X_redundant, X_noise])

# Non-linear decision boundary (XOR-like)
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

feature_names = [f"Info_{i}" for i in range(5)] + [f"Red_{i}" for i in range(5)] + [f"Noise_{i}" for i in range(5)]

print(f"Dataset: XOR-like boundary on features 0,1")
print(f"Features: 5 informative, 5 redundant, 5 noise")

# Train models
models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'LogisticReg': LogisticRegression(max_iter=1000, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: accuracy = {model.score(X_test, y_test):.3f}")

# Key insight: XGB can model XOR, Logistic cannot
print(f"\nNote: Logistic Regression CANNOT model XOR perfectly (linear boundary)")

# Find agreement instances
preds = {n: m.predict(X_test) for n, m in models.items()}
agree_mask = preds['XGBoost'] == preds['RandomForest']
agree_idx = np.where(agree_mask)[0][:50]

# Compute SHAP
print("\n--- Computing SHAP ---")
shap_vals = {}
for name, model in models.items():
    if name == 'LogisticReg':
        exp = shap.LinearExplainer(model, X_train)
    else:
        exp = shap.TreeExplainer(model)
    sv = exp.shap_values(X_test[agree_idx])
    # Handle different SHAP output formats
    if isinstance(sv, list):
        sv = sv[1]  # Class 1 for multi-output
    # Ensure 2D array
    if sv.ndim == 3:
        sv = sv[:, :, 1]  # Take class 1
    shap_vals[name] = sv

#######################################################################
# ANALYSIS 1: Feature Attribution Patterns
#######################################################################
print("\n" + "=" * 70)
print("ANALYSIS 1: Mean Feature Attributions")
print("=" * 70)

mean_attrs = {}
for name in models:
    mean_attrs[name] = np.abs(shap_vals[name]).mean(axis=0)

attr_df = pd.DataFrame(mean_attrs, index=feature_names)
print(attr_df.round(3).to_string())

#######################################################################
# ANALYSIS 2: Interaction Detection Capability
#######################################################################
print("\n" + "=" * 70)
print("ANALYSIS 2: Interaction Detection")
print("=" * 70)

print("""
Ground Truth: XOR interaction between Info_0 and Info_1

TREES can detect interactions via:
  - Splits: If Info_0 > 0, check Info_1
  - Attribution reflects conditional dependencies

LINEAR models cannot detect interactions:
  - Only additive effects: β₀ + β₁x₁ + β₂x₂
  - XOR requires: y = x₁ ⊕ x₂ (non-additive)
""")

# Compare Info_0, Info_1 attributions
for name in models:
    info0 = np.abs(shap_vals[name][:, 0]).mean()
    info1 = np.abs(shap_vals[name][:, 1]).mean()
    ratio = info0 / info1 if info1 > 0 else float('inf')
    print(f"  {name}: Info_0 = {info0:.3f}, Info_1 = {info1:.3f}, ratio = {ratio:.2f}")

#######################################################################
# ANALYSIS 3: Decision Boundary Geometry
#######################################################################
print("\n" + "=" * 70)
print("ANALYSIS 3: Decision Boundary Geometry")
print("=" * 70)

print("""
TREE Decision Boundary:
  - Axis-aligned splits: x₁ > θ₁, x₂ > θ₂
  - Creates rectangular regions
  - SHAP reflects local split importance

LINEAR Decision Boundary:
  - Hyperplane: β·x + b = 0
  - Creates half-spaces
  - SHAP = gradient-based (constant direction)

CONSEQUENCE:
  Same prediction ≠ Same attribution
  Trees attribute based on WHICH splits were taken
  Linear attributes based on GLOBAL coefficients
""")

#######################################################################
# ANALYSIS 4: Redundant Feature Handling
#######################################################################
print("\n" + "=" * 70)
print("ANALYSIS 4: Redundant Feature Handling")
print("=" * 70)

print("Ground Truth: Red_0-4 are redundant copies of Info_0-4")

for name in models:
    info_sum = np.abs(shap_vals[name][:, :5]).sum(axis=1).mean()
    red_sum = np.abs(shap_vals[name][:, 5:10]).sum(axis=1).mean()
    ratio = info_sum / red_sum if red_sum > 0 else float('inf')
    print(f"  {name}: Info sum = {info_sum:.3f}, Red sum = {red_sum:.3f}, ratio = {ratio:.2f}")

print("""
INSIGHT: 
- Trees may prefer original or redundant features arbitrarily
- Linear models distribute weight based on coefficients
- Different handling → Different attributions
""")

#######################################################################
# ANALYSIS 5: Per-Instance Divergence
#######################################################################
print("\n" + "=" * 70)
print("ANALYSIS 5: Per-Instance Divergence Patterns")
print("=" * 70)

# Compare tree-tree vs tree-linear
tree_tree_corrs = []
tree_linear_corrs = []

for i in range(len(agree_idx)):
    xgb_shap = shap_vals['XGBoost'][i]
    rf_shap = shap_vals['RandomForest'][i]
    lr_shap = shap_vals['LogisticReg'][i]
    
    rho_tt, _ = stats.spearmanr(xgb_shap, rf_shap)
    rho_tl, _ = stats.spearmanr(xgb_shap, lr_shap)
    
    if not np.isnan(rho_tt):
        tree_tree_corrs.append(rho_tt)
    if not np.isnan(rho_tl):
        tree_linear_corrs.append(rho_tl)

print(f"Tree-Tree (XGB vs RF): ρ = {np.mean(tree_tree_corrs):.3f} ± {np.std(tree_tree_corrs):.3f}")
print(f"Tree-Linear (XGB vs LR): ρ = {np.mean(tree_linear_corrs):.3f} ± {np.std(tree_linear_corrs):.3f}")

#######################################################################
# SUMMARY
#######################################################################
print("\n" + "=" * 70)
print("MECHANISTIC CONCLUSION")
print("=" * 70)

print(f"""
ROOT CAUSES OF EXPLANATION DIVERGENCE:

1. HYPOTHESIS CLASS MISMATCH
   - Trees: piecewise constant, axis-aligned boundaries
   - Linear: smooth, hyperplane boundaries
   → Same prediction, different decision process

2. INTERACTION MODELING
   - Trees detect feature interactions (XOR, AND, etc.)
   - Linear models are strictly additive
   → Fundamentally different sensitivity patterns

3. REDUNDANT FEATURE HANDLING  
   - Trees arbitrarily select among correlated features
   - Linear distributes coefficients
   → Different feature credit assignment

4. LOCAL vs GLOBAL ATTRIBUTIONS
   - Tree SHAP depends on path taken in tree
   - Linear SHAP is the same for all instances (gradient)
   → Instance-level variation differs

IMPLICATION:
Cross-model explanation disagreement is STRUCTURALLY INEVITABLE
when models come from different hypothesis classes.
This is not a bug—it reflects genuine differences in how models make decisions.
""")

# Save results
results = {
    'tree_tree_mean': float(np.mean(tree_tree_corrs)),
    'tree_linear_mean': float(np.mean(tree_linear_corrs)),
    'gap': float(np.mean(tree_tree_corrs) - np.mean(tree_linear_corrs)),
    'causes': ['hypothesis_class_mismatch', 'interaction_modeling', 'redundant_handling', 'local_vs_global']
}

import json
with open(RESULTS_DIR / "mechanistic_analysis.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to: {RESULTS_DIR}")
print("MECHANISTIC ANALYSIS COMPLETE!")
