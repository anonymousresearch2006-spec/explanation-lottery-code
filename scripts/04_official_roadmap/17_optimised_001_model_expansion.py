"""
=============================================================================
17_OPTIMISED_001: EXTENSIVE MODEL EXPANSION
=============================================================================
Tier C -- Item 17 | Impact: 2/5 | Effort: 3-5 days

Goal: Expand model families to include neural networks (MLP),
additional ensemble methods, and compare with existing results.

Output: results/optimised_001/17_model_expansion/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                               AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_wine
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '17_model_expansion')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 17: MODEL EXPANSION")
print("=" * 70)

# =============================================================================
# EXPANDED MODEL SET
# =============================================================================

def create_expanded_models(random_state=42):
    """Create expanded set of models including neural networks."""
    return {
        # Original tree-based
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=random_state),
        'extra_trees': ExtraTreesClassifier(n_estimators=100, max_depth=8, random_state=random_state),
        'adaboost': AdaBoostClassifier(n_estimators=100, random_state=random_state),
        
        # Linear models
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=random_state),
        
        # Neural networks
        'mlp_small': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=random_state),
        'mlp_medium': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state),
    }

MODEL_FAMILIES = {
    'random_forest': 'Tree',
    'gradient_boosting': 'Tree',
    'extra_trees': 'Tree',
    'adaboost': 'Tree',
    'logistic_regression': 'Linear',
    'mlp_small': 'Neural',
    'mlp_medium': 'Neural',
}

# =============================================================================
# TEST DATASETS
# =============================================================================

DATASETS = {
    'breast_cancer': load_breast_cancer(return_X_y=True),
    'wine_binary': None,  # Will process below
}

# Make wine binary (class 0 vs rest)
wine = load_wine()
wine_mask = (wine.target == 0) | (wine.target == 1)
DATASETS['wine_binary'] = (wine.data[wine_mask], wine.target[wine_mask])

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================
all_results = {}

for ds_name, (X_raw, y_raw) in DATASETS.items():
    print(f"\n{'='*70}")
    print(f"DATASET: {ds_name}")
    print(f"{'='*70}")
    
    X_df = pd.DataFrame(X_raw, columns=[f'f{i}' for i in range(X_raw.shape[1])])
    y = pd.Series(y_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    
    # Scale for neural networks
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_df.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_df.columns, index=X_test.index)
    
    print(f"  Shape: {X_df.shape}")
    
    # Train models
    models = create_expanded_models()
    accuracies = {}
    trained_models = {}
    
    for name, model in models.items():
        try:
            if name.startswith('mlp'):
                model.fit(X_train_scaled, y_train)
                acc = accuracy_score(y_test, model.predict(X_test_scaled))
            else:
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
            accuracies[name] = float(acc)
            trained_models[name] = model
            print(f"  {name}: accuracy = {acc:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
    
    # Find agreement instances (over all models that trained)
    preds = {}
    for name, model in trained_models.items():
        X_pred = X_test_scaled if name.startswith('mlp') else X_test
        preds[name] = model.predict(X_pred)
    
    if len(preds) < 2:
        print(f"  [!WARN!] Not enough models trained, skipping")
        continue
    
    first_model = list(preds.keys())[0]
    agree = np.ones(len(X_test), dtype=bool)
    for name in preds:
        agree &= (preds[name] == preds[first_model])
    
    X_agree = X_test[agree].iloc[:50]
    X_agree_scaled = X_test_scaled.loc[X_agree.index]
    print(f"  Agreement instances: {agree.sum()}, using {len(X_agree)}")
    
    if len(X_agree) < 5:
        print(f"  [!WARN!] Too few agreement instances, skipping")
        continue
    
    # Compute SHAP
    bg = X_train.sample(n=min(50, len(X_train)), random_state=42)
    bg_scaled = X_train_scaled.loc[bg.index]
    
    shap_vals = {}
    for name, model in trained_models.items():
        try:
            if name == 'logistic_regression':
                ex = shap.LinearExplainer(model, bg)
                sv = ex.shap_values(X_agree)
            elif name.startswith('mlp'):
                ex = shap.KernelExplainer(model.predict_proba, bg_scaled.values[:20])
                sv = ex.shap_values(X_agree_scaled.values)
                if isinstance(sv, list):
                    sv = sv[1]
                if len(getattr(sv, 'shape', [])) == 3:
                    sv = sv[:, :, 1]
            else:
                ex = shap.TreeExplainer(model)
                sv = ex.shap_values(X_agree)
                if isinstance(sv, list):
                    sv = sv[1]
                if len(getattr(sv, 'shape', [])) == 3:
                    sv = sv[:, :, 1]
            shap_vals[name] = sv
            print(f"  {name}: SHAP computed")
        except Exception as e:
            print(f"  {name}: SHAP failed ({e})")
    
    # Pairwise agreement
    model_names = list(shap_vals.keys())
    pair_results = []
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m_a, m_b = model_names[i], model_names[j]
            rhos = []
            for k in range(min(len(X_agree), shap_vals[m_a].shape[0], shap_vals[m_b].shape[0])):
                rho, _ = stats.spearmanr(shap_vals[m_a][k], shap_vals[m_b][k])
                if not np.any(np.isnan(np.atleast_1d(rho))):
                    rhos.append(rho)
            
            if rhos:
                family_a = MODEL_FAMILIES.get(m_a, 'Unknown')
                family_b = MODEL_FAMILIES.get(m_b, 'Unknown')
                pair_type = f'{family_a}-{family_b}' if family_a <= family_b else f'{family_b}-{family_a}'
                
                pair_results.append({
                    'pair': f'{m_a}-{m_b}',
                    'pair_type': pair_type,
                    'mean_rho': float(np.mean(rhos)),
                    'lottery_rate': float(np.mean([r < 0.5 for r in rhos]) * 100)
                })
                print(f"  {m_a} vs {m_b} [{pair_type}]: rho = {np.mean(rhos):.3f}")
    
    # Summarize by pair type
    pair_df = pd.DataFrame(pair_results)
    if len(pair_df) > 0:
        print(f"\n  Summary by pair type:")
        for pt in pair_df['pair_type'].unique():
            pt_data = pair_df[pair_df['pair_type'] == pt]
            print(f"    {pt}: rho = {pt_data['mean_rho'].mean():.3f}, lottery = {pt_data['lottery_rate'].mean():.1f}%")
    
    all_results[ds_name] = {
        'accuracies': accuracies,
        'pair_results': pair_results,
        'n_agreement': int(agree.sum()),
        'n_models_trained': len(trained_models)
    }

# =============================================================================
# CROSS-FAMILY COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("CROSS-FAMILY COMPARISON: Tree vs Linear vs Neural")
print("=" * 70)

all_pairs = []
for ds_name, result in all_results.items():
    if 'pair_results' in result:
        for p in result['pair_results']:
            p['dataset'] = ds_name
            all_pairs.append(p)

if all_pairs:
    pair_df = pd.DataFrame(all_pairs)
    print(f"\n  {'Pair Type':<20} {'Mean rho':<10} {'Lottery %':<12} {'N pairs'}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*8}")
    for pt in sorted(pair_df['pair_type'].unique()):
        pt_data = pair_df[pair_df['pair_type'] == pt]
        print(f"  {pt:<20} {pt_data['mean_rho'].mean():.3f}     {pt_data['lottery_rate'].mean():.1f}%        {len(pt_data)}")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'datasets': {k: {kk: vv for kk, vv in v.items()} for k, v in all_results.items()},
    'model_families': MODEL_FAMILIES,
    'models_tested': list(MODEL_FAMILIES.keys())
}

output_file = os.path.join(OUTPUT_DIR, '17_model_expansion_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 17 COMPLETE")
print("=" * 70)
