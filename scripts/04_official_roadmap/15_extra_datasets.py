"""
=============================================================================
15_OPTIMISED_001: MULTIPLE EXTRA DATASETS (CIFAR, NLP proxy)
=============================================================================
Tier C -- Item 15 | Impact: 2/5 | Effort: 1 week

Goal: Test lottery effect on additional non-tabular data using
      feature-extracted versions (since MNIST is in Exp 10).

Output: results/optimised_001/15_extra_datasets/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '15_extra_datasets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 15: EXTRA DATASETS")
print("=" * 70)

all_results = {}

# =============================================================================
# DATASET 1: DIGITS (8x8 images -- like mini-MNIST)
# =============================================================================
print("\n" + "=" * 70)
print("DATASET 1: DIGITS (8x8 images, 64 features)")
print("=" * 70)

try:
    digits = load_digits(n_class=2)  # Binary: 0 vs 1
    X_digits = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
    y_digits = pd.Series(digits.target)
    
    X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)
    
    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name}: accuracy = {acc:.4f}")
    
    # Agreement instances
    preds = {name: model.predict(X_test) for name, model in models.items()}
    agree = np.all([preds[n] == preds['rf'] for n in preds], axis=0)
    X_agree = X_test[agree].iloc[:50]
    print(f"  Agreement instances: {agree.sum()}")
    
    # SHAP
    bg = X_train.sample(n=min(50, len(X_train)), random_state=42)
    shap_vals = {}
    for name, model in models.items():
        try:
            if name == 'lr':
                ex = shap.LinearExplainer(model, bg)
            else:
                ex = shap.TreeExplainer(model)
            sv = ex.shap_values(X_agree)
            if isinstance(sv, list):
                sv = sv[1]
            shap_vals[name] = sv
        except Exception as e:
            print(f"  {name} SHAP failed: {e}")
    
    # Pairwise agreement
    digit_rhos = []
    model_names = list(shap_vals.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            rhos = [stats.spearmanr(shap_vals[model_names[i]][k], shap_vals[model_names[j]][k])[0] 
                    for k in range(len(X_agree))]
            rhos = [r for r in rhos if not np.isnan(r)]
            if rhos:
                mean_rho = np.mean(rhos)
                digit_rhos.extend(rhos)
                print(f"  {model_names[i]} vs {model_names[j]}: rho = {mean_rho:.3f}")
    
    all_results['digits'] = {
        'mean_rho': float(np.mean(digit_rhos)) if digit_rhos else None,
        'lottery_rate': float(np.mean([r < 0.5 for r in digit_rhos]) * 100) if digit_rhos else None,
        'n_instances': len(X_agree),
        'n_features': 64
    }
    print(f"  Overall: rho = {np.mean(digit_rhos):.3f}, lottery = {np.mean([r<0.5 for r in digit_rhos])*100:.1f}%")

except Exception as e:
    print(f"  DIGITS experiment failed: {e}")
    all_results['digits'] = {'error': str(e)}

# =============================================================================
# DATASET 2: TEXT DATA (20 Newsgroups, TF-IDF features)
# =============================================================================
print("\n" + "=" * 70)
print("DATASET 2: TEXT DATA (20 Newsgroups, binary, TF-IDF)")
print("=" * 70)

try:
    # Binary: sci.space vs comp.graphics
    cats = ['sci.space', 'comp.graphics']
    newsgroups = fetch_20newsgroups(subset='all', categories=cats, remove=('headers', 'footers', 'quotes'))
    
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_text = vectorizer.fit_transform(newsgroups.data).toarray()
    y_text = newsgroups.target
    
    feature_names = vectorizer.get_feature_names_out()
    X_text_df = pd.DataFrame(X_text, columns=feature_names)
    y_text_s = pd.Series(y_text)
    
    print(f"  Samples: {len(X_text_df)}, Features: {X_text_df.shape[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_text_df, y_text_s, test_size=0.2, random_state=42)
    
    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'lr': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"  {name}: accuracy = {acc:.4f}")
    
    # Agreement
    preds = {name: model.predict(X_test) for name, model in models.items()}
    agree = np.all([preds[n] == preds['rf'] for n in preds], axis=0)
    X_agree = X_test[agree].iloc[:50]
    print(f"  Agreement instances: {agree.sum()}")
    
    # SHAP
    bg = X_train.sample(n=min(50, len(X_train)), random_state=42)
    shap_vals = {}
    for name, model in models.items():
        try:
            if name == 'lr':
                ex = shap.LinearExplainer(model, bg)
            else:
                ex = shap.TreeExplainer(model)
            sv = ex.shap_values(X_agree)
            if isinstance(sv, list):
                sv = sv[1]
            shap_vals[name] = sv
        except Exception as e:
            print(f"  {name} SHAP failed: {e}")
    
    text_rhos = []
    model_names = list(shap_vals.keys())
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            rhos = [stats.spearmanr(shap_vals[model_names[i]][k], shap_vals[model_names[j]][k])[0]
                    for k in range(len(X_agree))]
            rhos = [r for r in rhos if not np.isnan(r)]
            if rhos:
                mean_rho = np.mean(rhos)
                text_rhos.extend(rhos)
                print(f"  {model_names[i]} vs {model_names[j]}: rho = {mean_rho:.3f}")
    
    all_results['text_newsgroups'] = {
        'mean_rho': float(np.mean(text_rhos)) if text_rhos else None,
        'lottery_rate': float(np.mean([r < 0.5 for r in text_rhos]) * 100) if text_rhos else None,
        'n_instances': len(X_agree),
        'n_features': 100
    }
    print(f"  Overall: rho = {np.mean(text_rhos):.3f}, lottery = {np.mean([r<0.5 for r in text_rhos])*100:.1f}%")

except Exception as e:
    print(f"  TEXT experiment failed: {e}")
    all_results['text_newsgroups'] = {'error': str(e)}

# =============================================================================
# COMPARISON WITH TABULAR
# =============================================================================
print("\n" + "=" * 70)
print("COMPARISON: TABULAR vs NON-TABULAR")
print("=" * 70)

# Load tabular baseline
tabular_df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
tabular_mean = tabular_df['spearman'].mean()
tabular_lottery = (tabular_df['spearman'] < 0.5).mean() * 100

print(f"\n  {'Dataset':<20} {'Mean rho':<12} {'Lottery Rate'}")
print(f"  {'-'*20} {'-'*12} {'-'*12}")
print(f"  {'Tabular (original)':<20} {tabular_mean:<12.3f} {tabular_lottery:.1f}%")

for name, result in all_results.items():
    if 'mean_rho' in result and result['mean_rho'] is not None:
        print(f"  {name:<20} {result['mean_rho']:<12.3f} {result['lottery_rate']:.1f}%")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'datasets': all_results,
    'tabular_baseline': {
        'mean_rho': float(tabular_mean),
        'lottery_rate': float(tabular_lottery)
    }
}

output_file = os.path.join(OUTPUT_DIR, '15_extra_datasets_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 15 COMPLETE")
print("=" * 70)
