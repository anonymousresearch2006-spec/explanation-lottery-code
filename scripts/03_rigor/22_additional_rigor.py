"""
22_additional_rigor.py
FINAL RIGOR CHECKS:
1. Instance Confidence (Difficulty) vs Agreement
   - Hypothesis: Disagreement is higher on "hard" instances (low confidence)
2. Noise Robustness (Data Perturbation)
   - Hypothesis: Feature ranking agreement should be robust to small input noise
   - If rho drops drastically with small noise, the explanation is unstable artifacts
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.datasets import fetch_openml
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("FINAL RIGOR CHECKS: CONFIDENCE & ROBUSTNESS")
print("=" * 70)

RESULTS_DIR = Path("results/rigor")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Diabetes Dataset
print("Loading Diabetes dataset...")
diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
X, y = diabetes.data, diabetes.target
y = (y == 'tested_positive').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
lr = LogisticRegression(random_state=42)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Find Agreement Instances
rf_pred = rf.predict(X_test)
lr_pred = lr.predict(X_test)
agree_mask = (rf_pred == lr_pred)
X_agree = X_test[agree_mask]
print(f"Analyzing {len(X_agree)} agreement instances")

# Compute SHAP
print("Computing SHAP values...")
rf_ex = shap.TreeExplainer(rf)
lr_ex = shap.LinearExplainer(lr, X_train)

# Helper to Extract Class 1 SHAP
def get_shap_class1(explainer, X):
    sv = explainer.shap_values(X)
    # Case 1: List (multiclass) -> predict_proba outputs
    if isinstance(sv, list):
        return sv[1]
    # Case 2: 3D Array (samples, features, classes) -> usually from predict_proba
    if len(sv.shape) == 3:
        return sv[:,:,1]
    # Case 3: 2D Array -> usually for binary cases or regression
    return sv

rf_shap = get_shap_class1(rf_ex, X_agree)
lr_shap = get_shap_class1(lr_ex, X_agree)

# Ensure shapes match
if rf_shap.shape != lr_shap.shape:
    print(f"Shape mismatch! RF: {rf_shap.shape}, LR: {lr_shap.shape}")
    # Sometimes LinearExplainer returns (N, M) and TreeExplainer (N, M) but might need transpose? 
    # Usually they match (N_samples, N_features). 
    # If LR is 1D (unlikely here), handle it.


# ---------------------------------------------------------
# EXPERIMENT 1: INSTANCE CONFIDENCE VS AGREEMENT
# ---------------------------------------------------------
print("\n--- EXPERIMENT 1: CONFIDENCE VS AGREEMENT ---")
# Calculate confidence (probability margin)
rf_probs = rf.predict_proba(X_agree)
lr_probs = lr.predict_proba(X_agree)

# Margin = |P(1) - 0.5| * 2 (ranges 0 to 1)
rf_margin = np.abs(rf_probs[:,1] - 0.5) * 2
lr_margin = np.abs(lr_probs[:,1] - 0.5) * 2
avg_margin = (rf_margin + lr_margin) / 2

# Calculate Agreement (Rho) per instance
rhos = []
for i in range(len(X_agree)):
    r, _ = stats.spearmanr(rf_shap[i], lr_shap[i])
    rhos.append(r)
rhos = np.array(rhos)

# Correlate Confidence with Agreement
corr_conf, p_conf = stats.spearmanr(avg_margin, rhos)

print(f"Correlation (Confidence vs Agreement): r = {corr_conf:.3f} (p={p_conf:.3f})")
if corr_conf > 0.1:
    print(">> Finding: Higher confidence → Higher agreement (Hard instances disagree more)")
elif corr_conf < -0.1:
    print(">> Finding: Higher confidence → Lower agreement (Unexpected)")
else:
    print(">> Finding: Agreement is independent of prediction confidence")

# ---------------------------------------------------------
# EXPERIMENT 2: NOISE ROBUSTNESS (PERTURBATION)
# ---------------------------------------------------------
print("\n--- EXPERIMENT 2: NOISE ROBUSTNESS ---")
# Add small Gaussian noise to test instances
noise_levels = [0.0, 0.01, 0.05, 0.10]
robustness_results = []

print(f"{'Noise (std)':<15} {'Mean Rho (Tree-Linear)':<25} {'Drop':<10}")

base_rho = np.mean(rhos)

for sigma in noise_levels:
    if sigma == 0:
        print(f"{sigma:<15.2f} {base_rho:<25.3f} {'-':<10}")
        continue
        
    # Perturb
    np.random.seed(42)
    noise = np.random.normal(0, sigma, X_agree.shape)
    X_perturbed = X_agree + noise
    
    # Re-compute SHAP (using same explainers)
    rf_shap_noise = get_shap_class1(rf_ex, X_perturbed)
    lr_shap_noise = get_shap_class1(lr_ex, X_perturbed)
    
    # Compute Agreement
    noise_rhos = []
    for i in range(len(X_agree)):
        r, _ = stats.spearmanr(rf_shap_noise[i], lr_shap_noise[i])
        noise_rhos.append(r)
    
    mean_noise_rho = np.nanmean(noise_rhos)
    drop = base_rho - mean_noise_rho
    
    robustness_results.append({'noise': sigma, 'rho': mean_noise_rho, 'drop': drop})
    print(f"{sigma:<15.2f} {mean_noise_rho:<25.3f} {drop:<10.3f}")

print("\nINTERPRETATION:")
if robustness_results[-1]['drop'] < 0.1:
    print(">> ROBUST: Agreement is stable under small perturbations.")
    print("   The disagreement is a structural property, not a fragility artifact.")
else:
    print(">> FRAGILE: Small noise destroys agreement.")

# Save results
import json
results = {
    'confidence_correlation': float(corr_conf),
    'confidence_p_value': float(p_conf),
    'robustness': robustness_results
}
with open(RESULTS_DIR / 'additional_rigor.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDONE.")
