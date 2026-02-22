"""
17_practitioner_guide.py
Q1 UPGRADE: Actionable practitioner guidance with decision framework
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import json
import matplotlib.pyplot as plt

print("=" * 70)
print("Q1 UPGRADE: PRACTITIONER DECISION FRAMEWORK")
print("Actionable Guidance for Real-World Use")
print("=" * 70)

RESULTS_DIR = Path("results")
GUIDE_DIR = RESULTS_DIR / "practitioner_guide"
FIGURES_DIR = Path("explanation_lottery/figures")
GUIDE_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(RESULTS_DIR / "combined_results.csv")

tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
df['pair_type'] = df.apply(
    lambda r: 'Tree-Tree' if r['model_a'] in tree_models and r['model_b'] in tree_models else 'Tree-Linear', 
    axis=1
)

#######################################################################
# 1. RELIABILITY SCORE FRAMEWORK
#######################################################################
print("\n" + "=" * 70)
print("1. RELIABILITY SCORE FRAMEWORK")
print("=" * 70)

print("""
DEFINITION: Reliability Score R(x)
For instance x explained by models M₁, ..., Mₖ:
  R(x) = mean(ρ(Mᵢ, Mⱼ, x) for all pairs i < j)

INTERPRETATION:
  R(x) > 0.7: HIGH reliability - explanations agree
  R(x) ∈ [0.5, 0.7]: MODERATE reliability - use with caution
  R(x) < 0.5: LOW reliability - explanation lottery territory
""")

# Compute per-instance reliability if available
if 'instance_idx' in df.columns and 'dataset_id' in df.columns:
    instance_R = df.groupby(['dataset_id', 'instance_idx'])['spearman'].mean()
    
    high_rel = (instance_R > 0.7).mean() * 100
    mod_rel = ((instance_R >= 0.5) & (instance_R <= 0.7)).mean() * 100
    low_rel = (instance_R < 0.5).mean() * 100
    
    print(f"\nEMPIRICAL DISTRIBUTION:")
    print(f"  High reliability (>0.7):   {high_rel:.1f}%")
    print(f"  Moderate (0.5-0.7):        {mod_rel:.1f}%")
    print(f"  Low reliability (<0.5):    {low_rel:.1f}%")

#######################################################################
# 2. DECISION TREE FOR PRACTITIONERS
#######################################################################
print("\n" + "=" * 70)
print("2. DECISION TREE FOR PRACTITIONERS")
print("=" * 70)

decision_tree = """
WHEN TO TRUST A FEATURE ATTRIBUTION EXPLANATION:

Q1: Are you using models from the SAME hypothesis class?
    ├─ YES (e.g., all tree-based) → Higher confidence (E[ρ] ≈ 0.65-0.70)
    └─ NO (e.g., tree + linear) → Lower confidence (E[ρ] ≈ 0.40-0.45)

Q2: How many features does your dataset have?
    ├─ Few (< 20 features) → Higher agreement expected
    └─ Many (> 50 features) → Lower agreement, more lottery effect

Q3: Is this a high-stakes decision (medical, legal, financial)?
    ├─ YES → Require R(x) > 0.7 AND consensus from multiple models
    └─ NO → R(x) > 0.5 may be acceptable

Q4: Do you need feature RANKING or just TOP-K features?
    ├─ Full ranking → More sensitive to lottery effect
    └─ Top-3 only → Use Top-K overlap, more robust

RECOMMENDATION MATRIX:
┌──────────────────┬─────────────────┬─────────────────┐
│                  │  Low Stakes     │  High Stakes    │
├──────────────────┼─────────────────┼─────────────────┤
│ Same-Class Models│  Trust if R>0.5 │  Trust if R>0.7 │
│ Cross-Class      │  Verify with R  │  Use consensus  │
└──────────────────┴─────────────────┴─────────────────┘
"""
print(decision_tree)

#######################################################################
# 3. CONSENSUS STRATEGIES
#######################################################################
print("\n" + "=" * 70)
print("3. CONSENSUS STRATEGIES")
print("=" * 70)

print("""
STRATEGY 1: Ensemble Explanation (Recommended for high-stakes)
  φ_ensemble(x) = (1/k) Σᵢ φ(Mᵢ, x)
  
  Pros: Averages out model-specific biases
  Cons: May dilute important model-specific signals

STRATEGY 2: Voting for Top-K Features
  Top-K_consensus = features appearing in majority of Top-K lists
  
  Pros: Robust, interpretable
  Cons: Loses fine-grained ranking info

STRATEGY 3: Reliability-Weighted Explanation  
  φ_weighted(x) = Σᵢ wᵢ · φ(Mᵢ, x) where wᵢ ∝ accuracy(Mᵢ)
  
  Pros: Weights by model quality
  Cons: Accuracy ≠ explanation quality

STRATEGY 4: Agreement-Filtered Features
  Report only features where sign(φᵢ) agrees across all models
  
  Pros: High confidence in reported features
  Cons: May miss important but contested features
""")

#######################################################################
# 4. EMPIRICAL THRESHOLDS
#######################################################################
print("\n" + "=" * 70)
print("4. EMPIRICALLY-DERIVED THRESHOLDS")
print("=" * 70)

# Compute thresholds from data
tt_mean = df[df['pair_type'] == 'Tree-Tree']['spearman'].mean()
tl_mean = df[df['pair_type'] == 'Tree-Linear']['spearman'].mean()
overall_mean = df['spearman'].mean()
overall_std = df['spearman'].std()

print(f"EMPIRICAL THRESHOLDS (from {len(df):,} observations):")
print(f"  Tree-Tree baseline: ρ = {tt_mean:.3f}")
print(f"  Tree-Linear baseline: ρ = {tl_mean:.3f}")
print(f"  Overall: μ = {overall_mean:.3f}, σ = {overall_std:.3f}")

thresholds = {
    'excellent': overall_mean + overall_std,
    'good': overall_mean,
    'moderate': overall_mean - 0.5 * overall_std,
    'poor': overall_mean - overall_std
}

print(f"\nRELIABILITY THRESHOLDS:")
print(f"  Excellent: ρ > {thresholds['excellent']:.3f}")
print(f"  Good:      ρ ∈ [{thresholds['good']:.3f}, {thresholds['excellent']:.3f}]")
print(f"  Moderate:  ρ ∈ [{thresholds['moderate']:.3f}, {thresholds['good']:.3f}]")
print(f"  Poor:      ρ < {thresholds['moderate']:.3f}")

#######################################################################
# 5. DOMAIN-SPECIFIC RECOMMENDATIONS
#######################################################################
print("\n" + "=" * 70)
print("5. DOMAIN-SPECIFIC RECOMMENDATIONS")
print("=" * 70)

recommendations = """
HEALTHCARE / MEDICAL DIAGNOSIS:
  - ALWAYS use ensemble of same-class models (trees OR linear, not both)
  - Require R(x) > 0.7 before clinical use
  - Report uncertainty: "Feature X is important (R=0.85)" vs "uncertain (R=0.45)"
  - Prefer Top-3 consensus over full ranking

FINANCIAL (Credit, Fraud):
  - Regulatory requirement: Explanation must be consistent
  - Use Tree-Tree pairs only (higher agreement)
  - Document model class in explanation reports
  - Consider ensemble SHAP for regulatory compliance

LEGAL (Recidivism, Bail):
  - Highest standards required
  - Multiple models + multiple seeds + R(x) threshold
  - Disclose explanation uncertainty to decision-makers
  - May need to avoid model explanations for low-R instances

GENERAL ML DEPLOYMENT:
  - Use R(x) as a quality filter for automated explanations
  - Log and monitor R(x) distribution in production
  - Alert when R(x) drops (model drift indicator)
"""
print(recommendations)

#######################################################################
# 6. REPORTING TEMPLATE
#######################################################################
print("\n" + "=" * 70)
print("6. EXPLANATION REPORT TEMPLATE")
print("=" * 70)

template = """
FEATURE ATTRIBUTION REPORT

Instance ID: [X]
Prediction: [Class Y with probability P]

Models Used: [XGBoost, RandomForest, LightGBM]
Model Class: [Tree-based ensemble]

Reliability Score: R = [0.XX]
Reliability Level: [High/Moderate/Low]

Top-5 Important Features:
  1. [Feature A] - Importance: [0.XX] - Agreement: [All 3 models]
  2. [Feature B] - Importance: [0.XX] - Agreement: [2/3 models]
  ...

Confidence Statement:
  [If R > 0.7]: "High confidence in feature attributions"
  [If R ∈ 0.5-0.7]: "Moderate confidence - interpret with caution"  
  [If R < 0.5]: "Low confidence - explanation lottery territory"

Technical Note:
  Full agreement ρ = [0.XX], Cross-model variance = [0.XX]
"""
print(template)

#######################################################################
# SAVE GUIDE
#######################################################################
print("\n" + "=" * 70)
print("SAVING PRACTITIONER GUIDE")
print("=" * 70)

guide = {
    'reliability_thresholds': thresholds,
    'tree_tree_baseline': float(tt_mean),
    'tree_linear_baseline': float(tl_mean),
    'consensus_strategies': ['ensemble', 'voting', 'weighted', 'filtered'],
    'domains': ['healthcare', 'financial', 'legal', 'general']
}

with open(GUIDE_DIR / "practitioner_guide.json", 'w') as f:
    json.dump(guide, f, indent=2)

# Save full text guide
with open(GUIDE_DIR / "practitioner_guide.txt", 'w', encoding='utf-8') as f:
    f.write("PRACTITIONER GUIDE: THE EXPLANATION LOTTERY\n")
    f.write("=" * 50 + "\n\n")
    f.write(decision_tree)
    f.write("\n" + recommendations)
    f.write("\n" + template)

print(f"Saved to: {GUIDE_DIR}")
print("PRACTITIONER GUIDE COMPLETE!")
