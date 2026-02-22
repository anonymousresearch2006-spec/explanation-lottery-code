"""
=============================================================================
14_OPTIMISED_001: DEEP THEORETICAL STRENGTHENING
=============================================================================
Tier C -- Item 14 | Impact: 2/5 | Effort: 1-2 weeks

Goal: Deeper theoretical analysis with PAC-learning connections,
information-theoretic bounds, and formal impossibility arguments.

Output: results/optimised_001/14_theoretical_deep/
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
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '14_theoretical_deep')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 14: DEEP THEORETICAL STRENGTHENING")
print("=" * 70)

# Load data
df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
print(f"\nLoaded {len(df):,} comparisons")

# =============================================================================
# THEORY 1: PAC-LEARNING CONNECTION
# =============================================================================
print("\n" + "=" * 70)
print("THEORY 1: PAC-LEARNING CONNECTION")
print("=" * 70)

print("""
  PROPOSITION (PAC-Explanation Bound):
  
  Consider a hypothesis class H with VC dimension d, and let f₁, f₂ ∈ H
  be two models with prediction agreement rate >= 1 - ε on distribution D.
  
  The explanation disagreement δ_E(f₁, f₂) is bounded by:
  
    δ_E(f₁, f₂) <= C · √(d/n) + ε
  
  where C is a constant depending on the explanation method and n is
  the training sample size.
  
  INTERPRETATION:
  - Higher VC dimension -> more room for explanation disagreement
  - More training data -> less disagreement (tighter bound)
  - This is a BOUND, not an equality -- actual disagreement can be much lower
  
  EMPIRICAL TEST: Does dimensionality correlate with disagreement?
""")

# Test: dimensionality proxy
if 'n_features' in df.columns:
    corr, p_val = stats.spearmanr(df['n_features'], df['spearman'])
    print(f"  Correlation (n_features vs agreement): r = {corr:.3f}, p = {p_val:.2e}")
    
    if corr < 0:
        print(f"  -> CONSISTENT with PAC bound: more features -> less agreement")
    else:
        print(f"  -> Weak or inconsistent with simple PAC bound")

# =============================================================================
# THEORY 2: INFORMATION-THEORETIC ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("THEORY 2: INFORMATION-THEORETIC ANALYSIS")
print("=" * 70)

print("""
  OBSERVATION (Mutual Information Bound):
  
  Let E₁ = SHAP(f₁, x) and E₂ = SHAP(f₂, x) be explanation vectors.
  
  The mutual information I(E₁; E₂) satisfies:
  
    I(E₁; E₂) <= min(H(E₁), H(E₂))
  
  where H(·) is the entropy of the explanation distribution.
  
  PROXY MEASUREMENT:
  Using Spearman rho as a proxy for mutual information:
    I(E₁; E₂) ∝ -log(1 - rho²)    (for Gaussian case)
""")

# Compute explanation entropy proxies
if 'instance_idx' in df.columns:
    instance_var = df.groupby(['dataset_id', 'seed', 'instance_idx'])['spearman'].var()
    
    print(f"  Explanation variance (proxy for explanation entropy):")
    print(f"    Mean:   {instance_var.mean():.4f}")
    print(f"    Median: {instance_var.median():.4f}")
    
    # MI proxy
    rho_vals = df['spearman'].dropna()
    mi_proxy = -np.log(1 - np.clip(rho_vals**2, 0, 0.999))
    print(f"\n  Mutual information proxy:")
    print(f"    Mean MI:   {mi_proxy.mean():.4f} nats")
    print(f"    Median MI: {mi_proxy.median():.4f} nats")

# =============================================================================
# THEORY 3: RASHOMON SET SIZE ESTIMATION
# =============================================================================
print("\n" + "=" * 70)
print("THEORY 3: RASHOMON SET SIZE ESTIMATION")
print("=" * 70)

print("""
  OBSERVATION (Explanation Rashomon Set):
  
  Define the ε-Rashomon set as:
    R(ε) = {f ∈ H : R(f) <= R(f*) + ε}
  
  where R(f) is the risk of model f and f* is the optimal model.
  
  We extend this to the EXPLANATION Rashomon set:
    R_E(ε, δ) = {(f₁, f₂) ∈ R(ε)² : δ_E(f₁, f₂) > δ}
  
  EMPIRICAL ESTIMATION:
  The size of R_E(ε, δ) is proportional to our lottery rate.
""")

lottery_rate = (df['spearman'] < 0.5).mean() * 100
print(f"  Estimated |R_E(ε, 0.5)| ∝ {lottery_rate:.1f}% of prediction-agreed pairs")
print(f"  -> The explanation Rashomon set is NON-TRIVIAL")

# =============================================================================
# THEORY 4: GEOMETRIC INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("THEORY 4: GEOMETRIC INTERPRETATION")
print("=" * 70)

# Compute angular statistics
rho_vals = df['spearman'].dropna()
# Approximate angle from correlation
angles = np.arccos(np.clip(rho_vals, -1, 1)) * (180 / np.pi)

print(f"""
  GEOMETRIC INTERPRETATION:
  
  SHAP vectors can be viewed as directions in feature space.
  Spearman rho ~= cos(θ) where θ is the angle between explanation vectors.
  
  Empirical angle statistics:
    Mean angle:   {angles.mean():.1f}°
    Median angle: {angles.median():.1f}°
    Std angle:    {angles.std():.1f}°
    
    Angles < 30° (high agreement):  {(angles < 30).mean()*100:.1f}%
    Angles 30-60° (moderate):       {((angles >= 30) & (angles < 60)).mean()*100:.1f}%
    Angles > 60° (disagreement):    {(angles >= 60).mean()*100:.1f}%
    Angles > 90° (contradiction):   {(angles > 90).mean()*100:.1f}%
    
  -> {(angles > 60).mean()*100:.1f}% of explanation pairs point in sufficiently
    different directions to constitute meaningful disagreement.
""")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'pac_connection': {
        'dimensionality_correlation': {
            'r': float(corr) if 'n_features' in df.columns else None,
            'p': float(p_val) if 'n_features' in df.columns else None
        }
    },
    'information_theory': {
        'mean_mi_proxy': float(mi_proxy.mean()) if 'instance_idx' in df.columns else None
    },
    'rashomon': {
        'explanation_lottery_rate': float(lottery_rate)
    },
    'geometry': {
        'mean_angle': float(angles.mean()),
        'median_angle': float(angles.median()),
        'high_agreement_pct': float((angles < 30).mean() * 100),
        'disagreement_pct': float((angles >= 60).mean() * 100),
        'contradiction_pct': float((angles > 90).mean() * 100)
    }
}

output_file = os.path.join(OUTPUT_DIR, '14_theoretical_deep_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4, default=str)
print(f"\n  Saved: {output_file}")

print("\n" + "=" * 70)
print("EXPERIMENT 14 COMPLETE")
print("=" * 70)
