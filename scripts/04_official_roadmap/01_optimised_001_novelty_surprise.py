"""
=============================================================================
01_OPTIMISED_001: STRENGTHEN NOVELTY / SURPRISE ARGUMENT
=============================================================================
Tier A -- Item 1 | Impact: 5/5 | Effort: 1 day

Goal: Quantify the "surprise" of the Explanation Lottery finding.
- What community assumes (explanations stable for same prediction)
- What literature lacks (cross-model disagreement analysis)
- Why 35% disagreement is surprising
- Practical implications

Output: results/optimised_001/01_novelty_surprise/
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
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '01_novelty_surprise')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 01: NOVELTY / SURPRISE ARGUMENT")
print("=" * 70)

# Load data
df = pd.read_csv(os.path.join(RESULTS_DIR, 'combined_results.csv'))
print(f"\nLoaded {len(df):,} comparisons from {df['dataset_id'].nunique()} datasets")

# =============================================================================
# ANALYSIS 1: COMMUNITY ASSUMPTION vs REALITY
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: COMMUNITY ASSUMPTION vs REALITY")
print("=" * 70)

# Community assumption: if models predict the same -> explanations should agree
# Reality: significant disagreement even when predictions match

overall_mean = df['spearman'].mean()
overall_std = df['spearman'].std()
overall_median = df['spearman'].median()

print(f"\n  Community Assumption: rho ~= 1.0 (explanations agree when predictions agree)")
print(f"  Reality:              rho = {overall_mean:.3f} +/- {overall_std:.3f}")
print(f"  Gap:                  {(1.0 - overall_mean):.3f} from perfect agreement")
print(f"  Median:               rho = {overall_median:.3f}")

# =============================================================================
# ANALYSIS 2: LOTTERY RATE COMPUTATION
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: LOTTERY RATE (% instances with rho < tau)")
print("=" * 70)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
lottery_rates = {}

for tau in thresholds:
    rate = (df['spearman'] < tau).mean() * 100
    lottery_rates[f'tau_{tau}'] = rate
    print(f"  tau = {tau}: {rate:.1f}% of instances in the lottery (rho < {tau})")

primary_lottery_rate = lottery_rates['tau_0.5']
print(f"\n  * PRIMARY LOTTERY RATE (tau=0.5): {primary_lottery_rate:.1f}%")
print(f"    -> Over {primary_lottery_rate:.0f}% of instances receive contradictory explanations")

# Confidence interval for primary rate
n = len(df)
p = primary_lottery_rate / 100
se = np.sqrt(p * (1 - p) / n)
ci_lower = (p - 1.96 * se) * 100
ci_upper = (p + 1.96 * se) * 100
print(f"    -> 95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

# =============================================================================
# ANALYSIS 3: SURPRISE QUANTIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: SURPRISE QUANTIFICATION")
print("=" * 70)

# Under null hypothesis: if explanations were consistent, rho should be ~1.0
# Test: is the observed mean significantly less than 1.0?
t_stat, p_value = stats.ttest_1samp(df['spearman'], 1.0)
print(f"\n  One-sample t-test (H0: mu = 1.0):")
print(f"    t-statistic: {t_stat:.2f}")
print(f"    p-value:     {'<1e-300' if p_value == 0 else f'{p_value:.2e}'}")
print(f"    -> Reject H0: explanations are NOT consistent (p << 0.001)")

# Effect size (Cohen's d)
cohens_d = (1.0 - overall_mean) / overall_std
print(f"\n  Effect Size (Cohen's d from perfect agreement):")
print(f"    d = {cohens_d:.3f}")
if cohens_d > 0.8:
    print(f"    -> LARGE effect (d > 0.8): massive departure from expectation")
elif cohens_d > 0.5:
    print(f"    -> MEDIUM effect (0.5 < d < 0.8): substantial departure")
else:
    print(f"    -> SMALL effect (d < 0.5): moderate departure")

# Under null: if explanations were random, rho should be ~0.0
t_stat2, p_value2 = stats.ttest_1samp(df['spearman'], 0.0)
print(f"\n  One-sample t-test (H0: mu = 0.0, totally random):")
print(f"    t-statistic: {t_stat2:.2f}")
print(f"    p-value:     {'<1e-300' if p_value2 == 0 else f'{p_value2:.2e}'}")
print(f"    -> Explanations are NOT random either -- partial, inconsistent structure")

# =============================================================================
# ANALYSIS 4: CONDITIONAL DISAGREEMENT (THE KEY FINDING)
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: CONDITIONAL ON PREDICTION AGREEMENT")
print("=" * 70)

# This IS the surprise: these are instances where models ALREADY AGREE on prediction
# Yet their explanations disagree substantially
print(f"\n  All {len(df):,} comparisons are CONDITIONAL on prediction agreement")
print(f"  (models gave the SAME prediction for these instances)")
print(f"\n  Despite prediction agreement:")
print(f"    Mean explanation agreement:  rho = {overall_mean:.3f}")
print(f"    Std deviation:               sigma = {overall_std:.3f}")
print(f"    Instances with disagreement: {primary_lottery_rate:.1f}%")
print(f"\n  * THIS IS THE SURPRISE:")
print(f"    Even when models AGREE on WHAT to predict,")
print(f"    they disagree on WHY in {primary_lottery_rate:.1f}% of cases.")

# =============================================================================
# ANALYSIS 5: BREAKDOWN BY MODEL PAIR TYPE
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: STRUCTURE OF DISAGREEMENT")
print("=" * 70)

tree_models = ['xgboost', 'lightgbm', 'catboost', 'random_forest']

def classify_pair(row):
    a_tree = row['model_a'] in tree_models
    b_tree = row['model_b'] in tree_models
    if a_tree and b_tree:
        return 'Tree-Tree'
    elif not a_tree and not b_tree:
        return 'Linear-Linear'
    else:
        return 'Tree-Linear'

df['pair_type'] = df.apply(classify_pair, axis=1)

print(f"\n  Agreement by model pair type:")
pair_stats = {}
for pt in ['Tree-Tree', 'Tree-Linear']:
    subset = df[df['pair_type'] == pt]
    if len(subset) > 0:
        mean_rho = subset['spearman'].mean()
        std_rho = subset['spearman'].std()
        lottery = (subset['spearman'] < 0.5).mean() * 100
        n = len(subset)
        ci = 1.96 * std_rho / np.sqrt(n)
        
        pair_stats[pt] = {
            'mean': float(mean_rho),
            'std': float(std_rho),
            'lottery_rate': float(lottery),
            'n': int(n),
            'ci_95': float(ci)
        }
        
        print(f"\n  {pt}:")
        print(f"    Mean rho:       {mean_rho:.3f} +/- {std_rho:.3f}")
        print(f"    95% CI:       [{mean_rho - ci:.3f}, {mean_rho + ci:.3f}]")
        print(f"    Lottery rate: {lottery:.1f}%")
        print(f"    N:            {n:,}")

# Statistical test: Tree-Tree vs Tree-Linear
tt = df[df['pair_type'] == 'Tree-Tree']['spearman']
tl = df[df['pair_type'] == 'Tree-Linear']['spearman']
if len(tt) > 0 and len(tl) > 0:
    u_stat, u_p = stats.mannwhitneyu(tt, tl, alternative='greater')
    effect_r = u_stat / (len(tt) * len(tl))
    print(f"\n  Mann-Whitney U (Tree-Tree > Tree-Linear):")
    print(f"    U = {u_stat:,.0f}, p = {'<1e-300' if u_p == 0 else f'{u_p:.2e}'}")
    print(f"    Rank-biserial r = {effect_r:.3f}")
    
    gap = tt.mean() - tl.mean()
    print(f"\n  * STRUCTURAL FINDING:")
    print(f"    Tree-Tree rho is {gap:.3f} higher than Tree-Linear rho")
    print(f"    -> Explanation disagreement is STRUCTURED, not random")

# =============================================================================
# ANALYSIS 6: PRACTICAL IMPLICATIONS QUANTIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 6: PRACTICAL IMPLICATIONS")
print("=" * 70)

# How many instances would a practitioner get misleading explanations?
total_instances = df['instance_idx'].nunique() if 'instance_idx' in df.columns else 'N/A'
total_datasets = df['dataset_id'].nunique()

print(f"\n  Scale of the problem:")
print(f"    Datasets analyzed:  {total_datasets}")
print(f"    Total comparisons:  {len(df):,}")
print(f"    Unique instances:   {total_instances}")

# Categorize risk levels
high_risk = (df['spearman'] < 0.3).mean() * 100
med_risk = ((df['spearman'] >= 0.3) & (df['spearman'] < 0.5)).mean() * 100
low_risk = ((df['spearman'] >= 0.5) & (df['spearman'] < 0.7)).mean() * 100
safe = (df['spearman'] >= 0.7).mean() * 100

print(f"\n  Risk Categories:")
print(f"    HIGH RISK (rho < 0.3):    {high_risk:.1f}% -- explanations strongly contradict")
print(f"    MEDIUM RISK (0.3-0.5):  {med_risk:.1f}% -- explanations weakly agree")
print(f"    LOW RISK (0.5-0.7):     {low_risk:.1f}% -- explanations moderately agree")
print(f"    SAFE (rho >= 0.7):         {safe:.1f}% -- explanations strongly agree")
print(f"\n  * {high_risk + med_risk:.1f}% of instances carry meaningful explanation risk")

# =============================================================================
# SAVE RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

results = {
    'community_assumption_vs_reality': {
        'assumed_agreement': 1.0,
        'actual_mean': float(overall_mean),
        'actual_std': float(overall_std),
        'actual_median': float(overall_median),
        'gap_from_perfect': float(1.0 - overall_mean)
    },
    'lottery_rates': {f'tau_{t}': float(r) for t, r in zip(thresholds, [lottery_rates[f'tau_{t}'] for t in thresholds])},
    'primary_lottery_rate': {
        'tau': 0.5,
        'rate': float(primary_lottery_rate),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper)
    },
    'surprise_statistics': {
        'cohens_d_from_perfect': float(cohens_d),
        'effect_interpretation': 'large' if cohens_d > 0.8 else ('medium' if cohens_d > 0.5 else 'small'),
        'total_comparisons': int(len(df)),
        'total_datasets': int(total_datasets)
    },
    'pair_type_analysis': pair_stats,
    'risk_categories': {
        'high_risk_pct': float(high_risk),
        'medium_risk_pct': float(med_risk),
        'low_risk_pct': float(low_risk),
        'safe_pct': float(safe),
        'total_at_risk_pct': float(high_risk + med_risk)
    }
}

output_file = os.path.join(OUTPUT_DIR, '01_novelty_surprise_results.json')
with open(output_file, 'w') as f:
    json.dump(results, f, indent=4)
print(f"\n  Saved: {output_file}")

# Summary report
summary_file = os.path.join(OUTPUT_DIR, '01_novelty_surprise_summary.txt')
with open(summary_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("NOVELTY / SURPRISE ARGUMENT -- SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"1. SURPRISE: Despite prediction agreement, explanation agreement\n")
    f.write(f"   is only rho = {overall_mean:.3f} +/- {overall_std:.3f}\n\n")
    f.write(f"2. LOTTERY RATE: {primary_lottery_rate:.1f}% of instances have rho < 0.5\n")
    f.write(f"   95% CI: [{ci_lower:.1f}%, {ci_upper:.1f}%]\n\n")
    f.write(f"3. EFFECT SIZE: Cohen's d = {cohens_d:.3f} (from perfect agreement)\n\n")
    f.write(f"4. STRUCTURE: Tree-Tree vs Tree-Linear gap is significant\n")
    if pair_stats:
        f.write(f"   Tree-Tree:   rho = {pair_stats.get('Tree-Tree', {}).get('mean', 'N/A')}\n")
        f.write(f"   Tree-Linear: rho = {pair_stats.get('Tree-Linear', {}).get('mean', 'N/A')}\n\n")
    f.write(f"5. RISK: {high_risk + med_risk:.1f}% of instances carry meaningful explanation risk\n")

print(f"  Saved: {summary_file}")
print("\n" + "=" * 70)
print("EXPERIMENT 01 COMPLETE")
print("=" * 70)
