"""
=============================================================================
04_OPTIMISED_001: SYNTHETIC GROUND TRUTH EXPERIMENT
=============================================================================
Tier A -- Item 4 | Impact: 5/5 | Effort: 3-5 days

Goal: Create synthetic datasets with KNOWN true feature importances.
Validate that explanation agreement correlates with explanation correctness.
This is the CRITICAL validation experiment.

Output: results/optimised_001/04_synthetic_ground_truth/
=============================================================================
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import json
import os
import warnings
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')

# Setup
PROJECT_DIR = 'results'
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'optimised_001', '04_synthetic_ground_truth')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTIMISED_001 -- EXPERIMENT 04: SYNTHETIC GROUND TRUTH")
print("=" * 70)

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIOS = {
    'linear_additive': {
        'name': 'Pure Linear / Additive',
        'description': 'y = 2*x0 + 3*x1 - 1*x2 -> expect HIGH agreement, esp. with linear model',
        'true_importances': {0: 2.0, 1: 3.0, 2: 1.0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'expected_agreement': 'high'
    },
    'interaction': {
        'name': 'Pure Interaction',
        'description': 'y = x0*x1 + x2*x3 -> expect LOW agreement between tree/linear',
        'true_importances': {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'expected_agreement': 'low'
    },
    'mixed': {
        'name': 'Mixed Mechanism',
        'description': 'y = 2*x0 + x1*x2 -> expect VARIABLE agreement',
        'true_importances': {0: 2.0, 1: 1.0, 2: 1.0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'expected_agreement': 'medium'
    },
    'redundant_features': {
        'name': 'Redundant Features',
        'description': 'y = x0 + x1 where x3 ~= x0, x4 ~= x1 -> expect confusion in importances',
        'true_importances': {0: 1.0, 1: 1.0, 2: 0, 3: 1.0, 4: 1.0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'expected_agreement': 'low'
    },
    'noisy_features': {
        'name': 'Signal + Noise',
        'description': 'y = 3*x0 + 2*x1 but x2-x9 are noise -> test robustness',
        'true_importances': {0: 3.0, 1: 2.0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
        'expected_agreement': 'high'
    }
}

def generate_synthetic_data(scenario, n_samples=2000, n_features=10, random_state=42):
    """Generate synthetic data with known ground truth."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    
    if scenario == 'linear_additive':
        z = 2 * X[:, 0] + 3 * X[:, 1] - 1 * X[:, 2]
    elif scenario == 'interaction':
        z = 5 * (X[:, 0] * X[:, 1]) + 5 * (X[:, 2] * X[:, 3])
    elif scenario == 'mixed':
        z = 2 * X[:, 0] + 3 * (X[:, 1] * X[:, 2])
    elif scenario == 'redundant_features':
        X[:, 3] = X[:, 0] + np.random.randn(n_samples) * 0.1  # x3 ~= x0
        X[:, 4] = X[:, 1] + np.random.randn(n_samples) * 0.1  # x4 ~= x1
        z = X[:, 0] + X[:, 1]
    elif scenario == 'noisy_features':
        z = 3 * X[:, 0] + 2 * X[:, 1]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    y = (z > np.median(z)).astype(int)
    feature_names = [f'f{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name='target')


def compute_shap_accuracy(shap_values, true_importances, n_features):
    """Compare SHAP values to ground truth importances."""
    # Normalize SHAP to absolute mean importance per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.sum() > 0:
        mean_abs_shap = mean_abs_shap / mean_abs_shap.sum()
    
    # Normalize ground truth
    true_vec = np.array([abs(true_importances.get(i, 0)) for i in range(n_features)])
    if true_vec.sum() > 0:
        true_vec = true_vec / true_vec.sum()
    
    # Spearman correlation with ground truth
    rho, p_val = stats.spearmanr(mean_abs_shap, true_vec)
    
    # Top-K overlap (correct features in top)
    n_true = int(np.sum(true_vec > 0))
    if n_true > 0:
        top_shap = set(np.argsort(mean_abs_shap)[-n_true:])
        top_true = set(np.argsort(true_vec)[-n_true:])
        topk_overlap = len(top_shap & top_true) / n_true
    else:
        topk_overlap = 1.0
    
    return {
        'spearman_with_truth': float(rho) if not np.isnan(rho) else 0.0,
        'topk_overlap': float(topk_overlap),
        'shap_importances': mean_abs_shap.tolist(),
        'true_importances': true_vec.tolist()
    }


# =============================================================================
# PER-SEED WORKER  (module-level so it is picklable for multiprocessing)
# =============================================================================

def run_scenario_seed(scenario_key, seed):
    """
    Run one (scenario, seed) combination and return a structured result dict.
    Called by multiprocessing.Pool.starmap.
    """
    import warnings
    warnings.filterwarnings('ignore')

    scenario_info = SCENARIOS[scenario_key]
    print(f"\n  [{scenario_key}] Seed {seed}:")

    X, y = generate_synthetic_data(scenario_key, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed),
        'lr': LogisticRegression(max_iter=1000, random_state=seed),
    }

    model_accs = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        model_accs[name] = acc
        print(f"    {name}: accuracy = {acc:.3f}")

    preds = {name: model.predict(X_test) for name, model in models.items()}
    agree_mask = (preds['rf'] == preds['gb']) & (preds['gb'] == preds['lr'])
    X_agree = X_test[agree_mask].iloc[:50]

    if len(X_agree) < 5:
        print(f"    [!WARN!] Only {len(X_agree)} agreement instances, skipping")
        return {
            'scenario_key': scenario_key,
            'seed': seed,
            'skipped': True,
            'model_accs': model_accs,
            'shap_accs': {},
            'pair_agreements': [],
            'overall_agreement': None,
            'n_agree_instances': len(X_agree),
        }

    print(f"    Agreement instances: {len(X_agree)}")

    shap_values = {}
    bg = X_train.sample(n=min(100, len(X_train)), random_state=seed)
    shap_accs = {}

    for name, model in models.items():
        try:
            if name == 'lr':
                explainer = shap.LinearExplainer(model, bg)
            else:
                explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_agree)
            if isinstance(sv, list):
                sv = sv[1]
            if len(getattr(sv, 'shape', [])) == 3:
                sv = sv[:, :, 1]
            shap_values[name] = sv

            acc_result = compute_shap_accuracy(sv, scenario_info['true_importances'], X.shape[1])
            shap_accs[name] = acc_result['spearman_with_truth']
            print(f"    {name} SHAP vs truth: rho = {acc_result['spearman_with_truth']:.3f}, "
                  f"top-K = {acc_result['topk_overlap']:.2f}")
        except Exception as e:
            print(f"    {name} SHAP failed: {e}")

    model_names = list(shap_values.keys())
    pair_agreements = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_a, m_b = model_names[i], model_names[j]
            instance_rhos = []
            for idx in range(len(X_agree)):
                rho, _ = stats.spearmanr(shap_values[m_a][idx], shap_values[m_b][idx])
                if not np.any(np.isnan(np.atleast_1d(rho))):
                    instance_rhos.append(rho)
            if instance_rhos:
                mean_rho = float(np.mean(instance_rhos))
                pair_agreements.append({
                    'pair': f'{m_a}-{m_b}',
                    'mean_rho': mean_rho,
                    'std_rho': float(np.std(instance_rhos)),
                    'n_instances': len(instance_rhos),
                })
                print(f"    {m_a} vs {m_b}: rho = {mean_rho:.3f}")

    overall_agreement = (
        float(np.mean([p['mean_rho'] for p in pair_agreements]))
        if pair_agreements else None
    )

    return {
        'scenario_key': scenario_key,
        'seed': seed,
        'skipped': False,
        'model_accs': model_accs,
        'shap_accs': shap_accs,
        'pair_agreements': pair_agreements,
        'overall_agreement': overall_agreement,
        'n_agree_instances': len(X_agree),
    }


# =============================================================================
# MAIN EXPERIMENT  (parallelised across scenario × seed combinations)
# =============================================================================

if __name__ == '__main__':
    combos = [
        (scenario_key, seed)
        for scenario_key in SCENARIOS
        for seed in [42, 123, 456]
    ]

    print(f"\nRunning {len(combos)} scenario-seed combinations "
          f"on {min(cpu_count(), 5)} workers...\n")

    with Pool(processes=min(cpu_count(), 5)) as pool:
        raw_results = pool.starmap(run_scenario_seed, combos)

    # -------------------------------------------------------------------------
    # Aggregate per-scenario (sequential)
    # -------------------------------------------------------------------------
    all_results = {}

    for scenario_key, scenario_info in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_info['name']}")
        print(f"  {scenario_info['description']}")
        print(f"  Expected agreement: {scenario_info['expected_agreement']}")
        print(f"{'='*70}")

        seed_results = [r for r in raw_results if r['scenario_key'] == scenario_key]

        scenario_results = {'info': scenario_info, 'seeds': {}}
        seed_agreements = []
        seed_accuracies = {'rf': [], 'gb': [], 'lr': []}
        seed_shap_accuracies = {'rf': [], 'gb': [], 'lr': []}

        for r in seed_results:
            seed = r['seed']
            if r['skipped']:
                continue

            scenario_results['seeds'][str(seed)] = {
                'n_agree_instances': r['n_agree_instances'],
                'pair_agreements': r['pair_agreements'],
            }

            for name, acc in r['model_accs'].items():
                seed_accuracies[name].append(acc)
                print(f"    {name}: accuracy = {acc:.3f}")

            for name, rho in r['shap_accs'].items():
                seed_shap_accuracies[name].append(rho)

            for p in r['pair_agreements']:
                print(f"    {p['pair'].replace('-', ' vs ')}: rho = {p['mean_rho']:.3f}")

            if r['overall_agreement'] is not None:
                seed_agreements.append(r['overall_agreement'])

        if seed_agreements:
            mean_agreement = float(np.mean(seed_agreements))
            std_agreement = float(np.std(seed_agreements))

            print(f"\n  SCENARIO SUMMARY: {scenario_info['name']}")
            print(f"    Mean agreement:   rho = {mean_agreement:.3f} +/- {std_agreement:.3f}")
            print(f"    Expected:         {scenario_info['expected_agreement']}")

            if scenario_info['expected_agreement'] == 'high' and mean_agreement > 0.5:
                print(f"    Validation:       [OK] MATCHES EXPECTATION")
            elif scenario_info['expected_agreement'] == 'low' and mean_agreement < 0.5:
                print(f"    Validation:       [OK] MATCHES EXPECTATION")
            elif scenario_info['expected_agreement'] == 'medium':
                print(f"    Validation:       ~ MIXED (as expected)")
            else:
                print(f"    Validation:       [FAIL] DOES NOT MATCH (interesting!)")

            scenario_results['summary'] = {
                'mean_agreement': mean_agreement,
                'std_agreement': std_agreement,
                'expected': scenario_info['expected_agreement'],
                'model_accuracies': {k: float(np.mean(v)) for k, v in seed_accuracies.items()},
                'shap_vs_truth': {
                    k: float(np.mean(v)) if v else None
                    for k, v in seed_shap_accuracies.items()
                },
            }

        all_results[scenario_key] = scenario_results

    # =========================================================================
    # CROSS-SCENARIO ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-SCENARIO ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Scenario':<25} {'Agreement':<15} {'Expected':<10} {'Match'}")
    print(f"  {'-'*25} {'-'*15} {'-'*10} {'-'*10}")

    for key, result in all_results.items():
        if 'summary' in result:
            s = result['summary']
            name = SCENARIOS[key]['name'][:25]
            agreement = f"rho = {s['mean_agreement']:.3f}"
            expected = s['expected']

            if expected == 'high' and s['mean_agreement'] > 0.5:
                match = "[OK]"
            elif expected == 'low' and s['mean_agreement'] < 0.5:
                match = "[OK]"
            elif expected == 'medium':
                match = "~"
            else:
                match = "[FAIL]"

            print(f"  {name:<25} {agreement:<15} {expected:<10} {match}")

    # KEY VALIDATION: Does agreement predict correctness?
    print(f"\n  KEY VALIDATION: Agreement ↔ SHAP Correctness")
    agreements = []
    shap_accuracies_all = []

    for key, result in all_results.items():
        if 'summary' in result:
            s = result['summary']
            agreements.append(s['mean_agreement'])
            truth_vals = [v for v in s['shap_vs_truth'].values() if v is not None]
            if truth_vals:
                shap_accuracies_all.append(np.mean(truth_vals))

    if len(agreements) >= 3:
        corr, p_val = stats.spearmanr(agreements, shap_accuracies_all)
        print(f"    Correlation (agreement vs SHAP accuracy): rho = {corr:.3f}, p = {p_val:.3f}")
        if corr > 0:
            print(f"    -> Higher agreement = better explanations [OK] (validates reliability)")
        else:
            print(f"    -> No clear relationship (more investigation needed)")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    output_file = os.path.join(OUTPUT_DIR, '04_synthetic_ground_truth_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4, default=str)
    print(f"\n  Saved: {output_file}")

    # Summary report
    summary_file = os.path.join(OUTPUT_DIR, '04_synthetic_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SYNTHETIC GROUND TRUTH VALIDATION -- SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        for key, result in all_results.items():
            if 'summary' in result:
                s = result['summary']
                f.write(f"Scenario: {SCENARIOS[key]['name']}\n")
                f.write(f"  Agreement:   rho = {s['mean_agreement']:.3f}\n")
                f.write(f"  Expected:    {s['expected']}\n")
                f.write(f"  Accuracies:  {s['model_accuracies']}\n")
                f.write(f"  SHAP truth:  {s['shap_vs_truth']}\n\n")

    print(f"  Saved: {summary_file}")
    print("\n" + "=" * 70)
    print("EXPERIMENT 04 COMPLETE")
    print("=" * 70)
