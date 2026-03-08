"""
=============================================================================
THEOREM VALIDATION — EXPERIMENT 1: COMPUTE Δ
=============================================================================
Purpose: Compute Δ = ρ_intra - ρ_inter across all datasets.

This script:
  1. Loads each OpenML dataset
  2. Trains tree models (XGB, LGB, CatBoost, RF) and linear models (LR)
  3. Finds consensus instances (all models predict correctly)
  4. Computes SHAP values for each model
  5. Calculates mean |SHAP| per model → Spearman correlations
  6. Separates within-class (tree-tree) vs cross-class (tree-linear)
  7. Computes Δ and statistical tests

Output: results/results/theorem_validation/01_delta_results.json
=============================================================================
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# sys.path guard
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.utils.data_loading import load_and_preprocess
from scripts.utils.model_training import train_models
from scripts.utils.shap_computation import get_shap_values, find_agreement_instances

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = "results"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "theorem_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_DATASET_IDS = [
    31,
    37,
    44,
    50,
    1046,
    1049,
    1050,
    1462,
    1464,
    1479,
    1480,
    1494,
    1510,
    1590,
    4534,
    40536,
    40975,
    41027,
    23512,
    1063,
]

TREE_MODELS = ["xgboost", "lightgbm", "catboost", "random_forest"]
LINEAR_MODELS = ["logistic_regression"]

SPLIT_SEED = 42
MAX_SHAP_INSTANCES = 100

BANNER = "=" * 70

# ── Helpers ────────────────────────────────────────────────────────────────


def classify_pair(model_a, model_b):
    """Return 'intra_tree', 'intra_linear', or 'inter'."""
    a_tree = model_a in TREE_MODELS
    b_tree = model_b in TREE_MODELS
    if a_tree and b_tree:
        return "intra_tree"
    elif not a_tree and not b_tree:
        return "intra_linear"
    else:
        return "inter"


def compute_delta_for_dataset(dataset_id):
    """
    Compute ρ_intra, ρ_inter, and Δ for one dataset.

    Returns dict with results, or None on failure.
    """
    result = load_and_preprocess(dataset_id, random_state=SPLIT_SEED)
    X_train, X_test, y_train, y_test, feature_names, dataset_name, meta = result

    if X_train is None:
        return None

    print(f"  {dataset_name} ({meta['n_instances']} inst, {meta['n_features']} feat)")

    # Train models
    models, _ = train_models(X_train, y_train, random_state=SPLIT_SEED)

    if len(models) < 3:
        print("    Too few models trained — skipping")
        return None

    # Find agreement instances
    agree_idx = find_agreement_instances(models, X_test, y_test)
    if len(agree_idx) < 10:
        print(f"    Only {len(agree_idx)} agreement instances — skipping")
        return None

    if len(agree_idx) > MAX_SHAP_INSTANCES:
        rng = np.random.default_rng(SPLIT_SEED)
        agree_idx = rng.choice(agree_idx, MAX_SHAP_INSTANCES, replace=False)

    X_agree = X_test[agree_idx]
    print(f"    Agreement instances: {len(agree_idx)}")

    # Compute SHAP values
    shap_dict = {}
    for name, model in models.items():
        sv = get_shap_values(model, X_agree, X_train, name)
        if sv is not None:
            # Mean absolute SHAP across instances → feature importance vector
            shap_dict[name] = np.abs(sv).mean(axis=0)

    if len(shap_dict) < 3:
        print("    Too few SHAP computations succeeded — skipping")
        return None

    # Compute pairwise Spearman correlations, classified by pair type
    intra_rhos = []
    inter_rhos = []
    pair_details = []

    model_names = list(shap_dict.keys())
    for i, ma in enumerate(model_names):
        for mb in model_names[i + 1 :]:
            rho, pval = spearmanr(shap_dict[ma], shap_dict[mb])
            if np.isnan(rho):
                continue

            pair_type = classify_pair(ma, mb)
            pair_details.append(
                {
                    "model_a": ma,
                    "model_b": mb,
                    "pair_type": pair_type,
                    "spearman": round(float(rho), 4),
                }
            )

            if pair_type == "intra_tree":
                intra_rhos.append(float(rho))
            elif pair_type == "inter":
                inter_rhos.append(float(rho))

    if not intra_rhos or not inter_rhos:
        print("    Missing intra or inter pairs — skipping")
        return None

    rho_intra = float(np.mean(intra_rhos))
    rho_inter = float(np.mean(inter_rhos))
    delta = rho_intra - rho_inter

    print(f"    ρ_intra (tree-tree) : {rho_intra:.3f}  ({len(intra_rhos)} pairs)")
    print(f"    ρ_inter (tree-lin)  : {rho_inter:.3f}  ({len(inter_rhos)} pairs)")
    print(f"    Δ                   : {delta:.3f}")

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "n_features": meta["n_features"],
        "n_instances": meta["n_instances"],
        "n_agreement": len(agree_idx),
        "rho_intra": round(rho_intra, 4),
        "rho_inter": round(rho_inter, 4),
        "delta": round(delta, 4),
        "n_intra_pairs": len(intra_rhos),
        "n_inter_pairs": len(inter_rhos),
        "pair_details": pair_details,
    }


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    print(BANNER)
    print("THEOREM VALIDATION — EXPERIMENT 1: COMPUTE Δ")
    print(BANNER)
    print(f"Datasets: {len(ALL_DATASET_IDS)}")
    print(f"Tree models: {TREE_MODELS}")
    print(f"Linear models: {LINEAR_MODELS}")
    print(BANNER)

    all_results = []
    for ds_id in ALL_DATASET_IDS:
        print(f"\n[Dataset {ds_id}]")
        result = compute_delta_for_dataset(ds_id)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nERROR: No datasets completed successfully.")
        return 1

    # ── Statistical Analysis ──────────────────────────────────────────────
    print("\n" + BANNER)
    print("STATISTICAL ANALYSIS")
    print(BANNER)

    deltas = [r["delta"] for r in all_results]
    rho_intras = [r["rho_intra"] for r in all_results]
    rho_inters = [r["rho_inter"] for r in all_results]

    # Test 1: Is Δ > 0?
    t_stat, p_value = stats.ttest_1samp(deltas, 0, alternative="greater")
    print(f"\nTest: Δ > 0 (one-sample t-test)")
    print(f"  Mean Δ      : {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
    print(f"  Median Δ    : {np.median(deltas):.4f}")
    print(f"  Min Δ       : {np.min(deltas):.4f}")
    print(f"  Max Δ       : {np.max(deltas):.4f}")
    print(f"  t-statistic : {t_stat:.2f}")
    print(f"  p-value     : {p_value:.2e}")
    print(
        f"  Result      : {'*** SIGNIFICANT ***' if p_value < 0.001 else 'NOT SIGNIFICANT'}"
    )

    # Test 2: Cohen's d
    mean_d = float(np.mean(deltas))
    std_d = float(np.std(deltas, ddof=1))
    cohens_d = mean_d / std_d if std_d > 0 else float("inf")
    effect = "Large" if cohens_d > 0.8 else "Medium" if cohens_d > 0.5 else "Small"
    print(f"\nEffect Size:")
    print(f"  Cohen's d   : {cohens_d:.2f} ({effect})")

    # Test 3: Consistency
    positive_count = sum(1 for d in deltas if d > 0)
    print(f"\nConsistency:")
    print(
        f"  Δ > 0 in    : {positive_count}/{len(deltas)} datasets ({positive_count/len(deltas)*100:.0f}%)"
    )

    # Test 4: 95% CI
    se = stats.sem(deltas)
    ci_low, ci_high = stats.t.interval(0.95, len(deltas) - 1, loc=mean_d, scale=se)
    print(f"\n95% Confidence Interval:")
    print(f"  [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  Excludes zero: {'YES' if ci_low > 0 else 'NO'}")

    # ── Per-dataset table ─────────────────────────────────────────────────
    print("\n" + BANNER)
    print("PER-DATASET RESULTS")
    print(BANNER)
    print(f"{'Dataset':<25} {'ρ_intra':>8} {'ρ_inter':>8} {'Δ':>8} {'d':>5}")
    print("-" * 58)
    for r in sorted(all_results, key=lambda x: x["delta"], reverse=True):
        print(
            f"  {r['dataset_name']:<23} {r['rho_intra']:>8.3f} {r['rho_inter']:>8.3f} {r['delta']:>8.3f} {r['n_features']:>5}"
        )

    # ── Save ──────────────────────────────────────────────────────────────
    summary = {
        "experiment": "01_compute_delta",
        "theorem_claim": "Δ = ρ_intra - ρ_inter > 0",
        "n_datasets": len(all_results),
        "mean_rho_intra": round(float(np.mean(rho_intras)), 4),
        "mean_rho_inter": round(float(np.mean(rho_inters)), 4),
        "mean_delta": round(mean_d, 4),
        "std_delta": round(std_d, 4),
        "median_delta": round(float(np.median(deltas)), 4),
        "min_delta": round(float(np.min(deltas)), 4),
        "max_delta": round(float(np.max(deltas)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": float(p_value),
        "cohens_d": round(cohens_d, 4),
        "effect_size": effect,
        "positive_count": positive_count,
        "ci_95_lower": round(ci_low, 4),
        "ci_95_upper": round(ci_high, 4),
        "verdict": "CONFIRMED" if p_value < 0.001 and ci_low > 0 else "NOT CONFIRMED",
        "per_dataset": all_results,
    }

    out_path = os.path.join(OUTPUT_DIR, "01_delta_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + BANNER)
    print(
        f"VERDICT: Δ > 0 is {'CONFIRMED' if summary['verdict'] == 'CONFIRMED' else 'NOT CONFIRMED'}"
    )
    print(f"  Δ = {mean_d:.4f}, p = {p_value:.2e}, Cohen's d = {cohens_d:.2f}")
    print(BANNER)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
