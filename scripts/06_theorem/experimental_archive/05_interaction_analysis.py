"""
=============================================================================
THEOREM VALIDATION — EXPERIMENT 5: INTERACTION INTELLIGENCE
=============================================================================
Purpose: Compute interaction strength (H-statistic and MI) across datasets
to correlate with the observed Agreement Gap (Δ).

This script performs Steps 1-4 of the User's Validation Plan:
  1. Computes H-statistic for feature pairs (Friedman H)
  2. Computes Pairwise Interaction Mutual Information
  3. Correlates metrics with Δ from Experiment 1
=============================================================================
"""

import sys
import io
import os
import json
import warnings
import numpy as np
import pandas as pd
import itertools
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.inspection import partial_dependence
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# sys.path guard
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.utils.data_loading import load_and_preprocess

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = "results"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "theorem_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 24 core datasets
DATASET_IDS = [
    1590,
    1461,
    31,
    1464,
    44,
    1510,
    37,
    1494,
    1489,
    1501,
    1497,
    4134,
    1462,
    1502,
    1468,
    1476,
    1478,
    1479,
    1480,
    1485,
    1486,
    1487,
    1491,
    1492,
]


def compute_h_statistic(model, X, feature_i, feature_j):
    """
    Computes the Friedman H-statistic for a pair of features.
    Ref: https://christophm.github.io/interpretable-ml-book/interaction.html
    """
    try:
        # Joint PDP
        pd_ij = partial_dependence(
            model, X, features=[feature_i, feature_j], grid_resolution=15
        )
        # Marginal PDPs
        pd_i = partial_dependence(model, X, features=[feature_i], grid_resolution=15)
        pd_j = partial_dependence(model, X, features=[feature_j], grid_resolution=15)

        # Center the PDPs
        val_ij = pd_ij["average"][0]
        val_i = pd_i["average"][0]
        val_j = pd_j["average"][0]

        # In sklearn, pd_ij['average'] is a 2D grid
        # We need to broadcast the marginals to match the joint grid
        # grid_i: grid_resolution, grid_j: grid_resolution

        # Simplified H-measure numerator (Interaction term)
        # PD_ij(xi, xj) - PD_i(xi) - PD_j(xj)

        # Since we just need a scalar strength, we compute the variance of the residuals
        # normalized by the variance of the joint PD.

        # Reshape to match joint grid
        gi = val_i.reshape(-1, 1)
        gj = val_j.reshape(1, -1)

        interaction_grid = val_ij - gi - gj

        numerator = np.sum(interaction_grid**2)
        denominator = np.sum(val_ij**2)

        h_squared = numerator / denominator if denominator > 0 else 0
        return np.sqrt(max(0, h_squared))
    except Exception:
        return 0.0


def dataset_interaction_mi(X, y):
    """Pairwise interaction MI as requested in Step 2."""
    n_features = X.shape[1]
    mi_scores = []

    # Subsample if too many features to avoid 2-hour compute
    max_features = 15
    if n_features > max_features:
        # Select features with highest individual MI
        base_mi = mutual_info_classif(X, y, random_state=42)
        top_idx = np.argsort(base_mi)[-max_features:]
        X_sub = X[:, top_idx]
    else:
        X_sub = X

    n_sub = X_sub.shape[1]

    for i, j in itertools.combinations(range(n_sub), 2):
        # Create interaction term (multiplicative for simple proxy)
        interaction = X_sub[:, i] * X_sub[:, j]
        mi = mutual_info_classif(interaction.reshape(-1, 1), y, random_state=42)[0]
        mi_scores.append(mi)

    if not mi_scores:
        return 0.0, 0.0
    return float(np.mean(mi_scores)), float(np.max(mi_scores))


def main():
    print("=" * 70)
    print("THEOREM VALIDATION — EXPERIMENT 5: INTERACTION INTELLIGENCE")
    print("=" * 70)

    # Load previously computed deltas
    delta_file = os.path.join(OUTPUT_DIR, "01_delta_results.json")
    if not os.path.exists(delta_file):
        print(f"ERROR: {delta_file} not found. Run Experiment 1 first.")
        return

    with open(delta_file, "r") as f:
        delta_data = json.load(f)

    # Map deltas by dataset name or ID
    observed_deltas = {r["dataset_id"]: r["delta"] for r in delta_data["per_dataset"]}

    h_stats = []
    mi_means = []
    mi_maxes = []
    deltas_to_match = []
    dataset_names = []

    process_ids = [d_id for d_id in DATASET_IDS if d_id in observed_deltas]
    print(f"Found {len(process_ids)} datasets with existing delta measurements.")

    for d_id in process_ids:
        print(f"\n[Processing Dataset ID: {d_id}]")
        result = load_and_preprocess(d_id, random_state=42)
        X_train, _, y_train, _, _, name, _ = result
        if X_train is None:
            continue

        # 1. H-Statistic (Step 1)
        print("  - Computing H-statistic...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train[:1000], y_train[:1000])  # Sample for speed

        # Check interactions of top 5 features
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[-5:]

        pairs = list(itertools.combinations(top_idx, 2))
        h_vals = [compute_h_statistic(rf, X_train[:500], p[0], p[1]) for p in pairs]
        h_score = np.max(h_vals) if h_vals else 0.0
        print(f"    Interaction strength (H): {h_score:.4f}")

        # 2. Mutual Information (Step 2)
        print("  - Computing Interaction MI...")
        mi_mean, mi_max = dataset_interaction_mi(X_train[:1000], y_train[:1000])
        print(f"    MI (mean/max): {mi_mean:.4f} / {mi_max:.4f}")

        h_stats.append(float(h_score))
        mi_means.append(float(mi_mean))
        mi_maxes.append(float(mi_max))
        deltas_to_match.append(float(observed_deltas[d_id]))
        dataset_names.append(name)

    # ── Correlation Analytics ───────────
    print("\n" + "=" * 70)
    print("STEP 4: CRITICAL CORRELATION CHECK")
    print("=" * 70)

    results_df = pd.DataFrame(
        {
            "dataset": dataset_names,
            "h_statistic": h_stats,
            "mi_interaction_mean": mi_means,
            "mi_interaction_max": mi_maxes,
            "observed_delta": deltas_to_match,
        }
    )

    # Pearson
    r_h, p_h = pearsonr(results_df["h_statistic"], results_df["observed_delta"])
    r_mi, p_mi = pearsonr(
        results_df["mi_interaction_max"], results_df["observed_delta"]
    )

    # Spearman (more robust)
    rho_h, _ = spearmanr(results_df["h_statistic"], results_df["observed_delta"])
    rho_mi, _ = spearmanr(
        results_df["mi_interaction_max"], results_df["observed_delta"]
    )

    print(
        f"H-statistic vs Δ: r={r_h:.3f} (Pearson), ρ={rho_h:.3f} (Spearman), p={p_h:.4f}"
    )
    print(
        f"MI (max) vs Δ   : r={r_mi:.3f} (Pearson), ρ={rho_mi:.3f} (Spearman), p={p_mi:.4f}"
    )

    # ── Decision Point ───────────
    print("\n[Decision Point]")
    r_val = max(r_h, r_mi)
    if r_val > 0.6 and p_h < 0.05:
        verdict = "GO: INFORMATION-THEORETIC THEOREM"
    elif r_val > 0.4:
        verdict = "BORDERLINE: WEAKER CLAIM"
    else:
        verdict = "ABANDON: OPTION 3 (ASYMPTOTIC TRUTH)"

    print(f"VERDICT: {verdict}")

    # ── Save Results ───────────
    out_path = os.path.join(OUTPUT_DIR, "05_interaction_results.json")
    summary = {
        "h_stat_correlation": {"r": r_h, "rho": rho_h, "p": p_h},
        "mi_correlation": {"r": r_mi, "rho": rho_mi, "p": p_mi},
        "verdict": verdict,
        "per_dataset": results_df.to_dict("records"),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal statistics saved to {out_path}")


if __name__ == "__main__":
    main()
