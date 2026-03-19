"""
=============================================================================
THEOREM VALIDATION — STEP 5.5: BOOTSTRAP CORRELATION ANALYSIS
=============================================================================
Purpose: Calculate 95% Confidence Interval for Spearman Rho (ρ = 0.624)
to confirm statistical significance despite small sample size (n=24).
=============================================================================
"""

import sys
import io
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, bootstrap

# sys.path guard
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = "results"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "theorem_validation")


def spearman_stat(x, y):
    return spearmanr(x, y)[0]


def main():
    print("=" * 70)
    print("STEP 5.5: BOOTSTRAP CONFIDENCE INTERVAL ANALYSIS")
    print("=" * 70)

    # Load previously computed results
    results_path = os.path.join(OUTPUT_DIR, "05_interaction_results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["per_dataset"])
    mi_scores = df["mi_interaction_max"].values
    deltas = df["observed_delta"].values

    print(f"Sample size (n): {len(deltas)}")
    print(f"Observed ρ:    {data['mi_correlation']['rho']:.4f}")

    # ── Bootstrap ──────────────────────────────────────────────────────────
    print("\nRunning bootstrap (n_resamples=10000)...")

    # We need a wrapper because scipy bootstrap expects a sequence of samples
    # For paired samples, we bootstrap the indices
    def bootstrap_spearman(indices):
        return spearmanr(mi_scores[indices], deltas[indices])[0]

    rng = np.random.default_rng(42)
    indices = np.arange(len(deltas))

    # Note: Scipy 1.7+ bootstrap
    res = bootstrap(
        (indices,),
        bootstrap_spearman,
        n_resamples=10000,
        confidence_level=0.95,
        random_state=rng,
    )

    ci_low, ci_high = res.confidence_interval

    print("\n" + "=" * 70)
    print("BOOTSTRAP RESULTS")
    print("=" * 70)
    print(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")

    significant = "YES" if ci_low > 0 or ci_high < 0 else "NO"
    print(f"Statistically Significant (Excludes 0): {significant}")

    # ── Save Results ──────────────────────────────────────────────────────────
    data["bootstrap_results"] = {
        "rho_observed": float(data["mi_correlation"]["rho"]),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "significant": significant == "YES",
    }

    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nFinal statistics updated in {results_path}")


if __name__ == "__main__":
    main()
