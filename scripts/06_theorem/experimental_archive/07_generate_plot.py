"""
=============================================================================
THEOREM VALIDATION — STEP 7: SCATTER PLOT & REGRESSION
=============================================================================
Purpose: Create the definitive scatter plot for the paper showing the
correlation between Interaction Strength and Attribution Gap (Δ).
=============================================================================
"""

import sys
import io
import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, spearmanr

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


def main():
    print("=" * 70)
    print("STEP 7: GENERATING THE DEFINITIVE SCATTER PLOT")
    print("=" * 70)

    # Load results
    results_path = os.path.join(OUTPUT_DIR, "05_interaction_results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["per_dataset"])

    # Check for outliers requested by user
    outliers = ["cnae9", "banknote-authentication", "ilpd", "german-credit"]

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white")

    # Scatter with regression line and 95% CI band
    ax = sns.regplot(
        data=df,
        x="mi_interaction_max",
        y="observed_delta",
        scatter_kws={"alpha": 0.6, "s": 100, "color": "#1f77b4"},
        line_kws={"color": "#d62728", "lw": 2},
        ci=95,
    )

    # Annotate Outliers
    for i, row in df.iterrows():
        if any(o in row["dataset"] for o in outliers):
            plt.annotate(
                row["dataset"].split("/")[-1],
                (row["mi_interaction_max"], row["observed_delta"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
                fontweight="bold",
            )

    # Statistics for the plot
    rho = data["mi_correlation"]["rho"]
    r_sq = data["mi_correlation"]["r"] ** 2

    plt.text(
        0.05,
        0.95,
        f"Spearman ρ = {rho:.3f}\nPearson R² = {r_sq:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    plt.title(
        "Information-Theoretic Evidence for the Explanation Lottery\nCorrelation Between Data Synergy and Attribution Gap (Δ)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel(
        "Interaction Mutual Information Score ($I(X_i \cdot X_j; Y)$)", fontsize=11
    )
    plt.ylabel("Observed Attribution Gap ($\Delta$)", fontsize=11)

    plt.grid(True, linestyle="--", alpha=0.3)

    fig_path = os.path.join(OUTPUT_DIR, "05_interaction_scatter_plot.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved scatter plot to: {fig_path}")


if __name__ == "__main__":
    main()
