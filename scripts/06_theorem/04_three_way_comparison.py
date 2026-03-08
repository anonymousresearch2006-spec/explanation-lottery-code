"""
=============================================================================
THEOREM VALIDATION — EXPERIMENT 4: 3-WAY HYPOTHESIS COMPARISON
=============================================================================
Purpose: Transform the study from Tree-vs-Linear to a general investigation
of "Hypothesis Class Boundaries" by adding Neural Networks (MLPs).

This script:
  1. Loads "Elite" datasets (including MNIST-like tabularized data)
  2. Trains 4 Trees, 4 Linears, and 2 MLPs
  3. Computes 3-way agreement (Tree vs Linear, Tree vs Neural, Linear vs Neural)
  4. Visualizes the "Hypothesis Class Boundary" effect

Output: results/results/theorem_validation/04_three_way_results.json
        results/results/theorem_validation/04_three_way_violin.png
=============================================================================
"""

import sys
import io
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# sys.path guard
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.utils.data_loading import load_and_preprocess
from scripts.utils.model_training import train_models
from scripts.utils.shap_computation import get_shap_values, find_agreement_instances

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = "results"
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "theorem_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Elite datasets spanning tabular, healthcare, finance, and vision-like
ELITE_DATASETS = [
    {"id": 1590, "name": "adult"},
    {"id": 31, "name": "german-credit"},
    {"id": 37, "name": "diabetes"},
    {"id": 1510, "name": "breast_cancer"},
    {"id": 1461, "name": "bank-marketing"},
    {"id": 1046, "name": "mnist_binary"},  # OpenML ID for skewed MNIST
]

TREE_FAM = ["xgboost", "lightgbm", "catboost", "random_forest"]
LINEAR_FAM = ["logistic_regression", "linear_svc", "ridge", "elasticnet"]
NEURAL_FAM = ["mlp_small", "mlp_medium"]

BANNER = "=" * 70


def get_class(model_name):
    if model_name in TREE_FAM:
        return "Tree"
    if model_name in LINEAR_FAM:
        return "Linear"
    if model_name in NEURAL_FAM:
        return "Neural"
    return "Other"


def run_3way_analysis():
    print(BANNER)
    print("EXPERIMENT 4: 3-WAY HYPOTHESIS CLASS COMPARISON")
    print(BANNER)

    all_rhos = []

    for ds_info in ELITE_DATASETS:
        ds_id = ds_info["id"]
        ds_name = ds_info["name"]
        print(f"\n[Dataset: {ds_name}]")

        result = load_and_preprocess(ds_id, random_state=42)
        X_train, X_test, y_train, y_test, features, _, meta = result
        if X_train is None:
            continue

        # 1. Train models
        models, _ = train_models(X_train, y_train, random_state=42)

        # 2. Find consensus
        agree_idx = find_agreement_instances(models, X_test, y_test)
        if len(agree_idx) < 10:
            print(f"  Only {len(agree_idx)} consensus instances. Skipping.")
            continue

        # Subsample for speed (KernelExplainer is slow)
        max_inst = 15
        rng = np.random.default_rng(42)
        use_idx = rng.choice(agree_idx, min(len(agree_idx), max_inst), replace=False)
        X_explain = X_test[use_idx]

        # 3. Compute SHAP
        shap_data = {}
        for name, model in models.items():
            print(f"  Computing SHAP for {name}...")
            sv = get_shap_values(model, X_explain, X_train, name)
            if sv is not None:
                shap_data[name] = sv

        # 4. Compute Pairwise Spearman
        names = list(shap_data.keys())
        for i, m_a in enumerate(names):
            class_a = get_class(m_a)
            for m_b in names[i + 1 :]:
                class_b = get_class(m_b)

                # Pair Label
                if class_a == class_b:
                    label = f"Intra-{class_a}"
                else:
                    sorted_classes = sorted([class_a, class_b])
                    label = f"{sorted_classes[0]} vs {sorted_classes[1]}"

                # Instance-level rhos
                for inst_i in range(len(X_explain)):
                    rho, _ = stats.spearmanr(
                        shap_data[m_a][inst_i], shap_data[m_b][inst_i]
                    )
                    if not np.isnan(rho):
                        all_rhos.append(
                            {
                                "Dataset": ds_name,
                                "Pair": f"{m_a} vs {m_b}",
                                "Comparison": label,
                                "Spearman ρ": float(rho),
                            }
                        )

    if not all_rhos:
        print("ERROR: No results generated.")
        return

    df = pd.DataFrame(all_rhos)

    # ── Visualization ─────────────────────────────────────────────────────
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    order = [
        "Intra-Tree",
        "Intra-Linear",
        "Intra-Neural",
        "Linear vs Tree",
        "Linear vs Neural",
        "Neural vs Tree",
    ]

    palette = {
        "Intra-Tree": "#2ca02c",
        "Intra-Linear": "#1f77b4",
        "Intra-Neural": "#ff7f0e",
        "Linear vs Tree": "#d62728",
        "Linear vs Neural": "#9467bd",
        "Neural vs Tree": "#8c564b",
    }

    sns.violinplot(
        data=df,
        x="Comparison",
        y="Spearman ρ",
        order=order,
        palette=palette,
        inner="box",
    )
    plt.axhline(0.5, color="black", alpha=0.3, ls="--")
    plt.title(
        "Explanation Lottery: 3-Way Hypothesis Class Comparison\nExpanding beyond Tree-vs-Linear",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylim(-0.2, 1.1)

    fig_path = os.path.join(OUTPUT_DIR, "04_three_way_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ── Summary JSON ──────────────────────────────────────────────────────
    summary = (
        df.groupby("Comparison")["Spearman ρ"].agg(["mean", "std"]).to_dict("index")
    )

    # Add Lottery Rate
    lottery_rates = {}
    for label in order:
        sub = df[df["Comparison"] == label]
        if not sub.empty:
            lottery_rates[label] = float((sub["Spearman ρ"] < 0.5).mean() * 100)
            summary[label]["lottery_rate"] = lottery_rates[label]

    out_json = os.path.join(OUTPUT_DIR, "04_three_way_results.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "experiment": "04_three_way_comparison",
                "summary": summary,
                "verdict": (
                    "CONFIRMED"
                    if summary.get("Linear vs Tree", {}).get("mean", 1.0) < 0.6
                    else "PARTIAL"
                ),
                "datasets": [d["name"] for d in ELITE_DATASETS],
            },
            f,
            indent=2,
        )

    print(f"\nSaved results to {out_json}")
    print(f"Saved plot to {fig_path}")


if __name__ == "__main__":
    run_3way_analysis()
