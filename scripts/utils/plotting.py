"""
Reusable plotting utilities for Explanation Lottery experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_agreement_heatmap(agreement_matrix, model_names, title, save_path):
    """
    Save a square heatmap of pairwise agreement scores.

    Args:
        agreement_matrix: 2-D numpy array (n_models × n_models).
        model_names:      List of model name labels.
        title:            Plot title string.
        save_path:        File path for the saved PNG (created if directory absent).
    """
    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        agreement_matrix,
        xticklabels=model_names,
        yticklabels=model_names,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Spearman Correlation"},
        ax=ax,
    )
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution(values, title, xlabel, save_path, mean_line=True):
    """
    Save a histogram of *values* with optional mean line.

    Args:
        values:    1-D array-like of numeric values.
        title:     Plot title string.
        xlabel:    X-axis label.
        save_path: File path for the saved PNG.
        mean_line: If True, draw a vertical dashed line at the mean.
    """
    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=50, edgecolor="black", alpha=0.7, color="steelblue")

    if mean_line:
        mean_val = values.mean()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax.legend()

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
