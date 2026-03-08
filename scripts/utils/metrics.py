"""
Model-pair classification and agreement metric utilities.
"""

import numpy as np
from scipy.stats import spearmanr

from scripts.utils.config import TREE_MODELS, LINEAR_MODELS


def classify_model_pair(model_a, model_b):
    """
    Classify a model pair by architecture family.

    Args:
        model_a: Name string (e.g. 'xgboost', 'logistic_regression').
        model_b: Name string.

    Returns:
        One of "Tree-Tree", "Tree-Linear", or "Linear-Linear".
    """
    a_tree = model_a in TREE_MODELS
    b_tree = model_b in TREE_MODELS
    a_linear = model_a in LINEAR_MODELS
    b_linear = model_b in LINEAR_MODELS

    if a_tree and b_tree:
        return "Tree-Tree"
    if a_linear and b_linear:
        return "Linear-Linear"
    return "Tree-Linear"


def spearman_top_k(shap_a, shap_b, k_values):
    """
    Compute Spearman rank correlation and top-K overlap between two SHAP vectors.

    Args:
        shap_a:   1-D numpy array of SHAP values for model A.
        shap_b:   1-D numpy array of SHAP values for model B.
        k_values: List of k values for top-K overlap calculation.

    Returns:
        Dict with keys 'spearman' and 'top_{k}_overlap' for each k.
    """
    results = {}

    rho, _ = spearmanr(shap_a, shap_b)
    results["spearman"] = float(rho) if not np.isnan(rho) else 0.0

    for k in k_values:
        if len(shap_a) >= k:
            top_a = set(np.argsort(-np.asarray(shap_a))[:k])
            top_b = set(np.argsort(-np.asarray(shap_b))[:k])
            results[f"top_{k}_overlap"] = round(len(top_a & top_b) / k, 4)

    return results
