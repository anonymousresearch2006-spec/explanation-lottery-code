"""
SHAP value computation and agreement metric utilities.
"""

import logging

import numpy as np
import shap
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _xgboost_native_shap(model, X_explain):
    """
    Use XGBoost's built-in pred_contribs to compute SHAP values.

    This bypasses shap.TreeExplainer, which fails on XGBoost >=3.x with
    shap <0.49 due to a UBJSON base_score format incompatibility.

    Returns absolute SHAP values (n_instances × n_features), bias column dropped.
    """
    import xgboost as xgb

    dmatrix = xgb.DMatrix(X_explain)
    sv = model.get_booster().predict(dmatrix, pred_contribs=True)
    # Last column is the bias term — drop it
    return np.abs(sv[:, :-1])


def get_shap_values(model, X_explain, X_background, model_name, n_background=100):
    """
    Compute absolute SHAP values for *model* on *X_explain*.

    Uses XGBoost's native pred_contribs for 'xgboost', TreeExplainer for
    other tree-based models, and LinearExplainer for linear models.

    Args:
        model:        Fitted classifier.
        X_explain:    Instances to explain (numpy array).
        X_background: Background / training data used by LinearExplainer.
        model_name:   One of 'xgboost', 'lightgbm', 'catboost',
                      'random_forest', 'logistic_regression', 'linear_svc',
                      'ridge_0.1', 'ridge_1.0', 'ridge_10.0', or 'elasticnet'.
        n_background: Max background samples for LinearExplainer (default 100).

    Returns:
        2-D numpy array of absolute SHAP values (n_instances × n_features),
        or None on failure.
    """
    try:
        if model_name == "xgboost":
            # Native XGBoost SHAP — avoids shap.TreeExplainer UBJSON bug on XGB 3.x
            return _xgboost_native_shap(model, X_explain)

        if model_name in ["lightgbm", "catboost", "random_forest"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
        elif model_name.startswith("mlp"):
            # MLPs require KernelExplainer (slow but robust)
            # Use the scaler from the wrapper if it exists
            if hasattr(model, "is_scaled_wrapper") and model.is_scaled_wrapper:
                explainer_model = model.inner_model
                explainer_bg = model.scaler.transform(X_background)
                X_exp_scaled = model.scaler.transform(X_explain)
            else:
                explainer_model = model
                explainer_bg = X_background
                X_exp_scaled = X_explain

            # Subsample background for speed
            bg_size = min(20, len(explainer_bg))
            bg_indices = np.random.choice(len(explainer_bg), bg_size, replace=False)
            bg_subset = explainer_bg[bg_indices]

            explainer = shap.KernelExplainer(explainer_model.predict_proba, bg_subset)
            shap_values = explainer.shap_values(X_exp_scaled)
        else:
            # LinearExplainer needs sampled background data
            actual_n = min(n_background, len(X_background))
            if len(X_background) > actual_n:
                indices = np.random.choice(len(X_background), actual_n, replace=False)
                background_sample = X_background[indices]
            else:
                background_sample = X_background

            # Handle ScaledModelWrapper
            if hasattr(model, "is_scaled_wrapper") and model.is_scaled_wrapper:
                explainer_model = model.inner_model
                explainer_bg = model.scaler.transform(background_sample)
                X_exp_final = model.scaler.transform(X_explain)
            else:
                explainer_model = model
                explainer_bg = background_sample
                X_exp_final = X_explain

            explainer = shap.LinearExplainer(explainer_model, explainer_bg)
            shap_values = explainer.shap_values(X_exp_final)

        # Normalise format: take class-1 values for binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        return np.abs(shap_values)

    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {str(e)[:100]}")
        return None


def find_agreement_instances(models, X_test, y_test):
    """
    Return indices of test instances where *all* models predict correctly.

    Args:
        models: Dict {name: fitted_model}.
        X_test: Test features (numpy array).
        y_test: True labels.

    Returns:
        1-D numpy array of integer indices.
    """
    all_correct = np.ones(len(y_test), dtype=bool)
    for name, model in models.items():
        try:
            preds = model.predict(X_test)
            all_correct &= preds == y_test
        except Exception as e:
            logger.warning(f"Prediction failed for {name}: {str(e)[:50]}")
            all_correct &= False
    return np.where(all_correct)[0]


def compute_agreement_metrics(shap_dict, instance_idx, n_features, top_k_values=None):
    """
    Compute pairwise SHAP agreement metrics for one instance.

    Metrics:
      - Spearman rank correlation
      - Top-K overlap (for each k in *top_k_values*)
      - Top-K Jaccard similarity
      - Weighted correlation
      - Cosine similarity

    Args:
        shap_dict:    Dict {model_name: shap_array} where shap_array is
                      (n_instances × n_features).
        instance_idx: Row index within each shap_array.
        n_features:   Number of features (used for top-K bounds check).
        top_k_values: List of k values (default [3, 5, 10]).

    Returns:
        List of dicts, one per model pair.
    """
    if top_k_values is None:
        top_k_values = [3, 5, 10]

    results = []
    model_names = list(shap_dict.keys())

    for i, model_a in enumerate(model_names):
        for model_b in model_names[i + 1 :]:
            try:
                shap_a = shap_dict[model_a][instance_idx]
                shap_b = shap_dict[model_b][instance_idx]

                if np.any(np.isnan(shap_a)) or np.any(np.isnan(shap_b)):
                    continue
                if np.all(shap_a == 0) or np.all(shap_b == 0):
                    continue

                # Spearman rank correlation
                rank_a = np.argsort(np.argsort(-shap_a))
                rank_b = np.argsort(np.argsort(-shap_b))
                if len(rank_a) > 1:
                    spearman_corr, spearman_pval = spearmanr(rank_a, rank_b)
                else:
                    spearman_corr, spearman_pval = 1.0, 0.0

                if np.isnan(spearman_corr):
                    spearman_corr = 0.0
                if np.isnan(spearman_pval):
                    spearman_pval = 1.0

                # Top-K overlap and Jaccard
                top_k_results = {}
                for k in top_k_values:
                    if n_features >= k:
                        top_a = set(np.argsort(-shap_a)[:k])
                        top_b = set(np.argsort(-shap_b)[:k])
                        overlap = len(top_a & top_b) / k
                        top_k_results[f"top_{k}_overlap"] = round(overlap, 4)
                        union = len(top_a | top_b)
                        jaccard = len(top_a & top_b) / union if union > 0 else 0
                        top_k_results[f"top_{k}_jaccard"] = round(jaccard, 4)

                # Weighted correlation by SHAP magnitude
                weights = (shap_a + shap_b) / 2
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights_norm = weights / weight_sum
                    weighted_corr = np.corrcoef(
                        shap_a * weights_norm, shap_b * weights_norm
                    )[0, 1]
                    if np.isnan(weighted_corr):
                        weighted_corr = 0.0
                else:
                    weighted_corr = 0.0

                # Cosine similarity
                norm_a = np.linalg.norm(shap_a)
                norm_b = np.linalg.norm(shap_b)
                cosine_sim = (
                    np.dot(shap_a, shap_b) / (norm_a * norm_b)
                    if norm_a > 0 and norm_b > 0
                    else 0.0
                )

                results.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "spearman": round(spearman_corr, 4),
                        "spearman_pval": round(spearman_pval, 6),
                        "weighted_corr": round(weighted_corr, 4),
                        "cosine_similarity": round(cosine_sim, 4),
                        **top_k_results,
                    }
                )

            except Exception:
                continue

    return results
