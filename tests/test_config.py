"""
Sanity tests for scripts/utils/config.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.config import (
    CONFIG,
    MODEL_NAMES,
    TREE_MODELS,
    LINEAR_MODELS,
    MODEL_PARAMS,
)


def test_all_model_names_present():
    for name in [
        "xgboost",
        "lightgbm",
        "catboost",
        "random_forest",
        "logistic_regression",
    ]:
        assert name in MODEL_NAMES, f"{name} missing from MODEL_NAMES"


def test_config_required_keys():
    required = [
        "random_seeds",
        "test_size",
        "top_k_values",
        "shap_background_samples",
        "max_instances_per_dataset",
        "min_agreement_instances",
    ]
    for key in required:
        assert key in CONFIG, f"CONFIG missing key: {key}"


def test_model_params_keys():
    for key in ["n_estimators", "max_depth", "learning_rate"]:
        assert key in MODEL_PARAMS, f"MODEL_PARAMS missing key: {key}"


def test_tree_plus_linear_equals_model_names():
    assert set(TREE_MODELS + LINEAR_MODELS) == set(
        MODEL_NAMES
    ), "TREE_MODELS + LINEAR_MODELS must equal MODEL_NAMES"


def test_tree_linear_disjoint():
    overlap = set(TREE_MODELS) & set(LINEAR_MODELS)
    assert len(overlap) == 0, f"TREE_MODELS and LINEAR_MODELS overlap: {overlap}"


def test_config_test_size_range():
    assert 0 < CONFIG["test_size"] < 1


def test_config_seeds_nonempty():
    assert len(CONFIG["random_seeds"]) > 0
