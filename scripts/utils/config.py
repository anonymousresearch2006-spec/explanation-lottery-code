"""
Centralised configuration constants for the Explanation Lottery experiments.
"""

CONFIG = {
    "random_seeds": [42, 123, 456],
    "test_size": 0.2,
    "max_instances_per_dataset": 200,
    "min_agreement_instances": 10,
    "top_k_values": [3, 5, 10],
    "shap_background_samples": 100,
}

MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
}

MODEL_NAMES = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "logistic_regression",
]

TREE_MODELS = ["xgboost", "lightgbm", "catboost", "random_forest"]

LINEAR_MODELS = ["logistic_regression"]
