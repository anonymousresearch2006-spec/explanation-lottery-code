"""
Model training and evaluation utilities for Explanation Lottery experiments.
"""

import logging
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from scripts.utils.config import MODEL_PARAMS as _DEFAULT_PARAMS

logger = logging.getLogger(__name__)


def train_models(X_train, y_train, random_state, model_params=None):
    """
    Train the 5 canonical classifiers with consistent hyperparameters.

    Args:
        X_train:      Training features (numpy array).
        y_train:      Training labels.
        random_state: Random seed for reproducibility.
        model_params: Dict with keys n_estimators, max_depth, learning_rate.
                      Defaults to MODEL_PARAMS from config.

    Returns:
        (models, train_times) where models is a dict {name: fitted_model}
        and train_times is a dict {name: seconds}.
    """
    if model_params is None:
        model_params = _DEFAULT_PARAMS

    models = {}
    train_times = {}
    params = model_params

    # XGBoost
    try:
        start = time.time()
        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train, y_train)
        models["xgboost"] = model
        train_times["xgboost"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"XGBoost failed: {str(e)[:100]}")

    # LightGBM
    try:
        start = time.time()
        model = LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=random_state,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(X_train, y_train)
        models["lightgbm"] = model
        train_times["lightgbm"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"LightGBM failed: {str(e)[:100]}")

    # CatBoost
    try:
        start = time.time()
        model = CatBoostClassifier(
            iterations=params["n_estimators"],
            depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_seed=random_state,
            verbose=False,
        )
        model.fit(X_train, y_train)
        models["catboost"] = model
        train_times["catboost"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"CatBoost failed: {str(e)[:100]}")

    # Random Forest
    try:
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        models["random_forest"] = model
        train_times["random_forest"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Random Forest failed: {str(e)[:100]}")

    # Logistic Regression
    try:
        start = time.time()
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1,
            solver="lbfgs",
        )
        model.fit(X_train, y_train)
        models["logistic_regression"] = model
        train_times["logistic_regression"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Logistic Regression failed: {str(e)[:100]}")

    # Linear SVM
    try:
        start = time.time()
        # LinearSVC and Ridge usually benefit from scaling
        from sklearn.preprocessing import StandardScaler

        svc_scaler = StandardScaler()
        X_train_svc = svc_scaler.fit_transform(X_train)
        model = LinearSVC(random_state=random_state, max_iter=2000)
        model.fit(X_train_svc, y_train)

        class ScaledModelWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
                self.inner_model = model
                self.is_scaled_wrapper = True

            def predict(self, X):
                return self.model.predict(self.scaler.transform(X))

            def decision_function(self, X):
                return self.model.decision_function(self.scaler.transform(X))

            def predict_proba(self, X):
                # LinearSVC doesn't have predict_proba by default
                if hasattr(self.model, "predict_proba"):
                    return self.model.predict_proba(self.scaler.transform(X))
                # Fallback to decision function for AUC
                df = self.decision_function(X)
                if len(df.shape) == 1:
                    return df
                return df

        models["linear_svc"] = ScaledModelWrapper(model, svc_scaler)
        train_times["linear_svc"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Linear SVC failed: {str(e)[:100]}")

    # Ridge Classifier (alpha=1.0)
    try:
        start = time.time()
        model = RidgeClassifier(alpha=1.0, random_state=random_state)
        model.fit(X_train, y_train)
        models["ridge"] = model
        train_times["ridge"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"Ridge failed: {str(e)[:100]}")

    # ElasticNet (via LogisticRegression)
    try:
        start = time.time()
        from sklearn.preprocessing import StandardScaler

        en_scaler = StandardScaler()
        X_train_en = en_scaler.fit_transform(X_train)
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=1000,
            random_state=random_state,
        )
        model.fit(X_train_en, y_train)
        models["elasticnet"] = ScaledModelWrapper(model, en_scaler)
        train_times["elasticnet"] = round(time.time() - start, 2)
    except Exception as e:
        logger.warning(f"ElasticNet failed: {str(e)[:100]}")

    # Neural Networks (MLP)
    from sklearn.neural_network import MLPClassifier

    for name, hidden in [("mlp_small", (50,)), ("mlp_medium", (100, 50))]:
        try:
            start = time.time()
            from sklearn.preprocessing import StandardScaler

            mlp_scaler = StandardScaler()
            X_train_mlp = mlp_scaler.fit_transform(X_train)
            model = MLPClassifier(
                hidden_layer_sizes=hidden, max_iter=1000, random_state=random_state
            )
            model.fit(X_train_mlp, y_train)
            models[name] = ScaledModelWrapper(model, mlp_scaler)
            train_times[name] = round(time.time() - start, 2)
        except Exception as e:
            logger.warning(f"MLP {name} failed: {str(e)[:100]}")

    return models, train_times


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models on the test set.

    Args:
        models: Dict {name: fitted_model}.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict {name: {'accuracy': float, 'auc': float}}.
    """
    results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
            results[name] = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "auc": round(roc_auc_score(y_test, y_proba), 4),
            }
        except Exception as e:
            logger.warning(f"Evaluation failed for {name}: {str(e)[:50]}")
            results[name] = {"accuracy": 0.0, "auc": 0.0}
    return results
