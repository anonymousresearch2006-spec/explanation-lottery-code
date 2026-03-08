"""
Unit tests for scripts/utils/shap_computation.py.
"""

import sys
import os

import numpy as np
import pytest
from unittest.mock import MagicMock
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.shap_computation import (
    find_agreement_instances,
    get_shap_values,
    compute_agreement_metrics,
)


class TestFindAgreementInstances:
    def _make_model(self, preds):
        m = MagicMock()
        m.predict.return_value = np.array(preds)
        return m

    def test_all_models_correct_returns_indices(self):
        m1 = self._make_model([0, 1, 1, 0])
        m2 = self._make_model([0, 1, 1, 0])
        X_test = np.zeros((4, 2))
        y_test = np.array([0, 1, 1, 0])
        idx = find_agreement_instances({"m1": m1, "m2": m2}, X_test, y_test)
        np.testing.assert_array_equal(np.sort(idx), [0, 1, 2, 3])

    def test_partial_disagreement_excludes_wrong(self):
        # instance 2: m2 predicts 0 but true label is 1
        m1 = self._make_model([0, 1, 1, 0])
        m2 = self._make_model([0, 1, 0, 0])
        X_test = np.zeros((4, 2))
        y_test = np.array([0, 1, 1, 0])
        idx = find_agreement_instances({"m1": m1, "m2": m2}, X_test, y_test)
        assert 2 not in idx
        assert 0 in idx
        assert 1 in idx
        assert 3 in idx

    def test_no_agreement_returns_empty(self):
        m1 = self._make_model([1, 0])
        m2 = self._make_model([1, 0])
        X_test = np.zeros((2, 2))
        y_test = np.array([0, 1])  # both models wrong on both instances
        idx = find_agreement_instances({"m1": m1, "m2": m2}, X_test, y_test)
        assert len(idx) == 0


class TestGetShapValues:
    def test_tree_model_returns_array(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 5))
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X[:50], y[:50])
        result = get_shap_values(model, X[50:], X[:50], "random_forest")
        assert result is not None
        assert result.shape == (10, 5)
        assert (result >= 0).all(), "Absolute SHAP values must be non-negative"

    def test_unknown_model_fails_gracefully(self):
        bad_model = MagicMock()
        bad_model.predict.side_effect = RuntimeError("boom")
        X = np.ones((5, 3))
        result = get_shap_values(bad_model, X, X, "unknown_model")
        # Should return None rather than raise
        assert result is None


class TestComputeAgreementMetrics:
    def _make_shap_dict(self):
        rng = np.random.default_rng(0)
        return {
            "model_a": rng.random((3, 5)),
            "model_b": rng.random((3, 5)),
        }

    def test_returns_list_of_dicts(self):
        shap_dict = self._make_shap_dict()
        results = compute_agreement_metrics(shap_dict, 0, 5)
        assert isinstance(results, list)
        assert len(results) == 1  # one pair

    def test_required_keys_present(self):
        shap_dict = self._make_shap_dict()
        r = compute_agreement_metrics(shap_dict, 0, 5)[0]
        for key in (
            "model_a",
            "model_b",
            "spearman",
            "weighted_corr",
            "cosine_similarity",
        ):
            assert key in r, f"Key '{key}' missing from result"

    def test_top_k_keys_present(self):
        shap_dict = self._make_shap_dict()
        r = compute_agreement_metrics(shap_dict, 0, 5, top_k_values=[3, 5])[0]
        assert "top_3_overlap" in r
        assert "top_5_overlap" in r

    def test_spearman_in_valid_range(self):
        shap_dict = self._make_shap_dict()
        r = compute_agreement_metrics(shap_dict, 0, 5)[0]
        assert -1.0 <= r["spearman"] <= 1.0

    def test_all_zeros_skipped(self):
        shap_dict = {
            "model_a": np.zeros((2, 5)),
            "model_b": np.ones((2, 5)),
        }
        results = compute_agreement_metrics(shap_dict, 0, 5)
        assert results == []  # all-zero SHAP → skipped

    def test_default_top_k_values(self):
        shap_dict = self._make_shap_dict()
        # No top_k_values arg → should use [3, 5, 10] but n_features=5 so k=10 skipped
        r = compute_agreement_metrics(shap_dict, 0, 5)[0]
        assert "top_3_overlap" in r
        assert "top_5_overlap" in r
        assert "top_10_overlap" not in r  # 5 features < 10
