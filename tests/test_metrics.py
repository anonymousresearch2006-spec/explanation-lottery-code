"""
Unit tests for scripts/utils/metrics.py.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.metrics import classify_model_pair, spearman_top_k


class TestClassifyModelPair:
    def test_tree_tree(self):
        assert classify_model_pair("xgboost", "lightgbm") == "Tree-Tree"

    def test_tree_tree_catboost_rf(self):
        assert classify_model_pair("catboost", "random_forest") == "Tree-Tree"

    def test_tree_linear(self):
        assert classify_model_pair("xgboost", "logistic_regression") == "Tree-Linear"

    def test_tree_linear_reversed(self):
        assert (
            classify_model_pair("logistic_regression", "random_forest") == "Tree-Linear"
        )

    def test_linear_linear(self):
        assert (
            classify_model_pair("logistic_regression", "logistic_regression")
            == "Linear-Linear"
        )


class TestSpearmanTopK:
    def test_perfect_agreement_rho(self):
        a = np.array([3.0, 2.0, 1.0, 0.5, 0.1])
        result = spearman_top_k(a, a, k_values=[3])
        assert result["spearman"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_disagreement_rho(self):
        a = np.array([3.0, 2.0, 1.0])
        b = np.array([1.0, 2.0, 3.0])
        result = spearman_top_k(a, b, k_values=[3])
        assert result["spearman"] == pytest.approx(-1.0, abs=1e-9)

    def test_top_k_perfect_overlap(self):
        a = np.array([3.0, 2.0, 1.0, 0.1, 0.0])
        result = spearman_top_k(a, a, k_values=[3])
        assert result["top_3_overlap"] == pytest.approx(1.0)

    def test_top_k_no_overlap(self):
        # top-3 of a: indices 0,1,2 ; top-3 of b: indices 2,3,4
        a = np.array([5.0, 4.0, 3.0, 0.1, 0.0])
        b = np.array([0.0, 0.1, 3.0, 4.0, 5.0])
        result = spearman_top_k(a, b, k_values=[2])
        # top-2 a: {0,1}, top-2 b: {3,4}  -> overlap = 0
        assert result["top_2_overlap"] == pytest.approx(0.0)

    def test_k_exceeds_length_skipped(self):
        a = np.array([1.0, 2.0])
        result = spearman_top_k(a, a, k_values=[5])
        assert "top_5_overlap" not in result

    def test_returns_spearman_key(self):
        a = np.array([1.0, 2.0, 3.0])
        result = spearman_top_k(a, a, k_values=[])
        assert "spearman" in result
