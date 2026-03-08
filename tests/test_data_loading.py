"""
Unit tests for scripts/utils/data_loading.py.
Mocks the OpenML API so no network access is required.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.data_loading import load_and_preprocess


def _make_openml_mock(n=100, n_features=5, binary=True):
    """Return a mock openml dataset object with synthetic data."""
    mock_ds = MagicMock()
    mock_ds.name = "mock_dataset"
    mock_ds.default_target_attribute = "target"

    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    if binary:
        y = pd.Series(rng.integers(0, 2, n).astype(str))
    else:
        y = pd.Series(rng.integers(0, 3, n).astype(str))

    mock_ds.get_data.return_value = (X, y, None, list(X.columns))
    return mock_ds


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_returns_correct_structure(mock_get):
    mock_get.return_value = _make_openml_mock(n=100)
    result = load_and_preprocess(99999, random_state=42)
    X_train, X_test, y_train, y_test, feature_names, dataset_name, metadata = result

    assert X_train is not None
    assert X_test is not None
    assert len(feature_names) == 5
    assert dataset_name == "mock_dataset"
    assert isinstance(metadata, dict)


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_train_test_split_size(mock_get):
    n = 200
    mock_get.return_value = _make_openml_mock(n=n)
    X_train, X_test, *_ = load_and_preprocess(99999, random_state=42, test_size=0.2)

    assert len(X_test) == pytest.approx(n * 0.2, abs=3)
    assert len(X_train) == pytest.approx(n * 0.8, abs=3)


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_scaling_applied(mock_get):
    mock_get.return_value = _make_openml_mock(n=300)
    X_train, X_test, *_ = load_and_preprocess(99999, random_state=42)

    # After StandardScaler the training mean should be very close to 0
    assert abs(X_train.mean()) < 0.2


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_metadata_keys(mock_get):
    mock_get.return_value = _make_openml_mock(n=100)
    *_, metadata = load_and_preprocess(99999, random_state=42)

    for key in (
        "dataset_id",
        "dataset_name",
        "n_instances",
        "n_features",
        "n_classes",
        "class_balance",
        "n_train",
        "n_test",
    ):
        assert key in metadata, f"metadata missing key: {key}"


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_binary_only_converts_multiclass(mock_get):
    mock_get.return_value = _make_openml_mock(n=150, binary=False)
    *_, metadata = load_and_preprocess(99999, random_state=42, binary_only=True)

    assert metadata["n_classes"] == 2


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_max_samples_caps_dataset(mock_get):
    mock_get.return_value = _make_openml_mock(n=500)
    X_train, X_test, *_ = load_and_preprocess(
        99999, random_state=42, max_samples=100, test_size=0.2
    )
    assert len(X_train) + len(X_test) == pytest.approx(100, abs=2)


@patch("scripts.utils.data_loading.openml.datasets.get_dataset")
def test_returns_none_on_exception(mock_get):
    mock_get.side_effect = RuntimeError("network error")
    result = load_and_preprocess(99999, random_state=42)
    assert all(v is None for v in result)
