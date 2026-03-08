"""
OpenML data loading and preprocessing for Explanation Lottery experiments.
"""

import logging
import traceback

import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


def load_and_preprocess(
    dataset_id, random_state, test_size=0.2, max_samples=10000, binary_only=True
):
    """
    Load a dataset from OpenML and preprocess it for binary classification.

    Steps:
      1. Download via OpenML API.
      2. Label-encode categorical columns.
      3. Fill missing values with column medians; replace infinities.
      4. Encode target with LabelEncoder; optionally convert multiclass → binary
         (class 0 vs rest).
      5. Optionally subsample large datasets to *max_samples* rows.
      6. Stratified train/test split + StandardScaler on features.

    Args:
        dataset_id:   OpenML dataset identifier.
        random_state: Random seed for reproducibility.
        test_size:    Fraction of data held out for testing (default 0.2).
        max_samples:  Cap dataset to this many rows before splitting;
                      pass None to disable (default 10 000).
        binary_only:  If True, convert multiclass targets to binary
                      (class 0 vs rest) (default True).

    Returns:
        Tuple (X_train, X_test, y_train, y_test, feature_names,
               dataset_name, metadata).
        All elements are None if loading fails.
    """
    try:
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        X, y, _, feature_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        if X is None or y is None:
            logger.error(f"Dataset {dataset_id} returned None data")
            return (None,) * 7

        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        n_original = len(X)
        n_features_original = len(feature_names)

        # Encode categorical columns
        categorical_cols = []
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                categorical_cols.append(col)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # Convert to numeric, handle missing / infinite values
        X = X.apply(pd.to_numeric, errors="coerce")
        n_missing = int(X.isna().sum().sum())
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        # Encode target
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y.astype(str))
        n_classes = len(np.unique(y_encoded))

        if n_classes > 2 and binary_only:
            y_encoded = (y_encoded == 0).astype(int)
            n_classes = 2

        class_balance = np.min(np.bincount(y_encoded)) / len(y_encoded)

        # Optionally subsample large datasets
        if max_samples is not None and len(X) > max_samples:
            np.random.seed(random_state)
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X.iloc[idx]
            y_encoded = y_encoded[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        metadata = {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "n_instances": n_original,
            "n_features": n_features_original,
            "n_categorical": len(categorical_cols),
            "n_missing_values": n_missing,
            "n_classes": n_classes,
            "class_balance": round(class_balance, 4),
            "n_train": len(X_train_scaled),
            "n_test": len(X_test_scaled),
        }

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            list(X.columns),
            dataset.name,
            metadata,
        )

    except Exception as e:
        logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return (None,) * 7
