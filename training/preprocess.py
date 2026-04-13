"""
Data loading, Great Expectations validation, and feature extraction.

This module is imported by training/train.py and is intentionally kept
separate from model training so that validation can also be run standalone
in the CI/CD pipeline before the training step starts.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
VALID_SPECIES = ["setosa", "versicolor", "virginica"]
SPECIES_MAP = {name: idx for idx, name in enumerate(VALID_SPECIES)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """
    Read a CSV file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    logger.info("Loading data from '%s' …", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows × %d columns.", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Great Expectations validation
# ---------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the dataset using Great Expectations (legacy PandasDataset API).

    Checks performed
    ----------------
    * All required columns are present.
    * No null values in any column.
    * Numeric feature values are within plausible ranges (0–30 cm).
    * ``species`` values are confined to the three known classes.
    * The dataset has at least 10 rows.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset to validate.

    Returns
    -------
    bool
        ``True`` if every expectation passes.

    Raises
    ------
    ValueError
        If one or more expectations fail, with a summary of failures.
    ImportError
        If great-expectations is not installed.
    """
    try:
        from great_expectations.dataset import PandasDataset  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "great-expectations is required for data validation. "
            "Install it with: pip install great-expectations"
        ) from exc

    logger.info("Running Great Expectations data validation …")
    ge_df = PandasDataset(df)

    expectations = [
        # ── Column presence ───────────────────────────────────────────────
        ge_df.expect_column_to_exist("sepal_length"),
        ge_df.expect_column_to_exist("sepal_width"),
        ge_df.expect_column_to_exist("petal_length"),
        ge_df.expect_column_to_exist("petal_width"),
        ge_df.expect_column_to_exist("species"),
        # ── No nulls ──────────────────────────────────────────────────────
        ge_df.expect_column_values_to_not_be_null("sepal_length"),
        ge_df.expect_column_values_to_not_be_null("sepal_width"),
        ge_df.expect_column_values_to_not_be_null("petal_length"),
        ge_df.expect_column_values_to_not_be_null("petal_width"),
        ge_df.expect_column_values_to_not_be_null("species"),
        # ── Value ranges (cm) ─────────────────────────────────────────────
        ge_df.expect_column_values_to_be_between("sepal_length", min_value=0, max_value=30),
        ge_df.expect_column_values_to_be_between("sepal_width", min_value=0, max_value=30),
        ge_df.expect_column_values_to_be_between("petal_length", min_value=0, max_value=30),
        ge_df.expect_column_values_to_be_between("petal_width", min_value=0, max_value=30),
        # ── Domain constraints ────────────────────────────────────────────
        ge_df.expect_column_values_to_be_in_set("species", VALID_SPECIES),
        # ── Row count ─────────────────────────────────────────────────────
        ge_df.expect_table_row_count_to_be_between(min_value=10),
    ]

    failures = [exp for exp in expectations if not exp["success"]]

    if failures:
        for failure in failures:
            cfg = failure.get("expectation_config", {})
            logger.error("FAILED: %s  kwargs=%s", cfg.get("expectation_type"), cfg.get("kwargs"))
        raise ValueError(
            f"Data validation failed: {len(failures)} / {len(expectations)} checks did not pass."
        )

    logger.info("All %d Great Expectations checks passed ✓", len(expectations))
    return True


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features_and_labels(df: pd.DataFrame):
    """
    Convert a validated DataFrame into NumPy arrays suitable for sklearn.

    Parameters
    ----------
    df : pd.DataFrame
        A validated DataFrame that contains FEATURE_COLUMNS and TARGET_COLUMN.

    Returns
    -------
    X : np.ndarray, shape (n_samples, 4)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Integer-encoded class labels.
    """
    df = df.copy()
    df["label"] = df[TARGET_COLUMN].map(SPECIES_MAP)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train / test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
