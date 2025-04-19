# pipeline.py
"""
Defines the scikit-learn preprocessing and modeling pipeline for house price category classification.
Note: Target variable binning happens *before* this pipeline in train.py.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler # Using StandardScaler for consistency

def create_classification_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
    model_params: dict = None,
) -> Pipeline:
    """
    Creates the scikit-learn pipeline combining preprocessing and the classification model.

    Args:
        numerical_features: List of names of numerical columns.
        categorical_features: List of names of categorical columns.
        model_params: Dictionary of parameters for the LogisticRegression model.

    Returns:
        A scikit-learn Pipeline object.
    """
    if model_params is None:
        # Default parameters for Logistic Regression
        model_params = {
            "solver": "liblinear", # Good for smaller datasets
            "random_state": 42,
            "max_iter": 1000, # Increased from default for convergence
            "C": 1.0, # Regularization strength
            "multi_class": "auto" # Handles multiclass if needed
        }

    # --- Preprocessing Steps ---

    # Strategy for numerical features:
    # 1. Impute missing values with the median.
    # 2. Scale features to have zero mean and unit variance.
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Strategy for categorical features:
    # 1. Impute missing values with a constant string 'missing'.
    # 2. Apply one-hot encoding, ignoring unknown categories during prediction.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # sparse=False easier for LogisticRegression input sometimes
        ]
    )

    # --- Column Transformer ---
    # Applies the defined transformers to the correct columns.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # --- Full Pipeline ---
    # Chains the preprocessor and the classification model.
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**model_params)),
        ]
    )

    return pipeline