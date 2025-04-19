# train.py (Corrected import and definition)
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             log_loss, roc_auc_score)

# Import ONLY the pipeline creation function
from pipeline import create_classification_pipeline

# --- Configuration ---
DATA_PATH = os.path.join("data", "AmesHousing.csv") # Ensure this path is correct
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "price_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list.joblib")
BIN_EDGES_PATH = os.path.join(MODEL_DIR, "bin_edges.joblib") # Path for bin edges

# Define Binner parameters
TARGET_COLUMN_ORIGINAL = "SalePrice"
TARGET_COLUMN_BINNED = "SalePrice_Category"
NUM_BINS = 3
BIN_LABELS = ['Low', 'Medium', 'High']

TEST_SIZE = 0.2
# --- Define RANDOM_STATE directly in this file ---
RANDOM_STATE = 42

# Columns to drop immediately
COLUMNS_TO_DROP = ["Order", "PID", "Utilities"]

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Custom Transformer for Binning ---
class PriceBinnerTransformer(BaseEstimator, TransformerMixin):
    """Bins the target column using pandas qcut."""
    def __init__(self, target_col='SalePrice', new_col_name='SalePrice_Category',
                 num_bins=3, labels=None):
        self.target_col = target_col
        self.new_col_name = new_col_name
        self.num_bins = num_bins
        self.labels = labels
        self.bin_edges_ = None # Will store the calculated bin edges

    def fit(self, X, y=None):
        if self.target_col not in X.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")
        # Use pd.qcut to find quantile-based bins and store the edges
        try:
            _, self.bin_edges_ = pd.qcut(X[self.target_col], q=self.num_bins,
                                          labels=self.labels, retbins=True, duplicates='drop')
            logging.info(f"Calculated bin edges for '{self.target_col}': {self.bin_edges_}")
        except ValueError as e:
            logging.error(f"Could not calculate bins with pd.qcut, likely due to non-unique edges: {e}")
            raise e # Re-raise to indicate a problem
        return self

    def transform(self, X):
        if self.bin_edges_ is None:
            raise RuntimeError("Transformer must be fitted before transforming.")
        X_transformed = X.copy()
        # Use pd.cut with the stored bin_edges_
        X_transformed[self.new_col_name] = pd.cut(X_transformed[self.target_col],
                                                 bins=self.bin_edges_,
                                                 labels=self.labels,
                                                 include_lowest=True,
                                                 duplicates='drop')
        # Ensure type consistency
        X_transformed[self.new_col_name] = X_transformed[self.new_col_name].astype('object')
        return X_transformed

# --- Evaluation Function ---
def evaluate_classification(model, X_test, y_test):
    """Calculates and logs classification metrics."""
    labels = model.classes_ # Get class labels from the fitted pipeline
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels)
    loss = log_loss(y_test, y_pred_proba, labels=labels)

    try:
        if len(labels) > 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted', labels=labels)
            roc_auc_type = "OvR Weighted"
        elif len(labels) == 2:
             positive_class_index = list(labels).index(labels[1])
             roc_auc = roc_auc_score(y_test, y_pred_proba[:, positive_class_index])
             roc_auc_type = "Binary"
        else:
             roc_auc = float('nan')
             roc_auc_type = "N/A"
    except ValueError as e:
        logging.warning(f"Could not calculate ROC AUC score: {e}")
        roc_auc = float('nan')
        roc_auc_type = "Error"

    logging.info(f"--- Classification Metrics ---")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Log Loss: {loss:.4f}")
    logging.info(f"ROC AUC ({roc_auc_type}): {roc_auc:.4f}")
    print(f"\nClassification Report:\n{report}") # Print report for better formatting
    logging.info(f"----------------------------")
    return {"accuracy": accuracy, "log_loss": loss, "roc_auc": roc_auc}

# --- Main Training Function ---
def run_training():
    """Loads data, bins target, trains the model pipeline, and saves artifacts."""
    logging.info("--- Starting Classification Model Training ---")

    # 1. Load Data
    logging.info(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {DATA_PATH}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    if TARGET_COLUMN_ORIGINAL not in df.columns:
        logging.error(f"Original target column '{TARGET_COLUMN_ORIGINAL}' not found.")
        return

    # --- Declare binner variable before try block ---
    binner = PriceBinnerTransformer(target_col=TARGET_COLUMN_ORIGINAL,
                                    new_col_name=TARGET_COLUMN_BINNED,
                                    num_bins=NUM_BINS, labels=BIN_LABELS)
    df_processed = None # Initialize df_processed

    # 2. Bin Target Variable
    logging.info(f"Binning target variable '{TARGET_COLUMN_ORIGINAL}'...")
    try:
        binner.fit(df) # Fit the binner to get bin_edges_
        df_binned = binner.transform(df)
        df_processed = df_binned.drop(columns=[TARGET_COLUMN_ORIGINAL] + COLUMNS_TO_DROP, errors='ignore')
        logging.info(f"Target binning complete. New shape: {df_processed.shape}")

        if df_processed[TARGET_COLUMN_BINNED].isnull().any():
            logging.warning(f"NaN values found in target column '{TARGET_COLUMN_BINNED}' after binning. Dropping affected rows.")
            original_count = len(df_processed)
            df_processed.dropna(subset=[TARGET_COLUMN_BINNED], inplace=True)
            logging.info(f"Dropped {original_count - len(df_processed)} rows with NaN target.")

        logging.info(f"Value counts for '{TARGET_COLUMN_BINNED}':\n{df_processed[TARGET_COLUMN_BINNED].value_counts(dropna=False)}")

    except Exception as e:
        logging.error(f"Error during target binning: {e}")
        return

    # --- Check if binner.bin_edges_ exists before proceeding ---
    if binner.bin_edges_ is None:
        logging.error("Bin edges were not calculated successfully. Cannot proceed.")
        return

    # 3. Feature Selection & Preparation
    logging.info("Preparing features and target...")
    X = df_processed.drop(TARGET_COLUMN_BINNED, axis=1)
    y = df_processed[TARGET_COLUMN_BINNED]

    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    logging.info(f"Identified features. Num: {len(numerical_features)}, Cat: {len(categorical_features)}.")
    feature_list = X.columns.tolist() # Save column names *before* pipeline transformations

    # 4. Split Data
    logging.info(f"Splitting data (Test size: {TEST_SIZE}, Random State: {RANDOM_STATE})...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    except ValueError as e:
         logging.warning(f"Could not stratify split: {e}. Splitting without stratification.")
         X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    # 5. Create and Train Pipeline
    logging.info("Creating classification pipeline...")
    pipeline = create_classification_pipeline(numerical_features, categorical_features)

    logging.info("Training pipeline...")
    try:
        pipeline.fit(X_train, y_train)
        logging.info("Pipeline training completed.")
    except Exception as e:
        logging.error(f"Error during pipeline training: {e}")
        return

    # 6. Evaluate Model
    logging.info("Evaluating model on the test set...")
    metrics = evaluate_classification(pipeline, X_test, y_test)

    # 7. Save Pipeline, Feature List, AND Bin Edges
    logging.info(f"Saving artifacts to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        joblib.dump(pipeline, MODEL_PATH)
        logging.info("Pipeline saved successfully.")
        joblib.dump(feature_list, FEATURES_PATH)
        logging.info("Feature list saved successfully.")
        # --- Save the bin edges ---
        joblib.dump(binner.bin_edges_, BIN_EDGES_PATH)
        logging.info("Bin edges saved successfully.")
    except Exception as e:
        logging.error(f"Error saving artifacts: {e}")

    logging.info("--- Classification Model Training Finished ---")

if __name__ == "__main__":
    run_training()