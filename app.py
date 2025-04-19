# app.py (Corrected: DataFrame.append fix + Vertical Price Ranges)
import logging
import os

import joblib
import numpy as np # Make sure numpy is imported
import pandas as pd
import streamlit as st

# --- Configuration ---
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "price_pipeline.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_list.joblib")
# --- Add path for bin edges ---
BIN_EDGES_PATH = os.path.join(MODEL_DIR, "bin_edges.joblib")

# Example options for dropdowns (should ideally match data)
NEIGHBORHOOD_OPTIONS = ['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst',
                        'Gilbert', 'NridgHt', 'Sawyer', 'NWAmes', 'SawyerW',
                        'BrkSide', 'Crawfor', 'Mitchel', 'NoRidge', 'Timber',
                        'IDOTRR', 'ClearCr', 'StoneBr', 'SWISU', 'MeadowV',
                        'Blmngtn', 'BrDale', 'Veenker', 'NPkVill', 'Blueste']
KITCHEN_QUAL_OPTIONS = ['TA', 'Gd', 'Ex', 'Fa', 'Po']


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Load Model, Feature List, and Bin Edges ---
@st.cache_resource # Cache loaded artifacts
def load_artifacts():
    """Loads the pipeline, feature list, and bin edges."""
    pipeline, feature_list, bin_edges = None, None, None # Initialize
    all_files_present = True

    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: Model file '{MODEL_PATH}' not found.")
        all_files_present = False
    if not os.path.exists(FEATURES_PATH):
        st.error(f"Error: Feature list file '{FEATURES_PATH}' not found.")
        all_files_present = False
    if not os.path.exists(BIN_EDGES_PATH):
        st.error(f"Error: Bin edges file '{BIN_EDGES_PATH}' not found.")
        all_files_present = False

    if not all_files_present:
         st.warning(f"One or more required files missing in '{MODEL_DIR}'. "
                  f"Please run `python train.py` first.")
         logging.error(f"One or more artifact files not found in {MODEL_DIR}.")
         return None, None, None

    try:
        pipeline = joblib.load(MODEL_PATH); logging.info("Pipeline loaded.")
        feature_list = joblib.load(FEATURES_PATH); logging.info("Feature list loaded.")
        bin_edges = joblib.load(BIN_EDGES_PATH); logging.info(f"Bin edges loaded: {bin_edges}") # Log loaded edges
        st.success("Loaded pre-trained model and configuration.")
        return pipeline, feature_list, bin_edges
    except Exception as e:
        st.error(f"An error occurred loading model artifacts: {e}")
        logging.error(f"Loading error: {e}")
        return None, None, None

pipeline, feature_list, bin_edges = load_artifacts()

# --- Helper function to format currency ---
def format_currency(value):
    """Formats a number as USD currency."""
    try:
        # Format with commas, no decimal places for cleaner look
        return "${:,.0f}".format(value)
    except (ValueError, TypeError):
        return "N/A" # Handle potential errors if value isn't numeric

# --- Streamlit UI ---
st.title("Ames Housing Price Category Prediction")

if pipeline and feature_list and bin_edges is not None: # Check all artifacts loaded
    st.write("Enter house details to predict its price category (Low, Medium, High).")

    # --- Display Price Ranges (Updated formatting: Vertical) ---
    try:
        # Assuming bin_edges is a numpy array like [min, edge1, edge2, max] from qcut
        if len(bin_edges) == 4:
            st.subheader("Price Category Ranges (based on training data):")
            # Use st.markdown for vertical stacking and standard text size
            st.markdown(f"**Low:** Up to {format_currency(bin_edges[1])}")
            st.markdown(f"**Medium:** {format_currency(bin_edges[1])} - {format_currency(bin_edges[2])}")
            st.markdown(f"**High:** Above {format_currency(bin_edges[2])}")
            st.markdown("---") # Add a visual separator
        else:
            st.warning("Loaded bin edges have an unexpected format. Cannot display ranges.")
            logging.warning(f"Unexpected bin_edges format: {bin_edges}")
    except Exception as e:
        st.error(f"Could not display price ranges due to an error: {e}")
        logging.error(f"Error displaying price ranges: {e}")


    # --- Input Form ---
    input_data = {}
    st.header("Key Property Details")
    col1, col2 = st.columns(2)
    # Define UI elements based on a subset of features present in feature_list
    # Ensure keys match the feature_list names
    ui_features = {
        'Gr Liv Area': {'type': 'number', 'min': 0, 'val': 1500, 'col': col1, 'label': 'Above Ground Living Area (SqFt)'},
        'Total Bsmt SF': {'type': 'number', 'min': 0, 'val': 1000, 'col': col1, 'label': 'Total Basement Area (SqFt)'},
        '1st Flr SF': {'type': 'number', 'min': 0, 'val': 1000, 'col': col1, 'label': 'First Floor Area (SqFt)'},
        'Garage Area': {'type': 'number', 'min': 0, 'val': 500, 'col': col1, 'label': 'Garage Area (SqFt)'},
        'Year Built': {'type': 'number', 'min': 1800, 'max': 2025, 'val': 1980, 'col': col1, 'label': 'Year Built'},
        'Overall Qual': {'type': 'slider', 'min': 1, 'max': 10, 'val': 5, 'col': col2, 'label': 'Overall Quality (1-10)'},
        'Full Bath': {'type': 'number', 'min': 0, 'val': 2, 'col': col2, 'label': 'Full Bathrooms'},
        'TotRms AbvGrd': {'type': 'number', 'min': 0, 'val': 6, 'col': col2, 'label': 'Total Rooms Above Ground'},
        'Garage Cars': {'type': 'number', 'min': 0, 'val': 2, 'col': col2, 'label': 'Garage Capacity (Cars)'},
        'Neighborhood': {'type': 'select', 'options': NEIGHBORHOOD_OPTIONS, 'col': col2, 'label': 'Neighborhood'},
        'Kitchen Qual': {'type': 'select', 'options': KITCHEN_QUAL_OPTIONS, 'col': col2, 'label': 'Kitchen Quality'}
    }

    # Create inputs only for features expected by the model
    for feature, config in ui_features.items():
        if feature in feature_list:
            with config['col']:
                if config['type'] == 'number':
                    current_value = input_data.get(feature, config['val'])
                    input_data[feature] = st.number_input(config['label'], min_value=config.get('min'), max_value=config.get('max'), value=current_value)
                elif config['type'] == 'slider':
                    current_value = input_data.get(feature, config['val'])
                    input_data[feature] = st.slider(config['label'], min_value=config['min'], max_value=config['max'], value=current_value)
                elif config['type'] == 'select':
                    current_value = input_data.get(feature, config['options'][0])
                    default_index = config['options'].index(current_value) if current_value in config['options'] else 0
                    input_data[feature] = st.selectbox(config['label'], config['options'], index=default_index)
        else:
            pass # Silently ignore UI features not found in the model's feature list

    st.markdown("---")
    st.write("_Note: Using a simplified input form. Provide values for the fields above._")

    # --- Prediction Logic (includes fix for DataFrame.append) ---
    if st.button("Predict Price Category"):
        try:
            # 1. Prepare the input data row as a dictionary
            row_data = {feature: np.nan for feature in feature_list}
            for feature, value in input_data.items():
                if feature in row_data: row_data[feature] = value
            logging.info(f"User input mapped: {row_data}")

            # 2. Create a single-row DataFrame
            input_df = pd.DataFrame([row_data], columns=feature_list)

            # 3. Type Conversion
            for col in feature_list:
                 if col in input_df.columns:
                     is_numeric_input = col in ui_features and ui_features[col]['type'] in ['number', 'slider']
                     is_processed_numeric = False
                     try: # Check if pipeline processes it numerically
                         # Adjust index if necessary based on actual pipeline structure
                         numeric_transformer_tuple = next((t for t in pipeline.named_steps['preprocessor'].transformers if t[0] == 'num'), None)
                         if numeric_transformer_tuple and col in numeric_transformer_tuple[2]:
                             is_processed_numeric = True
                     except Exception: pass # Ignore errors checking pipeline structure

                     if is_processed_numeric or is_numeric_input:
                        try: input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                        except Exception as E: logging.error(f"Conv fail {col}: {E}")
            logging.info(f"DF sent to pipeline:\n{input_df.to_string()}\n{input_df.dtypes}")

            # 4. Make Prediction
            prediction = pipeline.predict(input_df)
            prediction_proba = pipeline.predict_proba(input_df)
            predicted_category = prediction[0]
            probabilities = prediction_proba[0]
            logging.info(f"Predict: {predicted_category}, Probs: {probabilities}")

            # 5. Display Results
            st.subheader("Prediction Result")
            st.success(f"Predicted Price Category: **{predicted_category}**")
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame([probabilities], columns=pipeline.classes_)
            expected_order = ['Low', 'Medium', 'High']
            ordered_cols = [c for c in expected_order if c in proba_df.columns] + [c for c in proba_df.columns if c not in expected_order]
            proba_df = proba_df[ordered_cols]
            st.dataframe(proba_df.style.format("{:.2%}"))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            logging.error(f"Prediction error: {e}", exc_info=True)
else:
     # This message shows if artifacts failed to load
     st.warning("Could not load model artifacts. Please ensure `python train.py` has been run successfully.")