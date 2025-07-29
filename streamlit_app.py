
import streamlit as st
import joblib
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

st.title("RxExplorer Prediction Demo")

@st.cache_resource
def load_model():
    return joblib.load("rx_model.joblib")

@st.cache_resource
def load_encoder():
    return joblib.load("target_encoder.joblib")

# load model and encoder
model = load_model()
encoder = load_encoder()

if "bounds_df" not in st.session_state:
    st.session_state.bounds_df = None
if "file_loaded" not in st.session_state:
    st.session_state.file_loaded = False

# create file loader
uploaded_file = st.file_uploader("Pick Boundaries.xlsx file...", type=["xlsx"])

# handle Boundaries file loading
if uploaded_file is not None:
    st.session_state.bounds_df = pd.read_excel(uploaded_file)
    st.session_state.file_loaded = True
elif st.button("Use default Boundaries.xlsx file"):
    default_path = Path("Boundaries.xlsx")
    if default_path.exists():
        st.session_state.bounds_df = pd.read_excel(default_path)
        st.session_state.file_loaded = True
    else:
        st.error("Default file Boundaries.xlsx not found.")

# If file loaded, show controls
if st.session_state.file_loaded:
    bounds = st.session_state.bounds_df

    bb = st.selectbox("Building Block (Bb)", bounds["Bb"].dropna().unique())
    cp = st.selectbox("Coupling Partner (CP)", bounds["CP"].dropna().unique())
    solvent = st.selectbox("Solvent", bounds["Solvent"].dropna().unique())
    electrode = st.selectbox("Electrode", bounds["Electrode"].dropna().unique())
    additive = st.selectbox("Additive", bounds["Additives"].dropna().unique())

    # Predict button handle
    if st.button("Predict"):
        # Prepare data for model
        row = pd.DataFrame([{
            'Bb': bb, 'CP': cp, 'Solvent': solvent,
            'Electrode': electrode, 'Additives': additive
        }])
        # Encode data for model
        row_encoded = encoder.transform(row)
        # Predict category
        prediction = model.predict(row_encoded)[0]
        # Probability of prediction
        proba = model.predict_proba(row_encoded)[0][1]

        st.success(f"Prediction: {'Success (1)' if prediction == 1 else 'Fail (0)'}")
        st.success(f"Probability of correct prediction: {proba:.2%}")

        # Plot feature importance
        importances = model.feature_importances_
        feat_names = row_encoded.columns
        fig, ax = plt.subplots()
        ax.barh(feat_names, importances)
        ax.set_title("Feature importance")
        st.pyplot(fig)
        