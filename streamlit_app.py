
import subprocess
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

@st.cache_resource
def load_training_data():
    return pd.read_csv("training_data.txt", sep="\t")

# load model and encoder
model = load_model()
encoder = load_encoder()
training_data = load_training_data()

if "bounds_df" not in st.session_state:
    st.session_state.bounds_df = None
if "bounds_file_loaded" not in st.session_state:
    st.session_state.bounds_file_loaded = False
if "bounds_default" not in st.session_state:
    st.session_state.bounds_default = False
if "segmented_selection" not in st.session_state:
    st.session_state.segmented_selection = False

def reset_predicted_state():
    st.session_state.segmented_selection = None

# handle Boundaries file loading
if st.session_state.bounds_df is not None:
    if st.button("Clear Boundaries.xlsx file", key="boundariesFileClear"):
        st.session_state.bounds_df = None
        st.session_state.bounds_file_loaded = False
        st.session_state.bounds_default = False
        st.rerun()
    if st.session_state.bounds_default:
        st.badge("Default Boundaries.xlsx loaded", icon=":material/check:", color="green")
    #elif bounds_uploaded_file:
    #   st.badge("Custom Boundaries.xlsx loaded", icon=":material/check:", color="green")
else:
    bounds_uploaded_file = st.file_uploader("Pick Boundaries.xlsx file...", type=["xlsx"])
    if bounds_uploaded_file is not None:
        st.session_state.bounds_df = pd.read_excel(bounds_uploaded_file)
        st.session_state.bounds_file_loaded = True
        st.session_state.bounds_default = False
        reset_predicted_state()
    elif st.session_state.bounds_default:
        st.badge("Default Boundaries.xlsx loaded", icon=":material/check:", color="green")
    elif st.button("Use default Boundaries.xlsx file", key="boundariesFileUpload"):
        default_path = Path("Boundaries.xlsx")
        if default_path.exists():
            st.session_state.bounds_df = pd.read_excel(default_path)
            st.session_state.bounds_file_loaded = True
            st.session_state.bounds_default = True
            reset_predicted_state()
            st.rerun()
        else:
            st.error("Default file Boundaries.xlsx not found.")
            st.session_state.bounds_default = False

st.divider()

if st.session_state.bounds_file_loaded :
    selection = st.segmented_control(
        label="", options=["📈 Scoring", "🧪 Most probable reactions", "💾 Constructor", "📊 Feature importance"], selection_mode="single", default="📈 Scoring"
    )
    st.session_state.segmented_selection = selection

    if selection == "📈 Scoring":
        st.text("Chemical reactions sorted by score")
        all_combos_filtered = pd.read_csv("all_combos_filtered.txt", sep="\t")
        all_combos_filtered = all_combos_filtered.sort_values(by="Prior_score", ascending=False)
        st.dataframe(all_combos_filtered, hide_index=True)
    elif selection == "🧪 Most probable reactions":
        st.text("Chemical reactions with most probable reactions")
        predictions = pd.read_csv("Predictions.txt", sep="\t")
        
        filtered = predictions[predictions["Prediction"] == 1]
        filtered = filtered.drop(columns=["Prediction", "Prediction_0"])
        filtered = filtered.sort_values(by="Prediction_1", ascending=False)
        filtered = filtered.rename(columns={"Prediction_1": "Probability"})
        
        st.dataframe(filtered, hide_index=True)
    elif selection == "💾 Constructor":
        st.text("Constructor to check prediction")
        bounds = st.session_state.bounds_df

        bb = st.selectbox("Building Block (Bb)", bounds["Bb"].dropna().unique())
        cp = st.selectbox("Coupling Partner (CP)", bounds["CP"].dropna().unique())
        solvent = st.selectbox("Solvent", bounds["Solvent"].dropna().unique())
        electrode = st.selectbox("Electrode", bounds["Electrode"].dropna().unique())
        additive = st.selectbox("Additive", bounds["Additives"].dropna().unique())

        # Predict button handle
        if st.button("Predict", key="tab3Predict"):
            # Check if this combination from training data
            matched_row = training_data[
                (training_data['Bb'] == bb) &
                (training_data['CP'] == cp) &
                (training_data['Solvent'] == solvent) &
                (training_data['Electrode'] == electrode) &
                (training_data['Additives'] == additive)
            ]

            if not matched_row.empty:
                productive_value = matched_row.iloc[0]['Productive']
                st.success(f"Productive: {'Success (1)' if productive_value == 1 else 'Fail (0)'}")
                st.info(f"This combination is from training data")
            else:
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
                proba = model.predict_proba(row_encoded)[0]

                st.success(f"Productive: {'Success (1)' if prediction == 1 else 'Fail (0)'}")
                st.info(f"Probability of class: 0 - {proba[0]:.2%}, 1 - {proba[1]:.2%}")
    elif selection == "📊 Feature importance":
        st.text("How much feature important to prediction")
        importances = model.feature_importances_
        feat_names = [col for col in training_data.columns if col != "Productive"]
        fig, ax = plt.subplots()
        ax.barh(feat_names, importances)
        ax.set_title("Feature importance")
        st.pyplot(fig)
