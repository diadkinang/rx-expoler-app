
import streamlit as st
import joblib
import pandas as pd
import yaml
import matplotlib.pyplot as plt

with open("Boundaries.yaml") as f:
    bounds = yaml.safe_load(f)

model = joblib.load("rx_model.joblib")

st.title("RxExplorer Prediction Demo")

# Форма выбора
bb = st.selectbox("Building Block (Bb)", bounds["Bb"])
cp = st.selectbox("Coupling Partner (CP)", bounds["CP"])
solvent = st.selectbox("Solvent", bounds["Solvent"])
electrode = st.selectbox("Electrode", bounds["Electrode"])
additive = st.selectbox("Additive", bounds["Additives"])

if st.button("Predict"):
    row = pd.DataFrame([{
        'Bb': bb, 'CP': cp, 'Solvent': solvent,
        'Electrode': electrode, 'Additives': additive
    }])
    encoder = joblib.load("target_encoder.joblib")
    row_encoded = encoder.transform(row)
    
    prediction = model.predict(row_encoded)[0]
    proba = model.predict_proba(row_encoded)[0][1]
    
    st.success(f"Prediction: {'Success' if prediction == 1 else 'Fail'}")
    st.success(f"Probability of success: {proba:.2%}")
    
    importances = model.feature_importances_
    feat_names = row_encoded.columns
    fig, ax = plt.subplots()
    ax.barh(feat_names, importances)
    ax.set_title("Feature importance")
    st.pyplot(fig)