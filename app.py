import streamlit as st
import pickle
import pandas as pd
import os
import shap
import numpy as np

st.title("GenAI Fraud Investigator")

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    current_dir,
    "..",
    "models",
    "xgboost_model.pkl"
)

explainer_path = os.path.join(
    current_dir,
    "..",
    "models",
    "shap_explainer.pkl"
)

data_path = os.path.join(
    current_dir,
    "..",
    "data",
    "creditcard.csv"
)

# Load model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load SHAP explainer
with open(explainer_path, "rb") as f:
    explainer = pickle.load(f)

# Load dataset
df = pd.read_csv(data_path)

st.success("Model Loaded Successfully")
st.success("SHAP Explainer Loaded Successfully")
st.success("Dataset Loaded Successfully")

# Select one sample transaction
transaction = df.drop("Class", axis=1).iloc[0]

st.subheader("Sample Transaction")
st.write(transaction)

if st.button("Predict Fraud"):

    prediction = model.predict([transaction])

    if prediction[0] == 1:
        st.error("Fraud Transaction Detected")
    else:
        st.success("Genuine Transaction")

    # SHAP Explanation
    shap_values = explainer.shap_values(
        transaction.values.reshape(1, -1)
    )

    feature_names = transaction.index

    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })

    shap_df["Absolute"] = abs(shap_df["SHAP Value"])

    top_features = shap_df.sort_values(
        by="Absolute",
        ascending=False
    ).head(5)

    st.subheader("Top Risk Features")

    st.dataframe(
        top_features[["Feature", "SHAP Value"]]
    )

    st.subheader("Fraud Investigation Report")

    for _, row in top_features.iterrows():

        st.write(
            f"Feature {row['Feature']} contributed "
            f"{row['SHAP Value']:.2f} "
            f"to the prediction."
        )