import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open("bankruptcy_model.pkl", "rb") as file:
        model = pk.load(file)
    return model

model = load_model()

st.title("📉 Bankruptcy Prediction")
st.sidebar.header("🔧 Input Parameters")

# Define the columns for the data table
def user_input():
    industrial_risk = st.sidebar.number_input("Industrial Risk", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    management_risk = st.sidebar.number_input("Management Risk", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    financial_flexibility = st.sidebar.number_input("Financial Flexibility", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    credibility = st.sidebar.number_input("Credibility", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    competitiveness = st.sidebar.number_input("Competitiveness", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    operating_risk = st.sidebar.number_input("Operating Risk", 0.0, 1.0, step=0.01, help="Enter a value between 0 and 1")
    
    return np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])

input_data = user_input()

if st.sidebar.button("Predict Bankruptcy"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Assuming class 1 is financially stable
    
    if probability > 0.5:
        st.success(f"The company is financially stable with {probability * 100:.2f}% confidence.")
    else:
        st.error(f"The company is at risk of bankruptcy with {(1 - probability) * 100:.2f}% confidence.")
        
# Footer
st.markdown("---")

st.info("Built using Streamlit, Pandas, and Scikit-learn.")

st.markdown("**About the Bankruptcy Prediction Model**")

st.markdown(
    "This model uses a logistic regression algorithm to predict the likelihood of a company going bankrupt based on financial ratios and risk factors. "
    "It was trained on a dataset containing both bankrupt and non-bankrupt companies. Please note that this model is a prototype and may not accurately predict bankruptcy in all cases."
)

st.markdown("**Deployed by Srikanth Lankemalla using Streamlit Cloud.**")