import streamlit as st
import numpy as np
import joblib
import os

MODEL_PATH = "artifacts/customer_conversion_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

def make_prediction(input_data):
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    return prediction[0], probability

# Streamlit app layout
st.set_page_config(page_title="Customer Conversion Dashboard", layout="wide")
st.title("Customer Conversion Dashboard")
st.markdown("Enter customer behaviour data to predict conversion probability.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Predictions"])

if page == "Model Predictions":
    st.header("Model Predictions")
    st.write("Please provide the following customer data:")

    # Arrange input fields in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        ads_clicks = st.number_input("Ads Clicks", 0, 10, 5)
    with col2:
        time_on_site = st.number_input("Time on Site (minutes)", 0, 60, 30)
    with col3:
        pages_visited = st.number_input("Pages Visited", 1, 20, 10)

    if st.button("Predict Conversion"):
        st.markdown("---")
        input_data = np.array([[ads_clicks, time_on_site, pages_visited]])
        prediction, probability = make_prediction(input_data)
        if prediction == 1:
            st.success(f"Likely to convert! (Probability: {probability:.2f})")
        else:
            st.error(f"Unlikely to convert. (Probability: {probability:.2f})")
    st.markdown("---")
    st.write("#### Sample Input Guide")
    st.info("Typical values: Ads Clicks (0-10), Time on Site (0-60 min), Pages Visited (1-20)")

    st.markdown("---")
    st.write("### Model Information")
    st.write("**Model Type:** Random Forest Classifier")
    st.write("**Training Data Size:** 10,000 samples")
    st.write("**Features Used:**")
    st.write("- Ads Clicks")
    st.write("- Time on Site")
    st.write("- Pages Visited")

    st.markdown("---")
    st.caption("Developed by Your Team • Powered by Streamlit")

