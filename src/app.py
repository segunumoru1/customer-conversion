import streamlit as st
import numpy as np
import joblib
import os

# Load the trained model
model_path = "..\\artifacts\\customer_conversion_model.pkl"
scaler_path = "..\\artifacts\\scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_model_and_scaler()

def make_prediction(input_data):
    if model is None:
        return None, None
    if scaler:
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
        if model is None or prediction is None:
            st.error("Prediction could not be made because the model is not loaded.")
        elif prediction == 1:
            st.success(f"The customer is likely to convert! (Probability: {probability:.2f})")
        else:
            st.error(f"The customer is unlikely to convert. (Probability: {probability:.2f})")
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

