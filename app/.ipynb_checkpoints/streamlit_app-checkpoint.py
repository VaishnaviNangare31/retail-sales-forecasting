import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Page setup
st.set_page_config(page_title="Retail Sales Forecast", layout="centered")
st.title("🛒 Retail Sales Forecasting Dashboard")

st.markdown("""
This app predicts **weekly sales** for a retail store using a machine learning model trained on historical data.  
Adjust the input features below to forecast sales.
""")

# Load the trained model
model_path = "sales_model.joblib"  # Must match the saved model file name
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error(f"❌ Model file not found at: {model_path}")
    st.stop()

# Input section
st.header("📥 Input Features")

temperature = st.slider("Temperature (°F)", 20.0, 120.0, 60.0)
fuel_price = st.slider("Fuel Price ($)", 2.0, 4.5, 3.0)
cpi = st.number_input("CPI", value=220.0)
unemployment = st.number_input("Unemployment Rate", value=7.5)
holiday_flag = st.selectbox("Is it a Holiday Week?", [0, 1])
date_input = st.date_input("Date", datetime(2012, 1, 6))
store = st.selectbox("Store ID", [1, 2, 3])  # Can extend this list if needed

# Extract date parts
month = date_input.month
year = date_input.year
week = date_input.isocalendar()[1]

# Prepare input dictionary (without Store_1)
input_data = {
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "Holiday_Flag": [holiday_flag],
    "Month": [month],
    "Year": [year],
    "Week": [week],
    "Store_2": [1 if store == 2 else 0],
    "Store_3": [1 if store == 3 else 0]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# ✅ FIX: Ensure correct column order as used in training
expected_columns = [
    "Temperature", "Fuel_Price", "CPI", "Unemployment", "Holiday_Flag",
    "Month", "Year", "Week", "Store_2", "Store_3"
]
input_df = input_df[expected_columns]

# Predict button
if st.button("🔮 Predict Weekly Sales"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"📈 Predicted Weekly Sales: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
