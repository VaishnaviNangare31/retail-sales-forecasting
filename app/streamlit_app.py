import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load trained model
model = joblib.load("sales_model.joblib")

# Set Streamlit page config
st.set_page_config(page_title="Retail Sales Forecast", layout="centered")

st.title("🛒 Retail Sales Forecasting Dashboard")
st.markdown("""
This app predicts **weekly sales** for a retail store using a machine learning model.
Adjust inputs below to forecast sales for a specific week.
""")

# Input Features
st.header("📥 Input Features")

temperature = st.slider("Temperature (°F)", 20.0, 120.0, 60.0)
fuel_price = st.slider("Fuel Price ($)", 2.0, 4.5, 3.0)
cpi = st.number_input("CPI", value=220.0)
unemployment = st.number_input("Unemployment Rate", value=7.5)
holiday_flag = st.selectbox("Is it a Holiday Week?", [0, 1])
date_input = st.date_input("Date", datetime(2012, 1, 6))

# Extract date features
month = date_input.month
year = date_input.year
week = date_input.isocalendar()[1]

# Store selection
store = st.selectbox("Store ID", [1, 2, 3])
store_2 = 1 if store == 2 else 0
store_3 = 1 if store == 3 else 0

# Build DataFrame in EXACT model order
input_data = pd.DataFrame([{
    "Holiday_Flag": holiday_flag,
    "Temperature": temperature,
    "Fuel_Price": fuel_price,
    "CPI": cpi,
    "Unemployment": unemployment,
    "Month": month,
    "Year": year,
    "Week": week,
    "Store_2": store_2,
    "Store_3": store_3
}])

# Predict Button
if st.button("🔮 Predict Weekly Sales"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"📈 Predicted Sales: **${prediction:,.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
