import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Loading trained model
model = joblib.load(r"artifacts\final-models\final-model-2025-07-12_13-17-43.pkl")

st.title("Power Consumption Predictor")

st.markdown("Enter the environmental and meteorological details below:")

# User Inputs
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=10.0, value=2.0)
general_diffuse_flows = st.number_input("General Diffuse Flows", value=100.0)
diffuse_flows = st.number_input("Diffuse Flows", value=50.0)
air_quality_index_pm = st.number_input("Air Quality Index (PM)", value=155.0)
cloudiness = st.selectbox("Cloudiness", options=[0, 1])

# Feature Engineering (same as training)
temp_squared=temperature**2
temp_cubed=temperature**3
temp_humidity_interaction = temperature * humidity
temp_wind_interaction=temperature*wind_speed
humidity_squared=humidity**2
humidity_wind_interaction=humidity*wind_speed
wind_speed_squared=wind_speed**2
wind_power=wind_speed**3

# Create input DataFrame
input_data = pd.DataFrame([{
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'general_diffuse_flows': general_diffuse_flows,
    'diffuse_flows': diffuse_flows,
    'air_quality_index': air_quality_index_pm,
    'cloudiness': cloudiness,
    'temp_squared':temp_squared,
    'temp_cubed': temp_cubed,
    'temp_humidity_interaction':temp_humidity_interaction,
    'temp_wind_interaction':temp_wind_interaction,
    'humidity_squared':humidity_squared,
    'humidity_wind_interaction':humidity_wind_interaction,
    'wind_speed_squared':wind_speed_squared,
    'wind_power':wind_power
}])

# Prediction
if st.button("Predict Power Consumption"):
    prediction = model.predict(input_data)[0]
    st.success(f"🔋 Predicted Power Consumption: {round(prediction, 2)} units")