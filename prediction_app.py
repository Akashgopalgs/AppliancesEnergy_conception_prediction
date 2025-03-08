import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model, scaler, and column names
with open('lin_model.pkl', 'rb') as f:
    lin_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)


# Function to preprocess user input
def preprocess_input(hour, lights, t6, rh_6):
    # Create a DataFrame with the user input
    data = pd.DataFrame({
        'hour': [hour],
        'lights': [lights],
        't6': [t6],
        'rh_6': [rh_6],
        # Add placeholders for any features not included in the user input
        'high_consum': [0],  # Placeholder
        'low_consum': [0],  # Placeholder
        't6rh6': [t6 * rh_6],  # Example calculation, adjust if needed
        'windspeed': [0]  # Placeholder
    })

    # Ensure the order of columns matches the trained model
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_columns]  # Reorder columns

    # Scale the features
    data_scaled = scaler.transform(data)

    return data_scaled


# Streamlit App
st.title("Energy Consumption Prediction")

# User input
hour = st.slider("Hour of the Day", 0, 23, 12)
lights = st.slider("Lights", 0, 100, 10)
t6 = st.slider("T6 Temperature", -10.0, 40.0, 20.0)
rh_6 = st.slider("RH_6", 0.0, 100.0, 50.0)

# Button to make prediction
if st.button("Predict Energy Consumption"):
    # Preprocess input
    preprocessed_data = preprocess_input(hour, lights, t6, rh_6)

    # Make prediction
    prediction = lin_model.predict(preprocessed_data)

    # Display result
    st.write(f"Predicted Energy Consumption: {prediction[0]:.2f} Wh")
