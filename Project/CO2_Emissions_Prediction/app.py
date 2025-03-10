import numpy as np
import pickle as pkl
import streamlit as st

# Title and description
st.title("CO2 Emission Prediction")
st.write("""
This app predicts the **CO2 Emissions (g/km)** of a vehicle based on its specifications. 
Please provide the required inputs below:
""")

# Load the trained model and encoders
with open("Model.pkl", "rb") as file:
    model = pkl.load(file)

# Load the fitted encoders
with open("Encoders.pkl", "rb") as file:
    encoders = pkl.load(file)  # Dictionary containing LabelEncoders for each categorical column

# Get user input for features
make = st.selectbox("Make:", encoders["Make"].classes_, help="Select the make of the car.")
model_val = st.text_input("Model:", help="Enter the car's model (case-sensitive).")
vehicle_class = st.selectbox("Vehicle Class:", encoders["Vehicle Class"].classes_, help="Select the vehicle class.")
engine_size = st.number_input("Engine Size (L):", min_value=0.0, step=0.1, help="Enter the engine size in liters.")
cylinders = st.number_input("Cylinders:", min_value=0, step=1, help="Enter the number of cylinders.")
transmission = st.selectbox("Transmission:", encoders["Transmission"].classes_, help="Select the transmission type.")
fuel_type = st.selectbox("Fuel Type:", encoders["Fuel Type"].classes_, help="Select the fuel type.")
fuel_consumption_city = st.number_input("Fuel Consumption City (L/100 km):", min_value=0.0, step=0.1, help="Enter the fuel consumption in the city.")
fuel_consumption_hwy = st.number_input("Fuel Consumption Hwy (L/100 km):", min_value=0.0, step=0.1, help="Enter the fuel consumption on the highway.")
fuel_consumption_comb = st.number_input("Fuel Consumption Comb (L/100 km):", min_value=0.0, step=0.1, help="Enter the combined fuel consumption.")
fuel_consumption_comb_mpg = st.number_input("Fuel Consumption Comb (mpg):", min_value=0.0, step=0.1, help="Enter the combined fuel consumption in mpg.")

# Prediction
if st.button("Predict CO2 Emissions"):
    try:
        # Encode categorical inputs
        make_encoded = encoders["Make"].transform([make])[0]
        model_encoded = encoders["Model"].transform([model_val])[0]
        vehicle_class_encoded = encoders["Vehicle Class"].transform([vehicle_class])[0]
        transmission_encoded = encoders["Transmission"].transform([transmission])[0]
        fuel_type_encoded = encoders["Fuel Type"].transform([fuel_type])[0]

        # Prepare input features
        input_features = np.array([
            make_encoded, model_encoded, vehicle_class_encoded, engine_size,
            cylinders, transmission_encoded, fuel_type_encoded,
            fuel_consumption_city, fuel_consumption_hwy,
            fuel_consumption_comb, fuel_consumption_comb_mpg
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_features)

        # Display the result
        st.success(f"Predicted CO2 Emission (g/km): {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
