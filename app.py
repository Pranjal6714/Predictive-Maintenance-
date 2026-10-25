
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.title("⚙️ Predictive Maintenance System")
st.write(
    "Enter machine measurements to predict possible machine failure."
)

st.subheader("Machine Measurements")

# Input fields
machine_type = st.selectbox(
    "Machine Type", ["L", "M", "H"]
)

air_temp = st.number_input(
    "Air Temperature (K)", value=300.0
)

process_temp = st.number_input(
    "Process Temperature (K)", value=310.0
)

speed = st.number_input(
    "Rotational Speed (rpm)", value=1500
)

torque = st.number_input(
    "Torque (Nm)", value=40.0
)

wear = st.number_input(
    "Tool Wear (minutes)", value=100
)

if st.button("Predict Machine Failure"):
    # Create features used during training
    data = {
        "Type": machine_type,
        "Air_temperature": air_temp,
        "Process_temperature": process_temp,
        "Rotational_speed": speed,
        "Torque": torque,
        "Tool_wear": wear,
        "Temp_diff": process_temp - air_temp,
        "Power": torque * speed * 2 * np.pi / 60,
        "Wear_x_Torque": wear * torque
    }

    input_df = pd.DataFrame([data])

    try:
        # Match the exact columns used during training
        expected = list(
            model.named_steps["prep"].feature_names_in_
        )
        input_df = input_df[expected]

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Machine failure predicted!")
        else:
            st.success("✅ No machine failure predicted.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            failure_index = classes.index(1)

            st.metric(
                "Estimated Failure Probability",
                f"{probabilities[failure_index] * 100:.2f}%"
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            "Check that the feature names and preprocessing "
            "match your trained model."
        )

