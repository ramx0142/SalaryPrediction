import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
# Make sure 'salary_model.pkl' is in the same folder as this app.py file
try:
    model = joblib.load('Salary.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run the notebook to save 'salary_model.pkl' and place it here.")
    st.stop()

# 2. App Title and Description
st.title("💰 Salary Prediction App")
st.write("This machine learning model predicts the estimated salary based on years of experience.")

# 3. Input for the user
# The model was trained on 'Years of Experience' (float)
years_exp = st.number_input(
    "Enter Years of Experience:",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5,
    format="%.1f"
)

# 4. Predict Button
if st.button("Predict Salary"):
    try:
        # The model expects a 2D array like [[5]]
        input_data = np.array([[years_exp]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Extract the scalar value from the array (e.g., [66143.76] -> 66143.76)
        salary_result = prediction[0]
        
        # Display the result
        st.success(f"The estimated salary for {years_exp} years of experience is:")
        st.header(f"${salary_result:,.2f}")
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

# 5. Footer
st.markdown("---")
st.caption("Model trained using Linear Regression on Salary Data.")
