# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Set up the Streamlit page ---
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a better look and feel
st.markdown("""
    <style>
    .main-header {
        font-size: 60px;
        font-weight: bold;
        color: #007bff; /* Changed to a bright blue for contrast */
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 30px; /* Increased font size */
        font-weight: bold;
        color: #333333; /* Changed to a brighter color */
        text-align: center;
        margin-bottom: 40px;
    }
    .result-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
        text-align: center;
    }
    .prediction-label {
        font-size: 20px;
        font-weight: bold;
        color: #333333;
    }
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load the trained models and scaler ---
# The models and scaler were saved from the model training script
try:
    linear_model = joblib.load('linear_regression_model.pkl')
    xgb_model = joblib.load('xgboost_regressor_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Error: The model or scaler file was not found. Please ensure the following files exist in the same directory: "
             f"'linear_regression_model.pkl', 'xgboost_regressor_model.pkl', 'scaler.pkl'.")
    st.stop()

# --- Define the user interface ---
def main():
    """
    Main function to run the Streamlit app.
    """
    st.markdown('<p class="main-header">Car Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter the vehicle\'s details below to get a price prediction.</p>', unsafe_allow_html=True)

    # --- Create input widgets for car features ---
    with st.form(key='prediction_form'):
        st.subheader("Car Details")
        
        # Split inputs into columns for a cleaner layout
        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.number_input("Year", min_value=1990, max_value=datetime.datetime.now().year, value=2015, step=1)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
            engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)

        with col2:
            max_power = st.number_input("Max Power (bhp)", min_value=10, max_value=500, value=90, step=1)
            seats = st.number_input("Number of Seats", min_value=1, max_value=10, value=5, step=1)
            
            fuel = st.selectbox(
                "Fuel Type", 
                options=['Diesel', 'Petrol', 'LPG', 'CNG']
            )

        with col3:
            seller_type = st.selectbox(
                "Seller Type", 
                options=['Individual', 'Dealer', 'Trustmark Dealer']
            )
            transmission = st.selectbox(
                "Transmission Type", 
                options=['Manual', 'Automatic']
            )
            owner = st.selectbox(
                "Ownership", 
                options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
            )

        # Submit button
        st.markdown("---")
        submit_button = st.form_submit_button(label="Predict Price", type="primary")

    if submit_button:
        # --- Preprocessing the user input ---
        
        # Create a dictionary to hold the input data
        input_data = {
            'km_driven': km_driven,
            'engine': engine,
            'max_power': max_power,
            'seats': seats,
            'car_age': datetime.datetime.now().year - year,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner
        }
        
        # Convert to a DataFrame
        input_df = pd.DataFrame([input_data])
        
        # The list of numerical and categorical columns must match the training script
        numerical_cols = ['km_driven', 'engine', 'max_power', 'seats', 'car_age']
        categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
        
        # Perform one-hot encoding manually, matching the training script's columns
        # Note: The training script used pd.get_dummies with drop_first=True
        # We need to manually create the corresponding columns for the single input row
        
        # Create a list of all columns that the model was trained on
        model_columns = list(linear_model.feature_names_in_)
        
        # Initialize a new DataFrame with the same columns as the training data, filled with zeros
        processed_input = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        
        # Fill in the numerical values
        processed_input[numerical_cols] = input_df[numerical_cols]
        
        # Fill in the one-hot encoded columns based on user selection
        for col in categorical_cols:
            option = input_df[col].iloc[0]
            # Construct the column name as it would be from get_dummies(drop_first=True)
            if f'{col}_{option}' in processed_input.columns:
                processed_input[f'{col}_{option}'] = 1
        
        # Scale the numerical features using the pre-trained scaler
        processed_input[numerical_cols] = scaler.transform(processed_input[numerical_cols])

        # --- Make Predictions ---
        
        # Predict using both models
        linear_prediction = linear_model.predict(processed_input)[0]
        xgb_prediction = xgb_model.predict(processed_input)[0]

        # --- Display the results ---
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<p class="prediction-label">Linear Regression Predicted Price:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-value">${linear_prediction:,.2f}</p>', unsafe_allow_html=True)
        st.markdown('<p class="prediction-label">XGBoost Predicted Price:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-value">${xgb_prediction:,.2f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("Disclaimer: These are predictions based on a simplified model and may not reflect the actual market value.")
        
        # --- Add Model Insights Section ---
        st.markdown("---")
        with st.expander("Model Insights"):
            st.subheader("Feature Importance (XGBoost)")
            st.markdown("This chart shows which features had the most influence on the predicted price.")
            
            # Get feature importances from the trained XGBoost model
            feature_importances = xgb_model.feature_importances_
            feature_names = processed_input.columns
            importance_df = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)

            # Create a bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance_df.values, y=importance_df.index, palette='viridis', ax=ax)
            ax.set_title('Feature Importance for XGBoost Model')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Features')
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
