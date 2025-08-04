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
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 30px;
        font-weight: bold;
        color: #333333;
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

# --- Load the trained models, scaler, and column list ---
try:
    linear_model = joblib.load('linear_regression_model.pkl')
    xgb_model = joblib.load('xgboost_regressor_model.pkl')
    scaler = joblib.load('scaler.pkl')
    training_columns = joblib.load('training_columns.pkl')
except FileNotFoundError as e:
    st.error(f"Error: A required file was not found. Please ensure the following files exist in the same directory: "
             f"'linear_regression_model.pkl', 'xgboost_regressor_model.pkl', 'scaler.pkl', 'training_columns.pkl'.")
    st.stop()

# --- Define the user interface ---
def main():
    st.markdown('<p class="main-header">Car Price Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter the vehicle\'s details below to get a price prediction.</p>', unsafe_allow_html=True)
    st.markdown("View the source code on GitHub: [Car Price Predictor](https://github.com/gbennett90/Car-Price-Prediction.git)")

    # The comprehensive list of brands from the dataset to ensure we match the training data.
    brands_list = [
        'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford',
        'Chevrolet', 'Toyota', 'Renault', 'Volkswagen', 'Nissan',
        'Skoda', 'Other', 'Audi', 'BMW', 'Mercedes-Benz', 'Datsun',
        'Mitsubishi', 'Fiat', 'Jeep', 'Land Rover', 'Jaguar',
        'Volvo', 'Ambassador', 'Isuzu', 'Force'
    ]

    # --- Create input widgets for car features ---
    with st.form(key='prediction_form'):
        st.subheader("Car Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            brand = st.selectbox("Car Brand", options=brands_list)
            year = st.number_input("Year", min_value=1990, max_value=datetime.datetime.now().year, value=2015, step=1)
            km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000)
            engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)

        with col2:
            max_power = st.number_input("Max Power (bhp)", min_value=10, max_value=500, value=90, step=1)
            seats = st.number_input("Number of Seats", min_value=1, max_value=10, value=5, step=1)
            fuel = st.selectbox("Fuel Type", options=['Diesel', 'Petrol', 'LPG', 'CNG'])

        with col3:
            seller_type = st.selectbox("Seller Type", options=['Individual', 'Dealer', 'Trustmark Dealer'])
            transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
            owner = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

        st.markdown("---")
        submit_button = st.form_submit_button(label="Predict Price", type="primary")

    if submit_button:
        # --- Preprocessing the user input (Revised and more robust) ---
        
        # 1. Create a DataFrame with a single row, containing all zeros.
        input_data_processed = pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns)

        # 2. Fill in the numerical features.
        input_data_processed.loc[0, 'km_driven'] = km_driven
        input_data_processed.loc[0, 'engine'] = engine
        input_data_processed.loc[0, 'max_power'] = max_power
        input_data_processed.loc[0, 'seats'] = seats
        input_data_processed.loc[0, 'car_age'] = datetime.datetime.now().year - year

        # 3. Fill in the one-hot encoded categorical features.
        if f'fuel_{fuel}' in input_data_processed.columns:
            input_data_processed.loc[0, f'fuel_{fuel}'] = 1

        if f'seller_type_{seller_type}' in input_data_processed.columns:
            input_data_processed.loc[0, f'seller_type_{seller_type}'] = 1

        if f'transmission_{transmission}' in input_data_processed.columns:
            input_data_processed.loc[0, f'transmission_{transmission}'] = 1

        if f'owner_{owner}' in input_data_processed.columns:
            input_data_processed.loc[0, f'owner_{owner}'] = 1
        
        # The feature logic
        if brand != 'Other' and f'brand_{brand}' in input_data_processed.columns:
            input_data_processed.loc[0, f'brand_{brand}'] = 1
        elif brand == 'Other' and 'brand_Other' in input_data_processed.columns:
            input_data_processed.loc[0, 'brand_Other'] = 1

        # 4. Scale the numerical features using the pre-trained scaler.
        numerical_cols = ['km_driven', 'engine', 'max_power', 'seats', 'car_age']
        input_data_processed[numerical_cols] = scaler.transform(input_data_processed[numerical_cols])

        # --- Make Predictions ---
        linear_prediction = linear_model.predict(input_data_processed)[0]
        xgb_prediction = xgb_model.predict(input_data_processed)[0]

        # --- Display the results ---
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown('<p class="prediction-label">Linear Regression Predicted Price:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-value">${linear_prediction:,.2f}</p>', unsafe_allow_html=True)
        st.markdown('<p class="prediction-label">XGBoost Predicted Price:</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="prediction-value">${xgb_prediction:,.2f}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.success("Disclaimer: These are predictions based on a simplified model and may not reflect the actual market value.")

        # --- Model Insights Section ---
        st.markdown("---")
        with st.expander("Model Insights"):
            st.subheader("Feature Importance (XGBoost)")
            st.markdown("This chart shows which features had the most influence on the predicted price.")

            feature_importances = xgb_model.feature_importances_
            feature_names = training_columns
            importance_df = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance_df.values, y=importance_df.index, palette='viridis', ax=ax)
            ax.set_title('Feature Importance for XGBoost Model')
            ax.set_xlabel('Importance Score')
            ax.set_ylabel('Features')
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
