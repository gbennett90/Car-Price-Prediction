# app.py

# --- Imports ---
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime
import requests
import os

# --- API Configuration ---
# API key for currency conversion, retrieved securely from Streamlit secrets.
# This prevents your key from being exposed in your code.
try:
    API_KEY = st.secrets["EXCHANGE_RATE_API_KEY"]
except KeyError:
    # Fallback for local development with an .env file
    API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

# Fetches the INR to USD exchange rate.
def get_exchange_rate(api_key):
    """Fetches the latest INR to USD exchange rate from an API."""
    url = f"https://api.exchangerate-api.com/v4/latest/INR?access_key={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200 and 'USD' in data['rates']:
            return data['rates']['USD']
        else:
            st.error(f"Error fetching currency data: {data.get('error', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while fetching currency data: {e}")
        return None

# --- Streamlit Page Configuration ---
# Sets up the page layout and title.
st.set_page_config(
    page_title="Car Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
# Applies custom CSS to style the app's components.
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
    .stForm > div > div > div > button {
        background-color: #007bff !important;
        border-color: #007bff !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Resource Loading ---
# Loads machine learning models and data.
@st.cache_resource
def load_resources():
    try:
        linear_model = joblib.load('linear_regression_model.pkl')
        xgb_model = joblib.load('xgboost_regressor_model.pkl')
        scaler = joblib.load('scaler.pkl')
        training_columns = joblib.load('training_columns.pkl')
        
        try:
            sample_data = pd.read_csv('cleaned_car_data.csv')
        except FileNotFoundError:
            data = {
                'selling_price': np.random.uniform(100000, 1000000, 500),
                'brand': np.random.choice(['Maruti', 'Hyundai', 'Tata', 'Honda', 'Mahindra'], 500),
            }
            sample_data = pd.DataFrame(data)
            
        return linear_model, xgb_model, scaler, training_columns, sample_data
    except FileNotFoundError as e:
        st.error(f"Error: A required model file was not found. Please ensure the following files exist in the same directory: "
                 f"'linear_regression_model.pkl', 'xgboost_regressor_model.pkl', 'scaler.pkl', 'training_columns.pkl'.")
        st.stop()

# Load resources once at app startup.
linear_model, xgb_model, scaler, training_columns, sample_data = load_resources()

# --- Main Application Logic ---
# The main function that structures and runs the app.
def main():
    # App title and header.
    st.markdown('<p class="main-header">Car Price Predictor 🚗</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get an estimated price for your car.</p>', unsafe_allow_html=True)
    
    # Initialize session state.
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
        st.session_state.user_input = {}
        st.session_state.predictions = {}

    # Create UI tabs.
    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Detailed Breakdown", "Summary"])

    # --- TAB 1: Price Prediction Form ---
    # User input and prediction display.
    with tab1:
        st.header("Enter Car Details")
        st.info("Note: All price predictions are in Indian Rupees (₹) with an estimated conversion to US Dollars ($).")
        st.markdown("---")
        with st.form(key='prediction_form'):
            col1, col2, col3 = st.columns(3)

            brands_list = ['Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Chevrolet', 'Toyota', 'Renault', 'Volkswagen', 'Nissan', 'Skoda', 'Other', 'Audi', 'BMW', 'Mercedes-Benz', 'Datsun', 'Mitsubishi', 'Fiat', 'Jeep', 'Land Rover', 'Jaguar', 'Volvo', 'Ambassador', 'Isuzu', 'Force']

            with col1:
                st.selectbox("Car Brand", options=brands_list, key='brand_input', help="Select the manufacturer of the car.")
                st.number_input("Year", min_value=1990, max_value=datetime.datetime.now().year, value=2015, step=1, key='year_input', help="The year the car was manufactured.")
                st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=50000, step=1000, key='km_driven_input', help="Total distance the car has traveled.")
                st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100, key='engine_input', help="Engine displacement in cubic centimeters.")

            with col2:
                st.number_input("Max Power (bhp)", min_value=10, max_value=500, value=90, step=1, key='max_power_input', help="The maximum power output of the engine in brake horsepower.")
                st.number_input("Number of Seats", min_value=1, max_value=10, value=5, step=1, key='seats_input', help="The total number of seats in the car.")
                st.selectbox("Fuel Type", options=['Diesel', 'Petrol', 'LPG', 'CNG'], key='fuel_input', help="The type of fuel the car uses.")

            with col3:
                st.selectbox("Seller Type", options=['Individual', 'Dealer', 'Trustmark Dealer'], key='seller_type_input', help="Who is selling the car.")
                st.selectbox("Transmission Type", options=['Manual', 'Automatic'], key='transmission_input', help="Is the car's transmission manual or automatic?")
                st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], key='owner_input', help="The number of previous owners.")

            st.markdown("---")
            submit_button = st.form_submit_button(label="Predict Price", type="primary")

        # Handles prediction after form submission.
        if submit_button:
            st.session_state.prediction_made = True
            
            user_input = {
                'brand': st.session_state.brand_input, 'year': st.session_state.year_input,
                'km_driven': st.session_state.km_driven_input, 'engine': st.session_state.engine_input,
                'max_power': st.session_state.max_power_input, 'seats': st.session_state.seats_input,
                'fuel': st.session_state.fuel_input, 'seller_type': st.session_state.seller_type_input,
                'transmission': st.session_state.transmission_input, 'owner': st.session_state.owner_input
            }
            st.session_state.user_input = user_input

            # Preprocesses user input.
            input_data_processed = pd.DataFrame(np.zeros((1, len(training_columns))), columns=training_columns)
            input_data_processed.loc[0, 'km_driven'] = user_input['km_driven']
            input_data_processed.loc[0, 'engine'] = user_input['engine']
            input_data_processed.loc[0, 'max_power'] = user_input['max_power']
            input_data_processed.loc[0, 'seats'] = user_input['seats']
            input_data_processed.loc[0, 'car_age'] = datetime.datetime.now().year - user_input['year']
            
            if f'fuel_{user_input["fuel"]}' in input_data_processed.columns: input_data_processed.loc[0, f'fuel_{user_input["fuel"]}'] = 1
            if f'seller_type_{user_input["seller_type"]}' in input_data_processed.columns: input_data_processed.loc[0, f'seller_type_{user_input["seller_type"]}'] = 1
            if f'transmission_{user_input["transmission"]}' in input_data_processed.columns: input_data_processed.loc[0, f'transmission_{user_input["transmission"]}'] = 1
            if f'owner_{user_input["owner"]}' in input_data_processed.columns: input_data_processed.loc[0, f'owner_{user_input["owner"]}'] = 1
            
            if user_input['brand'] != 'Other' and f'brand_{user_input["brand"]}' in input_data_processed.columns:
                input_data_processed.loc[0, f'brand_{user_input["brand"]}'] = 1
            elif user_input['brand'] == 'Other' and 'brand_Other' in input_data_processed.columns:
                input_data_processed.loc[0, 'brand_Other'] = 1
            
            # Scales numerical features.
            numerical_cols = ['km_driven', 'engine', 'max_power', 'seats', 'car_age']
            input_data_processed[numerical_cols] = scaler.transform(input_data_processed[numerical_cols])

            # Gets predictions from both models.
            linear_prediction_inr = linear_model.predict(input_data_processed)[0]
            xgb_prediction_inr = xgb_model.predict(input_data_processed)[0]

            INR_TO_USD_RATE = get_exchange_rate(API_KEY)

            # Display prediction results.
            if INR_TO_USD_RATE:
                linear_prediction_usd = linear_prediction_inr * INR_TO_USD_RATE
                xgb_prediction_usd = xgb_prediction_inr * INR_TO_USD_RATE
                st.session_state.predictions = {
                    'linear_inr': linear_prediction_inr, 'xgb_inr': xgb_prediction_inr,
                    'linear_usd': linear_prediction_usd, 'xgb_usd': xgb_prediction_usd,
                }
                
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<p class="prediction-label">Linear Regression Predicted Price:</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{linear_prediction_inr:,.2f} ($ {linear_prediction_usd:,.2f})</p>', unsafe_allow_html=True)
                st.markdown('<p class="prediction-label">XGBoost Predicted Price:</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{xgb_prediction_inr:,.2f} ($ {xgb_prediction_usd:,.2f})</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Could not fetch a live exchange rate. Displaying prices in INR only.")
                st.session_state.predictions = {
                    'linear_inr': linear_prediction_inr, 'xgb_inr': xgb_prediction_inr,
                    'linear_usd': None, 'xgb_usd': None,
                }
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown('<p class="prediction-label">Linear Regression Predicted Price:</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{linear_prediction_inr:,.2f}</p>', unsafe_allow_html=True)
                st.markdown('<p class="prediction-label">XGBoost Predicted Price:</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{xgb_prediction_inr:,.2f}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.success("Disclaimer: These are predictions based on our machine learning models and may not reflect the exact market value.")

    # --- TAB 2: Detailed Breakdown ---
    # Compares the prediction to market averages.
    with tab2:
        if st.session_state.prediction_made:
            st.header("Detailed Price Breakdown 📊")
            st.write("Here is a more detailed look at your car's price prediction, comparing it to our dataset.")
            st.markdown("---")
            
            user_input = st.session_state.user_input
            predictions = st.session_state.predictions

            st.subheader("Your Car's Predicted Value")
            st.markdown(f"The best estimated price for your **{user_input['year']} {user_input['brand']}** is: ")
            
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            if predictions['xgb_usd'] is not None:
                st.markdown(f'<p class="prediction-value">₹{predictions["xgb_inr"]:,.2f} ($ {predictions["xgb_usd"]:,.2f})</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="prediction-value">₹{predictions["xgb_inr"]:,.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")

            st.subheader("Market Comparison")
            if sample_data is not None:
                total_avg_price = sample_data['selling_price'].mean()
                brand_data = sample_data[sample_data['brand'] == user_input['brand']]
                brand_avg_price = brand_data['selling_price'].mean() if not brand_data.empty else None

                col_a, col_b = st.columns(2)

                with col_a:
                    st.metric(
                        label="Average Car Price (All Brands)",
                        value=f"₹{total_avg_price:,.2f}"
                    )
                    st.write("This is the average selling price across our entire dataset of cars.")
                
                with col_b:
                    if brand_avg_price is not None:
                        st.metric(
                            label=f"Average {user_input['brand']} Price",
                            value=f"₹{brand_avg_price:,.2f}",
                            delta=f"₹{(predictions['xgb_inr'] - brand_avg_price):,.2f}",
                            delta_color="normal"
                        )
                        st.write(f"This is the average price for all {user_input['brand']} cars in our data.")
                        st.write("The `delta` shows the difference between your car's predicted price and the average for its brand.")
                    else:
                        st.info(f"No average price data available for {user_input['brand']}.")
            else:
                st.warning("Could not load or generate any sample data for market comparisons.")

        else:
            st.info("Please enter your car's details and click 'Predict Price' in the 'Price Prediction' tab to see a detailed breakdown.")

    # --- TAB 3: Summary ---
    # Shows a personalized summary, key features, and pros/cons.
    with tab3:
        if st.session_state.prediction_made:
            st.header("Your Personalized Prediction Summary 📝")
            st.markdown("---")
            
            user_input = st.session_state.user_input
            predictions = st.session_state.predictions

            st.markdown(
                f"For a **{user_input['year']} {user_input['brand']}** with **{user_input['km_driven']:,} km** "
                f"driven, our models have generated the following price estimates:"
            )

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            if predictions['xgb_usd'] is not None:
                st.markdown(f'<p class="prediction-label">Best Estimated Price (XGBoost):</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{predictions["xgb_inr"]:,.2f} ($ {predictions["xgb_usd"]:,.2f})</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="prediction-label">Best Estimated Price (XGBoost):</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="prediction-value">₹{predictions["xgb_inr"]:,.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.subheader("How Your Prediction Was Reached")
            st.write("Our advanced model (XGBoost) carefully analyzed your car's features to give you this price. Think of it like a car expert who knows which details matter most. Our model found the most important factors for your car's value were:")
            
            # Displays top influential features.
            feature_importances = xgb_model.feature_importances_
            feature_names = training_columns
            importance_df = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
            
            relevant_features = []
            for feature in importance_df.index:
                if feature in ['engine', 'max_power', 'car_age', 'km_driven', 'seats'] or f'brand_{user_input["brand"]}' == feature:
                    relevant_features.append(feature.replace('_', ' ').title())
                elif 'brand_' in feature and len(relevant_features) < 3:
                    relevant_features.append(f'Its specific **brand**')
            
            if relevant_features:
                st.info(f"🥇 **Top Influential Features:** {', '.join(relevant_features[:3])}")
            
            st.write(
                "It's important to remember that the model also considers less obvious factors "
                "and patterns in the data to make a more accurate prediction. Things like a car's "
                "service history or optional features can also affect its final value."
            )
            
            st.markdown("---")
            
            # Displays a list of brand-specific pros and cons.
            st.subheader(f"Pros and Cons of a {user_input['brand']} Car 🚗")
            
            # --- Car Brand Pros and Cons Data ---
            
            pros_cons_data = {
                
                'Maruti': {
                    'Pros': ["Excellent fuel efficiency", "Low maintenance costs", "Extensive service network", "High resale value"],
                    'Cons': ["Some models have basic interiors and lack premium feel", "Safety features may be limited in base models", "Long waiting periods for popular models"]
                },

                'Hyundai': {
                    'Pros': ["Modern design and premium features", "Good build quality", "Strong after-sales support"],
                    'Cons': ["Maintenance costs can be slightly higher than competitors", "Some models have lower fuel efficiency"]
                },

                'Mahindra': {
                    'Pros': ["Robust and rugged build quality", "Powerful engines", "High ground clearance"],
                    'Cons': ["Lower fuel efficiency in some models", "Ride quality can be a bit bumpy"]
                },

                'Tata': {
                    'Pros': ["Excellent build quality and safety ratings", "Spacious and comfortable interiors", "Value for money"],
                    'Cons': ["After-sales service experience can be inconsistent", "Some models have minor quality control issues"]
                },

                'Honda': {
                    'Pros': ["Refined and reliable engines", "Spacious and comfortable cabins", "High resale value"],
                    'Cons': ["Features and interior design can feel dated in some models", "More expensive than some competitors"]
                },

                'Force': {
                    'Pros': ["Exceptional off-road capability", "Spacious interior for large families", "Durable and reliable build"],
                    'Cons': ["Poor fuel economy", "Basic interior and features", "Limited service network"]
                },

                'Chevrolet': {
                    'Pros': ["Good safety features", "Comfortable ride", "Strong after-sales support"],
                    'Cons': ["Lower resale value compared to segment leaders", "Limited model lineup"]
                },

                'Toyota': {
                    'Pros': ["Legendary reliability and durability", "Low maintenance costs", "High resale value"],
                    'Cons': ["Some models are not feature-rich", "Can be more expensive to purchase"]
                },

                'Renault': {
                    'Pros': ["Stylish design", "Value-for-money pricing", "Good fuel efficiency"],
                    'Cons': ["Limited features in some variants", "After-sales support is not as widespread"]
                },

                'Volkswagen': {
                    'Pros': ["Solid build quality and premium feel", "Strong performance and handling", "Safety features"],
                    'Cons': ["Higher maintenance costs", "More expensive spare parts"]
                },

                'Skoda': {
                    'Pros': ["Premium interiors and design", "Powerful and efficient engines", "Great ride quality"],
                    'Cons': ["Expensive maintenance and spare parts", "Fewer service centers"]
                },

                'Nissan': {
                    'Pros': ["Reliable engines and comfortable ride", "Fuel efficiency", "Spacious interiors"],
                    'Cons': ["Interior quality is not as good as competitors", "Limited features in some models"]
                },

                'Audi': {
                    'Pros': ["Luxurious interiors and high-quality materials", "Advanced technology and features", "Strong performance"],
                    'Cons': ["Very high maintenance costs", "Expensive parts and repairs", "Rapid depreciation in value"]
                },

                'BMW': {
                    'Pros': ["Exceptional driving dynamics", "Luxurious and premium cabins", "Advanced technology"],
                    'Cons': ["High ownership costs", "Expensive to maintain and repair", "Specific model years have known issues"]
                },

                'Mercedes-Benz': {
                    'Pros': ["Ultimate luxury and comfort", "Superior safety and technology", "Strong brand image"],
                    'Cons': ["Extremely high purchase price and maintenance costs", "Depreciation can be significant"]
                },

                'Other': {
                    'Pros': ["Unique styling and features"],
                    'Cons': ["Can be difficult to find parts", "Limited resale value"]
                },

                'Datsun': {
                    'Pros': ["Affordable pricing", "Low running costs", "Compact and easy to drive in the city"],
                    'Cons': ["Basic interiors and features", "Sub-par safety ratings", "Limited resale value"]
                },

                'Mitsubishi': {
                    'Pros': ["Reliable engines and strong build", "Good off-road capabilities in some models", "Long-term durability"],
                    'Cons': ["Outdated technology and features", "Limited service network", "Lower fuel efficiency"]
                },

                'Fiat': {
                    'Pros': ["Stylish and unique design", "Fun to drive with good handling", "Good fuel efficiency"],
                    'Cons': ["Poor resale value", "Limited service centers and expensive parts", "Known for some electrical issues"]
                },

                'Jeep': {
                    'Pros': ["Excellent off-road performance", "Iconic design", "High build quality"],
                    'Cons': ["High maintenance costs", "Lower fuel economy", "Expensive spare parts"]
                },

                'Land Rover': {
                    'Pros': ["Premium luxury and comfort", "Exceptional off-road capability", "High-quality materials"],
                    'Cons': ["Very expensive to maintain and repair", "High purchase price", "Some models have reliability issues"]
                },

                'Jaguar': {
                    'Pros': ["Elegant and luxurious design", "High-performance engines", "Refined ride quality"],
                    'Cons': ["High maintenance and repair costs", "Poor reliability in some models", "Significant depreciation"]
                },

                'Volvo': {
                    'Pros': ["Top-tier safety features", "Comfortable and minimalist design", "Good fuel efficiency"],
                    'Cons': ["Higher initial cost", "Expensive parts and repairs", "Less sporty than some rivals"]
                },

                'Ambassador': {
                    'Pros': ["Iconic design", "Robust build", "Spacious interior"],
                    'Cons': ["Old technology", "Poor fuel efficiency", "Lack of modern safety and convenience features"]
                },

                'Isuzu': {
                    'Pros': ["Durable and reliable engines", "Strong towing capacity", "Good off-road performance"],
                    'Cons': ["Basic interior and features", "Limited service network", "Low resale value"]
                },
            }
            
            # --- End of Car Brand Pros and Cons Data ---

            brand_data = pros_cons_data.get(user_input['brand'], {'Pros': ['No specific data available for this brand.'], 'Cons': ['No specific data available for this brand.']})
            
            col_pros, col_cons = st.columns(2)
            with col_pros:
                st.markdown("##### 👍 Pros")
                st.markdown("")
                for pro in brand_data['Pros']:
                    st.write(f"- {pro}")
            with col_cons:
                st.markdown("##### 👎 Cons")
                st.markdown("")
                for con in brand_data['Cons']:
                    st.write(f"- {con}")

        else:
            st.info("Please enter your car's details and click 'Predict Price' in the 'Price Prediction' tab to see a personalized summary.")
            
        st.markdown("---")
        st.markdown("View the source code on GitHub: [Car Price Predictor](https://github.com/gbennett90/Car-Price-Prediction.git)")

# --- Application Entry Point ---
# The entry point to run the Streamlit app.
if __name__ == "__main__":
    main()
