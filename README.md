
# Car Price Predictor

This project is a web application built using Python and Streamlit that predicts the selling price of a used car based on various features. The prediction is powered by a machine learning model, specifically an XGBoost Regressor, trained on a comprehensive public dataset of used car listings.

The application provides a user-friendly interface to input car details, get a price prediction, and understand the key factors influencing the predicted value.

Features:
Interactive Interface: A clean and intuitive Streamlit interface for seamless user interaction.

Price Prediction: Predicts car prices using a trained machine learning model.

Professional Styling: Uses custom CSS for a polished and professional look and feel.

Technologies Used
Python: The core programming language for the entire project.

Streamlit: For building the interactive web application.

Scikit-learn: For data preprocessing and model handling.

XGBoost: The machine learning algorithm used for the regression model.

Pandas & NumPy: For data manipulation and numerical operations.

Joblib: To save and load the trained model and scaler.

Matplotlib & Seaborn: For creating the visualizations within the app.

How to Run the App Locally
To run this application on your local machine, follow these steps:

Clone the repository from GitHub.

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install the dependencies listed in requirements.txt.

pip install -r requirements.txt

Ensure you have the necessary model files in the same directory:

linear_regression_model.pkl

xgboost_regressor_model.pkl

scaler.pkl

Run the Streamlit app from your terminal.

streamlit run app.py

This will open the application in your default web browser.

How to Use the App
Open the application in your browser.

Use the dropdown menus to select the details of the car you want to get a price prediction for, including:

Year
Kilometers Driven
Engine (CC)
Max Power (bhp)
Number of Seats
Fuel Type
Seller Type
Transmission Type
Ownership

Click the "Predict Price" button.

The predicted prices from both the Linear Regression and XGBoost models will be displayed.

Deployment
This application is deployed on Streamlit Cloud and can be accessed at the following URL:

[https://car-price-prediction-xxkh63jersoo4hnugyuwow.streamlit.app/]

Dataset
The model was trained on a publicly available dataset. You can find the dataset used for this project at the following link:

[Kaggle Car Price Prediction Dataset](https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset/data)

Note: This project is a demonstration and should be used for educational purposes only. The predictions are based on a simplified model and may not reflect the actual market value.
=======
# Car-Price-Prediction
Data mining course project
>>>>>>> 83472f1ce334d6e2c31e12d0fcb66a73d8ffeb23
