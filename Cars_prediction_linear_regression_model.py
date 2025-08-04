# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import joblib
import re
import datetime
import xgboost as xgb

# Ignore warnings
warnings.filterwarnings('ignore')

# Set display options for pandas
pd.set_option('display.max_columns', None)
plt.style.use('default')

# -- Load data --
df = pd.read_csv("https://raw.githubusercontent.com/gbennett90/Car-Price-Prediction/refs/heads/main/Cars_dataset.csv")
print("\n--- First 5 rows of the dataset ---")
print(df.head())

# -- Data exploration --

# Summary of dataframe, type and non-null values
print("\n--- DataFrame Information ---")
df.info()

# Get descriptive statistics of the numerical columns
print("\n--- Descriptive Statistics ---")
print(df.describe())

# Check for missing values in the dataframe
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# Check for missing values percentage
print("\n--- Checking for Missing Values (Percentage) ---")
print(100 * df.isnull().sum() / len(df))


# -- Data cleaning and preprocessing --

# Drop rows with any missing values
print("\n--- Dropping rows with missing values ---")
# Get the initial number of rows
initial_rows = len(df)
df.dropna(inplace=True)
final_rows = len(df)
print(f"Dropped {initial_rows - final_rows} rows.")

# Verify that there are no more missing values
print("\n--- Re-checking for Missing Values (Count) after dropping ---")
print(df.isnull().sum())

# Clean and standardize column names
def clean_col_names(name):
    # Convert to lowercase, replace special characters and spaces with underscores
    clean_name = name.lower()
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_name)
    clean_name = clean_name.replace(' ', '_')
    return clean_name

df.columns = [clean_col_names(col) for col in df.columns]

print("\n--- DataFrame columns after cleaning ---")
print(df.columns)

print("\n--- DataFrame Information after cleaning ---")
df.info()

# --- Clean Numerical Features by removing units ---
print("\n--- Cleaning Numerical Features by removing units ---")
def clean_numerical_column(column_series):
    # Use a regular expression to extract the number
    cleaned_series = column_series.astype(str).str.extract('(\d+\.?\d*)').astype(float)
    return cleaned_series

df['mileagekmltrkg'] = clean_numerical_column(df['mileagekmltrkg'])
df['max_power'] = clean_numerical_column(df['max_power'])
df['engine'] = clean_numerical_column(df['engine'])
df['seats'] = clean_numerical_column(df['seats'])
print("Cleaned numerical columns by removing units.")

# --- Extract Car Brand from 'name' and handle low-frequency brands ---
print("\n--- Extracting Car Brand and handling low-frequency brands ---")
df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
brand_counts = df['brand'].value_counts()
other_brands = brand_counts[brand_counts < 10].index
df['brand'] = df['brand'].replace(other_brands, 'Other')
print("Extracted car 'brand' and grouped less common ones.")


# -- Exploratory data analysis and Visualization --

# Plot a correlation heatmap of numerical features to understand relationships
print("\n--- Generating Correlation Heatmap ---")
plt.figure(figsize=(10, 8))
# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=np.number)
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Visualize the distribution of the target variable 'selling_price'
print("\n--- Visualizing selling_price Distribution ---")
plt.figure(figsize=(10, 6))
sns.histplot(df['selling_price'], kde=True, bins=20)
plt.title('Distribution of Car selling_price')
plt.xlabel('selling_price')
plt.ylabel('Frequency')
plt.show()

# Create a pairplot to visualize relationships between all numerical features
print("\n--- Generating Pairplot ---")
# sns.pairplot(df) will show all numerical columns, but you can specify a subset if needed.
sns.pairplot(df[['year', 'engine', 'mileagekmltrkg', 'max_power', 'selling_price']])
plt.suptitle('Pairplot of Numerical Features', y=1.02)
plt.show()

# -- Feature engineering and categorical encoding --

print("\n--- Creating New Features ---")
# Calculate car age from the 'year' column
current_year = datetime.datetime.now().year
df['car_age'] = current_year - df['year']
print(f"Created 'car_age' feature. Current year used: {current_year}")

print("\n--- Performing One-Hot Encoding on Categorical Features ---")
# Identify the categorical columns, INCLUDING THE NEW 'brand' FEATURE
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
# Perform one-hot encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("Categorical features have been one-hot encoded.")

# Drop columns that are no longer needed for modeling
# We now drop the original 'name' and 'year' columns.
df.drop(['name', 'year'], axis=1, inplace=True)
print("Dropped 'name' and original 'year' columns.")

print("\n--- DataFrame after Feature Engineering and Encoding ---")
print(df.head())
print("\n--- Updated DataFrame Information ---")
df.info()

# -- Splitting the data and scaling --

print("\n--- Cleaning Numerical Data for Scaling ---")
# Identify numerical columns to clean
numerical_cols = ['km_driven', 'engine', 'max_power', 'seats', 'car_age']
for col in numerical_cols:
    # Convert column to numeric, coercing errors to NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill any remaining NaN values in numerical columns with the mean of that column
df.fillna(df.mean(numeric_only=True), inplace=True)
print("Numerical features cleaned and missing values imputed.")

print("\n--- Splitting Data into Training and Testing Sets ---")
# Define features (X) and target (y)
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Split the data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

print("\n--- Scaling Numerical Features ---")
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
print("Numerical features have been scaled.")

print("\n--- X_train after scaling ---")
print(X_train.head())

# -- Training and evaluation --

print("\n--- Training the Linear Regression Model ---")
# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print("Linear Regression model training complete.")

print("\n--- Making Predictions (Linear Regression) ---")
# Make predictions on the test set
y_pred = linear_model.predict(X_test)
print("Predictions on test data generated.")

print("\n--- Evaluating Linear Regression Model Performance ---")
# Calculate and print evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# -- Cross validation --

print("\n--- Performing Cross-Validation for XGBoost Model ---")
# Initialize the XGBoost Regressor
xgb_model_cv = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Set up K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and get R-squared scores
cv_scores = cross_val_score(xgb_model_cv, X, y, cv=kf, scoring='r2')

print(f"Cross-Validation R-squared scores: {cv_scores}")
print(f"Mean CV R-squared: {np.mean(cv_scores):.2f}")
print(f"Standard Deviation of CV R-squared: {np.std(cv_scores):.2f}")

# -- Model Training and Evaluation (XGBoost Regressor) --

print("\n--- Training the XGBoost Regressor Model ---")
# Initialize and train the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
print("XGBoost Regressor model training complete.")

print("\n--- Making Predictions with XGBoost Regressor ---")
# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)
print("Predictions with XGBoost Regressor generated.")

print("\n--- Evaluating XGBoost Regressor Model Performance ---")
# Calculate and print evaluation metrics for XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost MAE: {mae_xgb:.2f}")
print(f"XGBoost MSE: {mse_xgb:.2f}")
print(f"XGBoost RMSE: {rmse_xgb:.2f}")
print(f"XGBoost R-squared (R2): {r2_xgb:.2f}")

# -- Feature Importance --

print("\n--- Analyzing Feature Importance of the XGBoost Model ---")
# Get feature importances from the trained XGBoost model
feature_importances = xgb_model.feature_importances_
# Get the feature names from the training data
feature_names = X_train.columns
# Create a pandas Series for easy handling and plotting
importance_df = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)
# Create a bar plot to visualize feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=importance_df.values, y=importance_df.index, palette='viridis')
plt.title('XGBoost Model Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

print("Feature importance analysis complete and visualization generated.")

# -- Save the model, scaler, and column list

print("\n--- Saving the Trained Models, Scaler, and Column List ---")
# Save the trained models, the scaler, and the column list to files
joblib.dump(linear_model, 'linear_regression_model.pkl')
joblib.dump(xgb_model, 'xgboost_regressor_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
# --- THIS IS THE NEW CRUCIAL LINE ---
joblib.dump(X_train.columns, 'training_columns.pkl')
print("Models, scaler, and training column list saved to files.")

# -- Visualize model prediction --

print("\n--- Visualizing XGBoost Predictions vs. Actual Values ---")
# Create a scatter plot of actual prices vs. predicted prices for XGBoost
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.title('Actual vs. Predicted Selling Price (XGBoost)')
plt.grid(True)
plt.show()

# Visualize the distribution of the residuals (errors) for XGBoost
residuals = y_test - y_pred_xgb
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.title('Distribution of Residuals (XGBoost)')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

print("Model prediction visualizations generated.")
