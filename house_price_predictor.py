import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib # To save the model

# 1. Simulate Data (In a real project, you'd load from CSV/API)
np.random.seed(42)
num_houses = 100
areas = np.random.randint(800, 3000, num_houses) # Square feet
bedrooms = np.random.randint(2, 5, num_houses)
bathrooms = np.random.randint(1, 4, num_houses)
# Simulate price with some noise and dependency on area, bedrooms, bathrooms
prices = 50 * areas + 20000 * bedrooms + 15000 * bathrooms + np.random.normal(0, 50000, num_houses)
prices = np.maximum(100000, prices) # Ensure prices are not too low

data = pd.DataFrame({
    'Area_sqft': areas,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

print("--- Sample Data ---")
print(data.head())
print("\n--- Data Info ---")
data.info()
print("\n--- Data Description ---")
print(data.describe())

# 2. Define Features (X) and Target (y)
X = data[['Area_sqft', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# 3. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# 4. Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Training Complete ---")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# 7. Visualize Predictions vs. Actual (for a single feature, e.g., Area)
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Area_sqft'], y_test, color='blue', label='Actual Prices')
plt.scatter(X_test['Area_sqft'], y_pred, color='red', label='Predicted Prices', alpha=0.7)
plt.title('Actual vs. Predicted House Prices (by Area)')
plt.xlabel('Area (sqft)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 8. Save the trained model
model_filename = 'house_price_model.pkl'
joblib.dump(model, model_filename)
print(f"\nModel saved to {model_filename}")

# Example of loading and using the model for a new prediction
print("\n--- Predicting for a new house ---")
loaded_model = joblib.load(model_filename)
new_house_features = pd.DataFrame([[1800, 3, 2]], columns=['Area_sqft', 'Bedrooms', 'Bathrooms'])
predicted_price = loaded_model.predict(new_house_features)
print(f"Predicted price for a 1800 sqft, 3 bed, 2 bath house: ${predicted_price[0]:.2f}")