# 🚗 Car Price Prediction System


A *machine learning project** built with Python that predicts the price of a car based on various features such as brand, model, year, fuel type, transmission, and mileage.  
This project uses data preprocessing, feature engineering, and regression models to deliver accurate price predictions.

---

## 🧠 Tech Stack

- **Python 3.9+**
- **Pandas**, **NumPy** – For data handling and preprocessing  
- **Matplotlib**, **Seaborn** – For data visualization  
- **Scikit-learn** – For machine learning models and evaluation  
- **Jupyter Notebook / VS Code** – For interactive development  

---

## 🔍 Features

- Predict car prices using machine learning regression models  
- Clean and preprocess raw dataset  
- Visualize feature relationships and correlations  
- Compare model performance (Linear Regression, Random Forest, etc.)  
- Export trained model using pickle for deployment  

---

## 📊 Dataset

The dataset contains car listings with attributes such as:

| Feature | Description |
|----------|--------------|
| Name | Model and brand of the car |
| Year | Manufacturing year |
| Selling_Price | Target variable (price to predict) |
| Present_Price | Current market price |
| Kms_Driven | Distance driven |
| Fuel_Type | Petrol, Diesel, or CNG |
| Transmission | Manual or Automatic |
| Owner | Number of previous owners |

*(You can use your own dataset or one from Kaggle such as “Car Data.csv”.)*

---

## ⚙️ How to Run

1. Clone the repository:

# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv('Car_Data.csv')
data = data.drop_duplicates().dropna()

# Select relevant features
X = data[['Year', 'Kms_Driven', 'Fuel_Type', 'Transmission', 'Owner']]
y = data['Selling_Price']

# Encoding categorical features and scaling numerical ones
ct = ColumnTransformer([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'), ['Fuel_Type', 'Transmission']),
    ('scaler', MinMaxScaler(), ['Year', 'Kms_Driven', 'Owner'])
], remainder='passthrough')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
pipe = Pipeline(steps=[('transformer', ct), ('model', LinearRegression())])
pipe.fit(X_train, y_train)

# Evaluate model
score = pipe.score(X_test, y_test)
print(f'R² Score: {score:.3f}')

# Example prediction
sample = pd.DataFrame({
    'Year': [2018],
    'Kms_Driven': [35000],
    'Fuel_Type': ['Petrol'],
    'Transmission': ['Manual'],
    'Owner': [1]
})
pred = pipe.predict(sample)
print(f'Predicted Price: ₹{pred[0]:.2f} Lakhs')

# Save trained model
joblib.dump(pipe, 'car_price_model.pkl')
print('Model saved successfully as car_price_model.pkl')
