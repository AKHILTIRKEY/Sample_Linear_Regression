🏠 Linear Regression — House Price Prediction

This project demonstrates a Multiple Linear Regression model to predict house prices based on features such as:

🏡 Square footage

🛏️ Number of bedrooms

🕒 Age of the house

The dataset is synthetically generated, and the model is implemented using Scikit-Learn.

🚀 Project Overview

The goal of this project is to show a clear, beginner-friendly example of:

Data generation and preprocessing

Splitting data into train & test sets

Training a Linear Regression model

Evaluating performance using:

R² Score

RMSE (Root Mean Squared Error)

Interpreting learned coefficients

This project is useful for learners who want to understand the end-to-end workflow of regression modeling in Python.

📂 Technologies Used

Python

NumPy

Pandas

Scikit-Learn

Matplotlib

🧠 Model Features
Feature	Description
square_feet	Total area of the house
bedrooms	Number of bedrooms
age	Age of the property
price	Target variable

The price is generated using a formula with added noise to simulate real-world behavior.

📌 Code
# Linear Regression Example - House Price Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
square_feet = np.random.uniform(500, 3500, 200)
bedrooms = np.random.randint(1, 6, 200)
age = np.random.uniform(0, 50, 200)

# Price formula with some noise
price = 50000 + 150*square_feet + 20000*bedrooms - 1000*age + np.random.normal(0, 30000, 200)

# Create DataFrame
df = pd.DataFrame({
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'age': age,
    'price': price
})

# Features and target
X = df[['square_feet', 'bedrooms', 'age']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f\"\nCoefficients:\")
for feature, coef in zip(X.columns, model.coef_):
    print(f\"  {feature}: {coef:.2f}\")
print(f\"Intercept: {model.intercept_:.2f}\")

📊 Model Evaluation Output (Example)

R² Score: Measures variance explained by the model

RMSE: Shows prediction error in price units

Coefficients: Show how each feature affects price

Intercept: Baseline price value

▶️ How to Run
pip install numpy pandas scikit-learn matplotlib
python house_price_regression.py

💡 Learning Outcomes

After completing this project, you will understand:

How linear regression works

How to evaluate regression models

How feature coefficients influence predictions

How to work with synthetic datasets

📜 License

This project is free to use for learning and educational purposes.

⭐ Contribution / Feedback

Feel free to:

Fork this project

Add visualizations

Experiment with new features

Submit improvements

If you like this project, give it a ⭐ on GitHub 🙂

If you want, I can also:
👉 add plots to the README
👉 convert this into a Jupyter Notebook version
👉 or help you write a project description for your resume.
