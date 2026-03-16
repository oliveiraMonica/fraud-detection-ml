"""
Fraud Detection Model Training

This script trains a machine learning model to detect
fraudulent credit card transactions.

Steps:

1. Import libraries
2. Load dataset
3. Data preparation
4. Train/test split
5. Train model
6. Model evaluation
7. Save trained model
"""

# ----------------------------------------------------
# 1. Import libraries
# ----------------------------------------------------

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib


# ----------------------------------------------------
# 2. Load dataset
# ----------------------------------------------------

print("Loading dataset...")

df = pd.read_csv("/home/user/PycharmProjects/fraud-detection-ml/data/creditcard.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ----------------------------------------------------
# 3. Data preparation
# ----------------------------------------------------

print("Preparing features and target variable...")

X = df.drop("Class", axis=1)
y = df["Class"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ----------------------------------------------------
# 4. Train/test split
# ----------------------------------------------------

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ----------------------------------------------------
# 5. Train model
# ----------------------------------------------------

print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed")


# ----------------------------------------------------
# 6. Model evaluation
# ----------------------------------------------------

print("Evaluating model...")

y_pred = model.predict(X_test)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


# ----------------------------------------------------
# 7. Save trained model
# ----------------------------------------------------

print("Saving trained model...")

joblib.dump(model, "/home/user/PycharmProjects/fraud-detection-ml/models/fraud_model.pkl")

print("Model saved successfully")