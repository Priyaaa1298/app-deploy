#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import uniform, loguniform
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Streamlit title
st.title("Titanic Survival Prediction")

# File uploader for datasets
train_file = st.file_uploader("Upload Train Dataset", type=["csv"])
test_file = st.file_uploader("Upload Test Dataset", type=["csv"])

if train_file is not None and test_file is not None:
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Display first few rows of the train dataset
    st.subheader("Train Dataset")
    st.write(train_df.head())

    # Handle missing values in 'Age' and 'Fare'
    si = SimpleImputer(strategy='mean')
    train_df[['Age']] = si.fit_transform(train_df[['Age']])
    test_df[['Age']] = si.fit_transform(test_df[['Age']])
    test_df[['Fare']] = si.fit_transform(test_df[['Fare']])

    # Remove irrelevant columns
    irrelevant_columns = ['Pclass', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    train_cleaned = train_df.drop(columns=irrelevant_columns, axis=1)
    test_cleaned = test_df.drop(columns=irrelevant_columns, axis=1)

    # One-Hot Encoding
    train_cleaned = pd.get_dummies(train_cleaned, columns=['Sex'], drop_first=False)

    # Feature selection and target variable
    X = train_cleaned[['PassengerId', 'Age', 'SibSp', 'Parch', 'Sex_female', 'Sex_male']]
    y = train_cleaned['Survived']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test_scaled)
    st.subheader('Model Evaluation')
    st.write(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    st.write(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    st.write(f'Classification Report:\n{classification_report(y_test, y_pred)}')

    # Hyperparameter Tuning with RandomizedSearchCV
    logreg = LogisticRegression(C=0.1)  # Regularization
    param_dist = {
        'C': loguniform(1e-4, 1e4),
        'solver': ['liblinear', 'lbfgs']
    }
    random_search = RandomizedSearchCV(logreg, param_distributions=param_dist, n_iter=25, cv=3, verbose=1, random_state=42, n_jobs=1)
    random_search.fit(X_train_scaled, y_train)
    best_params = random_search.best_params_
    st.write(f"Best Hyperparameters: {best_params}")

    # Train the model with best hyperparameters
    best_logreg = LogisticRegression(C=best_params['C'], solver=best_params['solver'])
    best_logreg.fit(X_train_scaled, y_train)

    # Evaluate the best model
    y_pred = best_logreg.predict(X_test_scaled)
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # Handle class imbalance with oversampling and undersampling
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    rus = RandomUnderSampler(random_state=42)
    X_train_resampled2, y_train_resampled2 = rus.fit_resample(X_train, y_train)

    # Logistic Regression with Oversampled Data
    model_ros = LogisticRegression(max_iter=200)
    model_ros.fit(X_train_resampled, y_train_resampled)
    y_pred_ros = model_ros.predict(X_test)

    # Logistic Regression with Undersampled Data
    model_rus = LogisticRegression(max_iter=200)
    model_rus.fit(X_train_resampled2, y_train_resampled2)
    y_pred_rus = model_rus.predict(X_test)

    # Evaluate with Oversampling and Undersampling
    st.write("Random Oversampling Evaluation:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_ros)}")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred_ros)}")

    st.write("Random Undersampling Evaluation:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred_rus)}")
    st.write(f"Classification Report:\n{classification_report(y_test, y_pred_rus)}")
