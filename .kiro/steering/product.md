# Product Overview

This is an insulin response modeling system that analyzes and predicts blood glucose responses after meals. The system compares statistical and machine learning models using public datasets to understand glucose dynamics across different health populations (healthy, pre-diabetic, and Type 2 diabetic individuals).

The system processes two data tracks:
- Track A: Tabular snapshots from UCI Diabetes dataset (pre/post-meal glucose, insulin doses)
- Track B: Continuous time-series from CGMacros dataset (CGM readings, meal macronutrients, activity, heart rate, demographics)

The goal is to predict post-meal glucose responses using both traditional statistical methods (Linear Regression, ARIMA) and modern ML approaches (Random Forest, SVM, XGBoost, LightGBM, LSTM, Transformer).
