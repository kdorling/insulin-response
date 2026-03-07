# Insulin Response Modeling on Public Datasets

This project aims to analyze and predict how insulin/blood glucose levels respond after eating meals, using a combination of statistical and machine learning models on open public datasets.

## User Review Required

> [!IMPORTANT]
> **Dataset limitations**: Many advanced datasets like OhioT1DM or D1NAMO require formal access requests (Data Use Agreements). For full automation and immediate results, I propose using the **UCI Machine Learning Repository - Diabetes Dataset** or generating a realistic synthetic dataset based on postprandial glucose dynamics if the open datasets lack clean post-meal time-series data. 
> 
> **Does this sound good, or do you have a specific dataset downloaded already?** If not, I will start by using the UCI dataset or a suitable Kaggle alternative that doesn't require authentication, and we can swap it out later if needed.

## Proposed Changes

We will build a Python pipeline to process the data, train both statistical and machine learning models, and compare their performance.

### Project Setup and Data Pipeline

#### [NEW] [requirements.txt](file:///home/kevin/code/insulin-response/requirements.txt)
Will contain the necessary Python dependencies: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`.

#### [NEW] [src/data_preprocessing.py](file:///home/kevin/code/insulin-response/src/data_preprocessing.py)
Script to download/load the public dataset, clean the data, and extract features such as time-since-meal, carbohydrate estimates, and baseline glucose.

#### [NEW] [notebooks/EDA.ipynb](file:///home/kevin/code/insulin-response/notebooks/EDA.ipynb)
Jupyter notebook to perform Exploratory Data Analysis (EDA) to understand the statistical properties of the data before building models. This will include:
- Summary statistics for glucose levels pre- and post-meal.
- Visualizations of the glucose curves over time after different mealtimes (breakfast, lunch, dinner).
- Correlation analysis between meal information (if available, e.g., carbs) and peak glucose/insulin levels.
- Checking for missing data, outliers, and data distributions to inform preprocessing steps.

### Modeling

#### [NEW] [src/statistical_models.py](file:///home/kevin/code/insulin-response/src/statistical_models.py)
Will implement traditional statistical approaches, such as:
- Linear Regression (Baseline)
- Autoregressive models (ARIMA or SARIMAX) if the data is high-resolution time-series.

#### [NEW] [src/ml_models.py](file:///home/kevin/code/insulin-response/src/ml_models.py)
Will implement machine learning and deep learning models, such as:
- Random Forest Regressor & Support Vector Machines (SVM) (e.g., [Predicting postprandial hypoglycemia using machine learning](https://d-nb.info/1273950157/34))
- Gradient Boosting Regressor (e.g., XGBoost, LightGBM) (e.g., [Personalized Postprandial Glycemic Response Prediction](https://www.frontiersin.org/articles/10.3389/fnut.2023.1118173/full))
- Recurrent Neural Networks (LSTM) for time-series modeling (e.g., [Deep Learning for Blood Glucose Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8838327/))
- Transformer-based models for predicting glucose peaks and nadirs (e.g., [Transformer-based deep learning models for glucose prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8838327/))

### Evaluation and Comparison

#### [NEW] [src/evaluate.py](file:///home/kevin/code/insulin-response/src/evaluate.py)
A script that runs all models using cross-validation (or a time-series split), calculates metrics (Root Mean Squared Error, Mean Absolute Error), and prints out a comparative report.

## Verification Plan

### Automated Tests
1. Run `python src/data_preprocessing.py` to ensure the dataset downloads and cleans successfully without errors.
2. Run `python src/evaluate.py` to ensure all models train successfully and produce evaluation metrics.

### Manual Verification
1. I will present the final output metrics (RMSE, MAE) and a summary of which model performed best in responding to the post-meal glucose spikes.
