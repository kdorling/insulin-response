# Requirements Document

## Introduction

This document specifies the requirements for an insulin response modeling system that analyzes and predicts blood glucose responses after meals. The system will compare statistical and machine learning models using public datasets to understand glucose dynamics in different health populations (healthy, pre-diabetic, and Type 2 diabetic individuals).

## Glossary

- **System**: The insulin response modeling system
- **Track_A**: Tabular dataset analysis track using UCI Diabetes dataset
- **Track_B**: Time-series dataset analysis track using CGMacros dataset
- **Data_Pipeline**: Component responsible for downloading, parsing, and preprocessing datasets
- **Statistical_Model**: Linear regression or ARIMA/SARIMAX baseline models
- **ML_Model**: Machine learning models including Random Forest, SVM, XGBoost, LightGBM, LSTM, and Transformer
- **Evaluator**: Component responsible for model evaluation and performance metrics
- **CGM**: Continuous Glucose Monitor - device that measures glucose at regular intervals
- **iAUC**: Incremental Area Under the Curve - measure of glucose response
- **Pre_Meal_Glucose**: Blood glucose measurement taken before a meal
- **Post_Meal_Glucose**: Blood glucose measurement taken after a meal
- **Macronutrient**: Carbohydrates, fats, or proteins in food
- **Health_Group**: Classification of participants (healthy, pre-diabetic, Type 2 diabetic)

## Requirements

### Requirement 1: Data Acquisition and Parsing

**User Story:** As a data scientist, I want to acquire and parse multiple diabetes datasets, so that I can analyze glucose response patterns across different data formats.

#### Acceptance Criteria

1. WHEN the Data_Pipeline is initialized for Track_A data, THE System SHALL download the UCI Diabetes dataset from the UCI Machine Learning Repository
2. WHEN the Data_Pipeline processes Track_A data, THE System SHALL extract pre-meal glucose, post-meal glucose, insulin doses, and meal event timestamps
3. WHEN the Data_Pipeline is initialized for Track_B, THE System SHALL clone or download the CGMacros dataset repository
4. WHEN the Data_Pipeline processes Track_B data, THE System SHALL parse per-participant CSV files containing CGM readings, meal macronutrients, activity data, heart rate, demographics, blood analysis, and gut microbiome data
5. IF participants 24, 25, 37, or 40 are encountered, THEN THE System SHALL exclude their data from processing
6. WHEN parsing CGMacros data, THE System SHALL categorize participants into health groups (15 healthy, 16 pre-diabetic, 14 Type 2 diabetic)
7. IF the CGMacros dataset is unavailable, THEN THE System SHALL log an error naming the expected path and required layout, and SHALL raise `FileNotFoundError` without substituting synthetic data — an absent dataset is an unrecoverable precondition failure, not a degraded mode

### Requirement 2: Feature Engineering

**User Story:** As a data scientist, I want to engineer relevant features from raw data, so that I can improve model prediction accuracy.

#### Acceptance Criteria

1. WHEN processing meal events, THE System SHALL calculate time-since-meal for each glucose measurement
2. WHEN processing meal data, THE System SHALL extract macronutrient breakdown (carbohydrates, fats, proteins) for each meal
3. WHEN processing glucose measurements, THE System SHALL identify baseline glucose levels for each participant
4. WHEN processing Track_A data, THE System SHALL extract insulin dose information for each meal event
5. WHEN processing participant data, THE System SHALL encode health group classification as a categorical feature
6. WHEN feature engineering is complete, THE System SHALL validate that all required features are present and within expected ranges

### Requirement 3: Exploratory Data Analysis

**User Story:** As a data scientist, I want to perform exploratory data analysis, so that I can understand data distributions and relationships before modeling.

#### Acceptance Criteria

1. WHEN EDA is performed, THE System SHALL generate summary statistics grouped by health group
2. WHEN visualizing glucose distributions, THE System SHALL create distribution plots and box plots segmented by meal type and macronutrient composition
3. WHEN analyzing feature relationships, THE System SHALL generate a correlation heatmap for all numeric features
4. WHEN auditing data quality, THE System SHALL identify and report missing data percentages for each feature
5. WHEN detecting outliers, THE System SHALL flag glucose measurements outside physiologically plausible ranges (20-600 mg/dL) by setting a boolean indicator column, retaining the original value rather than dropping or altering the row
6. WHEN visualizing Track_B data, THE System SHALL create overlay plots of CGM glucose curves for multiple participants
7. WHEN detecting outliers, IF a glucose value is missing (NaN), THEN THE System SHALL NOT flag it as an outlier; missing values are reported by the missing-data audit in 3.4 instead

### Requirement 4: Statistical Model Implementation

**User Story:** As a data scientist, I want to implement baseline statistical models, so that I can establish performance benchmarks for comparison with ML models.

#### Acceptance Criteria

1. WHEN training Track_A models, THE Statistical_Model SHALL implement Linear Regression to predict post-meal glucose rise from pre-meal snapshots
2. WHEN training Track_B models, THE Statistical_Model SHALL implement ARIMA or SARIMAX to predict glucose time-series
3. WHEN fitting statistical models, THE System SHALL use appropriate train-test splits to prevent data leakage
4. WHEN statistical models are trained, THE System SHALL store model parameters and coefficients for interpretation

### Requirement 5: Machine Learning Model Implementation

**User Story:** As a data scientist, I want to implement multiple ML models, so that I can compare different approaches for glucose prediction.

#### Acceptance Criteria

1. WHEN training Track_A models, THE ML_Model SHALL implement Random Forest for post-meal glucose prediction
2. WHEN training Track_A models, THE ML_Model SHALL implement Support Vector Machine (SVM) for post-meal glucose prediction
3. WHEN training Track_A models, THE ML_Model SHALL implement XGBoost for post-meal glucose prediction
4. WHEN training Track_A models, THE ML_Model SHALL implement LightGBM for post-meal glucose prediction
5. WHEN training Track_B models, THE ML_Model SHALL implement LSTM networks for glucose time-series prediction
6. WHEN training Track_B models, THE ML_Model SHALL implement Transformer models for glucose time-series prediction
7. WHEN training any ML model, THE System SHALL use appropriate data splits to prevent temporal leakage in time-series data

### Requirement 6: Model Evaluation and Validation

**User Story:** As a data scientist, I want to evaluate models using rigorous validation techniques, so that I can assess their real-world performance accurately.

#### Acceptance Criteria

1. WHEN evaluating Track_A models, THE Evaluator SHALL use k-fold cross-validation with k >= 5
2. WHEN evaluating Track_B models, THE Evaluator SHALL use time-series split validation to respect temporal ordering
3. WHEN computing performance metrics, THE Evaluator SHALL calculate RMSE (Root Mean Squared Error) for all models
4. WHEN computing performance metrics, THE Evaluator SHALL calculate MAE (Mean Absolute Error) for all models
5. WHEN computing performance metrics, THE Evaluator SHALL calculate R² (coefficient of determination) for all models
6. WHEN computing performance metrics, THE Evaluator SHALL calculate MAPE (Mean Absolute Percentage Error) for all models
7. WHEN evaluating models, THE System SHALL ensure each metric is computed on held-out test data not used during training

### Requirement 7: Hyperparameter Optimization

**User Story:** As a data scientist, I want to optimize model hyperparameters systematically, so that I can achieve the best possible model performance.

#### Acceptance Criteria

1. WHEN optimizing ML models, THE System SHALL implement GridSearchCV or RandomizedSearchCV for hyperparameter tuning
2. WHEN performing hyperparameter search, THE System SHALL use cross-validation within the search process
3. WHEN hyperparameter optimization completes, THE System SHALL store the best parameters for each model
4. WHEN hyperparameter optimization completes, THE System SHALL log the performance improvement over default parameters

### Requirement 8: Results Visualization

**User Story:** As a data scientist, I want to visualize model results comprehensively, so that I can communicate findings and identify model strengths and weaknesses.

#### Acceptance Criteria

1. WHEN generating evaluation visualizations, THE System SHALL create bar charts comparing metrics across all models
2. WHEN generating prediction visualizations, THE System SHALL create predicted-vs-actual scatter plots for each model
3. WHEN generating diagnostic visualizations, THE System SHALL create residual plots to identify systematic prediction errors
4. WHEN creating visualizations, THE System SHALL use consistent color schemes and labeling for clarity
5. WHEN saving visualizations, THE System SHALL store them in a designated output directory with descriptive filenames

### Requirement 9: Project Structure and Dependencies

**User Story:** As a developer, I want a well-organized project structure with clear dependencies, so that I can set up and maintain the system easily.

#### Acceptance Criteria

1. THE System SHALL organize code as an installable `insulin_response` package under `src/`, containing the modules: data_preprocessing.py, statistical_models.py, ml_models.py, and evaluate.py
2. THE System SHALL include a requirements file specifying all dependencies: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, xgboost, lightgbm, torch, jupyter, pytest, hypothesis
3. THE System SHALL include a notebooks directory containing EDA.ipynb for exploratory analysis
4. THE System SHALL include a README file documenting setup instructions and usage examples
5. WHEN dependencies are installed, THE System SHALL verify that all required packages are available and compatible

### Requirement 10: Data Validation and Error Handling

**User Story:** As a developer, I want robust data validation and error handling, so that the system fails gracefully with informative messages.

#### Acceptance Criteria

1. WHEN loading datasets, IF a file is missing or corrupted, THEN THE System SHALL raise a descriptive error message
2. WHEN parsing data, IF required columns are missing, THEN THE System SHALL raise a descriptive error message listing missing columns
3. WHEN processing glucose measurements, IF values are outside physiological ranges (20-600 mg/dL), THEN THE System SHALL log a warning and flag the data point
4. WHEN training models, IF insufficient data is available, THEN THE System SHALL raise an error indicating minimum data requirements
5. WHEN evaluating models, IF predictions contain NaN or infinite values, THEN THE System SHALL raise an error with diagnostic information
