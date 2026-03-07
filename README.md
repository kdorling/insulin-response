# Insulin Response Modeling

A machine learning system for analyzing and predicting blood glucose responses after meals. This project compares statistical and ML models using public datasets to understand glucose dynamics across different health populations.

## Overview

The system processes two data tracks:
- **Track A**: Tabular snapshots from UCI Diabetes dataset (pre/post-meal glucose, insulin doses)
- **Track B**: Continuous time-series from CGMacros dataset (CGM readings, meal macronutrients, activity, heart rate, demographics)

Models include traditional statistical methods (Linear Regression, ARIMA) and modern ML approaches (Random Forest, SVM, XGBoost, LightGBM, LSTM, Transformer).

## Project Structure

```
src/                    # Core implementation modules
  data_preprocessing.py # Data loading and feature engineering
  statistical_models.py # Baseline statistical models
  ml_models.py         # Machine learning models
  evaluate.py          # Model evaluation and metrics
tests/
  unit/               # Unit tests for specific scenarios
  property/           # Property-based tests for correctness
  integration/        # End-to-end pipeline tests
notebooks/            # Jupyter notebooks for EDA
  EDA.ipynb          # Exploratory data analysis
data/                # Dataset storage
outputs/             # Visualizations and results
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
```bash
python src/data_preprocessing.py
```

### Model Evaluation
```bash
python src/evaluate.py
```

### Exploratory Analysis
```bash
jupyter notebook notebooks/EDA.ipynb
```

## Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property tests only
pytest tests/property/

# Run with coverage
pytest --cov=src
```

## Technology Stack

- **Language**: Python 3.x
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost, lightgbm, torch
- **Statistics**: statsmodels
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, hypothesis

## Data Sources

- UCI Diabetes Dataset (Track A)
- CGMacros Dataset (Track B)

Place datasets in the `data/` directory before running the pipeline.
