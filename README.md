# Insulin Response Modeling System

A Python-based data science platform that analyzes and predicts blood glucose responses after meals. The system processes two distinct data tracks: tabular snapshots (Track A) and continuous time-series data (Track B), applying both statistical and machine learning models to understand glucose dynamics across different health populations.

## Overview

The system compares statistical and machine learning models using public datasets to understand glucose dynamics in healthy, pre-diabetic, and Type 2 diabetic individuals.

### Data Tracks

- **Track A**: Tabular snapshots from UCI Diabetes dataset (pre/post-meal glucose, insulin doses)
- **Track B**: Continuous time-series from CGMacros dataset (CGM readings, meal macronutrients, activity, heart rate, demographics)

### Models

**Statistical Models:**
- Linear Regression (Track A)
- ARIMA/SARIMAX (Track B)

**Machine Learning Models:**
- Random Forest, SVM, XGBoost, LightGBM (Track A)
- LSTM, Transformer (Track B)

## Setup Instructions

### Prerequisites

- Python 3.9 or higher
- pip package manager
- **A POSIX platform (Linux or macOS) for data loading.** The dataset loaders store
  sensitive health data and enforce `0o700` permissions on the data directory. Because
  those permission bits cannot be set on non-POSIX platforms, `DatasetLoader` fails
  closed and raises `RuntimeError` on Windows rather than writing health data to a
  directory whose access it could not restrict. Windows users should run the data
  loading steps under WSL. The setup steps below still apply to Windows for the rest
  of the project.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd insulin-response-modeling
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the project package (editable). The code lives in a `src/` layout, so
   `import insulin_response...` — including from the test suite — only resolves
   after this step:
```bash
pip install -e .
```

5. Verify installation:
```bash
python verify_dependencies.py
```

## Project Structure

```
src/                        # Container directory (src-layout; not a package)
  insulin_response/         # Core implementation package
    data_preprocessing.py   # Data loading and feature engineering
    statistical_models.py   # Baseline statistical models
    ml_models.py            # Machine learning models
    evaluate.py             # Model evaluation and metrics
tests/
  unit/               # Unit tests for specific scenarios
  property/           # Property-based tests for correctness
  integration/        # End-to-end pipeline tests
notebooks/            # Jupyter notebooks for EDA
  EDA.ipynb          # Exploratory data analysis
data/                # Dataset storage
outputs/             # Visualizations and results
```

## Usage Examples

### Data Processing

```python
from insulin_response.data_preprocessing import UCIDiabetesLoader, CGMacrosLoader

# Load Track A data
track_a_loader = UCIDiabetesLoader()
track_a_data = track_a_loader.load()

# Load Track B data
track_b_loader = CGMacrosLoader()
track_b_data = track_b_loader.load()

# Feature engineering (planned — not yet implemented)
# from insulin_response.data_preprocessing import FeatureEngineer
# engineer = FeatureEngineer()
# track_a_features = engineer.calculate_time_since_meal(track_a_data)
```

### Model Training

```python
from insulin_response.statistical_models import LinearRegressionModel
from insulin_response.ml_models import RandomForestModel

# Train statistical model
lr_model = LinearRegressionModel()
lr_model.fit(X_train, y_train)
predictions = lr_model.predict(X_test)

# Train ML model
rf_model = RandomForestModel(n_estimators=100)
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
```

### Model Evaluation

```python
from insulin_response.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.cross_validate_track_a(rf_model, X, y, k=5)
print(f"RMSE: {results['rmse']:.2f}")
print(f"MAE: {results['mae']:.2f}")
print(f"R²: {results['r2']:.3f}")
```

### Hyperparameter Optimization

```python
from insulin_response.evaluate import HyperparameterTuner

tuner = HyperparameterTuner()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None]
}
best_params = tuner.grid_search(rf_model, param_grid, X, y, cv=5)
print(f"Best parameters: {best_params}")
```

### Visualization

```python
from insulin_response.evaluate import Visualizer

viz = Visualizer()
viz.plot_metric_comparison(results, 'outputs/metrics_comparison.png')
viz.plot_predicted_vs_actual(y_test, predictions, 'Random Forest', 'outputs/rf_predictions.png')
viz.plot_residuals(y_test, predictions, 'Random Forest', 'outputs/rf_residuals.png')
```

## Testing

Run all tests:
```bash
pytest
```

Run specific test suites:
```bash
# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/

# Integration tests only
pytest tests/integration/
```

Run with coverage:
```bash
pytest --cov=insulin_response --cov-report=html
```

## Exploratory Data Analysis

Launch Jupyter notebook for exploratory analysis:
```bash
jupyter notebook notebooks/EDA.ipynb
```

## Data Validation Rules

- Glucose values must be in range [20, 600] mg/dL
- Timestamps must be monotonically increasing within participants
- CGMacros participants 24, 25, 37, 40 are excluded (dropouts)
- Required columns must be present in datasets

## Error Handling

The system implements comprehensive error handling:
- Missing/corrupted files: Descriptive errors with file paths
- Missing columns: ValueError listing all missing columns
- Out-of-range values: Warnings logged, data points flagged
- Insufficient data: ValueError with minimum requirements
- NaN/Inf predictions: RuntimeError with diagnostic information

## Contributing

When contributing to this project:
1. Follow the existing code structure and naming conventions
2. Write both unit tests and property-based tests for new features
3. Ensure all tests pass before submitting changes
4. Update documentation as needed

## License

[Add license information here]

## Contact

[Add contact information here]
