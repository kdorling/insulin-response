# Technology Stack

## Language and Runtime
- Python 3.x

## Core Dependencies
- pandas, numpy — data manipulation
- scikit-learn — ML models and evaluation
- statsmodels — statistical models (ARIMA/SARIMAX)
- xgboost, lightgbm — gradient boosting models
- torch (PyTorch) — deep learning (LSTM, Transformer)
- matplotlib, seaborn — visualization
- jupyter — exploratory analysis notebooks
- hypothesis — property-based testing
- pytest — unit and integration testing

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

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Testing
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

## Testing Philosophy
- Property-based tests verify universal correctness properties across randomized inputs (minimum 100 iterations)
- Unit tests validate specific examples, edge cases, and integration points
- Both approaches are complementary and necessary for comprehensive coverage
