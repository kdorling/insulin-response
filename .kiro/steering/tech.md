# Technology Stack

## Language and Runtime
- Python 3.x
- Create the project virtual environment once with `python -m venv venv` (`py -m venv venv` on Windows), then always use it for Python commands and package installs.
  - POSIX: `./venv/bin/python`, `./venv/bin/pip` (and `./venv/bin/pytest`, `./venv/bin/jupyter`, etc.).
  - Windows: the equivalents under `venv\Scripts\` — `venv\Scripts\python.exe`, `venv\Scripts\pip.exe`, `venv\Scripts\pytest.exe`, `venv\Scripts\jupyter.exe`.
  - Prefer these over bare command names so the correct environment is always used.

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

## Common Commands

Paths below use the POSIX venv layout (`./venv/bin/<tool>`). On Windows, use the
equivalent under `venv\Scripts\` (e.g. `venv\Scripts\pytest.exe`,
`venv\Scripts\python.exe`, `venv\Scripts\jupyter.exe`).

### Setup
```bash
# Create the virtual environment once (before using ./venv/... below)
python -m venv venv                             # Windows: py -m venv venv

# Install dependencies
./venv/bin/pip install -r requirements.txt      # Windows: venv\Scripts\pip.exe install -r requirements.txt
```

### Testing
```bash
# Run all tests
./venv/bin/pytest

# Run unit tests only
./venv/bin/pytest tests/unit/

# Run property tests only
./venv/bin/pytest tests/property/

# Run with coverage
./venv/bin/pytest --cov=insulin_response
```

### Data Processing
```bash
./venv/bin/python src/insulin_response/data_preprocessing.py
```

### Model Evaluation
```bash
./venv/bin/python src/insulin_response/evaluate.py
```

### Exploratory Analysis
```bash
./venv/bin/jupyter notebook notebooks/EDA.ipynb
```

## Testing Philosophy
- Property-based tests verify universal correctness properties across randomized inputs (minimum 100 iterations)
- Unit tests validate specific examples, edge cases, and integration points
- Both approaches are complementary and necessary for comprehensive coverage
