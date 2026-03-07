# Project Structure and Organization

## Directory Layout

```
.kiro/
  specs/
    insulin-response-modeling/
      requirements.md    # Feature requirements and acceptance criteria
      design.md         # Architecture, components, and correctness properties
      tasks.md          # Implementation task breakdown
  steering/            # Project guidance documents (this directory)

plans/                 # Implementation planning documents

src/                   # Core implementation (to be created)
  data_preprocessing.py
  statistical_models.py
  ml_models.py
  evaluate.py

tests/                 # Test suite (to be created)
  unit/
  property/
  integration/

notebooks/             # Jupyter notebooks (to be created)
  EDA.ipynb

data/                  # Dataset storage (to be created)
outputs/               # Generated visualizations and results (to be created)
```

## Module Organization

### Data Pipeline (`src/data_preprocessing.py`)
- `DatasetLoader` — base class for dataset loading
- `UCIDiabetesLoader` — Track A data loader
- `CGMacrosLoader` — Track B data loader
- `FeatureEngineer` — feature transformation and validation

### Statistical Models (`src/statistical_models.py`)
- `BaselineModel` — abstract base class
- `LinearRegressionModel` — Track A baseline
- `ARIMAModel` — Track B baseline

### ML Models (`src/ml_models.py`)
- `MLModel` — abstract base class
- Track A: `RandomForestModel`, `SVMModel`, `XGBoostModel`, `LightGBMModel`
- Track B: `LSTMModel`, `TransformerModel`

### Evaluation (`src/evaluate.py`)
- `ModelEvaluator` — cross-validation and metrics computation
- `HyperparameterTuner` — hyperparameter optimization
- `Visualizer` — result visualization

## Naming Conventions

- Classes: PascalCase (e.g., `DatasetLoader`, `MLModel`)
- Functions/methods: snake_case (e.g., `calculate_time_since_meal`, `compute_metrics`)
- Constants: UPPER_SNAKE_CASE (e.g., `MIN_GLUCOSE`, `MAX_GLUCOSE`)
- Files: snake_case (e.g., `data_preprocessing.py`, `ml_models.py`)

## Data Validation Rules

- Glucose values must be in range [20, 600] mg/dL
- Timestamps must be monotonically increasing within participants
- CGMacros participants 24, 25, 37, 40 must be excluded (dropouts)
- Required columns must be present (raise ValueError with missing column names)

## Error Handling Patterns

- Missing/corrupted files: Raise descriptive errors with file paths
- Missing columns: Raise ValueError listing all missing columns
- Out-of-range values: Log warning, flag data point, continue processing
- Insufficient data: Raise ValueError with minimum requirements
- NaN/Inf predictions: Raise RuntimeError with diagnostic information

## Testing Organization

- Property tests tagged with: `# Feature: insulin-response-modeling, Property N: [property text]`
- Each property test validates specific requirements (documented in design.md)
- Unit tests focus on edge cases, error conditions, and integration points
- Integration tests validate end-to-end pipeline execution
