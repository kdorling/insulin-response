# Implementation Plan: Insulin Response Modeling System

## Overview

This implementation plan breaks down the insulin response modeling system into discrete coding tasks. The system will be built incrementally, starting with data infrastructure, then models, and finally evaluation components. Each task builds on previous work, with checkpoints to validate progress.

## Tasks

- [ ] 1. Set up project structure and dependencies (in progress)
  - [x] 1.1 Create project scaffolding and dependencies
    - Create directory structure: `src/insulin_response/`, `tests/unit/`, `tests/property/`, `tests/integration/`, `notebooks/`, `data/`, `outputs/`
    - Create `requirements.txt` with dependencies: pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, xgboost, lightgbm, torch, jupyter, hypothesis, pytest
    - Create `README.md` with setup instructions and usage examples
    - Create dependency verification script to check all packages are importable
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_
  
  - [ ] 1.2 Write property test for dependency verification
    - **Property 18: Dependency Verification**
    - **Validates: Requirements 9.5**
    - Note: Property 18 was defined in the design but referenced by no task

- [ ] 2. Implement data loading infrastructure (in progress)
  - [x] 2.1 Create base DatasetLoader class with abstract methods
    - Define abstract `download()`, `validate()`, and `load()` contracts that
      concrete loaders must implement (the base class provides no concrete
      download/validate/load — subclasses supply them)
    - Provide shared behavior on the base class (glucose validation/cleaning,
      secure data-directory setup) and error handling for missing/corrupted files
    - _Requirements: 1.1, 10.1_
  
  - [x] 2.2 Implement UCIDiabetesLoader for Track A data
    - Parse CSV to extract pre_meal_glucose, post_meal_glucose, insulin_dose, meal_timestamp
    - Validate required columns are present
    - Note: `download()` raises `NotImplementedError`; users must provide a pre-processed CSV
    - _Requirements: 1.1, 1.2, 10.2_
  
  - [x] 2.3 Write property test for Track A field extraction
    - **Property 1: Complete Field Extraction**
    - **Validates: Requirements 1.2**
  
  - [x] 2.4 Implement CGMacrosLoader for Track B data
    - Parse per-participant CSV files
    - Extract CGM readings, meal macronutrients, activity, heart rate, demographics
    - Exclude participants 24, 25, 37, 40
    - Categorize participants into health groups
    - Add fallback error handling if dataset unavailable
    - Note: `download()` raises `NotImplementedError`; users must provide dataset files manually
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7_
  
  - [x] 2.5 Write property test for Track B field extraction
    - **Property 1: Complete Field Extraction**
    - **Validates: Requirements 1.4**
  
  - [x] 2.6 Write property test for health group categorization
    - **Property 2: Health Group Categorization**
    - **Validates: Requirements 1.6**
  
  - [x] 2.7 Write unit tests for data loading edge cases
    - Test missing file error handling
    - Test corrupted file error handling
    - Test empty and header-only file rejection
    - _Requirements: 10.1_
  
  - [x] 2.8 Write property test for dropout participant exclusion
    - **Property 20: Dropout Participant Exclusion**
    - **Validates: Requirements 1.5**
    - Note: supersedes the former unit-test bullet in 2.7. Exclusion is a universal
      invariant over any mix of present participant files, so it belongs in a
      property rather than a single example (see design "Testing Balance")

  - [ ] 2.9 Implement CGMacros ingest: PhysioNet layout -> normalized schema
    - Walk the published dataset layout (`CGMacros-0XX/CGMacros-0YY.csv` plus the
      cohort `bio.csv`) rather than requiring top-level `participant_*.csv`
    - Map raw PhysioNet columns onto `CGMacrosLoader.required_columns`
      (e.g. CGM reading -> `glucose`, `HR` -> `heart_rate`, macronutrient columns
      -> `carbs`/`fat`/`protein`), including unit and timestamp normalization
    - Derive `participant_id` from the directory/file naming rather than the caller
    - Keep the normalization in the ingest step, not in `_parse_participant()`:
      per design.md the loaders consume an already-normalized on-disk schema, and
      that boundary is what task 2.4 was built against
    - Blocked on: access to PhysioNet CGMacros v1.0.0, so the column mapping can be
      verified against the real files instead of written from the published docs
    - Rationale: until this exists, Track B is only runnable against hand-normalized
      files, and `_discover_participant_ids()` returns an empty mapping for a
      straight unpack of the published dataset (raised in PR #1 review)
    - _Requirements: 1.3, 1.4_

- [ ] 3. Implement feature engineering
  - [ ] 3.1 Create FeatureEngineer class
    - Implement `calculate_time_since_meal()` to compute time differences
    - Implement `extract_macronutrients()` to parse carbs, fat, protein
    - Implement `compute_baseline_glucose()` to calculate participant baselines
    - Implement `encode_health_group()` for categorical encoding
    - Implement `validate_features()` to check required fields and ranges
    - Extract insulin dose per meal event for Track A records
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  
  - [ ] 3.2 Write property test for time-since-meal calculation
    - **Property 3: Time-Since-Meal Calculation**
    - **Validates: Requirements 2.1**
  
  - [ ] 3.3 Write property test for baseline glucose computation
    - **Property 4: Baseline Glucose Computation**
    - **Validates: Requirements 2.3**
  
  - [ ] 3.4 Write property test for health group encoding
    - **Property 5: Health Group Encoding**
    - **Validates: Requirements 2.5**
  
  - [ ] 3.5 Write property test for feature validation
    - **Property 6: Feature Validation**
    - **Validates: Requirements 2.6**

- [ ] 4. Implement data validation and outlier detection
  - [ ] 4.1 Create data validation functions
    - Implement outlier flagging: add a boolean indicator column, retaining the
      original value (do not drop or alter rows)
    - Treat missing (NaN) glucose as not-an-outlier
    - Implement missing data percentage calculation
    - Note: glucose range validation (20-600 mg/dL) and out-of-range warning
      logging already shipped with task 2 in `_validate_and_clean_glucose()`.
      What remains here is the *flag* column and the missing-data audit — do not
      reimplement the range check
    - _Requirements: 3.4, 3.5, 3.7, 10.3_
  
  - [ ] 4.2 Write property test for missing data calculation
    - **Property 7: Missing Data Calculation**
    - **Validates: Requirements 3.4**
  
  - [ ] 4.3 Write property test for outlier detection
    - **Property 8: Outlier Detection**
    - **Validates: Requirements 3.5, 3.7, 10.3**

- [ ] 5. Checkpoint - Ensure data pipeline works end-to-end
  - Run data loading and feature engineering on sample data
  - Verify all required features are present
  - Ensure all tests pass, ask the user if questions arise

- [ ] 6. Implement exploratory data analysis
  - [ ] 6.1 Implement `Visualizer.plot_glucose_curves()` for CGM overlay plots
    - Implements Requirement 3.6. Owned here, not in task 13.1 — it is an EDA
      plot, and implementing it in the module (rather than inline in the
      notebook) keeps it unit-testable and resolves the former duplicate
      ownership between tasks 6.1 and 13.1
    - _Requirements: 3.6_
  
  - [ ] 6.2 Create EDA.ipynb notebook
    - Generate summary statistics grouped by health group
    - Create distribution plots and box plots by meal type and macronutrients
    - Generate correlation heatmap for numeric features
    - Create missing data audit report
    - Call `Visualizer.plot_glucose_curves()` for CGM overlays (from 6.1)
    - Keep the notebook a thin caller over module-level functions so EDA logic
      stays testable
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 7. Implement statistical baseline models
  - [ ] 7.1 Create BaselineModel abstract class
    - Define interface: `fit()`, `predict()`, `get_params()`
    - _Requirements: 4.1, 4.2_
  
  - [ ] 7.2 Implement LinearRegressionModel for Track A
    - Implement fit method for training
    - Implement predict method for post-meal glucose rise prediction
    - Store model coefficients and parameters
    - _Requirements: 4.1, 4.4_
  
  - [ ] 7.3 Implement ARIMAModel for Track B
    - Implement fit method with configurable order parameters
    - Implement predict method for time-series forecasting
    - Store model parameters
    - _Requirements: 4.2, 4.4_
  
  - [ ] 7.4 Write property test for model parameter storage
    - **Property 10: Model Parameter Storage**
    - **Validates: Requirements 4.4**
  
  - [ ] 7.5 Write property test for statistical model training
    - **Property 11: Model Training and Prediction**
    - **Validates: Requirements 4.1, 4.2**

- [ ] 8. Implement machine learning models
  - [ ] 8.1 Create MLModel abstract class
    - Define interface: `fit()`, `predict()`, `get_hyperparams()`, `set_hyperparams()`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [ ] 8.2 Implement Track A models (Random Forest, SVM, XGBoost, LightGBM)
    - Create RandomForestModel class with configurable parameters
    - Create SVMModel class with kernel and C parameters
    - Create XGBoostModel class with n_estimators and learning_rate
    - Create LightGBMModel class with n_estimators and learning_rate
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [ ] 8.3 Implement Track B deep learning models (LSTM, Transformer)
    - Create LSTMModel class with PyTorch implementation
    - Create TransformerModel class with PyTorch implementation
    - Implement proper sequence handling and batching
    - _Requirements: 5.5, 5.6_
  
  - [ ] 8.4 Write property test for ML model training
    - **Property 11: Model Training and Prediction**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

- [ ] 9. Implement train-test splitting with leakage prevention
  - [ ] 9.1 Create data splitting utilities
    - Implement k-fold cross-validation for Track A
    - Implement time-series split for Track B with temporal ordering
    - Add validation to ensure no train-test overlap
    - _Requirements: 4.3, 5.7, 6.1, 6.2, 6.7_
  
  - [ ] 9.2 Write property test for train-test separation
    - **Property 9: Train-Test Separation**
    - **Validates: Requirements 4.3, 5.7, 6.7**
  
  - [ ] 9.3 Write property test for temporal ordering
    - **Property 12: Temporal Ordering in Time-Series Splits**
    - **Validates: Requirements 6.2**
  
  - [ ] 9.4 Write property test for cross-validation fold count
    - **Property 13: Cross-Validation Fold Count**
    - **Validates: Requirements 6.1**

- [ ] 10. Checkpoint - Ensure all models train successfully
  - Train each model on sample data
  - Verify predictions are generated
  - Ensure all tests pass, ask the user if questions arise

- [ ] 11. Implement model evaluation and metrics
  - [ ] 11.1 Create ModelEvaluator class
    - Implement `cross_validate_track_a()` with k-fold CV
    - Implement `cross_validate_track_b()` with time-series split
    - Implement `compute_metrics()` for RMSE, MAE, R², MAPE
    - Ensure metrics computed only on held-out test data
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_
  
  - [ ] 11.2 Write property test for metric calculation correctness
    - **Property 14: Metric Calculation Correctness**
    - **Validates: Requirements 6.3, 6.4, 6.5, 6.6**
  
  - [ ] 11.3 Write unit tests for evaluation edge cases
    - Test evaluation with minimal data
    - Test handling of NaN/Inf predictions
    - _Requirements: 10.4, 10.5_

- [ ] 12. Implement hyperparameter optimization
  - [ ] 12.1 Create HyperparameterTuner class
    - Implement `grid_search()` using GridSearchCV
    - Implement `random_search()` using RandomizedSearchCV
    - Ensure cross-validation is used within search
    - Store best parameters and log performance improvement
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  
  - [ ] 12.2 Write property test for hyperparameter search execution
    - **Property 15: Hyperparameter Search Execution**
    - **Validates: Requirements 7.1, 7.3, 7.4**
  
  - [ ] 12.3 Write property test for CV in hyperparameter search
    - **Property 16: Cross-Validation in Hyperparameter Search**
    - **Validates: Requirements 7.2**

- [ ] 13. Implement visualization components
  - [ ] 13.1 Create Visualizer class
    - Implement `plot_metric_comparison()` for bar charts
    - Implement `plot_predicted_vs_actual()` for scatter plots
    - Implement `plot_residuals()` for residual plots
    - Note: `plot_glucose_curves()` is implemented in task 6.1, not here
    - Ensure consistent color schemes and labeling
    - Save all visualizations to designated output directory with descriptive filenames
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ] 13.2 Write property test for visualization file storage
    - **Property 17: Visualization File Storage**
    - **Validates: Requirements 8.5**
  
  - [ ] 13.3 Write unit tests for visualization creation
    - Test that each visualization type creates output files
    - Test error handling for visualization failures
    - _Requirements: 8.1, 8.2, 8.3_

- [ ] 14. Implement error handling and validation
  - [ ] 14.1 Add comprehensive error handling
    - Add missing column error reporting with column names
    - Add insufficient data error with minimum requirements
    - Add NaN/Inf prediction error with diagnostics
    - Note: no network retry logic. The system performs no network I/O — both
      `download()` methods raise `NotImplementedError` by design and the loaders
      consume a pre-processed on-disk schema. Do not add retry/backoff until a
      real `download()` is in scope
    - _Requirements: 10.1, 10.2, 10.4, 10.5_
  
  - [ ] 14.2 Write property test for missing column error reporting
    - **Property 19: Missing Column Error Reporting**
    - **Validates: Requirements 10.2**

- [ ] 15. Create end-to-end integration
  - [ ] 15.1 Create main execution script
    - Wire together data loading, feature engineering, model training, and evaluation
    - Add command-line interface for running different tracks
    - Add configuration file support for model parameters
    - _Requirements: 9.1, 9.4_ (orchestration only; the script exercises the
      Requirement 1-8 pipeline end-to-end but validates no acceptance criterion
      that is not already owned by an earlier task)
  
  - [ ] 15.2 Write integration tests
    - Test complete Track A pipeline
    - Test complete Track B pipeline
    - Test error recovery and fallback strategies
    - _Requirements: 1.7, 10.1, 10.4, 10.5_

- [ ] 16. Set up continuous integration
  - Run the full test suite on every commit
  - Run property tests at 100 iterations in CI, 1000 iterations nightly
  - Enforce the 85% coverage target for core modules
  - Track model training time and prediction latency as performance benchmarks
  - Note: the design specifies a Continuous Integration section that no task
    previously implemented
  - _Requirements: 9.5_

- [ ] 17. Final checkpoint - Run complete system
  - Execute full pipeline on both Track A and Track B
  - Generate all visualizations and metrics
  - Verify all property tests pass with 100 iterations
  - Ensure all tests pass, ask the user if questions arise

## Notes

- Each task references specific requirements for traceability
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples and edge cases
- Checkpoints ensure incremental validation throughout development
- The system uses Python with standard data science libraries
- Deep learning models (LSTM, Transformer) use PyTorch
