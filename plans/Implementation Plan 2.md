# Insulin Response Modeling on Public Datasets

Analyze and predict how blood glucose levels respond after meals, comparing statistical and machine learning models on public datasets.

## User Review Required

> [!IMPORTANT]
> **CGMacros availability**: The CGMacros dataset is listed as "under review in Physionet" on the repo. If the data files are not yet publicly downloadable, we may need to fall back to the synthetic CGM approach for Track B, or check Physionet directly for access. We will attempt to clone the repo and use any available data first.

## Datasets

| Track | Dataset | What it provides | Source |
|-------|---------|-----------------|--------|
| **A (Tabular)** | UCI Diabetes | Pre/post-meal glucose, insulin doses, meal events | [UCI ML Repository](https://archive.ics.uci.edu/dataset/34/diabetes) |
| **B (Time-Series)** | CGMacros | Dual-CGM glucose at regular intervals, meal macros (carbs/fat/protein), activity, heart rate, demographics, blood analysis, gut microbiome | [GitHub: PSI-TAMU/CGMacros](https://github.com/PSI-TAMU/CGMacros) |

**CGMacros details**: 45 participants (15 healthy, 16 pre-diabetes, 14 T2D), 10 consecutive days of free-living data with known meal compositions. Participants 24, 25, 37, 40 dropped out.

## Prediction Targets

- **Track A**: Post-meal glucose rise (mg/dL) from a single pre-meal snapshot.
- **Track B**: Post-meal glucose curve (iAUC / AUC), or next-N-steps glucose prediction from CGM time-series.

## Proposed Changes

### Project Setup

#### [NEW] [requirements.txt](file:///home/kevin/code/insulin-response/requirements.txt)
`pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`, `torch`, `jupyter`.

---

### Data Pipeline

#### [NEW] [src/data_preprocessing.py](file:///home/kevin/code/insulin-response/src/data_preprocessing.py)
- Download/parse the UCI Diabetes dataset (Track A).
- Clone and parse CGMacros per-participant CSV files (Track B).
- Feature engineering: time-since-meal, macronutrient breakdown, baseline glucose, insulin dose, participant health group.

---

### Exploratory Data Analysis

#### [NEW] [notebooks/EDA.ipynb](file:///home/kevin/code/insulin-response/notebooks/EDA.ipynb)
- Summary statistics (mean, std, quartiles) for glucose levels by health group.
- Distribution and box plots for glucose by meal type and macronutrient composition.
- Correlation heatmap (carbs, fat, protein, insulin dose, baseline glucose vs. glucose rise / iAUC).
- Missing-data audit and outlier detection.
- Overlay plots of CGM glucose curves stratified by meal composition and health group.

---

### Modeling

#### [NEW] [src/statistical_models.py](file:///home/kevin/code/insulin-response/src/statistical_models.py)
- **Linear Regression** — baseline for Track A.
- **ARIMA / SARIMAX** — baseline for Track B time-series.

#### [NEW] [src/ml_models.py](file:///home/kevin/code/insulin-response/src/ml_models.py)
- **Random Forest & SVM** — Track A ([Ref](https://d-nb.info/1273950157/34)).
- **Gradient Boosting (XGBoost / LightGBM)** — Track A ([Ref](https://www.frontiersin.org/articles/10.3389/fnut.2023.1118173/full)).
- **LSTM** — Track B ([Ref](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8838327/)).
- **Transformer** — Track B ([Ref](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8838327/)).

---

### Evaluation and Comparison

#### [NEW] [src/evaluate.py](file:///home/kevin/code/insulin-response/src/evaluate.py)
- **Cross-validation**: k-fold for Track A; time-series split for Track B.
- **Metrics**: RMSE, MAE, R², MAPE.
- **Hyperparameter tuning** via `GridSearchCV` / `RandomizedSearchCV`.
- **Visualizations**: metric comparison bar charts, predicted-vs-actual scatter plots, residual plots.

## Verification Plan

### Automated Tests
1. `python src/data_preprocessing.py` — confirm both datasets load without errors.
2. `python src/evaluate.py` — confirm all models train and produce metrics.

### Manual Verification
1. Review EDA notebook outputs for statistical soundness.
2. Review final comparison table and visualizations.
