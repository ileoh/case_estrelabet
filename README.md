# EstrelaBet - Redeposit Prediction Model

A machine learning solution to predict customer redeposit behavior after their first bet, enabling proactive retention strategies in the online betting industry.

## Table of Contents

- [Overview](#overview)
- [Business Context](#business-context)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [File Descriptions](#file-descriptions)
- [Assumptions and Limitations](#assumptions-and-limitations)

## Overview

This project develops a predictive model to identify customers who will **NOT make a redeposit** after their first betting session. By identifying at-risk customers early, the business can implement targeted retention campaigns to maximize Customer Lifetime Value (CLV).

### Dataset Summary

| Metric | Value |
|--------|-------|
| Total Sessions | 10,000 |
| Unique Users | 1,446 |
| Variables | 29 |
| Time Period | July 2023 - December 2024 |

## Business Context

Customer retention is critical in the online betting market:

- **Customer Acquisition Cost (CAC)**: R$150-300 per user
- **Customer Lifetime Value (CLV)**: R$500 to R$20,000 depending on tier
- **Retention Impact**: Each retained customer can generate significant recurring revenue

The model enables:
1. Early identification of churn risk after first session
2. Targeted retention campaigns with optimized ROI
3. Personalized interventions based on user behavior patterns

## Project Structure

```
case_estrelabet/
├── data/
│   └── test_dataset.csv              # Raw session-level data
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb   # Complete EDA
│   ├── 02_feature_engineering.ipynb          # Feature construction
│   └── 03_modeling_and_evaluation.ipynb      # Modeling & evaluation
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── data_loader.py                # Data loading utilities
│   └── metrics.py                    # ML and business metrics
├── reports/
│   └── executive_summary.md          # Executive summary report
├── docs/
│   ├── Data Scientist Technical Assessment.pdf
│   └── Dataset Overview.pdf
├── app.py                            # Streamlit dashboard application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup

1. Clone or download the repository:
```bash
cd case_estrelabet
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

Or install individually:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn plotly shap lifelines optuna streamlit joblib
```

### Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | ML models and metrics |
| xgboost | Gradient boosting |
| lightgbm | Gradient boosting |
| matplotlib | Static visualizations |
| seaborn | Statistical visualizations |
| plotly | Interactive visualizations |
| shap | Model interpretability |
| lifelines | Survival analysis |
| optuna | Hyperparameter tuning |
| streamlit | Interactive web dashboard |
| joblib | Model serialization |

## Usage

### Running the Analysis

Execute the notebooks in sequence:

```bash
jupyter notebook
```

1. **01_exploratory_data_analysis.ipynb**
   - Load and explore the dataset
   - Understand data quality and distributions
   - Identify key patterns and insights

2. **02_feature_engineering.ipynb**
   - Construct the target variable
   - Create features from session data
   - Prepare data for modeling

3. **03_modeling_and_evaluation.ipynb**
   - Train and evaluate models
   - Optimize thresholds for business metrics
   - Generate interpretability analysis

### Using the Source Modules

```python
from src.data_loader import DataLoader, create_target
from src.metrics import MetricsCalculator, find_optimal_threshold

# Load data
loader = DataLoader('data')
df = loader.load('test_dataset.csv')
print(loader.get_summary())

# Create target variable
target, df_processed = create_target(df)

# Calculate metrics
calculator = MetricsCalculator(avg_clv=1000)
metrics = calculator.calculate_all_metrics(y_true, y_pred, y_proba)
```

## Streamlit Dashboard

The project includes an interactive Streamlit dashboard for real-time churn predictions and retention analytics.

### Running the Dashboard

1. First, ensure you have run the notebooks to generate the required files:
   - `data/features_engineered.csv` (from notebook 02)
   - `data/best_model.joblib` (from notebook 03)

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

### Dashboard Features

| Page | Description |
|------|-------------|
| **Dashboard** | Overview of retention analytics with key metrics, target distribution, first win effect analysis, and feature importance |
| **Individual Prediction** | Enter user characteristics to get real-time churn probability, risk classification, and intervention recommendations |
| **Batch Prediction** | Analyze churn risk across the entire user base with filtering, export capabilities, and business impact analysis |
| **Model Information** | View model details, performance metrics, feature list, and business rules |

### Dashboard Screenshots

**Main Dashboard**
- Key metrics: Total users, redeposit rate, churn rate, model performance
- Visual analytics: Target distribution pie chart, first win effect, sessions distribution
- Feature importance visualization

**Individual Prediction**
- Input form for first session characteristics
- Real-time churn probability gauge
- Risk level classification (Low/Medium/High/Critical)
- ROI calculation for intervention
- Personalized recommendations

**Batch Prediction**
- Filter by risk level and user characteristics
- Risk distribution visualization
- Downloadable CSV with all predictions
- Business impact analysis (value at risk, potential savings)

### Risk Classification

| Risk Level | Churn Probability | Recommended Action |
|------------|-------------------|-------------------|
| Low | < 30% | Standard engagement |
| Medium | 30% - 50% | Email campaigns |
| High | 50% - 70% | Bonus offers |
| Critical | > 70% | Personal outreach |

## Methodology

### Target Variable Definition

The target variable indicates whether a user **churned** (did NOT make a redeposit after their first betting session):

- **Churn = 1**: User did NOT make a redeposit after their first session (churned)
- **Churn = 0**: User made at least one redeposit after their first session (retained)

### Feature Engineering

**36 first-session features** were engineered to predict churn without data leakage:

| Category | Examples | Count |
|----------|----------|-------|
| Temporal | first_session_hour, first_session_day_of_week, first_session_weekend | ~6 |
| Financial | first_session_bet_amount, first_session_win_amount, first_session_net_result | ~8 |
| Behavioral | first_session_games_played, first_session_length, first_session_bonus_used | ~6 |
| Demographics | first_session_vip_tier, first_session_device, first_session_country | ~16 |

**Note**: Only first-session data is used to avoid data leakage. This ensures the model can make predictions immediately after a user's first betting session.

### Missing Values Treatment

| Variable | Missing % | Strategy |
|----------|-----------|----------|
| sport_type | 84% | Binary flag (is_sports_betting) |
| deposit_amount | 72% | Fill with 0 + flag (has_deposit) |
| withdrawal_amount | 91% | Fill with 0 + flag (has_withdrawal) |
| user_age | 7.65% | Median imputation by country |
| payment_method | 2.8% | Mode imputation |

### Models Evaluated

1. **Baseline**: DummyClassifier (stratified)
2. **Logistic Regression**: With L1/L2/ElasticNet regularization
3. **Random Forest**: Ensemble of decision trees
4. **XGBoost**: Gradient boosting with regularization
5. **LightGBM**: Gradient boosting with native categorical support

### Evaluation Strategy

- **Split**: Temporal split (65% train, 15% validation, 20% test)
- **Cross-Validation**: RepeatedStratifiedKFold (5 folds, 3 repeats)
- **Primary Metric**: ROC-AUC
- **Secondary Metrics**: PR-AUC, F2-Score, Recall, Precision, Brier Score

## Key Findings

### 1. The "First Win Effect"

Customers who **win on their first bet** have a significantly higher redeposit rate:

| First Bet Result | Redeposit Rate | Lift |
|------------------|----------------|------|
| Lost | ~40% | Baseline |
| Won | ~55-60% | +38-50% |

**Recommendation**: Implement "First Win" strategies such as favorable odds or cashback on first losses.

### 2. Campaign Effectiveness

| Campaign Type | Redeposit Rate | ROI |
|---------------|----------------|-----|
| Cashback | ~55% | Best |
| Welcome Bonus | ~50% | Good |
| Free Spins | ~45% | Average |
| None | ~40% | Baseline |

### 3. High Redeposit Indicators

- Longer first session duration (>45 minutes)
- Multiple games played in first session (>10)
- Bonus usage in first session
- Weekend activity
- Higher initial bet amounts

### 4. Churn Risk Indicators

- Short first session (<15 minutes)
- Single game type preference
- No bonus usage
- Large first-session losses
- Mobile-only usage

## Model Performance

### Best Model: Logistic Regression / LightGBM

| Metric | Validation | Test |
|--------|------------|------|
| ROC-AUC | 0.66 | 0.56 |
| PR-AUC | 0.45 | 0.34 |
| F2-Score | 0.56 | 0.50 |
| Recall | 0.58 | 0.54 |
| Precision | 0.42 | 0.38 |

**Note**: These metrics reflect the challenging task of predicting churn using only first-session data (no data leakage). The model provides meaningful business value despite moderate metrics.

### Top Predictive Features (First Session Only)

1. `first_session_net_result` - Net result of first betting session
2. `first_session_bet_amount` - Amount bet in first session
3. `first_session_win_amount` - Winnings in first session
4. `first_session_length` - Duration of first session
5. `first_session_games_played` - Games played in first session
6. `first_session_won` - Whether user won on first bet
7. `first_session_bonus_used` - Whether bonus was used
8. `first_session_deposited` - Whether deposit was made in first session

### Business Impact

For every 1,000 users predicted as at-risk churners:

| Metric | Value |
|--------|-------|
| True churners (38% precision) | 380 users |
| Saved with email campaign (27.5%) | 105 users |
| Value saved | R$105,000 |
| Campaign cost | R$25,000 |
| **Net value** | **R$80,000** |

## File Descriptions

### Notebooks

| File | Description |
|------|-------------|
| `01_exploratory_data_analysis.ipynb` | Complete EDA including data quality assessment, statistical analysis, temporal patterns, survival analysis, and RFM segmentation |
| `02_feature_engineering.ipynb` | Target variable construction, feature creation (80+ features), missing value handling, and categorical encoding |
| `03_modeling_and_evaluation.ipynb` | Model training, hyperparameter tuning, threshold optimization, business metrics, feature importance, and SHAP analysis |

### Source Code

| File | Description |
|------|-------------|
| `src/data_loader.py` | `DataLoader` class for loading and validating data, `create_target()` function for target variable construction |
| `src/metrics.py` | `MetricsCalculator` class for ML and business metrics, `find_optimal_threshold()` function, business constants (CLV, campaign costs) |

### Application

| File | Description |
|------|-------------|
| `app.py` | Streamlit dashboard with 4 pages: Dashboard, Individual Prediction, Batch Prediction, Model Information |
| `requirements.txt` | Python package dependencies for the entire project |

### Reports

| File | Description |
|------|-------------|
| `reports/executive_summary.md` | 2-3 page executive summary with key findings, business impact analysis, and actionable recommendations |

## Assumptions and Limitations

### Assumptions

1. **Churn Definition**: A user is considered churned if they make no redeposit during the observation period
2. **First Session**: The first recorded session represents the user's actual first betting activity
3. **Missing Deposits**: Null values in `deposit_amount` indicate no deposit was made in that session

### Limitations

1. **Dataset Size**: Relatively small dataset (~1,446 users) may limit model generalization
2. **Temporal Coverage**: 18-month period may not capture full seasonality effects
3. **Missing Data**: No user acquisition cost data available
4. **Pre-Registration**: No data on user behavior before registration

### Recommendations for Future Work

- Collect longer historical data (2+ years)
- Track acquisition channel costs
- Implement real-time feature computation
- Develop complementary LTV prediction model
- A/B test retention strategies based on model predictions

---

## Contact

For questions about this analysis or model implementation, please contact the Data Science team.

---

*This project was developed as part of the EstrelaBet Data Scientist Technical Assessment.*
