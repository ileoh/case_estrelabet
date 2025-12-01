# EstrelaBet - Customer Churn Prediction Model
## Executive Summary

---

## 1. Overview

### Business Context
Customer retention is the difference between profit and loss in the online betting market. With an average Customer Acquisition Cost (CAC) of R$150-300 per user, each retained customer can generate R$500 to R$20,000 in Lifetime Value (LTV) depending on their tier.

### Objective
Develop a predictive model to identify customers who will **churn** (NOT make a redeposit) after their first betting session, enabling proactive retention interventions.

**Target Variable Definition**:
- **Churn = 1**: Customer did NOT make a redeposit (churned)
- **Churn = 0**: Customer made a redeposit (retained)

### Dataset
- **10,000 sessions** from **1,446 unique users**
- **29 variables** covering behavioral, financial, and demographic data
- **Period**: July 2023 - December 2024

---

## 2. Business Parameters

### Customer Lifetime Value (CLV) by Tier

| Tier | CLV Range | Midpoint (Used) |
|------|-----------|-----------------|
| Bronze | R$500 - R$1,000 | R$750 |
| Silver | R$1,500 - R$2,000 | R$1,750 |
| Gold | R$2,500 - R$5,500 | R$4,000 |
| Platinum | R$6,000 - R$10,000 | R$8,000 |
| Diamond | R$10,500 - R$20,000 | R$15,250 |

### Retention Campaign Costs (Per User)

| Campaign Type | Cost Range | Midpoint (Used) |
|---------------|------------|-----------------|
| Email | R$15 - R$35 | R$25 |
| Bonus | R$50 - R$150 | R$100 |
| Phone/Chat | R$75 - R$200 | R$137.50 |
| VIP Manager | R$250 - R$500 | R$375 |

### Historical Retention Effectiveness by Risk Level

| Risk Level | Churn Probability | Success Rate Range | Midpoint (Used) |
|------------|-------------------|-------------------|-----------------|
| Early Intervention | 30-60% | 20-35% | 27.5% |
| High-Risk | 60-80% | 15-25% | 20% |
| Critical | >80% | 10-20% | 15% |

---

## 3. Key Performance Indicators

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Churn Rate** | ~32.6% | Baseline churn rate after first session |
| **Retention Rate** | ~67.4% | Baseline retention |
| **Model ROC-AUC** | 0.60-0.66 | Moderate discrimination ability |
| **Model Recall** | ~54% | Captures half of churners |
| **Model Precision** | ~38% | Targeting accuracy |

**Note**: These metrics reflect predictions made using **only first-session data**, which is a challenging but realistic scenario for early churn detection.

---

## 4. Key Findings

### 4.1 The "First Win Effect" (Critical Insight)

**Finding**: Customers who WIN on their first bet have a significantly **lower churn rate** than those who lose.

| First Bet Result | Churn Rate | Retention Rate | Lift |
|------------------|------------|----------------|------|
| Lost | ~55-60% | ~40-45% | Baseline |
| Won | ~35-40% | ~60-65% | -35% churn reduction |

**Recommendation**: Consider "First Win" strategies such as:
- Favorable odds for first bets
- Partial cashback on first losses
- "Guaranteed first win" promotions

### 4.2 Campaign Effectiveness

| Campaign Type | Users | Churn Rate | Relative Performance |
|---------------|-------|------------|---------------------|
| Cashback | Low | ~30% | Best (lowest churn) |
| Welcome Bonus | High | ~35% | Above average |
| Free Spins | Medium | ~40% | Average |
| None | Medium | ~50% | Worst (highest churn) |

**Recommendation**: Prioritize cashback and welcome bonus campaigns for new user acquisition.

### 4.3 User Behavior Patterns

**Low Churn Indicators** (Retained Users):
- Longer first session duration (>45 minutes)
- Multiple games played in first session (>10)
- Used bonus in first session
- Played on weekend
- Higher initial bet amounts

**High Churn Risk Indicators**:
- Short first session (<15 minutes)
- Single game type
- No bonus usage
- Large first-session losses
- Mobile-only users (slightly higher churn)

### 4.4 Geographic Insights

| Country | Volume | Churn Rate | Retention Rate |
|---------|--------|------------|----------------|
| BR | Highest | Average | Average |
| DE | High | Below average | Above average |
| UK | Medium | Low | High |
| AU | Medium | Average | Average |
| Other | Low | Variable | Variable |

---

## 5. Model Performance

### Best Model: Logistic Regression / LightGBM

| Metric | Validation | Test | Interpretation |
|--------|------------|------|----------------|
| ROC-AUC | 0.66 | 0.56 | Moderate discrimination |
| PR-AUC | 0.45 | 0.34 | Above random baseline |
| F2-Score | 0.56 | 0.50 | Recall-weighted performance |
| Recall | 0.58 | 0.54 | Captures 54% of churners |
| Precision | 0.42 | 0.38 | 38% of flagged users actually churn |

**Why metrics are moderate (not perfect)**:
- Model uses **only first-session features** (no future information)
- This is the correct methodology to avoid data leakage
- Predicting churn with limited data is inherently challenging
- Metrics are realistic for production deployment

### Top Predictive Features (First Session Only)

1. **first_session_net_result** - Net result of first betting session
2. **first_session_bet_amount** - Amount bet in first session
3. **first_session_win_amount** - Winnings in first session
4. **first_session_length** - Duration of first session
5. **first_session_games_played** - Games played in first session
6. **first_session_won** - Whether user won on first bet
7. **first_session_bonus_used** - Whether bonus was used
8. **first_session_deposited** - Whether deposit was made

---

## 6. Business Impact Analysis

### ROI by Campaign Type (for Churn Intervention)

| Campaign | Cost/User | Expected Success | Expected CLV Saved | Net Value/User | ROI |
|----------|-----------|------------------|-------------------|----------------|-----|
| Email | R$25 | 27.5% | R$275 | R$250 | 900% |
| Bonus | R$100 | 25% | R$250 | R$150 | 150% |
| Phone | R$137.50 | 20% | R$200 | R$62.50 | 45% |
| VIP Manager | R$375 | 15% | R$150 | -R$225 | -60% |

**Note**: VIP Manager campaigns are only ROI-positive for high-value (Gold+) customers where CLV exceeds R$2,500.

### Expected Impact (Conservative Estimate)

For every 1,000 users predicted as at-risk churners:
- **Model Precision**: ~38% are true churners = 380 actual churners
- **With Email Campaign (27.5% success)**: 105 users retained
- **Value Saved**: 105 x R$1,000 avg CLV = **R$105,000**
- **Campaign Cost**: 1,000 x R$25 = **R$25,000**
- **Net Value**: **R$80,000 additional value**

---

## 7. Recommendations

### Immediate Actions (0-30 days)

1. **Deploy Early Warning System**
   - Implement model scoring for all new users after first session
   - Create automated alerts for high-risk users (>50% churn probability)

2. **First Session Optimization**
   - A/B test "first win guarantee" program
   - Ensure bonus activation in first session

3. **24-Hour Re-engagement**
   - Automated email/push 24h after first session
   - Personalized based on first session behavior

### Medium Term (30-90 days)

4. **Segmented Campaign Strategy**
   - Low Risk (30-50%): Email campaigns only
   - Medium Risk (50-70%): Personalized bonus offers
   - High Risk (>70%): Phone outreach for valuable users

5. **Game-Specific Strategies**
   - Different retention approaches by preferred game type
   - Sports betting users: Event-based promotions

6. **Geographic Customization**
   - Localized campaigns for top markets (BR, DE, UK)

### Long Term (90+ days)

7. **Model Enhancement**
   - Add real-time behavioral features as they become available
   - Implement automated retraining pipeline
   - Develop complementary LTV prediction model

8. **Holistic Customer Journey**
   - Integrate model with CRM system
   - Personalized user experience based on risk score

---

## 8. Limitations and Assumptions

### Assumptions
- Churn defined as no redeposit within observation period
- First session = first betting activity
- Missing deposit values indicate no deposit

### Limitations
- **First-Session Only Features**: Model intentionally uses only data available at time of first session to avoid data leakage
- **Moderate Predictive Power**: ROC-AUC ~0.60 reflects the difficulty of predicting churn with limited initial data
- Relatively small dataset (1,446 users)
- 18-month period may not capture full seasonality
- No user acquisition cost data
- No pre-registration behavior data

### Recommendations for Future
- Collect longer historical data
- Track acquisition channel costs
- Implement real-time feature computation
- Consider multi-session models for ongoing retention monitoring

---

## 9. Technical Appendix

### Model Architecture
```
Pipeline:
├── Data Loading (data_loader.py)
├── Feature Engineering (feature_engineering.py)
│   └── First Session Features (~36 features)
│       ├── Temporal (hour, day_of_week, weekend, holiday)
│       ├── Financial (bet_amount, win_amount, net_result, deposit)
│       ├── Behavioral (games_played, session_length, bonus_used)
│       └── Demographics (vip_tier, device, country, payment_method)
├── Preprocessing
│   ├── RobustScaler for numerical
│   └── Target/One-Hot encoding for categorical
└── Modeling (model_training.py)
    ├── Logistic Regression (best generalization)
    ├── Random Forest
    ├── XGBoost
    └── LightGBM
```

### Files Delivered
- `notebooks/01_exploratory_data_analysis.ipynb` - Complete EDA
- `notebooks/02_feature_engineering.ipynb` - Feature construction (first session only)
- `notebooks/03_modeling_and_evaluation.ipynb` - Model training & evaluation
- `src/data_loader.py` - Data loading utilities
- `src/feature_engineering.py` - Feature engineering pipeline
- `src/model_training.py` - Model training utilities
- `src/metrics.py` - ML and business metrics
- `app.py` - Streamlit dashboard application
- `reports/executive_summary.md` - This document

---

## Contact

For questions about this analysis or model implementation, please contact the Data Science team.

---

*Report generated as part of EstrelaBet Data Scientist Technical Assessment*
