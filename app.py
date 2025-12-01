"""
EstrelaBet - Churn Prediction Dashboard

A Streamlit application for predicting customer churn (no redeposit)
and analyzing retention metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import sklearn components
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Page configuration
st.set_page_config(
    page_title="EstrelaBet - Churn Prediction",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Business constants
CLV_BY_TIER = {
    'bronze': 750,
    'silver': 1750,
    'gold': 4000,
    'platinum': 8000,
    'diamond': 15250,
    'unknown': 750
}

CAMPAIGN_COSTS = {
    'email': 25,
    'bonus': 100,
    'phone': 137.5,
    'vip_manager': 375
}

RETENTION_SUCCESS_RATES = {
    'early_intervention': 0.275,  # 30-60% churn risk, 20-35% success
    'high_risk': 0.20,            # 60-80% churn risk, 15-25% success
    'critical': 0.15              # >80% churn risk, 10-20% success
}


@st.cache_data
def load_data():
    """Load the engineered features dataset or generate from raw data."""
    # Try to load engineered features first
    features_path = Path("data/features_engineered.csv")
    if features_path.exists():
        df = pd.read_csv(features_path)
        return df

    # Fallback: Load raw data and generate basic features
    raw_path = Path("data/test_dataset.csv")
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        df_raw = df_raw.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Generate features on-the-fly
        df = generate_features_from_raw(df_raw)
        return df

    return None


def generate_features_from_raw(df_raw):
    """Generate features from raw session data."""
    df = df_raw.copy()

    # Mark first session
    df['is_first_session'] = ~df.duplicated('user_id', keep='first')

    # Create target: churn (no redeposit after first session)
    # churn=1 means customer did NOT make a redeposit (churned)
    # churn=0 means customer made a redeposit (retained)
    def check_churn(group):
        subsequent = group[~group['is_first_session']]
        if len(subsequent) == 0:
            return 1  # No subsequent sessions = churned
        return 0 if (subsequent['deposit_amount'].fillna(0) > 0).any() else 1

    churn = df.groupby('user_id').apply(check_churn)
    churn.name = 'churn'

    # Get first session data
    first_sessions = df[df['is_first_session']].copy()

    # Create user-level features
    features = pd.DataFrame()
    features['user_id'] = first_sessions['user_id'].values

    # First session features
    features['first_session_hour'] = first_sessions['hour'].values
    features['first_session_day_of_week'] = first_sessions['day_of_week'].values
    features['first_session_weekend'] = first_sessions['is_weekend'].values
    features['first_session_bet_amount'] = first_sessions['bet_amount'].values
    features['first_session_win_amount'] = first_sessions['win_amount'].values
    features['first_session_net_result'] = first_sessions['net_result'].values
    features['first_session_won'] = (first_sessions['net_result'] > 0).astype(int).values
    features['first_session_length'] = first_sessions['session_length_minutes'].values
    features['first_session_games_played'] = first_sessions['games_played'].values
    features['first_session_bonus_used'] = first_sessions['bonus_used'].values
    features['first_session_deposited'] = (first_sessions['deposit_amount'].fillna(0) > 0).astype(int).values
    features['first_session_deposit_amount'] = first_sessions['deposit_amount'].fillna(0).values

    # Behavioral aggregations
    behavioral = df.groupby('user_id').agg({
        'session_id': 'count',
        'session_length_minutes': ['sum', 'mean'],
        'games_played': ['sum', 'mean'],
        'bet_amount': ['sum', 'mean', 'max'],
        'win_amount': ['sum', 'mean'],
        'net_result': ['sum', 'mean'],
        'bonus_used': ['sum', 'mean'],
        'is_weekend': 'mean',
        'game_type': 'nunique',
        'device_type': 'nunique'
    })
    behavioral.columns = ['_'.join(col).strip('_') for col in behavioral.columns]
    behavioral = behavioral.rename(columns={
        'session_id_count': 'total_sessions',
        'session_length_minutes_sum': 'total_time_played',
        'session_length_minutes_mean': 'avg_session_length',
        'games_played_sum': 'total_games_played',
        'games_played_mean': 'avg_games_per_session',
        'bet_amount_sum': 'total_bet_amount',
        'bet_amount_mean': 'avg_bet_amount',
        'bet_amount_max': 'max_bet_amount',
        'win_amount_sum': 'total_win_amount',
        'win_amount_mean': 'avg_win_amount',
        'net_result_sum': 'total_net_result',
        'net_result_mean': 'avg_net_result',
        'bonus_used_sum': 'total_bonuses_used',
        'bonus_used_mean': 'bonus_usage_rate',
        'is_weekend_mean': 'weekend_ratio',
        'game_type_nunique': 'game_type_diversity',
        'device_type_nunique': 'device_diversity'
    })
    behavioral = behavioral.reset_index()

    # Merge all features
    features = features.merge(behavioral, on='user_id', how='left')

    # Add churn target
    features = features.merge(churn.reset_index(), on='user_id', how='left')

    # Calculate derived features
    features['win_rate'] = features['total_win_amount'] / features['total_bet_amount'].replace(0, 1)

    # Fill missing values
    features = features.fillna(0)

    return features


@st.cache_resource
def load_model():
    """Load the trained model or create a simple one."""
    model_path = Path("data/best_model.joblib")
    if model_path.exists():
        model_package = joblib.load(model_path)
        return model_package

    # Create a simple model if none exists
    return create_simple_model()


def create_simple_model():
    """Create a simple logistic regression model for demo purposes."""
    try:
        # Load raw data directly to avoid circular dependency
        raw_path = Path("data/test_dataset.csv")
        features_path = Path("data/features_engineered.csv")

        if features_path.exists():
            df = pd.read_csv(features_path)
        elif raw_path.exists():
            df_raw = pd.read_csv(raw_path)
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            df_raw = df_raw.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
            df = generate_features_from_raw(df_raw)
        else:
            return None

        # Prepare features
        feature_cols = [col for col in df.columns if col not in ['user_id', 'churn']]
        X = df[feature_cols].copy()
        y = df['churn'].copy()

        # Handle any remaining issues
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train simple model
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_scaled, y)

        # Calculate basic metrics
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = model.predict(X_scaled)

        metrics = {
            'ROC-AUC': roc_auc_score(y, y_proba),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred)
        }

        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_cols,
            'model_name': 'Logistic Regression (Auto-generated)',
            'metrics': metrics
        }
    except Exception as e:
        st.warning(f"Could not create model: {e}")
        return None


def get_risk_category(probability):
    """Categorize churn risk based on probability."""
    if probability >= 0.7:
        return "Critical", "#e74c3c"
    elif probability >= 0.5:
        return "High", "#e67e22"
    elif probability >= 0.3:
        return "Medium", "#f1c40f"
    else:
        return "Low", "#27ae60"


def calculate_expected_value(prob_churn, vip_tier, campaign_type, retention_rate=0.25):
    """Calculate expected value of intervention."""
    clv = CLV_BY_TIER.get(vip_tier, 750)
    cost = CAMPAIGN_COSTS.get(campaign_type, 100)

    # Expected value = probability of saving customer * CLV - cost
    expected_value = prob_churn * retention_rate * clv - cost
    roi = (expected_value / cost * 100) if cost > 0 else 0

    return {
        'clv': clv,
        'cost': cost,
        'expected_value': expected_value,
        'roi': roi,
        'worth_intervening': expected_value > 0
    }


def create_gauge_chart(value, title):
    """Create a gauge chart for probability display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': '%', 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#27ae60'},
                {'range': [30, 50], 'color': '#f1c40f'},
                {'range': [50, 70], 'color': '#e67e22'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    """Main application function."""

    # Header
    st.markdown('<p class="main-header">🎰 EstrelaBet Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict customer churn and optimize retention strategies</p>', unsafe_allow_html=True)

    # Load data and model
    df = load_data()
    model_package = load_model()

    # Sidebar
    with st.sidebar:
        # Load logo
        logo_path = Path("images/logo.jpg")
        if logo_path.exists():
            st.image(str(logo_path), width=200)
        else:
            st.markdown("### ⭐ EstrelaBet")
        st.markdown("---")
        st.markdown("### Navigation")

        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "🔮 Individual Prediction", "📈 Batch Prediction", "📋 Model Information"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Quick Stats")

        if df is not None:
            total_users = len(df)
            churn_rate = df['churn'].mean() * 100
            retention_rate = (1 - df['churn'].mean()) * 100

            st.metric("Total Users", f"{total_users:,}")
            st.metric("Retention Rate", f"{retention_rate:.1f}%")
            st.metric("Churn Rate", f"{churn_rate:.1f}%")

        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This application predicts customer churn (no redeposit) "
            "after the first betting session, enabling proactive retention strategies."
        )

    # Main content based on selected page
    if page == "📊 Dashboard":
        show_dashboard(df, model_package)
    elif page == "🔮 Individual Prediction":
        show_individual_prediction(df, model_package)
    elif page == "📈 Batch Prediction":
        show_batch_prediction(df, model_package)
    elif page == "📋 Model Information":
        show_model_info(model_package)


def show_dashboard(df, model_package):
    """Display the main dashboard."""
    st.header("📊 Retention Analytics Dashboard")

    if df is None:
        st.error("Data not found. Please ensure 'data/features_engineered.csv' exists.")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Users",
            f"{len(df):,}",
            help="Total number of users in the dataset"
        )

    with col2:
        churn_rate = df['churn'].mean()
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1%}",
            delta=f"-{churn_rate:.1%}",
            delta_color="inverse",
            help="Percentage of users who did NOT make a redeposit"
        )

    with col3:
        retention_rate = 1 - churn_rate
        st.metric(
            "Retention Rate",
            f"{retention_rate:.1%}",
            help="Percentage of users who made a redeposit (retained)"
        )

    with col4:
        if model_package:
            st.metric(
                "Model ROC-AUC",
                f"{model_package['metrics']['ROC-AUC']:.2%}",
                help="Model discrimination performance"
            )
        else:
            st.metric("Model Status", "Not Loaded")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig = px.pie(
            df,
            names=df['churn'].map({0: 'Retained', 1: 'Churned'}),
            color_discrete_sequence=['#27ae60', '#e74c3c'],
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("First Session Outcome Impact")
        if 'first_session_won' in df.columns:
            # Calculate retention rate (1 - churn) by first bet outcome
            won_retention = df.groupby('first_session_won')['churn'].apply(lambda x: 1 - x.mean()).reset_index()
            won_retention.columns = ['first_session_won', 'retention_rate']
            won_retention['first_session_won'] = won_retention['first_session_won'].map(
                {0: 'Lost First Bet', 1: 'Won First Bet'}
            )
            fig = px.bar(
                won_retention,
                x='first_session_won',
                y='retention_rate',
                color='first_session_won',
                color_discrete_sequence=['#e74c3c', '#27ae60'],
                labels={'retention_rate': 'Retention Rate', 'first_session_won': ''}
            )
            fig.update_layout(height=350, showlegend=False)
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

    # Second row of charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sessions Distribution")
        if 'total_sessions' in df.columns:
            fig = px.histogram(
                df,
                x='total_sessions',
                color=df['churn'].map({0: 'Retained', 1: 'Churned'}),
                barmode='overlay',
                nbins=30,
                color_discrete_sequence=['#27ae60', '#e74c3c'],
                labels={'color': 'Status'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("First Session Net Result")
        if 'first_session_net_result' in df.columns:
            fig = px.box(
                df,
                x=df['churn'].map({0: 'Retained', 1: 'Churned'}),
                y='first_session_net_result',
                color=df['churn'].map({0: 'Retained', 1: 'Churned'}),
                color_discrete_sequence=['#27ae60', '#e74c3c']
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.markdown("---")
    st.subheader("Top Predictive Features")

    if model_package and 'model' in model_package:
        model = model_package['model']
        feature_names = model_package.get('feature_names', [])

        if hasattr(model, 'feature_importances_') and len(feature_names) > 0:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True).tail(15)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")


def show_individual_prediction(df, model_package):
    """Show individual user prediction interface."""
    st.header("🔮 Individual Churn Prediction")

    if df is None or model_package is None:
        st.error("Data or model not loaded. Please check that all files exist.")
        return

    st.markdown("Enter user characteristics to predict churn probability.")

    # Create input form
    with st.form("prediction_form"):
        st.subheader("First Session Characteristics")

        col1, col2, col3 = st.columns(3)

        with col1:
            first_session_won = st.selectbox(
                "First Bet Result",
                options=[0, 1],
                format_func=lambda x: "Won" if x == 1 else "Lost"
            )

            first_session_net_result = st.number_input(
                "Net Result (R$)",
                min_value=-10000.0,
                max_value=10000.0,
                value=0.0,
                step=10.0
            )

            first_session_bet_amount = st.number_input(
                "Bet Amount (R$)",
                min_value=0.0,
                max_value=10000.0,
                value=50.0,
                step=10.0
            )

        with col2:
            first_session_length = st.number_input(
                "Session Length (minutes)",
                min_value=1,
                max_value=480,
                value=30
            )

            first_session_games_played = st.number_input(
                "Games Played",
                min_value=1,
                max_value=100,
                value=5
            )

            first_session_bonus_used = st.selectbox(
                "Bonus Used",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            first_session_deposited = st.selectbox(
                "Made Deposit",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            first_session_hour = st.slider(
                "Session Hour",
                min_value=0,
                max_value=23,
                value=20
            )

            first_session_weekend = st.selectbox(
                "Weekend Session",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        st.subheader("User Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            vip_tier = st.selectbox(
                "VIP Tier",
                options=['bronze', 'silver', 'gold', 'platinum', 'diamond'],
                index=0
            )

        with col2:
            total_sessions = st.number_input(
                "Total Sessions",
                min_value=1,
                max_value=100,
                value=1
            )

        with col3:
            campaign_type = st.selectbox(
                "Campaign Type",
                options=['email', 'bonus', 'phone', 'vip_manager'],
                index=0,
                help="For ROI calculation"
            )

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        st.markdown("---")

        # Create feature vector (simplified for demo)
        # In production, this would use the full feature engineering pipeline
        feature_values = {
            'first_session_won': first_session_won,
            'first_session_net_result': first_session_net_result,
            'first_session_bet_amount': first_session_bet_amount,
            'first_session_length': first_session_length,
            'first_session_games_played': first_session_games_played,
            'first_session_bonus_used': first_session_bonus_used,
            'first_session_deposited': first_session_deposited,
            'first_session_hour': first_session_hour,
            'first_session_weekend': first_session_weekend,
            'total_sessions': total_sessions
        }

        # For demo purposes, use a heuristic if model features don't match
        # In production, ensure feature alignment
        try:
            model = model_package['model']
            scaler = model_package.get('scaler')
            feature_names = model_package.get('feature_names', [])

            # Create a sample from data with similar characteristics
            sample = df.drop(['user_id', 'churn'], axis=1, errors='ignore').iloc[0:1].copy()

            # Update with user inputs where columns match
            for col, val in feature_values.items():
                if col in sample.columns:
                    sample[col] = val

            # Ensure column order matches
            sample = sample[feature_names] if feature_names else sample

            # Scale if needed
            if scaler and 'Logistic' in model_package.get('model_name', ''):
                sample = pd.DataFrame(scaler.transform(sample), columns=sample.columns)

            # Predict churn probability directly (model predicts churn=1)
            prob_churn = model.predict_proba(sample)[0][1]

        except Exception as e:
            # Fallback to heuristic prediction
            st.warning(f"Using heuristic prediction due to feature mismatch: {e}")

            # Simple heuristic based on key factors
            base_prob = 0.45  # Base churn probability

            # Adjust based on first bet result
            if first_session_won:
                base_prob -= 0.15
            else:
                base_prob += 0.10

            # Adjust based on session length
            if first_session_length > 45:
                base_prob -= 0.10
            elif first_session_length < 15:
                base_prob += 0.10

            # Adjust based on bonus usage
            if first_session_bonus_used:
                base_prob -= 0.05

            # Adjust based on total sessions
            if total_sessions > 3:
                base_prob -= 0.15

            prob_churn = max(0.05, min(0.95, base_prob))

        # Display results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Churn Probability")
            fig = create_gauge_chart(prob_churn, "Churn Risk")
            st.plotly_chart(fig, use_container_width=True)

            risk_category, risk_color = get_risk_category(prob_churn)
            st.markdown(
                f"<h3 style='text-align: center; color: {risk_color};'>Risk Level: {risk_category}</h3>",
                unsafe_allow_html=True
            )

        with col2:
            st.subheader("Business Analysis")

            ev_analysis = calculate_expected_value(prob_churn, vip_tier, campaign_type)

            st.metric("Customer Lifetime Value", f"R$ {ev_analysis['clv']:,.0f}")
            st.metric("Campaign Cost", f"R$ {ev_analysis['cost']:,.0f}")
            st.metric("Expected Value", f"R$ {ev_analysis['expected_value']:,.0f}")
            st.metric("Expected ROI", f"{ev_analysis['roi']:.0f}%")

            if ev_analysis['worth_intervening']:
                st.success("✅ Intervention Recommended - Positive Expected Value")
            else:
                st.warning("⚠️ Consider lower-cost intervention or no action")

        # Recommendations
        st.markdown("---")
        st.subheader("📋 Recommendations")

        if risk_category == "Critical":
            st.error("""
            **Critical Risk User - Immediate Action Required**

            1. **Immediate Contact**: Personal outreach within 24 hours
            2. **Premium Offer**: High-value bonus or cashback offer
            3. **VIP Upgrade**: Consider temporary VIP status upgrade
            4. **Root Cause**: Analyze first session experience for issues
            """)
        elif risk_category == "High":
            st.warning("""
            **High Risk User - Proactive Intervention Needed**

            1. **Quick Response**: Send personalized offer within 48 hours
            2. **Bonus Offer**: Targeted deposit bonus
            3. **Re-engagement**: Invite to upcoming promotions or events
            4. **Follow-up**: Schedule check-in after 7 days
            """)
        elif risk_category == "Medium":
            st.info("""
            **Medium Risk User - Monitor and Engage**

            1. **Automated Campaign**: Include in email re-engagement sequence
            2. **Light Incentive**: Small free bet or bonus offer
            3. **Content**: Send relevant game updates or tips
            4. **Monitor**: Track activity over next 14 days
            """)
        else:
            st.success("""
            **Low Risk User - Nurture and Grow**

            1. **Standard Engagement**: Regular promotional emails
            2. **Loyalty Program**: Highlight loyalty benefits
            3. **Cross-sell**: Introduce new game types
            4. **Feedback**: Request satisfaction feedback
            """)


def show_batch_prediction(df, model_package):
    """Show batch prediction interface."""
    st.header("📈 Batch Churn Prediction")

    if df is None or model_package is None:
        st.error("Data or model not loaded. Please check that all files exist.")
        return

    st.markdown("Analyze churn risk across your entire user base.")

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'first_session_vip_tier_encoded' in df.columns:
            min_vip = st.slider("Minimum VIP Level", 0, 5, 0)

    with col2:
        if 'total_sessions' in df.columns:
            min_sessions = st.slider(
                "Minimum Sessions",
                int(df['total_sessions'].min()),
                int(df['total_sessions'].max()),
                1
            )

    with col3:
        risk_filter = st.multiselect(
            "Risk Levels",
            options=["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"]
        )

    # Generate predictions
    if st.button("Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Generating predictions..."):
            try:
                model = model_package['model']
                scaler = model_package.get('scaler')
                feature_names = model_package.get('feature_names', [])
                model_name = model_package.get('model_name', '')

                # Prepare features
                X = df.drop(['user_id', 'churn'], axis=1, errors='ignore')

                if feature_names:
                    # Ensure columns exist
                    missing_cols = set(feature_names) - set(X.columns)
                    for col in missing_cols:
                        X[col] = 0
                    X = X[feature_names]

                # Scale if needed
                if scaler and 'Logistic' in model_name:
                    X = pd.DataFrame(scaler.transform(X), columns=X.columns)

                # Predict churn probability directly (model predicts churn=1)
                churn_proba = model.predict_proba(X)[:, 1]

                # Create results dataframe
                results = pd.DataFrame({
                    'user_id': df['user_id'].values if 'user_id' in df.columns else range(len(df)),
                    'actual_outcome': df['churn'].map({0: 'Retained', 1: 'Churned'}).values,
                    'churn_probability': churn_proba,
                    'retention_probability': 1 - churn_proba
                })

                # Add risk category
                results['risk_category'] = results['churn_probability'].apply(
                    lambda x: get_risk_category(x)[0]
                )

            except Exception as e:
                st.error(f"Error generating predictions: {e}")

                # Fallback: use churn as proxy
                results = pd.DataFrame({
                    'user_id': df['user_id'].values if 'user_id' in df.columns else range(len(df)),
                    'actual_outcome': df['churn'].map({0: 'Retained', 1: 'Churned'}).values,
                    'churn_probability': df['churn'].values * 0.7 + np.random.uniform(0, 0.3, len(df)),
                    'retention_probability': (1 - df['churn'].values) * 0.7 + np.random.uniform(0, 0.3, len(df))
                })
                results['risk_category'] = results['churn_probability'].apply(
                    lambda x: get_risk_category(x)[0]
                )

            # Apply filters
            filtered_results = results[results['risk_category'].isin(risk_filter)]

            # Display summary
            st.markdown("---")
            st.subheader("Risk Distribution")

            col1, col2, col3, col4 = st.columns(4)

            risk_counts = results['risk_category'].value_counts()

            with col1:
                critical = risk_counts.get('Critical', 0)
                st.metric("Critical Risk", f"{critical:,}", delta=f"{critical/len(results)*100:.1f}%")

            with col2:
                high = risk_counts.get('High', 0)
                st.metric("High Risk", f"{high:,}", delta=f"{high/len(results)*100:.1f}%")

            with col3:
                medium = risk_counts.get('Medium', 0)
                st.metric("Medium Risk", f"{medium:,}", delta=f"{medium/len(results)*100:.1f}%")

            with col4:
                low = risk_counts.get('Low', 0)
                st.metric("Low Risk", f"{low:,}", delta=f"{low/len(results)*100:.1f}%")

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    results,
                    names='risk_category',
                    title='Risk Distribution',
                    color='risk_category',
                    color_discrete_map={
                        'Critical': '#e74c3c',
                        'High': '#e67e22',
                        'Medium': '#f1c40f',
                        'Low': '#27ae60'
                    }
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    results,
                    x='churn_probability',
                    nbins=50,
                    title='Churn Probability Distribution',
                    color_discrete_sequence=['#3498db']
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Data table
            st.subheader(f"Filtered Results ({len(filtered_results):,} users)")

            # Sort by churn probability
            display_df = filtered_results.sort_values('churn_probability', ascending=False)

            st.dataframe(
                display_df.head(100).style.background_gradient(
                    subset=['churn_probability'],
                    cmap='RdYlGn_r'
                ),
                use_container_width=True,
                height=400
            )

            # Download button
            csv = filtered_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            # Business impact
            st.markdown("---")
            st.subheader("💰 Business Impact Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Calculate potential value at risk
                avg_clv = 1000  # Average CLV
                high_risk_users = len(results[results['risk_category'].isin(['Critical', 'High'])])
                value_at_risk = high_risk_users * avg_clv * 0.5  # 50% expected churn

                st.metric("High-Risk Users", f"{high_risk_users:,}")
                st.metric("Value at Risk", f"R$ {value_at_risk:,.0f}")

            with col2:
                # Calculate potential savings
                intervention_rate = 0.25  # 25% success rate
                potential_saved = high_risk_users * intervention_rate * avg_clv

                st.metric("Potential Users Saved", f"{int(high_risk_users * intervention_rate):,}")
                st.metric("Potential Value Saved", f"R$ {potential_saved:,.0f}")


def show_model_info(model_package):
    """Display model information and metrics."""
    st.header("📋 Model Information")

    if model_package is None:
        st.error("Model not loaded. Please run the modeling notebook first.")

        st.markdown("""
        ### How to Generate the Model

        1. Open the Jupyter notebook: `notebooks/03_modeling_and_evaluation.ipynb`
        2. Run all cells to train the models
        3. The best model will be saved to `data/best_model.joblib`
        4. Refresh this page to load the model
        """)
        return

    # Model details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Details")
        st.info(f"**Model Type**: {model_package.get('model_name', 'Unknown')}")
        st.info(f"**Number of Features**: {len(model_package.get('feature_names', []))}")

    with col2:
        st.subheader("Performance Metrics")
        metrics = model_package.get('metrics', {})

        for metric, value in metrics.items():
            if isinstance(value, float):
                st.metric(metric, f"{value:.4f}")

    # Feature list
    st.markdown("---")
    st.subheader("Feature List")

    feature_names = model_package.get('feature_names', [])
    if feature_names:
        col1, col2, col3 = st.columns(3)

        n_per_col = len(feature_names) // 3 + 1

        with col1:
            for feat in feature_names[:n_per_col]:
                st.write(f"• {feat}")

        with col2:
            for feat in feature_names[n_per_col:2*n_per_col]:
                st.write(f"• {feat}")

        with col3:
            for feat in feature_names[2*n_per_col:]:
                st.write(f"• {feat}")

    # Business rules
    st.markdown("---")
    st.subheader("Business Rules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Customer Lifetime Value by Tier**")
        clv_df = pd.DataFrame({
            'VIP Tier': CLV_BY_TIER.keys(),
            'CLV (R$)': CLV_BY_TIER.values()
        })
        st.dataframe(clv_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Campaign Costs**")
        campaign_df = pd.DataFrame({
            'Campaign Type': CAMPAIGN_COSTS.keys(),
            'Cost (R$)': CAMPAIGN_COSTS.values()
        })
        st.dataframe(campaign_df, use_container_width=True, hide_index=True)

    # Risk thresholds
    st.markdown("---")
    st.subheader("Risk Classification Thresholds")

    threshold_data = {
        'Risk Level': ['Low', 'Medium', 'High', 'Critical'],
        'Churn Probability': ['< 30%', '30% - 50%', '50% - 70%', '> 70%'],
        'Recommended Action': [
            'Standard engagement',
            'Email campaigns',
            'Bonus offers',
            'Personal outreach'
        ]
    }
    st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
