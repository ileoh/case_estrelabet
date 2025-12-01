"""
Feature Engineering Module for EstrelaBet Churn Prediction

This module provides utilities for creating features from raw session data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline for churn prediction.

    By default, creates features using ONLY first session data to avoid data leakage.
    This is critical because the target (churn) is defined as "no redeposit after first session".

    First session features include:
    - Temporal (hour, day_of_week, weekend, holiday)
    - Financial (bet_amount, win_amount, net_result, deposit)
    - Behavioral (games_played, session_length, bonus_used)
    - Demographics (vip_tier, device, country, payment_method)

    Note: The class also contains methods for behavioral, financial, engagement, and trend
    features which use ALL sessions. These are kept for reference but should NOT be used
    for churn prediction as they cause data leakage.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_names: List[str] = []
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []

    def extract_first_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from the user's first session.

        Parameters
        ----------
        df : pd.DataFrame
            Processed session data with 'is_first_session' column

        Returns
        -------
        pd.DataFrame
            First session features for each user
        """
        first_sessions = df[df['is_first_session']].copy()

        features = pd.DataFrame()
        features['user_id'] = first_sessions['user_id']

        # Temporal features
        features['first_session_hour'] = first_sessions['hour'].values
        features['first_session_day_of_week'] = first_sessions['day_of_week'].values
        features['first_session_weekend'] = first_sessions['is_weekend'].values
        features['first_session_holiday'] = first_sessions['is_holiday'].values

        # Game and device features
        features['first_session_game_type'] = first_sessions['game_type'].values
        features['first_session_is_sports'] = (first_sessions['game_type'] == 'sports_betting').astype(int).values
        features['first_session_device'] = first_sessions['device_type'].values
        features['first_session_country'] = first_sessions['country'].values

        # Payment and account features
        features['first_session_payment_method'] = first_sessions['payment_method'].values
        features['first_session_account_age'] = first_sessions['account_age_days'].values
        features['first_session_vip_tier'] = first_sessions['vip_tier'].values
        features['first_session_user_age'] = first_sessions['user_age'].values

        # Campaign and bonus features
        features['first_session_campaign'] = first_sessions['campaign_type'].values
        features['first_session_bonus_used'] = first_sessions['bonus_used'].values

        # Betting behavior features
        features['first_session_bet_amount'] = first_sessions['bet_amount'].values
        features['first_session_win_amount'] = first_sessions['win_amount'].values
        features['first_session_net_result'] = first_sessions['net_result'].values
        features['first_session_won'] = (first_sessions['net_result'] > 0).astype(int).values

        # Session behavior features
        features['first_session_length'] = first_sessions['session_length_minutes'].values
        features['first_session_games_played'] = first_sessions['games_played'].values

        # Deposit/Withdrawal features
        features['first_session_deposited'] = (first_sessions['deposit_amount'].fillna(0) > 0).astype(int).values
        features['first_session_deposit_amount'] = first_sessions['deposit_amount'].fillna(0).values
        features['first_session_withdrew'] = (first_sessions['withdrawal_amount'].fillna(0) > 0).astype(int).values

        logger.info(f"Extracted {len(features.columns) - 1} first session features")
        return features

    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract aggregated behavioral features per user.

        Parameters
        ----------
        df : pd.DataFrame
            Session-level data

        Returns
        -------
        pd.DataFrame
            Behavioral features for each user
        """
        # Session aggregations
        session_agg = df.groupby('user_id').agg({
            'session_id': 'count',
            'timestamp': ['min', 'max']
        })
        session_agg.columns = ['total_sessions', 'first_session_time', 'last_session_time']

        # Session characteristics
        session_chars = df.groupby('user_id').agg({
            'session_length_minutes': ['sum', 'mean', 'std', 'max', 'min'],
            'games_played': ['sum', 'mean', 'std', 'max']
        })
        session_chars.columns = [
            'total_time_played', 'avg_session_length', 'std_session_length',
            'max_session_length', 'min_session_length',
            'total_games_played', 'avg_games_per_session', 'std_games_per_session', 'max_games_per_session'
        ]

        # Session gap aggregations
        gap_agg = df.groupby('user_id')['previous_session_gap_hours'].agg(['mean', 'std', 'max', 'min'])
        gap_agg.columns = ['avg_session_gap_hours', 'std_session_gap_hours', 'max_session_gap_hours', 'min_session_gap_hours']

        # Temporal patterns
        def get_mode(x):
            return x.mode()[0] if len(x.mode()) > 0 else x.median()

        temporal_agg = df.groupby('user_id').agg({
            'hour': get_mode,
            'day_of_week': get_mode,
            'is_weekend': 'mean',
            'is_holiday': 'mean'
        })
        temporal_agg.columns = ['preferred_hour', 'preferred_day', 'weekend_ratio', 'holiday_ratio']

        # Diversity metrics
        diversity_agg = df.groupby('user_id').agg({
            'game_type': 'nunique',
            'device_type': 'nunique',
            'bonus_used': ['sum', 'mean']
        })
        diversity_agg.columns = ['game_type_diversity', 'device_diversity', 'total_bonuses_used', 'bonus_usage_rate']

        # Merge all
        features = session_agg.join(session_chars).join(gap_agg).join(temporal_agg).join(diversity_agg)

        # Derived features
        features['active_days'] = (features['last_session_time'] - features['first_session_time']).dt.days + 1
        features['sessions_per_day'] = features['total_sessions'] / features['active_days'].replace(0, 1)

        # Drop timestamp columns
        features = features.drop(['first_session_time', 'last_session_time'], axis=1)
        features = features.reset_index()

        logger.info(f"Extracted {len(features.columns) - 1} behavioral features")
        return features

    def extract_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract financial and betting features per user.

        Parameters
        ----------
        df : pd.DataFrame
            Session-level data

        Returns
        -------
        pd.DataFrame
            Financial features for each user
        """
        # Betting aggregations
        betting_agg = df.groupby('user_id').agg({
            'bet_amount': ['sum', 'mean', 'std', 'max', 'min'],
            'win_amount': ['sum', 'mean', 'max'],
            'net_result': ['sum', 'mean', 'std', 'max', 'min']
        })
        betting_agg.columns = [
            'total_bet_amount', 'avg_bet_amount', 'std_bet_amount', 'max_bet_amount', 'min_bet_amount',
            'total_win_amount', 'avg_win_amount', 'max_win_amount',
            'total_net_result', 'avg_net_result', 'std_net_result', 'max_net_result', 'min_net_result'
        ]

        # Deposit aggregations
        deposit_agg = df.groupby('user_id')['deposit_amount'].agg([
            ('total_deposits', lambda x: x.fillna(0).sum()),
            ('deposit_count', lambda x: (x.fillna(0) > 0).sum()),
            ('avg_deposit_amount', lambda x: x.fillna(0).mean())
        ])

        # Withdrawal aggregations
        withdrawal_agg = df.groupby('user_id')['withdrawal_amount'].agg([
            ('total_withdrawals', lambda x: x.fillna(0).sum()),
            ('withdrawal_count', lambda x: (x.fillna(0) > 0).sum())
        ])

        # Lifetime metrics
        lifetime_agg = df.groupby('user_id').agg({
            'lifetime_deposits': 'last',
            'lifetime_bets': 'last',
            'avg_bet_size': 'last'
        })
        lifetime_agg.columns = ['lifetime_deposits', 'lifetime_bets', 'lifetime_avg_bet_size']

        # Merge all
        features = betting_agg.join(deposit_agg).join(withdrawal_agg).join(lifetime_agg)

        # Derived features
        features['net_deposits'] = features['total_deposits'] - features['total_withdrawals']
        features['win_rate'] = features['total_win_amount'] / features['total_bet_amount'].replace(0, 1)
        features['return_on_bets'] = features['total_net_result'] / features['total_bet_amount'].replace(0, 1)
        features['deposit_to_bet_ratio'] = features['total_deposits'] / features['total_bet_amount'].replace(0, 1)

        # Winning session rate
        win_sessions = df.groupby('user_id')['net_result'].apply(lambda x: (x > 0).sum())
        total_sessions = df.groupby('user_id').size()
        features['winning_session_rate'] = win_sessions / total_sessions
        features = features.reset_index()

        logger.info(f"Extracted {len(features.columns) - 1} financial features")
        return features

    def extract_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract engagement and preference features per user.

        Parameters
        ----------
        df : pd.DataFrame
            Session-level data

        Returns
        -------
        pd.DataFrame
            Engagement features for each user
        """
        features = pd.DataFrame()
        features['user_id'] = df['user_id'].unique()

        # Favorite game type
        favorite_game = df.groupby('user_id')['game_type'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        ).reset_index()
        favorite_game.columns = ['user_id', 'favorite_game_type']
        features = features.merge(favorite_game, on='user_id', how='left')

        # Favorite device
        favorite_device = df.groupby('user_id')['device_type'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        ).reset_index()
        favorite_device.columns = ['user_id', 'primary_device']
        features = features.merge(favorite_device, on='user_id', how='left')

        # Primary payment method
        favorite_payment = df.groupby('user_id')['payment_method'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 and x.notna().any() else 'unknown'
        ).reset_index()
        favorite_payment.columns = ['user_id', 'primary_payment_method']
        features = features.merge(favorite_payment, on='user_id', how='left')

        # Mobile ratio
        mobile_ratio = df.groupby('user_id')['device_type'].apply(
            lambda x: (x == 'mobile').mean()
        ).reset_index()
        mobile_ratio.columns = ['user_id', 'mobile_ratio']
        features = features.merge(mobile_ratio, on='user_id', how='left')

        # Night owl ratio
        night_ratio = df.groupby('user_id')['hour'].apply(
            lambda x: ((x >= 22) | (x <= 6)).mean()
        ).reset_index()
        night_ratio.columns = ['user_id', 'night_owl_ratio']
        features = features.merge(night_ratio, on='user_id', how='left')

        # Sports betting ratio
        sports_ratio = df.groupby('user_id')['game_type'].apply(
            lambda x: (x == 'sports_betting').mean()
        ).reset_index()
        sports_ratio.columns = ['user_id', 'sports_betting_ratio']
        features = features.merge(sports_ratio, on='user_id', how='left')

        # Uses crypto
        uses_crypto = df.groupby('user_id')['payment_method'].apply(
            lambda x: (x == 'crypto').any()
        ).astype(int).reset_index()
        uses_crypto.columns = ['user_id', 'uses_crypto']
        features = features.merge(uses_crypto, on='user_id', how='left')

        # Payment method diversity
        payment_diversity = df.groupby('user_id')['payment_method'].nunique().reset_index()
        payment_diversity.columns = ['user_id', 'payment_method_diversity']
        features = features.merge(payment_diversity, on='user_id', how='left')

        # Campaign diversity
        campaign_diversity = df.groupby('user_id')['campaign_type'].nunique().reset_index()
        campaign_diversity.columns = ['user_id', 'campaign_diversity']
        features = features.merge(campaign_diversity, on='user_id', how='left')

        logger.info(f"Extracted {len(features.columns) - 1} engagement features")
        return features

    def extract_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract trend features that capture behavioral changes over time.

        Parameters
        ----------
        df : pd.DataFrame
            Session-level data

        Returns
        -------
        pd.DataFrame
            Trend features for each user
        """
        def calculate_trend(series):
            """Calculate linear trend slope."""
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope, _, _, _, _ = stats.linregress(x, series)
                return slope
            except:
                return 0

        def calculate_first_vs_last_half(series):
            """Compare first half vs second half average."""
            if len(series) < 2:
                return 1
            mid = len(series) // 2
            first_half_avg = series[:mid].mean()
            second_half_avg = series[mid:].mean()
            if first_half_avg == 0:
                return 1
            return second_half_avg / first_half_avg

        features = pd.DataFrame()
        features['user_id'] = df['user_id'].unique()

        # Filter users with multiple sessions
        multi_session_users = df.groupby('user_id').filter(lambda x: len(x) >= 3)

        if len(multi_session_users) > 0:
            # Bet amount trend
            bet_trend = multi_session_users.groupby('user_id')['bet_amount'].apply(calculate_trend).reset_index()
            bet_trend.columns = ['user_id', 'bet_amount_trend']
            features = features.merge(bet_trend, on='user_id', how='left')

            # Session length trend
            session_trend = multi_session_users.groupby('user_id')['session_length_minutes'].apply(calculate_trend).reset_index()
            session_trend.columns = ['user_id', 'session_length_trend']
            features = features.merge(session_trend, on='user_id', how='left')

            # Games played trend
            games_trend = multi_session_users.groupby('user_id')['games_played'].apply(calculate_trend).reset_index()
            games_trend.columns = ['user_id', 'games_played_trend']
            features = features.merge(games_trend, on='user_id', how='left')

            # Decay features
            bet_decay = multi_session_users.groupby('user_id')['bet_amount'].apply(calculate_first_vs_last_half).reset_index()
            bet_decay.columns = ['user_id', 'bet_amount_decay']
            features = features.merge(bet_decay, on='user_id', how='left')

            session_decay = multi_session_users.groupby('user_id')['session_length_minutes'].apply(calculate_first_vs_last_half).reset_index()
            session_decay.columns = ['user_id', 'session_length_decay']
            features = features.merge(session_decay, on='user_id', how='left')

        # Session gap trend
        gap_df = df[df['previous_session_gap_hours'].notna()]
        if len(gap_df) > 0:
            gap_trend = gap_df.groupby('user_id')['previous_session_gap_hours'].apply(calculate_trend).reset_index()
            gap_trend.columns = ['user_id', 'session_gap_trend']
            features = features.merge(gap_trend, on='user_id', how='left')

        # Fill NaN with neutral values
        trend_cols = ['bet_amount_trend', 'session_length_trend', 'games_played_trend',
                      'bet_amount_decay', 'session_length_decay', 'session_gap_trend']
        for col in trend_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)
            else:
                features[col] = 0

        logger.info(f"Extracted {len(features.columns) - 1} trend features")
        return features

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        first_session_only: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature matrix from raw session data.

        Parameters
        ----------
        df : pd.DataFrame
            Processed session-level data with 'is_first_session' column
        target : pd.Series
            Target variable (churn) indexed by user_id
        first_session_only : bool
            If True (default), use only first session features to avoid data leakage.
            Set to False only for experimental purposes.

        Returns
        -------
        pd.DataFrame
            Complete feature matrix with all features and target
        """
        logger.info("Building feature matrix...")

        # Extract first session features (always included)
        first_session_features = self.extract_first_session_features(df)
        features_df = first_session_features.copy()

        # IMPORTANT: By default, only use first session features to avoid data leakage
        # The behavioral, financial, engagement, and trend features use data from ALL sessions,
        # which would leak future information when predicting churn after first session.
        if not first_session_only:
            logger.warning("Using all features - this may cause data leakage!")
            behavioral_features = self.extract_behavioral_features(df)
            financial_features = self.extract_financial_features(df)
            engagement_features = self.extract_engagement_features(df)
            trend_features = self.extract_trend_features(df)

            features_df = features_df.merge(behavioral_features, on='user_id', how='left')
            features_df = features_df.merge(financial_features, on='user_id', how='left')
            features_df = features_df.merge(engagement_features, on='user_id', how='left')
            features_df = features_df.merge(trend_features, on='user_id', how='left')

        # Add target
        features_df = features_df.merge(target.reset_index(), on='user_id', how='left')

        logger.info(f"Feature matrix built: {features_df.shape}")
        if first_session_only:
            logger.info("Using first-session features only (no data leakage)")

        return features_df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on feature type.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with potential missing values

        Returns
        -------
        pd.DataFrame
            Feature matrix with missing values handled
        """
        df = df.copy()

        # Numerical features - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # Categorical features - fill with 'unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna('unknown')

        logger.info("Missing values handled")
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        target_col: str = 'churn'
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix with categorical columns
        target_col : str
            Name of target column for target encoding

        Returns
        -------
        pd.DataFrame
            Feature matrix with encoded categorical variables
        """
        df = df.copy()

        # VIP tier - ordinal encoding
        vip_order = {'unknown': 0, 'bronze': 1, 'silver': 2, 'gold': 3, 'platinum': 4, 'diamond': 5}
        if 'first_session_vip_tier' in df.columns:
            df['first_session_vip_tier_encoded'] = df['first_session_vip_tier'].map(vip_order).fillna(0)

        # One-hot encode low cardinality features (first session only to avoid leakage)
        low_cardinality_cols = [
            'first_session_device', 'first_session_game_type', 'first_session_campaign'
        ]

        for col in low_cardinality_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        # Target encode high cardinality features (first session only to avoid leakage)
        high_cardinality_cols = [
            'first_session_country', 'first_session_payment_method'
        ]

        for col in high_cardinality_cols:
            if col in df.columns and target_col in df.columns:
                target_means = df.groupby(col)[target_col].mean()
                df[f'{col}_encoded'] = df[col].map(target_means).fillna(df[target_col].mean())

        # Identify and drop original categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if 'user_id' in categorical_cols:
            categorical_cols.remove('user_id')

        df = df.drop(columns=categorical_cols, errors='ignore')

        logger.info("Categorical variables encoded")
        return df

    def add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sin/cos encoding for cyclical features.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix

        Returns
        -------
        pd.DataFrame
            Feature matrix with cyclical encodings
        """
        df = df.copy()

        # Hour encoding
        if 'first_session_hour' in df.columns:
            df['first_session_hour_sin'] = np.sin(2 * np.pi * df['first_session_hour'] / 24)
            df['first_session_hour_cos'] = np.cos(2 * np.pi * df['first_session_hour'] / 24)

        if 'preferred_hour' in df.columns:
            df['preferred_hour_sin'] = np.sin(2 * np.pi * df['preferred_hour'] / 24)
            df['preferred_hour_cos'] = np.cos(2 * np.pi * df['preferred_hour'] / 24)

        # Day of week encoding
        if 'first_session_day_of_week' in df.columns:
            df['first_session_dow_sin'] = np.sin(2 * np.pi * df['first_session_day_of_week'] / 7)
            df['first_session_dow_cos'] = np.cos(2 * np.pi * df['first_session_day_of_week'] / 7)

        if 'preferred_day' in df.columns:
            df['preferred_day_sin'] = np.sin(2 * np.pi * df['preferred_day'] / 7)
            df['preferred_day_cos'] = np.cos(2 * np.pi * df['preferred_day'] / 7)

        logger.info("Cyclical features added")
        return df


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data, create_target

    # Load data
    df = load_data('../data/test_dataset.csv')
    target, df_processed = create_target(df)

    # Create feature engineer
    fe = FeatureEngineer()

    # Build feature matrix
    features = fe.build_feature_matrix(df_processed, target)

    # Handle missing values
    features = fe.handle_missing_values(features)

    # Encode categorical
    features = fe.encode_categorical(features)

    # Add cyclical features
    features = fe.add_cyclical_features(features)

    print(f"Final feature matrix: {features.shape}")
