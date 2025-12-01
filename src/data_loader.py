"""
Data Loading Module for EstrelaBet Churn Prediction

This module provides utilities for loading and validating data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader with validation and preprocessing.
    """

    EXPECTED_COLUMNS = [
        'user_id', 'session_id', 'timestamp', 'date', 'hour', 'day_of_week',
        'is_weekend', 'is_holiday', 'game_type', 'sport_type', 'country',
        'device_type', 'payment_method', 'user_age', 'account_age_days',
        'vip_tier', 'campaign_type', 'bet_amount', 'win_amount', 'net_result',
        'session_length_minutes', 'games_played', 'bonus_used', 'deposit_amount',
        'withdrawal_amount', 'previous_session_gap_hours', 'lifetime_deposits',
        'lifetime_bets', 'avg_bet_size'
    ]

    def __init__(self, data_path: str):
        """
        Initialize DataLoader.

        Parameters
        ----------
        data_path : str
            Path to the data directory
        """
        self.data_path = Path(data_path)
        self.df = None

    def load(self, filename: str = 'test_dataset.csv') -> pd.DataFrame:
        """
        Load and validate dataset.

        Parameters
        ----------
        filename : str
            Name of the CSV file to load

        Returns
        -------
        pd.DataFrame
            Loaded and validated dataset
        """
        file_path = self.data_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        self.df = pd.read_csv(file_path)

        # Validate columns
        self._validate_columns()

        # Convert data types
        self._convert_types()

        logger.info(f"Loaded {len(self.df):,} rows and {len(self.df.columns)} columns")

        return self.df

    def _validate_columns(self) -> None:
        """Validate that all expected columns are present."""
        missing = set(self.EXPECTED_COLUMNS) - set(self.df.columns)
        if missing:
            logger.warning(f"Missing columns: {missing}")

    def _convert_types(self) -> None:
        """Convert columns to appropriate data types."""
        # Convert timestamps
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])

        # Sort by user and timestamp
        if 'user_id' in self.df.columns and 'timestamp' in self.df.columns:
            self.df = self.df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    def get_summary(self) -> dict:
        """
        Get summary statistics of the dataset.

        Returns
        -------
        dict
            Summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'n_users': self.df['user_id'].nunique(),
            'n_sessions': self.df['session_id'].nunique(),
            'date_range': (
                self.df['date'].min().strftime('%Y-%m-%d'),
                self.df['date'].max().strftime('%Y-%m-%d')
            ),
            'missing_values': self.df.isnull().sum().to_dict()
        }


def create_target(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Create the target variable: churn (no redeposit within 30 days).

    A user is considered churned (target=1) if they did NOT make any deposit
    after their first session. Users who made a redeposit are retained (target=0).

    Parameters
    ----------
    df : pd.DataFrame
        Raw session-level data

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Target variable (user-level) and processed DataFrame
        - target=1: Churned (no redeposit)
        - target=0: Retained (made redeposit)
    """
    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()

    # Mark first session
    df_sorted['is_first_session'] = ~df_sorted.duplicated('user_id', keep='first')

    # Check for churn (no redeposit) after first session
    def check_churn(group):
        subsequent = group[~group['is_first_session']]
        if len(subsequent) == 0:
            return 1  # No subsequent sessions = churned
        # If any deposit > 0 in subsequent sessions, user is retained (0), else churned (1)
        return 0 if (subsequent['deposit_amount'].fillna(0) > 0).any() else 1

    target = df_sorted.groupby('user_id').apply(check_churn)
    target.name = 'churn'

    churn_count = target.sum()
    retained_count = len(target) - churn_count
    logger.info(f"Target created: {churn_count:,} churned, {retained_count:,} retained out of {len(target):,} users (Churn rate: {target.mean():.2%})")

    return target, df_sorted


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('../data')
    df = loader.load()
    print(loader.get_summary())
