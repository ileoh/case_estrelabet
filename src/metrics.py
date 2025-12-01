"""
Metrics Module for EstrelaBet Churn Prediction

This module provides ML and business metrics calculation utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)


# Business constants from case document
# CLV by Tier (using midpoint of ranges from assessment)
CLV_BY_TIER = {
    'bronze': 750,      # R$500-R$1,000
    'silver': 1750,     # R$1,500-R$2,000
    'gold': 4000,       # R$2,500-R$5,500
    'platinum': 8000,   # R$6,000-R$10,000
    'diamond': 15250,   # R$10,500-R$20,000
    'unknown': 750      # Default to bronze
}

# Campaign Costs per User (using midpoint of ranges)
CAMPAIGN_COSTS = {
    'email': 25,        # R$15-R$35
    'bonus': 100,       # R$50-R$150
    'phone': 137.5,     # R$75-R$200
    'vip_manager': 375  # R$250-R$500
}

# Retention Success Rates by Risk Category
RETENTION_SUCCESS_RATES = {
    'early_intervention': 0.275,  # 30-60% churn probability, 20-35% success rate
    'high_risk': 0.20,            # 60-80% churn probability, 15-25% success rate
    'critical': 0.15              # >80% churn probability, 10-20% success rate
}


class MetricsCalculator:
    """
    Calculate ML and business metrics for model evaluation.
    """

    def __init__(self, avg_clv: float = 1000):
        """
        Initialize MetricsCalculator.

        Parameters
        ----------
        avg_clv : float
            Average Customer Lifetime Value
        """
        self.avg_clv = avg_clv

    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate machine learning metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities

        Returns
        -------
        Dict[str, float]
            Dictionary of ML metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'pr_auc': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }

    def calculate_confusion_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Calculate confusion matrix metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels

        Returns
        -------
        Dict[str, int]
            Confusion matrix values
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        campaign_type: str = 'bonus',
        avg_retention_rate: float = 0.25
    ) -> Dict[str, Any]:
        """
        Calculate business metrics for churn prediction.

        Parameters
        ----------
        y_true : np.ndarray
            True labels (1 = churned, 0 = retained)
        y_pred : np.ndarray
            Predicted labels (1 = predicted churn, 0 = predicted retained)
        y_proba : np.ndarray
            Predicted probabilities of churn
        campaign_type : str
            Type of retention campaign
        avg_retention_rate : float
            Average success rate of retention campaigns

        Returns
        -------
        Dict[str, Any]
            Business metrics
        """
        campaign_cost = CAMPAIGN_COSTS.get(campaign_type, 100)

        # Users targeted (predicted as churners)
        # target=1 means churn, target=0 means retained
        # We target users predicted as churners (y_pred == 1)
        users_targeted = (y_pred == 1).sum()

        # True positives (correctly identified churners)
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()

        # False positives (incorrectly identified as churners)
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()

        # Total campaign cost
        total_cost = users_targeted * campaign_cost

        # Expected users saved (only true positives can be saved)
        users_saved = true_positives * avg_retention_rate

        # Value saved
        value_saved = users_saved * self.avg_clv

        # Net value
        net_value = value_saved - total_cost

        # ROI
        roi = (net_value / total_cost * 100) if total_cost > 0 else 0

        return {
            'users_targeted': int(users_targeted),
            'true_churners_found': int(true_positives),
            'false_alarms': int(false_positives),
            'total_campaign_cost': total_cost,
            'expected_users_saved': users_saved,
            'value_saved': value_saved,
            'net_value': net_value,
            'roi_percentage': roi
        }

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        campaign_type: str = 'bonus'
    ) -> Dict[str, Any]:
        """
        Calculate all metrics (ML + business).

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities
        campaign_type : str
            Type of retention campaign

        Returns
        -------
        Dict[str, Any]
            All metrics
        """
        ml_metrics = self.calculate_ml_metrics(y_true, y_pred, y_proba)
        confusion_metrics = self.calculate_confusion_metrics(y_true, y_pred)
        business_metrics = self.calculate_business_metrics(
            y_true, y_pred, y_proba, campaign_type
        )

        return {
            'ml_metrics': ml_metrics,
            'confusion_metrics': confusion_metrics,
            'business_metrics': business_metrics
        }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f2'
) -> float:
    """
    Find optimal classification threshold.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize ('f1', 'f2', 'youden')

    Returns
    -------
    float
        Optimal threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        elif metric == 'youden':
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            # Find threshold that maximizes Youden's J
            j_scores = tpr - fpr
            score = j_scores.max()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.random(100)
    y_pred = (y_proba >= 0.5).astype(int)

    calculator = MetricsCalculator()
    metrics = calculator.calculate_all_metrics(y_true, y_pred, y_proba)

    print("ML Metrics:")
    for k, v in metrics['ml_metrics'].items():
        print(f"  {k}: {v:.4f}")

    print("\nBusiness Metrics:")
    for k, v in metrics['business_metrics'].items():
        print(f"  {k}: {v}")
