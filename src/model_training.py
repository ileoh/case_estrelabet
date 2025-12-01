"""
Model Training Module for EstrelaBet Churn Prediction

This module provides utilities for training and evaluating churn prediction models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logger = logging.getLogger(__name__)

# Optional imports for boosting models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed")


class ChurnModelTrainer:
    """
    Model training pipeline for churn prediction.

    Supports training multiple model types with cross-validation,
    hyperparameter configuration, and model persistence.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize ChurnModelTrainer.

        Parameters
        ----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: List[str] = []

    def initialize_models(self, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Initialize all models with default hyperparameters.

        Parameters
        ----------
        y_train : np.ndarray
            Training target for calculating class weights

        Returns
        -------
        Dict[str, Any]
            Dictionary of initialized models
        """
        models = {}

        # Baseline models
        models['Baseline (Majority)'] = DummyClassifier(
            strategy='most_frequent',
            random_state=self.random_state
        )

        models['Baseline (Stratified)'] = DummyClassifier(
            strategy='stratified',
            random_state=self.random_state
        )

        # Logistic Regression
        models['Logistic Regression'] = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state
        )

        # Random Forest
        models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        # XGBoost
        if XGBOOST_AVAILABLE:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            models['XGBoost'] = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='auc',
                use_label_encoder=False
            )

        # LightGBM
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                is_unbalance=True,
                random_state=self.random_state,
                verbose=-1
            )

        self.models = models
        logger.info(f"Initialized {len(models)} models: {list(models.keys())}")
        return models

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.20,
        val_size: float = 0.15
    ) -> Tuple[np.ndarray, ...]:
        """
        Prepare data for training with train/val/test split.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        test_size : float
            Proportion of data for test set
        val_size : float
            Proportion of data for validation set

        Returns
        -------
        Tuple
            X_train, X_val, X_test, y_train, y_val, y_test (scaled and unscaled)
        """
        self.feature_names = X.columns.tolist()

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: train vs validation
        val_proportion = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_proportion, random_state=self.random_state, stratify=y_temp
        )

        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train, X_val, X_test,
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train, y_val, y_test)

    def train(
        self,
        X_train: pd.DataFrame,
        X_train_scaled: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all initialized models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Unscaled training features
        X_train_scaled : pd.DataFrame
            Scaled training features (for linear models)
        y_train : pd.Series
            Training target

        Returns
        -------
        Dict[str, Any]
            Dictionary of trained models
        """
        if not self.models:
            self.initialize_models(y_train.values)

        self.trained_models = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Use scaled data for Logistic Regression
            X_tr = X_train_scaled if 'Logistic' in name else X_train

            # Train model
            model.fit(X_tr, y_train)
            self.trained_models[name] = model

        logger.info(f"Trained {len(self.trained_models)} models")
        return self.trained_models

    def cross_validate(
        self,
        X_train: pd.DataFrame,
        X_train_scaled: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation on all models.

        Parameters
        ----------
        X_train : pd.DataFrame
            Unscaled training features
        X_train_scaled : pd.DataFrame
            Scaled training features
        y_train : pd.Series
            Training target
        cv : int
            Number of cross-validation folds

        Returns
        -------
        Dict[str, Dict[str, float]]
            Cross-validation results for each model
        """
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_results = {}

        for name, model in self.models.items():
            if 'Baseline' in name:
                continue

            logger.info(f"Cross-validating {name}...")

            X_cv = X_train_scaled if 'Logistic' in name else X_train
            scores = cross_val_score(model, X_cv, y_train, cv=cv_splitter, scoring='roc_auc', n_jobs=-1)

            cv_results[name] = {
                'mean_roc_auc': scores.mean(),
                'std_roc_auc': scores.std(),
                'scores': scores.tolist()
            }

            logger.info(f"  {name}: ROC-AUC = {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        return cv_results

    def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        X_scaled: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model to use
        X : pd.DataFrame
            Unscaled features
        X_scaled : pd.DataFrame
            Scaled features

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Predicted labels and probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")

        model = self.trained_models[model_name]
        X_eval = X_scaled if 'Logistic' in model_name else X

        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, 'predict_proba') else None

        return y_pred, y_proba

    def get_best_model(self, results: Dict[str, Dict]) -> str:
        """
        Get the name of the best performing model.

        Parameters
        ----------
        results : Dict[str, Dict]
            Evaluation results for each model

        Returns
        -------
        str
            Name of the best model
        """
        best_model = None
        best_score = 0

        for name, metrics in results.items():
            if 'Baseline' in name:
                continue

            score = metrics.get('roc_auc', 0) if isinstance(metrics, dict) else 0
            if score > best_score:
                best_score = score
                best_model = name

        return best_model

    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Optional[pd.DataFrame]
            Feature importance DataFrame or None
        """
        if model_name not in self.trained_models:
            return None

        model = self.trained_models[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            return importance

        elif hasattr(model, 'coef_'):
            importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': np.abs(model.coef_[0])
            }).sort_values('Importance', ascending=False)
            return importance

        return None

    def save_model(
        self,
        model_name: str,
        filepath: str,
        include_metrics: Optional[Dict] = None
    ) -> None:
        """
        Save a trained model to disk.

        Parameters
        ----------
        model_name : str
            Name of the model to save
        filepath : str
            Path to save the model
        include_metrics : Optional[Dict]
            Optional metrics to include in the package
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not trained")

        model_package = {
            'model': self.trained_models[model_name],
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': model_name,
            'metrics': include_metrics
        }

        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Load a saved model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model

        Returns
        -------
        Dict[str, Any]
            Model package with model, scaler, and metadata
        """
        model_package = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model_package


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f2'
) -> Tuple[float, float]:
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
    Tuple[float, float]
        Optimal threshold and corresponding score
    """
    from sklearn.metrics import f1_score, fbeta_score, roc_curve

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
            j_scores = tpr - fpr
            score = j_scores.max()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


if __name__ == "__main__":
    # Example usage
    from data_loader import load_data, create_target
    from feature_engineering import FeatureEngineer
    from metrics import MetricsCalculator

    # Load data
    df = load_data('../data/test_dataset.csv')
    target, df_processed = create_target(df)

    # Build features
    fe = FeatureEngineer()
    features = fe.build_feature_matrix(df_processed, target)
    features = fe.handle_missing_values(features)
    features = fe.encode_categorical(features)
    features = fe.add_cyclical_features(features)

    # Prepare for training
    feature_cols = [col for col in features.columns if col not in ['user_id', 'churn']]
    X = features[feature_cols]
    y = features['churn']

    # Train models
    trainer = ChurnModelTrainer()
    (X_train, X_val, X_test,
     X_train_scaled, X_val_scaled, X_test_scaled,
     y_train, y_val, y_test) = trainer.prepare_data(X, y)

    trainer.initialize_models(y_train.values)
    trainer.train(X_train, X_train_scaled, y_train)

    # Evaluate
    calc = MetricsCalculator()
    for name in trainer.trained_models:
        y_pred, y_proba = trainer.predict(name, X_test, X_test_scaled)
        if y_proba is not None:
            metrics = calc.calculate_ml_metrics(y_test.values, y_pred, y_proba)
            print(f"{name}: ROC-AUC={metrics['roc_auc']:.4f}")
