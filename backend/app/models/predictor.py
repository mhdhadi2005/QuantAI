"""
AI Prediction Engine
Phase 1: XGBoost + RandomForest classification and regression models.
Predicts probability of UP/DOWN movement and expected price change.
"""
import os
import logging
import pickle
from typing import Optional, Tuple, List
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from app.config import settings
from app.models.lstm_model import LSTMPredictorModel

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../models_saved")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "rsi_14", "macd", "macd_signal", "macd_histogram",
    "stoch_rsi_k", "stoch_rsi_d",
    "bb_pct", "bb_width",
    "atr_14", "sma_20", "ema_9", "ema_21",
    "trend_up", "above_sma_50", "price_vs_sma20",
    "momentum_score", "obv",
    "open", "high", "low", "close", "volume",
]


class AIPredictor:
    """
    AI-powered price direction predictor.
    
    Trains on historical OHLCV + indicator data.
    Predicts: probability of price being higher N bars from now.
    """

    def __init__(self, symbol: str, horizon: int = 5):
        self.symbol = symbol
        self.horizon = horizon  # bars ahead to predict
        self.classifier = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_metrics = {}
        self.feature_importance = {}
        self._load_model()

    def _model_path(self, suffix: str = "") -> str:
        name = f"{self.symbol}_h{self.horizon}{suffix}.pkl"
        return os.path.join(MODEL_DIR, name)

    def _save_model(self):
        try:
            with open(self._model_path("_clf"), "wb") as f:
                pickle.dump(self.classifier, f)
            with open(self._model_path("_scaler"), "wb") as f:
                pickle.dump(self.scaler, f)
            with open(self._model_path("_meta"), "wb") as f:
                pickle.dump({
                    "model_metrics": self.model_metrics,
                    "feature_importance": self.feature_importance,
                }, f)
            logger.info(f"Model saved for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def _load_model(self):
        try:
            clf_path = self._model_path("_clf")
            scaler_path = self._model_path("_scaler")
            meta_path = self._model_path("_meta")
            if os.path.exists(clf_path) and os.path.exists(scaler_path):
                with open(clf_path, "rb") as f:
                    self.classifier = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                        self.model_metrics = meta.get("model_metrics", {})
                        self.feature_importance = meta.get("feature_importance", {})
                self.is_trained = True
                logger.info(f"Loaded pre-trained model for {self.symbol}")
        except Exception as e:
            logger.warning(f"Could not load model for {self.symbol}: {e}")

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare feature matrix X and target labels y from OHLCV + indicator DataFrame.
        Target: 1 if close[t+horizon] > close[t], else 0.
        """
        df = df.copy()

        # Create target variable
        df["future_close"] = df["close"].shift(-self.horizon)
        df["target"] = (df["future_close"] > df["close"]).astype(int)

        # Compute percentage returns
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)

        # Lags of 1-day returns
        df["return_1_lag1"] = df["return_1"].shift(1)
        df["return_1_lag2"] = df["return_1"].shift(2)
        df["return_1_lag3"] = df["return_1"].shift(3)

        # Volatility features
        df["volatility_5"] = df["return_1"].rolling(5).std()
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()

        # Volume momentum
        df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["volume_momentum"] = df["volume"].pct_change(5)

        # Broad Market Context (Index features - Nifty 50 for Indian stocks, SPY for US stocks)
        index_ticker = "^NSEI" if (self.symbol.endswith(".NS") or self.symbol == "^NSEI") else "SPY"
        if self.symbol != index_ticker:
            try:
                from app.data.data_loader import get_ohlcv
                index_df = get_ohlcv(index_ticker, timeframe="1d")
                if not index_df.empty:
                    index_df = index_df.rename(columns={
                        "close": "index_close",
                        "volume": "index_volume"
                    })
                    index_df["index_return_1"] = index_df["index_close"].pct_change(1)
                    index_df["index_above_sma50"] = (index_df["index_close"] > index_df["index_close"].rolling(50).mean()).astype(int)
                    df = df.join(index_df[["index_close", "index_return_1", "index_above_sma50"]], how="left")
            except Exception as e:
                logger.error(f"Error merging index features: {e}")

        # Features to include
        additional_cols = [
            "return_1", "return_5", "return_10", 
            "return_1_lag1", "return_1_lag2", "return_1_lag3",
            "volatility_5", "volatility_10", "volatility_20",
            "volume_sma_ratio", "volume_momentum",
            "index_close", "index_return_1", "index_above_sma50"
        ]
        
        feature_cols = [c for c in FEATURE_COLUMNS + additional_cols if c in df.columns]

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaN targets or features
        df.dropna(subset=feature_cols + ["target"], inplace=True)
        X = df[feature_cols]
        y = df["target"]
        return X, y

    def train(self, df: pd.DataFrame) -> dict:
        """Train the classifier on historical data. Returns training metrics."""
        if len(df) < 100:
            logger.warning(f"Too little data to train model for {self.symbol}: {len(df)} rows")
            return {}

        X, y = self.prepare_features(df)

        if len(X) < 80:
            logger.warning(f"Insufficient training samples after feature prep: {len(X)}")
            return {}

        # Time-series split for validation
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Build base model
        if XGB_AVAILABLE:
            base_clf = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            base_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )

        base_clf.fit(X_train_scaled, y_train)

        # Build LSTM model
        lstm_clf = LSTMPredictorModel(
            input_dim=X_train_scaled.shape[1],
            sequence_length=10,
            epochs=15,
        )
        lstm_clf.fit(X_train_scaled, y_train)

        # Evaluate ensemble validation predictions
        base_probas = base_clf.predict_proba(X_test_scaled)[:, 1]
        
        # Concatenate train/test to generate continuous sequence context for LSTM test predictions
        X_full_scaled = np.concatenate([X_train_scaled, X_test_scaled], axis=0)
        lstm_probas = []
        for i in range(len(X_test_scaled)):
            seq_end_idx = len(X_train_scaled) + i + 1
            X_seq = X_full_scaled[:seq_end_idx]
            p_lstm = lstm_clf.predict_proba(X_seq)[0, 1]
            lstm_probas.append(p_lstm)
            
        lstm_probas = np.array(lstm_probas)
        
        # Average probability predictions
        y_proba = (base_probas + lstm_probas) / 2.0
        y_pred = (y_proba >= 0.5).astype(int)

        model_name = f"Ensemble ({'XGBoost' if XGB_AVAILABLE else 'RandomForest'}+LSTM)"

        self.model_metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "class_balance": float(y_train.mean()),
            "trained_at": datetime.utcnow().isoformat(),
            "model_type": model_name,
        }

        # Feature importance from base model
        if hasattr(base_clf, "feature_importances_"):
            self.feature_importance = dict(zip(X.columns.tolist(), 
                                               base_clf.feature_importances_.tolist()))

        # Save both models in dictionary
        self.classifier = {
            "base": base_clf,
            "lstm": lstm_clf,
        }

        self.is_trained = True
        self._save_model()
        logger.info(f"Trained ensemble model for {self.symbol}: accuracy={self.model_metrics['accuracy']:.3f}")
        return self.model_metrics

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Make a prediction on the most recent data point.
        Returns: direction (UP/DOWN), confidence, and raw probabilities.
        """
        if not self.is_trained or self.classifier is None:
            return {"direction": "NEUTRAL", "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5}

        df = df.copy()

        # Compute same auxiliary features as training
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_10"] = df["close"].pct_change(10)
        df["return_1_lag1"] = df["return_1"].shift(1)
        df["return_1_lag2"] = df["return_1"].shift(2)
        df["return_1_lag3"] = df["return_1"].shift(3)
        df["volatility_5"] = df["return_1"].rolling(5).std()
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_20"] = df["return_1"].rolling(20).std()
        df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["volume_momentum"] = df["volume"].pct_change(5)

        # Broad Market Context (Index features - Nifty 50 for Indian stocks, SPY for US stocks)
        index_ticker = "^NSEI" if (self.symbol.endswith(".NS") or self.symbol == "^NSEI") else "SPY"
        if self.symbol != index_ticker:
            try:
                from app.data.data_loader import get_ohlcv
                index_df = get_ohlcv(index_ticker, timeframe="1d")
                if not index_df.empty:
                    index_df = index_df.rename(columns={
                        "close": "index_close",
                        "volume": "index_volume"
                    })
                    index_df["index_return_1"] = index_df["index_close"].pct_change(1)
                    index_df["index_above_sma50"] = (index_df["index_close"] > index_df["index_close"].rolling(50).mean()).astype(int)
                    df = df.join(index_df[["index_close", "index_return_1", "index_above_sma50"]], how="left")
            except Exception as e:
                logger.error(f"Error merging index features in predict: {e}")

        additional_cols = [
            "return_1", "return_5", "return_10", 
            "return_1_lag1", "return_1_lag2", "return_1_lag3",
            "volatility_5", "volatility_10", "volatility_20",
            "volume_sma_ratio", "volume_momentum",
            "index_close", "index_return_1", "index_above_sma50"
        ]

        feature_cols = [c for c in FEATURE_COLUMNS + additional_cols if c in df.columns]

        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        df_clean = df[feature_cols].dropna()
        if df_clean.empty:
            return {"direction": "NEUTRAL", "confidence": 0.5, "up_prob": 0.5, "down_prob": 0.5}

        X_all = df_clean[feature_cols]
        X_scaled_all = self.scaler.transform(X_all)

        if isinstance(self.classifier, dict) and "base" in self.classifier and "lstm" in self.classifier:
            base_clf = self.classifier["base"]
            lstm_clf = self.classifier["lstm"]

            # Predict base probability on last step
            X_scaled_last = X_scaled_all[-1:]
            proba_base = base_clf.predict_proba(X_scaled_last)[0]

            # Predict LSTM probability on all scaled sequence
            proba_lstm = lstm_clf.predict_proba(X_scaled_all)[0]

            proba = (proba_base + proba_lstm) / 2.0
            model_name = f"Ensemble ({'XGBoost' if XGB_AVAILABLE else 'RandomForest'}+LSTM)"
        else:
            # Fallback to single model
            X_scaled_last = X_scaled_all[-1:]
            proba = self.classifier.predict_proba(X_scaled_last)[0]
            model_name = "XGBoost" if XGB_AVAILABLE else "RandomForest"

        up_prob = float(proba[1])
        down_prob = float(proba[0])
        direction = "UP" if up_prob >= 0.5 else "DOWN"
        confidence = max(up_prob, down_prob)

        return {
            "direction": direction,
            "confidence": confidence,
            "up_prob": up_prob,
            "down_prob": down_prob,
            "threshold": settings.AI_CONFIDENCE_THRESHOLD,
            "signal": direction if confidence >= settings.AI_CONFIDENCE_THRESHOLD else "NEUTRAL",
            "model_type": model_name,
        }

    def get_status(self) -> dict:
        if isinstance(self.classifier, dict) and "base" in self.classifier:
            model_name = f"Ensemble ({'XGBoost' if XGB_AVAILABLE else 'RandomForest'}+LSTM)"
        else:
            model_name = "XGBoost" if XGB_AVAILABLE else "RandomForest"

        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "is_trained": self.is_trained,
            "metrics": self.model_metrics,
            "feature_importance": dict(sorted(
                self.feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]),
            "model_type": model_name,
        }


# Global predictor registry
_predictors: dict = {}


def get_predictor(symbol: str, horizon: int = 5) -> AIPredictor:
    """Get or create a predictor for the given symbol."""
    key = f"{symbol}:{horizon}"
    if key not in _predictors:
        _predictors[key] = AIPredictor(symbol, horizon)
    return _predictors[key]


def train_all_predictors(symbol_data: dict) -> dict:
    """Train predictors for all symbols. symbol_data: {symbol: DataFrame}"""
    results = {}
    for symbol, df in symbol_data.items():
        predictor = get_predictor(symbol)
        metrics = predictor.train(df)
        results[symbol] = metrics
    return results
