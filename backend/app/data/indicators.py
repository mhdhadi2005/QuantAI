"""
Technical Indicators Engine
Computes all required indicators using pandas/numpy and the 'ta' library.
All functions accept a DataFrame with columns: open, high, low, close, volume
and return a new DataFrame with all indicator columns appended.
"""
import logging
import pandas as pd
import numpy as np

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full suite of technical indicators and append them to the DataFrame.
    
    Indicators computed:
    - SMA (20, 50, 200)
    - EMA (9, 21, 55)
    - RSI (14)
    - MACD (12, 26, 9)
    - Bollinger Bands (20, 2)
    - ATR (14)
    - VWAP (rolling)
    - Stochastic RSI (14, 3, 3)
    - OBV
    """
    if df.empty or len(df) < 30:
        logger.warning("Insufficient data for indicator computation")
        return df

    df = df.copy()

    try:
        # =====================================================
        # TREND INDICATORS
        # =====================================================
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_55"] = df["close"].ewm(span=55, adjust=False).mean()

        # =====================================================
        # MOMENTUM INDICATORS
        # =====================================================
        # RSI
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Stochastic RSI
        rsi = df["rsi_14"]
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        stoch_rsi_raw = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)
        df["stoch_rsi_k"] = stoch_rsi_raw.rolling(window=3).mean() * 100
        df["stoch_rsi_d"] = df["stoch_rsi_k"].rolling(window=3).mean()

        # =====================================================
        # VOLATILITY INDICATORS
        # =====================================================
        # Bollinger Bands
        bb_sma = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = bb_sma + 2 * bb_std
        df["bb_middle"] = bb_sma
        df["bb_lower"] = bb_sma - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = true_range.ewm(com=13, adjust=False).mean()

        # =====================================================
        # VOLUME INDICATORS
        # =====================================================
        # OBV (On-Balance Volume)
        direction = df["close"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        df["obv"] = (direction * df["volume"]).cumsum()

        # VWAP (Rolling 14-period approximation)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).rolling(window=14).sum() / df["volume"].rolling(window=14).sum()

        # =====================================================
        # DERIVED SIGNALS
        # =====================================================
        # Trend direction
        df["trend_up"] = (df["ema_9"] > df["ema_21"]).astype(int)
        df["above_sma_50"] = (df["close"] > df["sma_50"]).astype(int)

        # Price position
        df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]

        # Momentum score: combines RSI, MACD, and trend
        rsi_norm = (df["rsi_14"] - 50) / 50  # Range: -1 to +1
        macd_norm = df["macd_histogram"] / (df["atr_14"] + 1e-10)
        df["momentum_score"] = (rsi_norm * 0.5 + macd_norm.clip(-1, 1) * 0.3 + df["trend_up"] * 0.2)

        logger.debug(f"Computed {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} indicators")

    except Exception as e:
        logger.error(f"Error computing indicators: {e}")

    return df


def get_latest_indicators(df: pd.DataFrame) -> dict:
    """Return the most recent row of indicator values as a dictionary."""
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {col: float(row[col]) if not pd.isna(row[col]) else None for col in df.columns}


def compute_support_resistance(df: pd.DataFrame, window: int = 20) -> dict:
    """
    Compute basic support and resistance levels.
    Uses rolling min/max over the lookback window.
    """
    if df.empty or len(df) < window:
        return {"support": None, "resistance": None}

    recent = df["close"].tail(window * 2)
    support = float(recent.rolling(window).min().iloc[-1])
    resistance = float(recent.rolling(window).max().iloc[-1])
    return {"support": support, "resistance": resistance}
