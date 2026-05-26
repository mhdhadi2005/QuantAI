"""
Data Loader: Fetches OHLCV data from yfinance, caches locally in DB,
handles rate limiting and data normalization.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import yfinance as yf
from sqlalchemy.orm import Session

from app.db.models import HistoricalPrice
from app.config import settings

logger = logging.getLogger(__name__)

# Simple in-memory cache: {(symbol, timeframe): (timestamp, DataFrame)}
_cache: Dict[str, tuple] = {}
CACHE_TTL_SECONDS = 60  # 1 minute for live data

# Live price feed simulation tracking
_live_prices: Dict[str, float] = {}


def set_live_price(symbol: str, price: float):
    _live_prices[symbol] = price


def _cache_key(symbol: str, timeframe: str, lookback_days: int) -> str:
    return f"{symbol}:{timeframe}:{lookback_days}"


def _is_cache_valid(key: str) -> bool:
    if key not in _cache:
        return False
    cached_at, _ = _cache[key]
    return (time.time() - cached_at) < CACHE_TTL_SECONDS


def get_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    lookback_days: int = None,
    db: Optional[Session] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a symbol.
    Returns DataFrame with columns: open, high, low, close, volume (lowercase).
    Index is DatetimeIndex.
    """
    lookback_days = lookback_days or settings.LOOKBACK_DAYS
    key = _cache_key(symbol, timeframe, lookback_days)

    if _is_cache_valid(key):
        logger.debug(f"Cache hit for {symbol} {timeframe}")
        _, df = _cache[key]
        return df.copy()

    logger.info(f"Fetching {symbol} ({timeframe}, {lookback_days}d)")

    yf_period_map = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "1h": "730d",
        "1d": f"{lookback_days}d",
    }
    yf_interval_map = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    period = yf_period_map.get(timeframe, f"{lookback_days}d")
    interval = yf_interval_map.get(timeframe, "1d")

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert("UTC")

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Cache the result
        _cache[key] = (time.time(), df)
        logger.info(f"Loaded {len(df)} bars for {symbol} ({timeframe})")
        return df.copy()

    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def get_latest_price(symbol: str) -> float:
    """Get the most recent close price for a symbol. Returns live simulated price if available."""
    if symbol in _live_prices and _live_prices[symbol] > 0:
        return _live_prices[symbol]
    df = get_ohlcv(symbol, "1d", lookback_days=5)
    if df.empty:
        return 0.0
    return float(df["close"].iloc[-1])


def get_multi_symbol_data(
    symbols: List[str],
    timeframe: str = "1d",
    lookback_days: int = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple symbols at once."""
    result = {}
    for symbol in symbols:
        df = get_ohlcv(symbol, timeframe, lookback_days)
        if not df.empty:
            result[symbol] = df
        time.sleep(0.1)  # Rate limiting - be gentle with yfinance
    return result


def get_fundamental_data(symbol: str) -> dict:
    """Fetch fundamental data for a ticker (P/E, EPS, Market Cap, etc.)."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "pe_ratio": info.get("trailingPE", None),
            "eps": info.get("trailingEps", None),
            "market_cap": info.get("marketCap", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "earnings_growth": info.get("earningsGrowth", None),
            "sector": info.get("sector", None),
            "industry": info.get("industry", None),
            "beta": info.get("beta", None),
        }
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {"symbol": symbol}


def invalidate_cache(symbol: str = None, timeframe: str = None):
    """Clear the in-memory cache for a specific symbol or all symbols."""
    if symbol and timeframe:
        keys_to_remove = [k for k in _cache if k.startswith(f"{symbol}:{timeframe}:")]
        for k in keys_to_remove:
            _cache.pop(k, None)
    elif symbol:
        keys_to_remove = [k for k in _cache if k.startswith(f"{symbol}:")]
        for k in keys_to_remove:
            _cache.pop(k, None)
    else:
        _cache.clear()
