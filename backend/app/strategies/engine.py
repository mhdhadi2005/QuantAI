"""
Strategy Engine
Evaluates multiple trading strategies and generates buy/sell/hold signals.

Strategies:
1. Momentum - EMA crossovers, RSI, MACD
2. Mean Reversion - RSI oversold/overbought, Bollinger Band touches
3. Breakout - Price breaks above resistance or below support
4. AI Confidence - Only trade when AI model exceeds confidence threshold
"""
import logging
from typing import Optional, List, Dict
from datetime import datetime

import pandas as pd
import numpy as np

from app.config import settings
from app.data.indicators import compute_support_resistance

logger = logging.getLogger(__name__)


class SignalResult:
    """Structured result from a strategy evaluation."""

    def __init__(
        self,
        symbol: str,
        action: str,  # "buy" | "sell" | "hold"
        strategy: str,
        confidence: float,
        indicators: dict,
        reason: str = "",
    ):
        self.symbol = symbol
        self.action = action
        self.strategy = strategy
        self.confidence = confidence
        self.indicators = indicators
        self.reason = reason
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strategy": self.strategy,
            "confidence": self.confidence,
            "indicators": self.indicators,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


def _safe_float(series_or_val, default=None):
    """Safely extract float from series or scalar."""
    try:
        if isinstance(series_or_val, pd.Series):
            return float(series_or_val.iloc[-1])
        return float(series_or_val)
    except Exception:
        return default


def strategy_momentum(df: pd.DataFrame, symbol: str) -> SignalResult:
    """
    Momentum Strategy with Volume confirmation:
    BUY when: EMA9 > EMA21 AND RSI > 50 AND MACD above signal line AND price > VWAP AND OBV trend is up
    SELL when: EMA9 < EMA21 AND RSI < 50 AND MACD below signal line AND price < VWAP AND OBV trend is down
    """
    if df.empty or len(df) < 30:
        return SignalResult(symbol, "hold", "momentum", 0.5, {}, "Insufficient data")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    ema9 = _safe_float(last.get("ema_9", 0))
    ema21 = _safe_float(last.get("ema_21", 0))
    rsi = _safe_float(last.get("rsi_14", 50))
    macd = _safe_float(last.get("macd", 0))
    macd_signal = _safe_float(last.get("macd_signal", 0))
    macd_hist = _safe_float(last.get("macd_histogram", 0))
    prev_macd_hist = _safe_float(prev.get("macd_histogram", 0))

    close_price = _safe_float(last.get("close", 0))
    vwap = _safe_float(last.get("vwap", close_price))
    obv = _safe_float(last.get("obv", 0))
    obv_ma = float(df["obv"].tail(5).mean()) if len(df) >= 5 else obv

    # Volume filters
    volume_confirm_buy = (close_price >= vwap) and (obv >= obv_ma)
    volume_confirm_sell = (close_price <= vwap) and (obv <= obv_ma)

    # MACD crossover detection
    macd_crossover_up = prev_macd_hist < 0 and macd_hist > 0
    macd_crossover_down = prev_macd_hist > 0 and macd_hist < 0

    bullish_conditions = [
        ema9 > ema21,
        rsi > 50,
        macd > macd_signal,
    ]
    bearish_conditions = [
        ema9 < ema21,
        rsi < 50,
        macd < macd_signal,
    ]

    bull_score = sum(bullish_conditions) / len(bullish_conditions)
    bear_score = sum(bearish_conditions) / len(bearish_conditions)

    indicators = {
        "ema_9": ema9, "ema_21": ema21, "rsi_14": rsi,
        "macd": macd, "macd_signal": macd_signal, "macd_histogram": macd_hist,
        "vwap": vwap, "obv": obv, "volume_confirm": volume_confirm_buy if bull_score > bear_score else volume_confirm_sell
    }

    if (bull_score >= 0.66 or macd_crossover_up) and volume_confirm_buy:
        confidence = min(0.5 + bull_score * 0.4 + (0.1 if macd_crossover_up else 0), 0.95)
        return SignalResult(symbol, "buy", "momentum", confidence, indicators,
                          f"Bull score: {bull_score:.2f}, MACD xover: {macd_crossover_up}, Volume confirmed")
    elif (bear_score >= 0.66 or macd_crossover_down) and volume_confirm_sell:
        confidence = min(0.5 + bear_score * 0.4 + (0.1 if macd_crossover_down else 0), 0.95)
        return SignalResult(symbol, "sell", "momentum", confidence, indicators,
                          f"Bear score: {bear_score:.2f}, MACD xover: {macd_crossover_down}, Volume confirmed")

    return SignalResult(symbol, "hold", "momentum", 0.5, indicators, "No clear momentum signal or volume confirmation failed")


def strategy_mean_reversion(df: pd.DataFrame, symbol: str) -> SignalResult:
    """
    Mean Reversion Strategy with Stochastic RSI check:
    BUY when: RSI < 30 (oversold) AND close < BB lower band AND Stochastic RSI %K turning upward (%K > %D)
    SELL when: RSI > 70 (overbought) AND close > BB upper band AND Stochastic RSI %K turning downward (%K < %D)
    """
    if df.empty or len(df) < 25:
        return SignalResult(symbol, "hold", "mean_reversion", 0.5, {}, "Insufficient data")

    last = df.iloc[-1]

    rsi = _safe_float(last.get("rsi_14", 50))
    bb_pct = _safe_float(last.get("bb_pct", 0.5))  # 0 = at lower band, 1 = at upper band
    close = _safe_float(last.get("close", 0))
    bb_lower = _safe_float(last.get("bb_lower", 0))
    bb_upper = _safe_float(last.get("bb_upper", 0))
    stoch_k = _safe_float(last.get("stoch_rsi_k", 50))
    stoch_d = _safe_float(last.get("stoch_rsi_d", 50))

    # Crossover/turnaround validation
    stoch_confirm_buy = stoch_k > stoch_d
    stoch_confirm_sell = stoch_k < stoch_d

    indicators = {
        "rsi_14": rsi, "bb_pct": bb_pct, "close": close,
        "bb_lower": bb_lower, "bb_upper": bb_upper,
        "stoch_rsi_k": stoch_k, "stoch_rsi_d": stoch_d,
    }

    # Oversold - potential buy
    if rsi < 30 and bb_pct < 0.1 and stoch_confirm_buy:
        confidence = 0.5 + (30 - rsi) / 60 + max(0, 0.1 - bb_pct) * 2
        confidence = min(confidence, 0.92)
        return SignalResult(symbol, "buy", "mean_reversion", confidence, indicators,
                          f"Oversold: RSI={rsi:.1f}, BB%={bb_pct:.2f}, Stoch confirmed")

    # Overbought - potential sell
    if rsi > 70 and bb_pct > 0.9 and stoch_confirm_sell:
        confidence = 0.5 + (rsi - 70) / 60 + max(0, bb_pct - 0.9) * 2
        confidence = min(confidence, 0.92)
        return SignalResult(symbol, "sell", "mean_reversion", confidence, indicators,
                          f"Overbought: RSI={rsi:.1f}, BB%={bb_pct:.2f}, Stoch confirmed")

    # Mild oversold
    if rsi < 40 and stoch_k < 20 and stoch_confirm_buy:
        return SignalResult(symbol, "buy", "mean_reversion", 0.58, indicators,
                          f"Mild oversold: RSI={rsi:.1f}, StochRSI={stoch_k:.1f}, Stoch confirmed")

    return SignalResult(symbol, "hold", "mean_reversion", 0.5, indicators, "No reversion signal or Stochastic confirmation failed")


def strategy_breakout(df: pd.DataFrame, symbol: str) -> SignalResult:
    """
    Breakout Strategy with SMA 50 Filter:
    BUY when price breaks above 20-day resistance with above-average volume AND price > SMA 50
    SELL when price breaks below 20-day support with above-average volume AND price < SMA 50
    """
    if df.empty or len(df) < 25:
        return SignalResult(symbol, "hold", "breakout", 0.5, {}, "Insufficient data")

    sr = compute_support_resistance(df, window=20)
    support = sr.get("support")
    resistance = sr.get("resistance")

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    close = _safe_float(last.get("close", 0))
    prev_close = _safe_float(prev.get("close", 0))
    volume = _safe_float(last.get("volume", 0))
    avg_volume = float(df["volume"].tail(20).mean())
    volume_ratio = volume / max(avg_volume, 1)

    sma50 = _safe_float(last.get("sma_50", close))
    trend_confirm_buy = close > sma50
    trend_confirm_sell = close < sma50

    indicators = {
        "close": close, "resistance": resistance, "support": support,
        "volume_ratio": volume_ratio, "prev_close": prev_close, "sma_50": sma50
    }

    if resistance and close > resistance and prev_close <= resistance and volume_ratio > 1.2 and trend_confirm_buy:
        confidence = min(0.55 + (volume_ratio - 1) * 0.2, 0.9)
        return SignalResult(symbol, "buy", "breakout", confidence, indicators,
                          f"Breakout above resistance {resistance:.2f}, vol ratio {volume_ratio:.2f}x, Trend confirmed")

    if support and close < support and prev_close >= support and volume_ratio > 1.2 and trend_confirm_sell:
        confidence = min(0.55 + (volume_ratio - 1) * 0.2, 0.9)
        return SignalResult(symbol, "sell", "breakout", confidence, indicators,
                          f"Breakdown below support {support:.2f}, vol ratio {volume_ratio:.2f}x, Trend confirmed")

    return SignalResult(symbol, "hold", "breakout", 0.5, indicators, "No breakout detected or trend confirmation failed")


def strategy_ai_confidence(df: pd.DataFrame, symbol: str, ai_prediction: dict) -> SignalResult:
    """
    AI Confidence Strategy:
    Trade only when AI model confidence exceeds the configured threshold.
    """
    if not ai_prediction or ai_prediction.get("direction") == "NEUTRAL":
        return SignalResult(symbol, "hold", "ai_confidence", 0.5, {}, "AI: NEUTRAL")

    direction = ai_prediction.get("direction", "NEUTRAL")
    confidence = ai_prediction.get("confidence", 0.0)
    threshold = settings.AI_CONFIDENCE_THRESHOLD

    close = _safe_float(df.iloc[-1].get("close", 0)) if not df.empty else 0

    indicators = {
        "ai_direction": direction,
        "ai_confidence": confidence,
        "ai_threshold": threshold,
        "close": close,
    }

    if confidence >= threshold:
        action = "buy" if direction == "UP" else "sell"
        return SignalResult(symbol, action, "ai_confidence", confidence, indicators,
                          f"AI: {direction} with {confidence:.1%} confidence (threshold: {threshold:.1%})")

    return SignalResult(symbol, "hold", "ai_confidence", confidence, indicators,
                       f"AI: {direction} but below threshold ({confidence:.1%} < {threshold:.1%})")


def strategy_news_sentiment(sentiment_score: float, symbol: str) -> SignalResult:
    """
    News Sentiment Strategy:
    BUY when sentiment score > 0.15 (bullish news)
    SELL when sentiment score < -0.15 (bearish news)
    """
    indicators = {"news_sentiment": sentiment_score}
    if sentiment_score > 0.15:
        # Scale confidence based on sentiment strength, capped at 0.85
        confidence = min(0.5 + sentiment_score * 0.35, 0.85)
        return SignalResult(symbol, "buy", "news_sentiment", confidence, indicators,
                            f"Bullish news sentiment: {sentiment_score:.3f}")
    elif sentiment_score < -0.15:
        confidence = min(0.5 + abs(sentiment_score) * 0.35, 0.85)
        return SignalResult(symbol, "sell", "news_sentiment", confidence, indicators,
                            f"Bearish news sentiment: {sentiment_score:.3f}")

    return SignalResult(symbol, "hold", "news_sentiment", 0.5, indicators,
                        f"Neutral news sentiment: {sentiment_score:.3f}")


def aggregate_signals(signals: List[SignalResult], momentum_score: float = 0.0) -> dict:
    """
    Aggregate multiple strategy signals into a single consensus action.
    Uses dynamic, market-regime adaptive weights.
    """
    if not signals:
        return {"action": "hold", "consensus_confidence": 0.5, "votes": {}}

    # Define baseline weights
    base_weights = {
        "ai_confidence": 2.5,
        "news_sentiment": 1.5,
        "momentum": 1.2,
        "mean_reversion": 1.0,
        "breakout": 1.0
    }

    # Dynamic Weight Adjustments based on trend strength (momentum_score)
    abs_momentum = abs(momentum_score)
    if abs_momentum > 0.35:  # Strong trending regime
        base_weights["momentum"] = 2.0
        base_weights["breakout"] = 1.5
        base_weights["mean_reversion"] = 0.2
    elif abs_momentum <= 0.15:  # Sideways ranging regime
        base_weights["momentum"] = 0.3
        base_weights["breakout"] = 0.2
        base_weights["mean_reversion"] = 2.0

    net_score = 0.0
    active_weight = 0.0
    votes = {"buy": [], "sell": [], "hold": []}

    for sig in signals:
        weight = base_weights.get(sig.strategy, 1.0)
        votes[sig.action].append({
            "strategy": sig.strategy,
            "confidence": sig.confidence,
            "weight": weight
        })

        if sig.action == "buy":
            net_score += weight * sig.confidence
            active_weight += weight
        elif sig.action == "sell":
            net_score -= weight * sig.confidence
            active_weight += weight

    if active_weight > 0:
        # Normalized score ranges from -1.0 (strong sell consensus) to +1.0 (strong buy consensus)
        consensus_score = net_score / active_weight
        
        # Determine consensus action based on score thresholds
        if consensus_score > 0.40:
            action = "buy"
            consensus_confidence = float(consensus_score)
        elif consensus_score < -0.40:
            action = "sell"
            consensus_confidence = float(abs(consensus_score))
        else:
            action = "hold"
            consensus_confidence = float(1.0 - abs(consensus_score))
    else:
        action = "hold"
        consensus_confidence = 0.5

    return {
        "action": action,
        "consensus_confidence": consensus_confidence,
        "votes": votes,
        "net_score": float(net_score),
        "active_weight": float(active_weight),
    }


class StrategyEngine:
    """
    Main strategy engine that orchestrates all strategies
    and produces aggregated trading signals.
    """

    def __init__(self):
        self.enabled_strategies = {
            "momentum": settings.STRATEGY_MOMENTUM_ENABLED,
            "mean_reversion": settings.STRATEGY_MEAN_REVERSION_ENABLED,
            "breakout": settings.STRATEGY_BREAKOUT_ENABLED,
            "ai_confidence": settings.STRATEGY_AI_ENABLED,
            "news_sentiment": settings.STRATEGY_NEWS_SENTIMENT_ENABLED,
        }

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        ai_prediction: Optional[dict] = None,
        news_sentiment: float = 0.0,
    ) -> dict:
        """
        Run all enabled strategies and return aggregated signal.
        """
        signals = []

        if self.enabled_strategies.get("momentum"):
            signals.append(strategy_momentum(df, symbol))

        if self.enabled_strategies.get("mean_reversion"):
            signals.append(strategy_mean_reversion(df, symbol))

        if self.enabled_strategies.get("breakout"):
            signals.append(strategy_breakout(df, symbol))

        if self.enabled_strategies.get("ai_confidence") and ai_prediction:
            signals.append(strategy_ai_confidence(df, symbol, ai_prediction))

        if self.enabled_strategies.get("news_sentiment"):
            signals.append(strategy_news_sentiment(news_sentiment, symbol))

        momentum_val = float(df.iloc[-1].get("momentum_score", 0.0)) if not df.empty else 0.0
        consensus = aggregate_signals(signals, momentum_score=momentum_val)
        last_close = _safe_float(df.iloc[-1].get("close", 0)) if not df.empty else 0

        return {
            "symbol": symbol,
            "price": last_close,
            "individual_signals": [s.to_dict() for s in signals],
            "consensus": consensus,
            "ai_prediction": ai_prediction,
            "news_sentiment": news_sentiment,
            "timestamp": datetime.utcnow().isoformat(),
        }
