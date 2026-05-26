"""
Risk Management System
Controls position sizing, stop-loss/take-profit placement,
daily loss limits, max open positions, and volatility filters.
"""
import logging
from typing import Optional, Tuple
from datetime import datetime, date

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Portfolio, Position, RiskLog

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Central risk management system.
    
    Key controls:
    - Max risk per trade: 1% of capital (configurable)
    - ATR-based stop loss and take profit
    - Max daily loss: 3% (halt trading if exceeded)
    - Max concurrent open positions: 5
    - Volatility filter: skip trades when ATR spike is excessive
    """

    def __init__(
        self,
        max_risk_per_trade_pct: float = None,
        max_daily_loss_pct: float = None,
        max_open_positions: int = None,
        stop_loss_atr_mult: float = None,
        take_profit_atr_mult: float = None,
    ):
        self.max_risk_per_trade_pct = max_risk_per_trade_pct or settings.MAX_RISK_PER_TRADE_PCT
        self.max_daily_loss_pct = max_daily_loss_pct or settings.MAX_DAILY_LOSS_PCT
        self.max_open_positions = max_open_positions or settings.MAX_OPEN_POSITIONS
        self.stop_loss_atr_mult = stop_loss_atr_mult or settings.STOP_LOSS_ATR_MULTIPLIER
        self.take_profit_atr_mult = take_profit_atr_mult or settings.TAKE_PROFIT_ATR_MULTIPLIER

    def check_daily_loss_limit(self, portfolio: Portfolio) -> Tuple[bool, str]:
        """
        Check if the portfolio has hit the maximum daily loss threshold.
        Returns (is_allowed, reason).
        """
        if portfolio.trading_halted:
            return False, "Trading is halted: daily loss limit was previously hit."

        if portfolio.daily_pnl_start_value <= 0:
            return True, "OK"

        daily_loss_pct = portfolio.daily_pnl / portfolio.daily_pnl_start_value
        if daily_loss_pct <= -self.max_daily_loss_pct:
            return False, (
                f"Daily loss limit reached: {daily_loss_pct:.2%} loss "
                f"(limit: {self.max_daily_loss_pct:.2%})"
            )

        return True, "OK"

    def check_max_positions(self, portfolio: Portfolio) -> Tuple[bool, str]:
        """Check if we can open more positions."""
        open_count = len(portfolio.positions)
        if open_count >= self.max_open_positions:
            return False, f"Max open positions reached ({open_count}/{self.max_open_positions})"
        return True, "OK"

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        atr: float,
        side: str = "long",
    ) -> dict:
        """
        Calculate position size using ATR-based risk management.
        
        Formula:
            Risk Amount = portfolio_value * max_risk_per_trade_pct
            Stop Distance = ATR * stop_loss_atr_mult
            Shares = Risk Amount / Stop Distance
            
        Returns dict with qty, stop_loss, take_profit, risk_amount.
        """
        if entry_price <= 0 or atr <= 0:
            return {"qty": 0, "stop_loss": None, "take_profit": None, "risk_amount": 0}

        risk_amount = portfolio_value * self.max_risk_per_trade_pct
        stop_distance = atr * self.stop_loss_atr_mult
        tp_distance = atr * self.take_profit_atr_mult

        qty = risk_amount / stop_distance
        qty = max(0, qty)

        # Ensure we don't spend more than we have
        max_affordable_qty = (portfolio_value * 0.25) / entry_price  # max 25% in single position
        qty = min(qty, max_affordable_qty)

        if side == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        return {
            "qty": round(qty, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(stop_distance, 4),
            "risk_reward_ratio": round(tp_distance / stop_distance, 2) if stop_distance > 0 else 0,
        }

    def check_volatility_filter(self, atr: float, avg_atr: float) -> Tuple[bool, str]:
        """
        Volatility filter: avoid trading when ATR is more than 2x the average.
        Indicates extreme market stress.
        """
        if avg_atr <= 0:
            return True, "OK"

        ratio = atr / avg_atr
        if ratio > 2.5:
            return False, f"Volatility too high: ATR is {ratio:.1f}x average"
        return True, "OK"

    def check_position_stop_loss(
        self, position: Position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if a position should be stopped out."""
        if position.stop_loss is None:
            return False, ""

        if position.side == "long" and current_price <= position.stop_loss:
            return True, f"Stop loss hit: price {current_price:.2f} <= SL {position.stop_loss:.2f}"
        elif position.side == "short" and current_price >= position.stop_loss:
            return True, f"Stop loss hit: price {current_price:.2f} >= SL {position.stop_loss:.2f}"

        return False, ""

    def check_position_take_profit(
        self, position: Position, current_price: float
    ) -> Tuple[bool, str]:
        """Check if a position should take profit."""
        if position.take_profit is None:
            return False, ""

        if position.side == "long" and current_price >= position.take_profit:
            return True, f"Take profit hit: price {current_price:.2f} >= TP {position.take_profit:.2f}"
        elif position.side == "short" and current_price <= position.take_profit:
            return True, f"Take profit hit: price {current_price:.2f} <= TP {position.take_profit:.2f}"

        return False, ""

    def validate_trade(
        self,
        portfolio: Portfolio,
        symbol: str,
        entry_price: float,
        qty: float,
        atr: float = None,
        avg_atr: float = None,
    ) -> Tuple[bool, str]:
        """
        Run all pre-trade risk checks.
        Returns (is_valid, reason).
        """
        # 1. Daily loss limit
        allowed, reason = self.check_daily_loss_limit(portfolio)
        if not allowed:
            return False, reason

        # 2. Max open positions
        allowed, reason = self.check_max_positions(portfolio)
        if not allowed:
            return False, reason

        # 3. Sufficient capital
        trade_value = entry_price * qty
        if trade_value > portfolio.cash:
            return False, f"Insufficient cash: need {trade_value:.2f}, have {portfolio.cash:.2f}"

        # 4. Volatility filter
        if atr and avg_atr:
            allowed, reason = self.check_volatility_filter(atr, avg_atr)
            if not allowed:
                return False, reason

        return True, "OK"

    def log_risk_event(
        self,
        db: Session,
        event_type: str,
        symbol: str = None,
        detail: str = None,
        portfolio_value: float = None,
    ):
        """Persist a risk management event to the database."""
        entry = RiskLog(
            event_type=event_type,
            symbol=symbol,
            detail=detail,
            portfolio_value=portfolio_value,
        )
        db.add(entry)
        db.commit()
        logger.warning(f"[RiskLog] {event_type}: {detail}")
