"""
Trade Execution Brokers
Provides abstract broker interface with two implementations:
1. SimulatedBroker - Paper trading with local state tracking (no API keys needed)
2. AlpacaBroker - Real/paper trading via Alpaca API
"""
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List

import httpx
from sqlalchemy.orm import Session

from app.db.models import Portfolio, Position, Trade, Signal, EquitySnapshot
from app.config import settings

logger = logging.getLogger(__name__)

COMMISSION_PER_TRADE = 0.0  # $0 commission (simulated)
SLIPPAGE_PCT = 0.0001  # 0.01% slippage simulation


class BaseBroker(ABC):
    """Abstract base class for all broker implementations."""

    @abstractmethod
    def place_order(
        self, db: Session, portfolio: Portfolio, symbol: str,
        side: str, qty: float, order_type: str = "market",
        limit_price: float = None, stop_price: float = None,
        stop_loss: float = None, take_profit: float = None,
        strategy: str = None, signal_id: int = None,
    ) -> dict:
        """Place a buy or sell order. Returns order result dict."""
        pass

    @abstractmethod
    def get_positions(self, db: Session, portfolio: Portfolio) -> List[dict]:
        """Get all current open positions."""
        pass

    @abstractmethod
    def close_position(
        self, db: Session, portfolio: Portfolio,
        position: Position, current_price: float, reason: str = ""
    ) -> dict:
        """Close an existing position."""
        pass

    @abstractmethod
    def get_account_info(self, db: Session, portfolio: Portfolio) -> dict:
        """Get account balance and info."""
        pass


class SimulatedBroker(BaseBroker):
    """
    Paper trading broker that simulates execution locally.
    Uses yfinance for live price feeds.
    All state is stored in the local database.
    """

    def place_order(
        self, db: Session, portfolio: Portfolio, symbol: str,
        side: str, qty: float, order_type: str = "market",
        limit_price: float = None, stop_price: float = None,
        stop_loss: float = None, take_profit: float = None,
        strategy: str = None, signal_id: int = None,
    ) -> dict:
        """Simulate order execution with slippage."""
        from app.data.data_loader import get_latest_price

        current_price = get_latest_price(symbol)
        if current_price <= 0:
            return {"success": False, "error": f"Could not fetch price for {symbol}"}

        # Apply slippage
        if side == "buy":
            execution_price = current_price * (1 + settings.SIM_SLIPPAGE_PCT)
        else:
            execution_price = current_price * (1 - settings.SIM_SLIPPAGE_PCT)

        trade_value = execution_price * qty
        commission = settings.SIM_COMMISSION

        if side == "buy":
            if portfolio.cash < trade_value + commission:
                return {"success": False, "error": "Insufficient cash"}

            # Deduct cash
            portfolio.cash -= (trade_value + commission)

            # Check if we already have a position
            existing = db.query(Position).filter(
                Position.portfolio_id == portfolio.id,
                Position.symbol == symbol
            ).first()

            if existing:
                # Average in
                total_value = existing.entry_price * existing.qty + execution_price * qty
                new_qty = existing.qty + qty
                existing.entry_price = total_value / new_qty
                existing.qty = new_qty
                existing.current_price = execution_price
                if stop_loss:
                    existing.stop_loss = stop_loss
                if take_profit:
                    existing.take_profit = take_profit
                existing.unrealized_pnl = (execution_price - existing.entry_price) * existing.qty
                db.add(existing)
            else:
                position = Position(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    qty=qty,
                    entry_price=execution_price,
                    current_price=execution_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    side="long",
                    strategy=strategy,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                )
                db.add(position)

        elif side == "sell":
            # Close or reduce position
            existing = db.query(Position).filter(
                Position.portfolio_id == portfolio.id,
                Position.symbol == symbol
            ).first()

            pnl = 0.0
            pnl_pct = 0.0

            if existing:
                sell_qty = min(qty, existing.qty)
                pnl = (execution_price - existing.entry_price) * sell_qty
                pnl_pct = (execution_price - existing.entry_price) / existing.entry_price

                proceeds = execution_price * sell_qty
                portfolio.cash += (proceeds - commission)
                portfolio.daily_pnl = (portfolio.daily_pnl or 0) + pnl
                portfolio.total_pnl = (portfolio.total_pnl or 0) + pnl

                if sell_qty >= existing.qty:
                    db.delete(existing)
                else:
                    existing.qty -= sell_qty
                    db.add(existing)
                db.flush()
            else:
                return {"success": False, "error": f"No position found for {symbol}"}

        else:
            return {"success": False, "error": f"Invalid side: {side}"}

        # Record the trade
        trade = Trade(
            portfolio_id=portfolio.id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=execution_price,
            order_type=order_type,
            status="filled",
            pnl=pnl if side == "sell" else 0.0,
            pnl_pct=pnl_pct if side == "sell" else 0.0,
            strategy=strategy,
            signal_id=signal_id,
            commission=commission,
            slippage=abs(execution_price - current_price) * qty,
            broker_order_id=str(uuid.uuid4()),
        )
        db.add(trade)

        # Update portfolio total value
        self._update_portfolio_value(db, portfolio)
        db.add(portfolio)
        db.commit()

        logger.info(f"[SimBroker] {side.upper()} {qty:.4f} {symbol} @ {execution_price:.2f}")
        return {
            "success": True,
            "order_id": trade.broker_order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "execution_price": execution_price,
            "trade_value": trade_value,
        }

    def _update_portfolio_value(self, db: Session, portfolio: Portfolio):
        """Recalculate portfolio total value from cash + open positions."""
        from app.data.data_loader import get_latest_price

        positions_value = 0.0
        positions = db.query(Position).filter(Position.portfolio_id == portfolio.id).all()

        for pos in positions:
            price = get_latest_price(pos.symbol)
            if price > 0:
                pos.current_price = price
                pos.unrealized_pnl = (price - pos.entry_price) * pos.qty
                pos.unrealized_pnl_pct = (price - pos.entry_price) / pos.entry_price
                db.add(pos)
            positions_value += pos.current_price * pos.qty

        portfolio.total_value = portfolio.cash + positions_value
        
        # Align initial capital and daily start value if they are unaligned or at defaults
        if portfolio.initial_capital == 1000.0 or portfolio.initial_capital < portfolio.total_value / 10:
            portfolio.initial_capital = portfolio.total_value
            portfolio.peak_value = portfolio.total_value
            portfolio.daily_pnl_start_value = portfolio.total_value
            
        portfolio.total_pnl = portfolio.total_value - portfolio.initial_capital
        portfolio.total_pnl_pct = portfolio.total_pnl / portfolio.initial_capital

        # Track peak and drawdown
        if portfolio.total_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.total_value
        if portfolio.peak_value > 0:
            portfolio.max_drawdown = min(
                portfolio.max_drawdown or 0.0,
                (portfolio.total_value - portfolio.peak_value) / portfolio.peak_value,
            )

    def get_positions(self, db: Session, portfolio: Portfolio) -> List[dict]:
        """Get all open positions with current P&L."""
        from app.data.data_loader import get_latest_price

        positions = db.query(Position).filter(Position.portfolio_id == portfolio.id).all()
        result = []
        for pos in positions:
            current_price = get_latest_price(pos.symbol)
            if current_price > 0:
                pos.current_price = current_price
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.qty
                pos.unrealized_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                db.add(pos)

            result.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "qty": pos.qty,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "side": pos.side,
                "strategy": pos.strategy,
                "unrealized_pnl": pos.unrealized_pnl,
                "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                "market_value": pos.current_price * pos.qty,
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
            })

        try:
            db.commit()
        except Exception:
            pass
        return result

    def close_position(
        self, db: Session, portfolio: Portfolio,
        position: Position, current_price: float, reason: str = ""
    ) -> dict:
        """Close a position at current market price."""
        return self.place_order(
            db=db, portfolio=portfolio, symbol=position.symbol,
            side="sell", qty=position.qty, order_type="market",
            strategy=reason or position.strategy,
        )

    def get_account_info(self, db: Session, portfolio: Portfolio) -> dict:
        """Return account summary."""
        self._update_portfolio_value(db, portfolio)
        db.add(portfolio)

        # Save equity snapshot (throttled to at most once per hour to prevent DB bloat)
        positions = db.query(Position).filter(Position.portfolio_id == portfolio.id).all()
        positions_value = sum(p.current_price * p.qty for p in positions)

        last_snapshot = (
            db.query(EquitySnapshot)
            .filter(EquitySnapshot.portfolio_id == portfolio.id)
            .order_by(EquitySnapshot.recorded_at.desc())
            .first()
        )
        now = datetime.utcnow()
        if not last_snapshot or (now - last_snapshot.recorded_at).total_seconds() >= 3600:
            snapshot = EquitySnapshot(
                portfolio_id=portfolio.id,
                total_value=portfolio.total_value,
                cash=portfolio.cash,
                positions_value=positions_value,
                daily_pnl=portfolio.daily_pnl,
            )
            db.add(snapshot)
            db.commit()

        return {
            "cash": portfolio.cash,
            "total_value": portfolio.total_value,
            "positions_value": positions_value,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_pct": portfolio.total_pnl_pct,
            "daily_pnl": portfolio.daily_pnl,
            "max_drawdown": portfolio.max_drawdown,
            "initial_capital": portfolio.initial_capital,
            "trading_halted": portfolio.trading_halted,
            "mode": "paper_sim",
        }


class AlpacaBroker(BaseBroker):
    """
    Alpaca broker integration for paper and live trading.
    Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in settings.
    """

    def __init__(self):
        self.base_url = settings.ALPACA_BASE_URL
        self.headers = {
            "APCA-API-KEY-ID": settings.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=10) as client:
                response = client.request(method, url, headers=self.headers, json=data)
                if response.status_code >= 400:
                    logger.error(f"Alpaca API error response: {response.text}")
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Alpaca API request failed: {e}")
            return {"error": str(e)}

    def place_order(
        self, db: Session, portfolio: Portfolio, symbol: str,
        side: str, qty: float, order_type: str = "market",
        limit_price: float = None, stop_price: float = None,
        stop_loss: float = None, take_profit: float = None,
        strategy: str = None, signal_id: int = None,
    ) -> dict:
        """Place order via Alpaca API."""
        is_bracket = bool(stop_loss or take_profit)
        
        if is_bracket:
            # Alpaca bracket orders do not support fractional shares; qty must be an integer
            qty_val = int(qty)
            if qty_val <= 0:
                logger.warning(f"Canceled bracket order for {symbol} because integer quantity is 0 (float qty: {qty})")
                return {"success": False, "error": f"Integer quantity is 0 for bracket order (float qty: {qty})"}
        else:
            qty_val = round(qty, 4)

        payload = {
            "symbol": symbol,
            "qty": str(qty_val),
            "side": side,
            "type": order_type,
            "time_in_force": "gtc",
        }

        if limit_price:
            payload["limit_price"] = f"{limit_price:.2f}"
        if stop_price:
            payload["stop_price"] = f"{stop_price:.2f}"
            
        if is_bracket:
            payload["order_class"] = "bracket"
            if stop_loss:
                payload["stop_loss"] = {"stop_price": f"{stop_loss:.2f}"}
            if take_profit:
                payload["take_profit"] = {"limit_price": f"{take_profit:.2f}"}

        logger.info(f"Placing Alpaca order for {symbol}: {payload}")
        result = self._request("POST", "/v2/orders", payload)
        if "error" in result:
            return {"success": False, "error": result["error"]}

        # Record the trade in local DB
        trade = Trade(
            portfolio_id=portfolio.id,
            symbol=symbol,
            side=side,
            qty=float(qty_val),
            price=float(result.get("filled_avg_price", 0) or result.get("price", 0) or 0),
            order_type=order_type,
            status=result.get("status", "pending"),
            strategy=strategy,
            signal_id=signal_id,
            broker_order_id=result.get("id"),
        )
        db.add(trade)
        db.commit()

        return {"success": True, "order": result}

    def get_positions(self, db: Session, portfolio: Portfolio) -> List[dict]:
        result = self._request("GET", "/v2/positions")
        if not isinstance(result, list):
            return []
            
        alpaca_positions = []
        active_symbols = set()
        
        for p in result:
            symbol = p["symbol"]
            qty = float(p["qty"])
            entry_price = float(p["avg_entry_price"]) if "avg_entry_price" in p else float(p["cost_basis"]) / qty if qty > 0 else 0.0
            current_price = float(p["current_price"])
            unrealized_pnl = float(p["unrealized_pl"])
            unrealized_pnl_pct = float(p["unrealized_plpc"]) if "unrealized_plpc" in p else 0.0
            
            active_symbols.add(symbol)
            
            # Find existing local position
            pos = db.query(Position).filter(
                Position.portfolio_id == portfolio.id,
                Position.symbol == symbol
            ).first()
            
            if pos:
                pos.qty = qty
                pos.entry_price = entry_price
                pos.current_price = current_price
                pos.unrealized_pnl = unrealized_pnl
                pos.unrealized_pnl_pct = unrealized_pnl_pct
            else:
                pos = Position(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    qty=qty,
                    entry_price=entry_price,
                    current_price=current_price,
                    side="long" if qty > 0 else "short",
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                )
                db.add(pos)
                
            alpaca_positions.append({
                "symbol": symbol,
                "qty": qty,
                "entry_price": entry_price,
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl,
            })
            
        # Remove positions that are no longer active in Alpaca
        db.query(Position).filter(
            Position.portfolio_id == portfolio.id,
            ~Position.symbol.in_(active_symbols)
        ).delete(synchronize_session=False)
        
        try:
            db.commit()
        except Exception as e:
            logger.error(f"Error syncing positions in get_positions: {e}")
            db.rollback()
            
        return alpaca_positions

    def close_position(self, db: Session, portfolio: Portfolio, position: Position, current_price: float, reason: str = "") -> dict:
        result = self._request("DELETE", f"/v2/positions/{position.symbol}")
        return {"success": "error" not in result}

    def get_account_info(self, db: Session, portfolio: Portfolio) -> dict:
        result = self._request("GET", "/v2/account")
        if "error" in result:
            return {}
        
        cash = float(result.get("cash", 0))
        total_value = float(result.get("portfolio_value", 0))
        buying_power = float(result.get("buying_power", 0))
        
        # Align initial capital and daily start value if they are unaligned or at defaults
        if portfolio.initial_capital == 1000.0 or portfolio.initial_capital < total_value / 10:
            portfolio.initial_capital = total_value
            portfolio.peak_value = total_value
            portfolio.daily_pnl_start_value = total_value
            
        portfolio.cash = cash
        portfolio.total_value = total_value
        portfolio.total_pnl = portfolio.total_value - portfolio.initial_capital
        if portfolio.initial_capital > 0:
            portfolio.total_pnl_pct = portfolio.total_pnl / portfolio.initial_capital
            
        # Calculate daily P&L
        if not portfolio.daily_pnl_start_value or portfolio.daily_pnl_start_value <= 0:
            portfolio.daily_pnl_start_value = total_value
        portfolio.daily_pnl = portfolio.total_value - portfolio.daily_pnl_start_value
        
        if portfolio.total_value > portfolio.peak_value:
            portfolio.peak_value = portfolio.total_value
        if portfolio.peak_value > 0:
            portfolio.max_drawdown = min(
                portfolio.max_drawdown or 0.0,
                (portfolio.total_value - portfolio.peak_value) / portfolio.peak_value,
            )
            
        db.add(portfolio)
        try:
            db.commit()
        except Exception as e:
            logger.error(f"Error committing portfolio update in get_account_info: {e}")
            
        return {
            "cash": cash,
            "total_value": total_value,
            "positions_value": total_value - cash,
            "total_pnl": portfolio.total_pnl,
            "total_pnl_pct": portfolio.total_pnl_pct,
            "daily_pnl": portfolio.daily_pnl,
            "max_drawdown": portfolio.max_drawdown,
            "initial_capital": portfolio.initial_capital,
            "trading_halted": portfolio.trading_halted,
            "buying_power": buying_power,
            "mode": "alpaca_paper" if "paper" in self.base_url else "alpaca_live",
        }


def get_broker() -> BaseBroker:
    """Factory function to get the appropriate broker based on settings."""
    mode = settings.TRADING_MODE
    if mode == "paper_sim" or not settings.ALPACA_API_KEY:
        logger.info("Using SimulatedBroker (paper_sim mode)")
        return SimulatedBroker()
    else:
        logger.info(f"Using AlpacaBroker ({mode} mode)")
        return AlpacaBroker()
