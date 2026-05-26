"""
Telegram Notification Module
Formats and transmits portfolio and trade alerts to a Telegram bot.
"""
import logging
import requests
from datetime import datetime
from sqlalchemy.orm import Session

from app.config import settings
from app.execution.broker import get_broker
from app.db.database import SessionLocal
from app.execution.orchestrator import get_or_create_portfolio

logger = logging.getLogger(__name__)


TELEGRAM_KEYBOARD = {
    "keyboard": [
        [{"text": "📊 Status"}, {"text": "💼 Positions"}, {"text": "📢 Signals"}],
        [{"text": "🔄 Run Cycle"}, {"text": "🧠 Train AI"}, {"text": "📜 Closed"}],
        [{"text": "🛑 Halt"}, {"text": "🚀 Resume"}, {"text": "❓ Help"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": False,
}


def send_telegram_message(message: str, chat_id: str = None, reply_markup: dict = None) -> bool:
    """Send a raw text message to the configured Telegram chat."""
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    
    if not token or not chat_id:
        logger.debug("Telegram notifications not configured (missing token or chat ID)")
        return False
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown",
        "reply_markup": reply_markup or TELEGRAM_KEYBOARD,
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info("Telegram notification sent successfully")
            return True
        else:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def send_portfolio_update_to_chat(chat_id: str = None, reply_markup: dict = None) -> bool:
    """Fetch latest portfolio data and send a formatted update to Telegram (or specific chat ID)."""
    if chat_id is None:
        chat_id = settings.TELEGRAM_CHAT_ID
        
    if not chat_id:
        logger.warning("No chat_id specified or configured for Telegram update")
        return False
        
    db = SessionLocal()
    try:
        portfolio = get_or_create_portfolio(db)
        broker = get_broker()
        
        # Get latest account info
        account = broker.get_account_info(db, portfolio)
        positions = broker.get_positions(db, portfolio)
        
        # Format message
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        total_val = account.get("total_value", 0.0)
        cash = account.get("cash", 0.0)
        daily_pnl = account.get("daily_pnl", 0.0)
        initial_cap = account.get("initial_capital", 1000.0)
        
        total_pnl = total_val - initial_cap
        total_pnl_pct = (total_pnl / initial_cap) * 100 if initial_cap > 0 else 0.0
        
        daily_pnl_start = account.get("daily_pnl_start_value", initial_cap)
        daily_pnl_pct = (daily_pnl / daily_pnl_start) * 100 if daily_pnl_start > 0 else 0.0
        
        halted = account.get("trading_halted", False)
        status_str = "⚠️ Halted" if halted else "✅ Active"
        
        msg = (
            f"📊 *QuantAI Portfolio Update*\n"
            f"⏰ `{now_str}`\n"
            f"───────────────────\n"
            f"💰 *Portfolio Value:* `${total_val:,.2f}` (Cash: `${cash:,.2f}`)\n"
            f"📈 *Daily P&L:* `{'+' if daily_pnl >= 0 else ''}{daily_pnl:,.2f}` (`{'+' if daily_pnl_pct >= 0 else ''}{daily_pnl_pct:.2f}%`)\n"
            f"📊 *Total P&L:* `{'+' if total_pnl >= 0 else ''}{total_pnl:,.2f}` (`{'+' if total_pnl_pct >= 0 else ''}{total_pnl_pct:.2f}%`)\n"
            f"🚨 *Status:* {status_str}\n\n"
            f"💼 *Open Positions ({len(positions)}):*\n"
        )
        
        if not positions:
            msg += "_No active open positions._"
        else:
            for p in positions:
                sym = p.get("symbol", "UNKNOWN")
                qty = p.get("qty", 0.0)
                entry = p.get("entry_price", 0.0)
                curr = p.get("current_price", entry)
                pnl = p.get("unrealized_pnl", 0.0)
                pnl_pct = p.get("unrealized_pnl_pct", 0.0) * 100
                side = p.get("side", "LONG")
                
                msg += (
                    f"• *{sym}* ({side} | `{qty:.4f}` shares)\n"
                    f"  Entry: `${entry:,.2f}` | Current: `${curr:,.2f}`\n"
                    f"  Unrealized P&L: `{'+' if pnl >= 0 else ''}{pnl:,.2f}` (`{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%`)\n"
                )
                
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            logger.debug("Telegram notifications not configured (missing token)")
            return False
            
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "Markdown",
            "reply_markup": reply_markup or TELEGRAM_KEYBOARD,
        }
        
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            logger.info(f"Telegram notification sent successfully to chat {chat_id}")
            return True
        else:
            logger.error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error constructing Telegram update: {e}")
        return False
    finally:
        db.close()


def send_hourly_portfolio_update() -> bool:
    """Fetch latest portfolio data and send a formatted update to Telegram."""
    return send_portfolio_update_to_chat()


def get_positions_message() -> str:
    """Format open positions summary for Telegram."""
    db = SessionLocal()
    try:
        portfolio = get_or_create_portfolio(db)
        broker = get_broker()
        positions = broker.get_positions(db, portfolio)
        if not positions:
            return "💼 *No open positions.*"
        
        msg = f"💼 *Open Positions ({len(positions)}):*\n"
        for p in positions:
            sym = p.get("symbol", "UNKNOWN")
            qty = p.get("qty", 0.0)
            entry = p.get("entry_price", 0.0)
            curr = p.get("current_price", entry)
            pnl = p.get("unrealized_pnl", 0.0)
            pnl_pct = p.get("unrealized_pnl_pct", 0.0) * 100
            side = p.get("side", "LONG")
            sl = p.get("stop_loss")
            tp = p.get("take_profit")
            
            sl_str = f"SL: `${sl:,.2f}`" if sl else "SL: None"
            tp_str = f"TP: `${tp:,.2f}`" if tp else "TP: None"
            
            msg += (
                f"• *{sym}* ({side})\n"
                f"  Qty: `{qty:.4f}` | Entry: `${entry:,.2f}`\n"
                f"  Current: `${curr:,.2f}` | P&L: `{'+' if pnl >= 0 else ''}{pnl:,.2f}` (`{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%`)\n"
                f"  {sl_str} | {tp_str}\n"
            )
        return msg
    except Exception as e:
        logger.error(f"Error formatting positions message: {e}")
        return "❌ Error fetching positions."
    finally:
        db.close()


def get_latest_signals_message() -> str:
    """Format latest 5 signals summary for Telegram."""
    db = SessionLocal()
    try:
        from app.db.models import Signal
        signals = db.query(Signal).order_by(Signal.created_at.desc()).limit(5).all()
        if not signals:
            return "📢 *No signals generated yet.*"
        
        msg = "📢 *Latest Trading Signals:*\n"
        for s in signals:
            action_emoji = "🟢 BUY" if s.action == "buy" else "🔴 SELL" if s.action == "sell" else "⚪ HOLD"
            time_str = s.created_at.strftime("%H:%M:%S UTC")
            msg += (
                f"• *{s.symbol}* | {action_emoji} | Price: `${s.price_at_signal:,.2f}`\n"
                f"  Strategy: `{s.strategy}` | Confidence: `{s.confidence * 100:.1f}%`\n"
                f"  Time: `{time_str}` | Executed: `{'Yes' if s.executed else 'No'}`\n"
            )
        return msg
    except Exception as e:
        logger.error(f"Error formatting signals message: {e}")
        return "❌ Error fetching signals."
    finally:
        db.close()


def set_trading_halt(halt: bool) -> str:
    """Halt or resume system trading via Telegram."""
    db = SessionLocal()
    try:
        portfolio = get_or_create_portfolio(db)
        portfolio.trading_halted = halt
        db.add(portfolio)
        db.commit()
        status = "⚠️ HALTED" if halt else "✅ ACTIVE"
        return f"System trading status updated to: *{status}*"
    except Exception as e:
        logger.error(f"Error setting trading halt: {e}")
        return "❌ Error updating trading status."
    finally:
        db.close()


def get_closed_positions_message(limit: int = 5) -> str:
    """Format recently closed positions (historical sell trades) for Telegram."""
    db = SessionLocal()
    try:
        from app.db.models import Trade
        # Fetch the latest sell trades (which represent closed positions)
        closed_trades = (
            db.query(Trade)
            .filter(Trade.side == "sell", Trade.status == "filled")
            .order_by(Trade.executed_at.desc())
            .limit(limit)
            .all()
        )
        
        if not closed_trades:
            return "📜 *No closed positions found.*"
            
        msg = f"📜 *Recently Closed Positions (Last {len(closed_trades)}):*\n"
        for t in closed_trades:
            pnl = t.pnl or 0.0
            pnl_pct = (t.pnl_pct or 0.0) * 100
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            time_str = t.executed_at.strftime("%Y-%m-%d %H:%M:%S UTC") if t.executed_at else "Unknown"
            
            # Translate close reason (strategy column stores close reason for exit trades)
            reason = t.strategy or "unknown"
            
            msg += (
                f"• *{t.symbol}* | {pnl_emoji} P&L: `{'+' if pnl >= 0 else ''}${pnl:,.2f}` (`{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%`)\n"
                f"  Closed At: `{time_str}`\n"
                f"  Price: `${t.price:,.2f}` | Qty: `{t.qty:.4f}`\n"
                f"  Reason: `{reason}`\n\n"
            )
        return msg
    except Exception as e:
        logger.error(f"Error formatting closed positions message: {e}")
        return "❌ Error fetching closed positions."
    finally:
        db.close()
