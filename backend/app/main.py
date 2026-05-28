"""
FastAPI Main Application
Provides REST API and WebSocket endpoints for the trading dashboard.
"""
import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db, init_db
from app.db.models import Portfolio, Position, Trade, Signal, AIPrediction, RiskLog, EquitySnapshot
from app.execution.orchestrator import (
    get_or_create_portfolio, run_trading_cycle, trigger_model_training
)
from app.data.data_loader import get_ohlcv, get_latest_price, get_fundamental_data
from app.data.indicators import compute_all_indicators, get_latest_indicators
from app.backtester.engine import BacktestEngine
from app.models.predictor import get_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Trigger reload to pick up new default symbols
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        disconnected = []
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                disconnected.append(conn)
        for conn in disconnected:
            self.disconnect(conn)

ws_manager = ConnectionManager()


# Background scheduler
_scheduler_running = False
_scheduler_thread = None
_main_loop = None
_last_telegram_time = 0.0


def _run_scheduler():
    """Background thread running the trading cycle periodically."""
    import time
    global _scheduler_running, _main_loop, _last_telegram_time
    while _scheduler_running:
        try:
            logger.info("Running trading cycle...")
            results = run_trading_cycle()
            # Broadcast results to WebSocket clients on the main event loop
            if _main_loop:
                asyncio.run_coroutine_threadsafe(
                    ws_manager.broadcast({
                        "type": "trading_cycle",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": results,
                    }),
                    _main_loop
                )
            else:
                logger.warning("Main event loop not set, skipping websocket broadcast")

            # Send hourly updates via Telegram
            from app.notifications import send_hourly_portfolio_update
            current_time = time.time()
            if current_time - _last_telegram_time >= 3600:
                logger.info("Sending hourly Telegram portfolio update...")
                if send_hourly_portfolio_update():
                    _last_telegram_time = current_time
                    logger.info("Telegram portfolio update sent successfully.")
                else:
                    logger.warning("Telegram update failed, will retry next cycle.")
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
        time.sleep(300)  # Run every 5 minutes


async def _run_live_price_ticker():
    """Background task generating and broadcasting live prices every 2 seconds."""
    import random
    import time
    from app.data.data_loader import _live_prices, set_live_price, get_latest_price

    failed_symbols = set()
    last_failed_retry_time = time.time()

    # Initialize prices
    for symbol in settings.symbols_list:
        price = await asyncio.to_thread(get_latest_price, symbol)
        if price > 0:
            set_live_price(symbol, price)
        else:
            failed_symbols.add(symbol)

    while _scheduler_running:
        try:
            # Periodically retry failed symbols every 5 minutes to be resilient
            current_time = time.time()
            if current_time - last_failed_retry_time > 300:
                logger.info(f"Retrying failed symbols in live price simulation: {list(failed_symbols)}")
                failed_symbols.clear()
                last_failed_retry_time = current_time

            updates = {}
            for symbol in settings.symbols_list:
                if symbol in failed_symbols:
                    continue

                current = _live_prices.get(symbol, 0.0)
                if current <= 0:
                    price = await asyncio.to_thread(get_latest_price, symbol)
                    if price > 0:
                        set_live_price(symbol, price)
                        current = price
                    else:
                        failed_symbols.add(symbol)
                        continue

                if current > 0:
                    # Fluctuate by random walk (-0.05% to +0.05% per tick)
                    change_pct = random.uniform(-0.0005, 0.0005)
                    new_price = current * (1 + change_pct)
                    set_live_price(symbol, new_price)
                    updates[symbol] = round(new_price, 4)

            if updates:
                await ws_manager.broadcast({
                    "type": "live_prices",
                    "prices": updates,
                    "timestamp": datetime.utcnow().isoformat(),
                })
        except Exception as e:
            logger.error(f"Live price ticker error: {e}")
        await asyncio.sleep(2)  # Tick every 2 seconds


async def _run_telegram_polling():
    """Background task polling for Telegram commands (/start, /status, etc.)."""
    import httpx
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        logger.info("Telegram Bot Token not configured, skipping Telegram polling task.")
        return

    # Register commands with Telegram to display in the menu
    try:
        commands_url = f"https://api.telegram.org/bot{token}/setMyCommands"
        commands_payload = {
            "commands": [
                {"command": "status", "description": "Get current portfolio summary"},
                {"command": "positions", "description": "List all open trading positions"},
                {"command": "signals", "description": "Get the latest 5 strategy signals"},
                {"command": "cycle", "description": "Trigger a manual trading cycle immediately"},
                {"command": "train", "description": "Trigger AI model retraining in background"},
                {"command": "halt", "description": "Emergency halt all trading activity"},
                {"command": "resume", "description": "Resume system trading"},
                {"command": "closed", "description": "List recently closed positions and exit reasons"},
                {"command": "settings", "description": "View current risk and strategy settings"},
                {"command": "buy", "description": "Place manual BUY order: /buy <symbol> <qty>"},
                {"command": "sell", "description": "Place manual SELL order: /sell <symbol> <qty>"},
                {"command": "close", "description": "Close position: /close <symbol>"},
                {"command": "setrisk", "description": "Modify risk setting: /setrisk <param> <val>"},
                {"command": "help", "description": "Show commands help menu"}
            ]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(commands_url, json=commands_payload, timeout=5)
            if resp.status_code == 200:
                logger.info("Telegram slash commands registered successfully.")
            else:
                logger.warning(f"Failed to register Telegram slash commands: {resp.status_code} - {resp.text}")
    except Exception as e:
        logger.error(f"Error registering Telegram slash commands: {e}")

    offset = 0
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    
    logger.info("Telegram command polling task started.")
    
    while _scheduler_running:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, 
                    params={"offset": offset, "timeout": 10}, 
                    timeout=15
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("ok") and data.get("result"):
                        for update in data["result"]:
                            offset = update["update_id"] + 1
                            message = update.get("message")
                            if not message:
                                continue
                            
                            chat = message.get("chat", {})
                            chat_id = chat.get("id")
                            text = message.get("text", "").strip()
                            
                            if not chat_id:
                                continue
                                
                            logger.info(f"Received Telegram message: '{text}' from chat {chat_id}")
                            
                            # Handle commands and reply keyboard texts
                            text_lower = text.lower()
                            is_authorized = str(chat_id) == str(settings.TELEGRAM_CHAT_ID)

                            if text.startswith("/start") or text.startswith("/help") or "help" in text_lower:
                                help_msg = (
                                    "🤖 *QuantAI Trading Bot Commands:*\n"
                                    "───────────────────\n"
                                    "📊 `/status` or `/update` - Get current portfolio summary\n"
                                    "💼 `/positions` - List all open trading positions\n"
                                    "📢 `/signals` - Get the latest 5 strategy signals\n"
                                    "🔄 `/cycle` - Trigger a manual trading cycle immediately\n"
                                    "🧠 `/train` - Trigger AI model retraining in background\n"
                                    "🛑 `/halt` - Emergency halt all trading activity\n"
                                    "🚀 `/resume` - Resume system trading\n"
                                    "📜 `/closed` or `/history` - List recently closed positions and exit reasons\n"
                                    "⚙️ `/settings` - View current risk and strategy settings\n"
                                    "🟢 `/buy <symbol> <qty>` - Place manual BUY order (e.g. `/buy AAPL 10`)\n"
                                    "🔴 `/sell <symbol> <qty>` - Place manual SELL order (e.g. `/sell AAPL 10`)\n"
                                    "❌ `/close <symbol>` - Close open position (e.g. `/close AAPL`)\n"
                                    "🛠️ `/setrisk <param> <val>` - Update risk parameters (e.g. `/setrisk max_open_positions 5`)\n"
                                    "❓ `/help` - Show this commands help menu"
                                )
                                from app.notifications import send_telegram_message
                                await asyncio.to_thread(send_telegram_message, help_msg, str(chat_id))

                            elif text.startswith("/status") or text.startswith("/update") or "status" in text_lower:
                                from app.notifications import send_portfolio_update_to_chat
                                await asyncio.to_thread(send_portfolio_update_to_chat, str(chat_id))

                            elif text.startswith("/positions") or "positions" in text_lower:
                                from app.notifications import get_positions_message, send_telegram_message
                                msg = await asyncio.to_thread(get_positions_message)
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))

                            elif text.startswith("/signals") or "signals" in text_lower:
                                from app.notifications import get_latest_signals_message, send_telegram_message
                                msg = await asyncio.to_thread(get_latest_signals_message)
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))

                            elif text.startswith("/train") or "train" in text_lower:
                                from app.notifications import send_telegram_message
                                await asyncio.to_thread(send_telegram_message, "🧠 *Model retraining triggered in the background...* This will take about 30-45 seconds.", str(chat_id))
                                from app.execution.orchestrator import trigger_model_training
                                threading.Thread(target=trigger_model_training, daemon=True).start()

                            elif text.startswith("/cycle") or "cycle" in text_lower or "run cycle" in text_lower:
                                from app.notifications import send_telegram_message
                                await asyncio.to_thread(send_telegram_message, "🔄 *Manual trading cycle triggered...*", str(chat_id))
                                from app.execution.orchestrator import run_trading_cycle
                                def run_cycle_and_notify():
                                    run_trading_cycle()
                                    from app.notifications import send_portfolio_update_to_chat
                                    send_portfolio_update_to_chat(str(chat_id))
                                threading.Thread(target=run_cycle_and_notify, daemon=True).start()

                            elif text.startswith("/halt") or "halt" in text_lower:
                                from app.notifications import set_trading_halt, send_telegram_message
                                msg = await asyncio.to_thread(set_trading_halt, True)
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))

                            elif text.startswith("/resume") or "resume" in text_lower:
                                from app.notifications import set_trading_halt, send_telegram_message
                                msg = await asyncio.to_thread(set_trading_halt, False)
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))
                                
                            elif text.startswith("/closed") or text.startswith("/history") or "closed" in text_lower:
                                from app.notifications import get_closed_positions_message, send_telegram_message
                                msg = await asyncio.to_thread(get_closed_positions_message)
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))

                            elif text.startswith("/settings") or "settings" in text_lower:
                                from app.notifications import send_telegram_message
                                msg = (
                                    f"⚙️ *QuantAI Settings Summary:*\n"
                                    f"───────────────────\n"
                                    f"🔌 *Trading Mode:* `{settings.TRADING_MODE}`\n"
                                    f"📈 *Active Symbols:* `{settings.DEFAULT_SYMBOLS}`\n"
                                    f"🧠 *AI Threshold:* `{settings.AI_CONFIDENCE_THRESHOLD * 100:.0f}%`\n"
                                    f"⚠️ *Max Risk / Trade:* `{settings.MAX_RISK_PER_TRADE_PCT * 100:.1f}%`\n"
                                    f"🛑 *Max Daily Loss:* `{settings.MAX_DAILY_LOSS_PCT * 100:.1f}%`\n"
                                    f"💼 *Max Positions:* `{settings.MAX_OPEN_POSITIONS}`\n"
                                    f"💰 *Sim Commission:* `${settings.SIM_COMMISSION:.2f}`\n"
                                    f"📉 *Sim Slippage:* `{settings.SIM_SLIPPAGE_PCT * 100:.4f}%`"
                                )
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))

                            elif text.startswith("/buy"):
                                from app.notifications import send_telegram_message
                                if not is_authorized:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Unauthorized:* You are not authorized to place trades.", str(chat_id))
                                    continue
                                
                                parts = text.split()
                                if len(parts) < 3:
                                    await asyncio.to_thread(send_telegram_message, "ℹ️ *Usage:* `/buy <symbol> <quantity>` (e.g. `/buy AAPL 10`)", str(chat_id))
                                    continue
                                
                                symbol = parts[1].upper()
                                try:
                                    qty = float(parts[2])
                                    if qty <= 0:
                                        raise ValueError()
                                except ValueError:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Error:* Quantity must be a positive number.", str(chat_id))
                                    continue
                                    
                                from app.db.database import SessionLocal
                                from app.execution.orchestrator import get_or_create_portfolio
                                from app.execution.broker import get_broker
                                
                                db = SessionLocal()
                                try:
                                    portfolio = get_or_create_portfolio(db)
                                    broker = get_broker()
                                    await asyncio.to_thread(send_telegram_message, f"⏳ Placing manual BUY order for {qty} {symbol}...", str(chat_id))
                                    res = await asyncio.to_thread(
                                        broker.place_order, db=db, portfolio=portfolio, symbol=symbol, side="buy", qty=qty, strategy="telegram_manual"
                                    )
                                    if res.get("success"):
                                        msg = (
                                            f"🟢 *BUY Order Filled!*\n"
                                            f"───────────────────\n"
                                            f"📈 *Symbol:* {symbol}\n"
                                            f"💼 *Qty:* {qty}\n"
                                            f"💵 *Price:* ${res.get('execution_price', 0.0):.2f}\n"
                                            f"💰 *Remaining Cash:* ${portfolio.cash:.2f}"
                                        )
                                    else:
                                        msg = f"❌ *Trade Failed:* {res.get('error', 'Unknown broker error')}"
                                    await asyncio.to_thread(send_telegram_message, msg, str(chat_id))
                                finally:
                                    db.close()

                            elif text.startswith("/sell"):
                                from app.notifications import send_telegram_message
                                if not is_authorized:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Unauthorized:* You are not authorized to place trades.", str(chat_id))
                                    continue
                                
                                parts = text.split()
                                if len(parts) < 3:
                                    await asyncio.to_thread(send_telegram_message, "ℹ️ *Usage:* `/sell <symbol> <quantity>` (e.g. `/sell AAPL 10`)", str(chat_id))
                                    continue
                                
                                symbol = parts[1].upper()
                                try:
                                    qty = float(parts[2])
                                    if qty <= 0:
                                        raise ValueError()
                                except ValueError:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Error:* Quantity must be a positive number.", str(chat_id))
                                    continue
                                    
                                from app.db.database import SessionLocal
                                from app.execution.orchestrator import get_or_create_portfolio
                                from app.execution.broker import get_broker
                                
                                db = SessionLocal()
                                try:
                                    portfolio = get_or_create_portfolio(db)
                                    broker = get_broker()
                                    await asyncio.to_thread(send_telegram_message, f"⏳ Placing manual SELL order for {qty} {symbol}...", str(chat_id))
                                    res = await asyncio.to_thread(
                                        broker.place_order, db=db, portfolio=portfolio, symbol=symbol, side="sell", qty=qty, strategy="telegram_manual"
                                    )
                                    if res.get("success"):
                                        msg = (
                                            f"🔴 *SELL Order Filled!*\n"
                                            f"───────────────────\n"
                                            f"📈 *Symbol:* {symbol}\n"
                                            f"💼 *Qty:* {qty}\n"
                                            f"💵 *Price:* ${res.get('execution_price', 0.0):.2f}\n"
                                            f"💰 *Remaining Cash:* ${portfolio.cash:.2f}"
                                        )
                                    else:
                                        msg = f"❌ *Trade Failed:* {res.get('error', 'Unknown broker error')}"
                                    await asyncio.to_thread(send_telegram_message, msg, str(chat_id))
                                finally:
                                    db.close()

                            elif text.startswith("/close"):
                                from app.notifications import send_telegram_message
                                if not is_authorized:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Unauthorized:* You are not authorized to close positions.", str(chat_id))
                                    continue
                                
                                parts = text.split()
                                if len(parts) < 2:
                                    await asyncio.to_thread(send_telegram_message, "ℹ️ *Usage:* `/close <symbol>` (e.g. `/close AAPL`)", str(chat_id))
                                    continue
                                
                                symbol = parts[1].upper()
                                from app.db.database import SessionLocal
                                from app.execution.orchestrator import get_or_create_portfolio
                                from app.execution.broker import get_broker
                                from app.db.models import Position
                                from app.data.data_loader import get_latest_price
                                
                                db = SessionLocal()
                                try:
                                    portfolio = get_or_create_portfolio(db)
                                    position = db.query(Position).filter(
                                        Position.portfolio_id == portfolio.id,
                                        Position.symbol == symbol
                                    ).first()
                                    
                                    if not position:
                                        await asyncio.to_thread(send_telegram_message, f"❌ *Error:* No open position found for {symbol}.", str(chat_id))
                                        continue
                                        
                                    broker = get_broker()
                                    current_price = get_latest_price(symbol)
                                    await asyncio.to_thread(send_telegram_message, f"⏳ Closing position for {symbol} at ${current_price:.2f}...", str(chat_id))
                                    res = await asyncio.to_thread(
                                        broker.close_position, db=db, portfolio=portfolio, position=position, current_price=current_price, reason="telegram_close"
                                    )
                                    if res.get("success"):
                                        msg = f"🟢 *Position Closed for {symbol}!* Filled @ ${current_price:.2f}."
                                    else:
                                        msg = f"❌ *Close Failed:* {res.get('error', 'Unknown broker error')}"
                                    await asyncio.to_thread(send_telegram_message, msg, str(chat_id))
                                finally:
                                    db.close()

                            elif text.startswith("/setrisk"):
                                from app.notifications import send_telegram_message
                                if not is_authorized:
                                    await asyncio.to_thread(send_telegram_message, "❌ *Unauthorized:* You are not authorized to modify settings.", str(chat_id))
                                    continue
                                
                                parts = text.split()
                                if len(parts) < 3:
                                    usage_msg = (
                                        "ℹ️ *Usage:* `/setrisk <parameter> <value>`\n"
                                        "Parameters:\n"
                                        "- `max_risk_per_trade_pct` (e.g. `0.02` for 2%)\n"
                                        "- `max_daily_loss_pct` (e.g. `0.05` for 5%)\n"
                                        "- `max_open_positions` (e.g. `10`)\n"
                                        "- `ai_confidence_threshold` (e.g. `0.70` for 70%)\n"
                                        "- `sim_commission` (e.g. `1.50`)\n"
                                        "- `sim_slippage_pct` (e.g. `0.0005` for 0.05%)"
                                    )
                                    await asyncio.to_thread(send_telegram_message, usage_msg, str(chat_id))
                                    continue
                                    
                                key = parts[1].strip().lower()
                                val_str = parts[2].strip()
                                
                                valid_keys = {
                                    "max_risk_per_trade_pct": float,
                                    "max_daily_loss_pct": float,
                                    "max_open_positions": int,
                                    "ai_confidence_threshold": float,
                                    "sim_commission": float,
                                    "sim_slippage_pct": float
                                }
                                
                                if key not in valid_keys:
                                    await asyncio.to_thread(send_telegram_message, f"❌ *Error:* Unknown parameter '{key}'.", str(chat_id))
                                    continue
                                    
                                try:
                                    val_type = valid_keys[key]
                                    val = val_type(val_str)
                                except ValueError:
                                    await asyncio.to_thread(send_telegram_message, f"❌ *Error:* Invalid value '{val_str}' for parameter '{key}'.", str(chat_id))
                                    continue
                                    
                                # Update in-memory
                                if key == "max_risk_per_trade_pct":
                                    settings.MAX_RISK_PER_TRADE_PCT = val
                                elif key == "max_daily_loss_pct":
                                    settings.MAX_DAILY_LOSS_PCT = val
                                elif key == "max_open_positions":
                                    settings.MAX_OPEN_POSITIONS = val
                                elif key == "ai_confidence_threshold":
                                    settings.AI_CONFIDENCE_THRESHOLD = val
                                elif key == "sim_commission":
                                    settings.SIM_COMMISSION = val
                                elif key == "sim_slippage_pct":
                                    settings.SIM_SLIPPAGE_PCT = val
                                    
                                # Persist to .env
                                from app.config import update_env_file
                                env_key = key.upper()
                                success = update_env_file({env_key: val})
                                
                                if success:
                                    msg = f"✅ *Success:* Set `{key}` to `{val}` and saved to disk."
                                else:
                                    msg = f"⚠️ *Warning:* Set `{key}` to `{val}` in-memory, but failed to save to `.env`."
                                await asyncio.to_thread(send_telegram_message, msg, str(chat_id))
        except Exception as e:
            logger.error(f"Telegram polling error: {e}")
        
        await asyncio.sleep(3)  # Poll every 3 seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    global _scheduler_running, _scheduler_thread, _main_loop
    _main_loop = asyncio.get_running_loop()

    # Initialize database
    logger.info(f"STARTUP SYMBOLS LIST: {settings.symbols_list}")
    logger.info("Initializing database...")
    init_db()

    # Create default portfolio
    from app.db.database import SessionLocal
    db = SessionLocal()
    try:
        get_or_create_portfolio(db)
    finally:
        db.close()

    # Start background scheduler
    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_run_scheduler, daemon=True)
    _scheduler_thread.start()
    logger.info("Background trading scheduler started")

    # Start live price ticker
    asyncio.create_task(_run_live_price_ticker())
    logger.info("Background live price simulation ticker started")

    # Start Telegram polling task
    asyncio.create_task(_run_telegram_polling())
    logger.info("Background Telegram polling task started")

    # Notify Telegram that server is online
    from app.notifications import send_telegram_message
    try:
        send_telegram_message("🚀 *QuantAI Backend Server is ONLINE*")
    except Exception as e:
        logger.error(f"Failed to send startup Telegram alert: {e}")

    yield

    # Shutdown
    _scheduler_running = False
    logger.info("Application shutting down")

    # Notify Telegram that server is offline
    try:
        send_telegram_message("⚠️ *QuantAI Backend Server is OFFLINE (Shutting down)*")
    except Exception as e:
        logger.error(f"Failed to send shutdown Telegram alert: {e}")


# Create FastAPI app
app = FastAPI(
    title="Autonomous AI Trading System",
    description="AI-powered algorithmic trading platform with real-time analytics",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class BacktestRequest(BaseModel):
    symbol: str = "SPY"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = "1d"
    lookback_days: int = 365
    initial_capital: float = 1000.0
    use_ai: bool = False


class ManualTradeRequest(BaseModel):
    symbol: str
    side: str  # buy | sell
    qty: float
    order_type: str = "market"


class AlpacaTestRequest(BaseModel):
    api_key: str
    secret_key: str
    base_url: str


class AlpacaSaveRequest(BaseModel):
    api_key: str
    secret_key: str
    base_url: str
    trading_mode: str


class SettingsUpdateRequest(BaseModel):
    max_risk_per_trade_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_open_positions: Optional[int] = None
    ai_confidence_threshold: Optional[float] = None
    symbols: Optional[str] = None
    stop_loss_atr_multiplier: Optional[float] = None
    take_profit_atr_multiplier: Optional[float] = None
    sim_commission: Optional[float] = None
    sim_slippage_pct: Optional[float] = None
    momentum: Optional[bool] = None
    mean_reversion: Optional[bool] = None
    breakout: Optional[bool] = None
    ai_confidence: Optional[bool] = None


# ============================================================
# HEALTH & STATUS
# ============================================================

@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "trading_mode": settings.TRADING_MODE,
        "version": "1.0.0",
    }


@app.get("/api/status")
async def get_status(db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    from app.execution.broker import get_broker
    broker = get_broker()
    account = broker.get_account_info(db, portfolio)
    return {
        "portfolio": account,
        "scheduler_running": _scheduler_running,
        "trading_mode": settings.TRADING_MODE,
        "symbols": settings.symbols_list,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================
# PORTFOLIO ENDPOINTS
# ============================================================

@app.get("/api/portfolio")
async def get_portfolio(db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    from app.execution.broker import get_broker
    broker = get_broker()
    account = broker.get_account_info(db, portfolio)
    return account


@app.get("/api/portfolio/equity-curve")
async def get_equity_curve(limit: int = 200, db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    snapshots = (
        db.query(EquitySnapshot)
        .filter(EquitySnapshot.portfolio_id == portfolio.id)
        .order_by(EquitySnapshot.recorded_at.asc())
        .limit(limit)
        .all()
    )
    return {
        "equity_curve": [
            {
                "timestamp": s.recorded_at.isoformat(),
                "total_value": s.total_value,
                "cash": s.cash,
                "positions_value": s.positions_value,
                "daily_pnl": s.daily_pnl,
            }
            for s in snapshots
        ]
    }


# ============================================================
# POSITIONS & TRADES
# ============================================================

@app.get("/api/positions")
async def get_positions(db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    from app.execution.broker import get_broker
    broker = get_broker()
    return {"positions": broker.get_positions(db, portfolio)}


@app.get("/api/trades")
async def get_trades(limit: int = 50, db: Session = Depends(get_db)):
    trades = (
        db.query(Trade)
        .order_by(Trade.executed_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "trades": [
            {
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": t.qty,
                "price": t.price,
                "order_type": t.order_type,
                "status": t.status,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "strategy": t.strategy,
                "commission": t.commission,
                "executed_at": t.executed_at.isoformat() if t.executed_at else None,
            }
            for t in trades
        ]
    }


@app.post("/api/trades/manual")
async def place_manual_trade(req: ManualTradeRequest, db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    from app.execution.broker import get_broker
    broker = get_broker()
    result = broker.place_order(
        db=db, portfolio=portfolio, symbol=req.symbol.upper(),
        side=req.side, qty=req.qty, order_type=req.order_type,
        strategy="manual",
    )
    return result


@app.delete("/api/positions/{symbol}")
async def close_position(symbol: str, db: Session = Depends(get_db)):
    portfolio = get_or_create_portfolio(db)
    position = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.symbol == symbol.upper()
    ).first()
    if not position:
        raise HTTPException(status_code=404, detail=f"No open position for {symbol}")
    from app.execution.broker import get_broker
    broker = get_broker()
    current_price = get_latest_price(symbol.upper())
    result = broker.close_position(db, portfolio, position, current_price, "manual_close")
    return result


# ============================================================
# MARKET DATA
# ============================================================

@app.get("/api/market/{symbol}/chart")
async def get_chart_data(
    symbol: str,
    timeframe: str = "1d",
    lookback_days: int = 180,
):
    df = get_ohlcv(symbol.upper(), timeframe, lookback_days)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")

    df = compute_all_indicators(df)

    result = []
    for idx, row in df.iterrows():
        bar = {
            "timestamp": idx.isoformat(),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": int(row["volume"]),
        }
        # Add indicators
        for col in ["sma_20", "ema_9", "ema_21", "bb_upper", "bb_lower", "bb_middle",
                    "rsi_14", "macd", "macd_signal", "macd_histogram", "atr_14", "vwap"]:
            if col in row.index and not pd.isna(row[col]):
                bar[col] = round(float(row[col]), 4)
        result.append(bar)

    return {"symbol": symbol.upper(), "timeframe": timeframe, "bars": result}


@app.get("/api/market/{symbol}/price")
async def get_current_price(symbol: str):
    price = get_latest_price(symbol.upper())
    return {"symbol": symbol.upper(), "price": price, "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/market/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    return get_fundamental_data(symbol.upper())


# ============================================================
# SIGNALS
# ============================================================

@app.get("/api/signals")
async def get_signals(limit: int = 50, db: Session = Depends(get_db)):
    signals = (
        db.query(Signal)
        .order_by(Signal.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "signals": [
            {
                "id": s.id,
                "symbol": s.symbol,
                "action": s.action,
                "strategy": s.strategy,
                "confidence": s.confidence,
                "price_at_signal": s.price_at_signal,
                "ai_prediction": s.ai_prediction,
                "ai_confidence": s.ai_confidence,
                "executed": s.executed,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in signals
        ]
    }


async def run_trading_cycle_and_broadcast():
    results = await asyncio.to_thread(run_trading_cycle)
    await ws_manager.broadcast({
        "type": "trading_cycle",
        "timestamp": datetime.utcnow().isoformat(),
        "data": results,
    })


@app.post("/api/signals/generate")
async def generate_signals(background_tasks: BackgroundTasks):
    """Trigger a manual trading cycle run."""
    background_tasks.add_task(run_trading_cycle_and_broadcast)
    return {"message": "Trading cycle triggered", "timestamp": datetime.utcnow().isoformat()}


# In-memory news cache
_cached_news = []
_last_news_fetch_time = 0.0

@app.get("/api/news")
async def get_market_news():
    global _cached_news, _last_news_fetch_time
    import time
    import yfinance as yf
    
    current_time = time.time()
    # Cache news for 5 minutes (300 seconds)
    if _cached_news and (current_time - _last_news_fetch_time < 300):
        return {"news": _cached_news, "cached": True}
        
    top_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    aggregated = []
    seen_ids = set()
    
    def fetch_symbol_news(sym):
        try:
            ticker = yf.Ticker(sym)
            return ticker.news or []
        except Exception as e:
            logger.error(f"Error fetching news for {sym} in endpoint: {e}")
            return []

    for symbol in top_symbols:
        news_list = await asyncio.to_thread(fetch_symbol_news, symbol)
        for article in news_list[:5]:
            art_id = article.get("id")
            if not art_id or art_id in seen_ids:
                continue
            seen_ids.add(art_id)
            
            content = article.get("content") or {}
            title = content.get("title") or ""
            pub_date = content.get("pubDate") or ""
            summary = content.get("summary") or ""
            provider = (content.get("provider") or {}).get("displayName") or ""
            click_url = (content.get("clickThroughUrl") or {}).get("url") or ""
            
            if not title.strip():
                continue
                
            aggregated.append({
                "id": art_id,
                "symbol": symbol,
                "title": title,
                "summary": summary,
                "pub_date": pub_date,
                "provider": provider,
                "url": click_url
            })
            
    if aggregated:
        aggregated.sort(key=lambda x: x["pub_date"], reverse=True)
        _cached_news = aggregated[:15]
        _last_news_fetch_time = current_time
        
    return {"news": _cached_news, "cached": False}


# ============================================================
# AI MODEL
# ============================================================

@app.get("/api/ai/status")
async def get_ai_status():
    result = {}
    for symbol in settings.symbols_list:
        predictor = get_predictor(symbol)
        status = predictor.get_status()
        
        # Calculate latest prediction dynamically for the UI
        if predictor.is_trained:
            try:
                df = get_ohlcv(symbol, "1d", lookback_days=settings.LOOKBACK_DAYS)
                if not df.empty:
                    df = compute_all_indicators(df)
                    df.dropna(inplace=True)
                    pred = predictor.predict(df)
                    status["prediction_direction"] = pred.get("direction")
                    status["prediction_confidence"] = pred.get("confidence")
            except Exception as e:
                logger.error(f"Error getting prediction for {symbol} in status: {e}")
                
        result[symbol] = status
    return result


@app.post("/api/ai/train")
async def train_models(background_tasks: BackgroundTasks):
    """Trigger AI model retraining in background."""
    background_tasks.add_task(trigger_model_training)
    return {"message": "Model training started", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/ai/predict/{symbol}")
async def get_prediction(symbol: str):
    df = get_ohlcv(symbol.upper(), "1d", lookback_days=settings.LOOKBACK_DAYS)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for {symbol}")
    df = compute_all_indicators(df)
    df.dropna(inplace=True)

    predictor = get_predictor(symbol.upper())
    if not predictor.is_trained:
        raise HTTPException(status_code=400, detail="Model not yet trained. Call /api/ai/train first.")

    prediction = predictor.predict(df)
    return {"symbol": symbol.upper(), "prediction": prediction}


# ============================================================
# BACKTESTING
# ============================================================

@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    engine = BacktestEngine(initial_capital=req.initial_capital)
    result = engine.run(
        symbol=req.symbol.upper(),
        start_date=req.start_date,
        end_date=req.end_date,
        timeframe=req.timeframe,
        lookback_days=req.lookback_days,
        use_ai=req.use_ai,
    )
    return result.to_dict()


# ============================================================
# RISK LOGS
# ============================================================

@app.get("/api/risk/logs")
async def get_risk_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(RiskLog).order_by(RiskLog.created_at.desc()).limit(limit).all()
    return {
        "logs": [
            {
                "id": l.id,
                "event_type": l.event_type,
                "symbol": l.symbol,
                "detail": l.detail,
                "portfolio_value": l.portfolio_value,
                "created_at": l.created_at.isoformat() if l.created_at else None,
            }
            for l in logs
        ]
    }


# ============================================================
# SETTINGS
# ============================================================

@app.get("/api/settings")
async def get_settings():
    return {
        "trading_mode": settings.TRADING_MODE,
        "initial_capital": settings.INITIAL_CAPITAL,
        "symbols": settings.symbols_list,
        "ai_confidence_threshold": settings.AI_CONFIDENCE_THRESHOLD,
        "max_risk_per_trade_pct": settings.MAX_RISK_PER_TRADE_PCT,
        "max_daily_loss_pct": settings.MAX_DAILY_LOSS_PCT,
        "max_open_positions": settings.MAX_OPEN_POSITIONS,
        "stop_loss_atr_multiplier": settings.STOP_LOSS_ATR_MULTIPLIER,
        "take_profit_atr_multiplier": settings.TAKE_PROFIT_ATR_MULTIPLIER,
        "sim_commission": settings.SIM_COMMISSION,
        "sim_slippage_pct": settings.SIM_SLIPPAGE_PCT,
        "alpaca_api_key": settings.ALPACA_API_KEY,
        "alpaca_secret_key": settings.ALPACA_SECRET_KEY,
        "alpaca_base_url": settings.ALPACA_BASE_URL,
        "strategies": {
            "momentum": settings.STRATEGY_MOMENTUM_ENABLED,
            "mean_reversion": settings.STRATEGY_MEAN_REVERSION_ENABLED,
            "breakout": settings.STRATEGY_BREAKOUT_ENABLED,
            "ai_confidence": settings.STRATEGY_AI_ENABLED,
        }
    }


@app.put("/api/settings")
async def update_settings(req: SettingsUpdateRequest):
    from app.config import update_env_file
    
    updates = {}
    
    if req.max_risk_per_trade_pct is not None:
        settings.MAX_RISK_PER_TRADE_PCT = req.max_risk_per_trade_pct
        updates["MAX_RISK_PER_TRADE_PCT"] = req.max_risk_per_trade_pct
        
    if req.max_daily_loss_pct is not None:
        settings.MAX_DAILY_LOSS_PCT = req.max_daily_loss_pct
        updates["MAX_DAILY_LOSS_PCT"] = req.max_daily_loss_pct
        
    if req.max_open_positions is not None:
        settings.MAX_OPEN_POSITIONS = req.max_open_positions
        updates["MAX_OPEN_POSITIONS"] = req.max_open_positions
        
    if req.ai_confidence_threshold is not None:
        settings.AI_CONFIDENCE_THRESHOLD = req.ai_confidence_threshold
        updates["AI_CONFIDENCE_THRESHOLD"] = req.ai_confidence_threshold
        
    if req.symbols is not None:
        settings.DEFAULT_SYMBOLS = req.symbols
        updates["DEFAULT_SYMBOLS"] = req.symbols
        
    if req.stop_loss_atr_multiplier is not None:
        settings.STOP_LOSS_ATR_MULTIPLIER = req.stop_loss_atr_multiplier
        updates["STOP_LOSS_ATR_MULTIPLIER"] = req.stop_loss_atr_multiplier
        
    if req.take_profit_atr_multiplier is not None:
        settings.TAKE_PROFIT_ATR_MULTIPLIER = req.take_profit_atr_multiplier
        updates["TAKE_PROFIT_ATR_MULTIPLIER"] = req.take_profit_atr_multiplier
        
    if req.sim_commission is not None:
        settings.SIM_COMMISSION = req.sim_commission
        updates["SIM_COMMISSION"] = req.sim_commission
        
    if req.sim_slippage_pct is not None:
        settings.SIM_SLIPPAGE_PCT = req.sim_slippage_pct
        updates["SIM_SLIPPAGE_PCT"] = req.sim_slippage_pct
        
    if req.momentum is not None:
        settings.STRATEGY_MOMENTUM_ENABLED = req.momentum
        updates["STRATEGY_MOMENTUM_ENABLED"] = str(req.momentum).lower()
        
    if req.mean_reversion is not None:
        settings.STRATEGY_MEAN_REVERSION_ENABLED = req.mean_reversion
        updates["STRATEGY_MEAN_REVERSION_ENABLED"] = str(req.mean_reversion).lower()
        
    if req.breakout is not None:
        settings.STRATEGY_BREAKOUT_ENABLED = req.breakout
        updates["STRATEGY_BREAKOUT_ENABLED"] = str(req.breakout).lower()
        
    if req.ai_confidence is not None:
        settings.STRATEGY_AI_ENABLED = req.ai_confidence
        updates["STRATEGY_AI_ENABLED"] = str(req.ai_confidence).lower()

    if updates:
        success = update_env_file(updates)
        if not success:
            return {"success": False, "error": "Failed to update the .env file on disk, but settings were updated in memory."}

    return {"success": True, "message": "Settings updated successfully."}


@app.post("/api/settings/alpaca/test")
async def test_alpaca_credentials(req: AlpacaTestRequest):
    import httpx
    url = f"{req.base_url}/v2/account"
    headers = {
        "APCA-API-KEY-ID": req.api_key,
        "APCA-API-SECRET-KEY": req.secret_key,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                return {"success": True, "message": "Connection successful!"}
            else:
                return {"success": False, "error": f"Alpaca returned status code {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to connect to Alpaca: {str(e)}"}


@app.post("/api/settings/alpaca/save")
async def save_alpaca_credentials(req: AlpacaSaveRequest):
    from app.config import update_env_file
    
    # Update in-memory settings
    settings.ALPACA_API_KEY = req.api_key
    settings.ALPACA_SECRET_KEY = req.secret_key
    settings.ALPACA_BASE_URL = req.base_url
    settings.TRADING_MODE = req.trading_mode
    
    # Update .env file on disk
    updates = {
        "ALPACA_API_KEY": req.api_key,
        "ALPACA_SECRET_KEY": req.secret_key,
        "ALPACA_BASE_URL": req.base_url,
        "TRADING_MODE": req.trading_mode,
    }
    
    success = update_env_file(updates)
    if success:
        return {"success": True, "message": f"Alpaca credentials saved and trading mode set to {req.trading_mode}."}
    else:
        return {"success": False, "error": "Failed to update the .env file on disk, but settings were updated in memory."}


# ============================================================
# WEBSOCKET REAL-TIME FEED
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await ws_manager.connect(websocket)
    try:
        # Send initial state on connect
        portfolio = get_or_create_portfolio(db)
        from app.execution.broker import get_broker
        broker = get_broker()
        account = broker.get_account_info(db, portfolio)
        positions = broker.get_positions(db, portfolio)

        await websocket.send_json({
            "type": "initial_state",
            "portfolio": account,
            "positions": positions,
            "settings": {
                "symbols": settings.symbols_list,
                "trading_mode": settings.TRADING_MODE,
            }
        })

        # Keep alive and handle client messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})

                elif msg.get("type") == "request_update":
                    portfolio = get_or_create_portfolio(db)
                    account = broker.get_account_info(db, portfolio)
                    positions = broker.get_positions(db, portfolio)
                    await websocket.send_json({
                        "type": "portfolio_update",
                        "portfolio": account,
                        "positions": positions,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)

