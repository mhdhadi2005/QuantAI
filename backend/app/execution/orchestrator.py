"""
Trading Orchestrator
Ties together: data loading → indicators → AI prediction → strategy signal →
risk management → trade execution → portfolio update.
This is the main trading loop called by the scheduler.
"""
import logging
from typing import List, Optional
from datetime import datetime, date

import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.data.data_loader import get_ohlcv, get_latest_price
from app.data.indicators import compute_all_indicators
from app.models.predictor import get_predictor, train_all_predictors
from app.strategies.engine import StrategyEngine
from app.data.sentiment import get_news_sentiment
from app.risk.manager import RiskManager
from app.execution.broker import get_broker
from app.db.database import SessionLocal
from app.db.models import Portfolio, Position, Signal, AIPrediction

logger = logging.getLogger(__name__)

strategy_engine = StrategyEngine()
risk_manager = RiskManager()


def get_or_create_portfolio(db: Session) -> Portfolio:
    """Get the portfolio for the current trading mode, or create it."""
    mode = settings.TRADING_MODE
    portfolio = db.query(Portfolio).filter(Portfolio.name == f"Portfolio {mode}").first()
    if not portfolio:
        # Check if we can rename a legacy 'Main Portfolio' to preserve history
        legacy_portfolio = db.query(Portfolio).filter(Portfolio.name == "Main Portfolio").first()
        if legacy_portfolio:
            legacy_portfolio.name = f"Portfolio {mode}"
            db.add(legacy_portfolio)
            db.commit()
            db.refresh(legacy_portfolio)
            logger.info(f"Renamed legacy 'Main Portfolio' to '{legacy_portfolio.name}'")
            return legacy_portfolio

        portfolio = Portfolio(
            name=f"Portfolio {mode}",
            cash=settings.INITIAL_CAPITAL,
            initial_capital=settings.INITIAL_CAPITAL,
            total_value=settings.INITIAL_CAPITAL,
            peak_value=settings.INITIAL_CAPITAL,
            daily_pnl_start_value=settings.INITIAL_CAPITAL,
        )
        db.add(portfolio)
        db.commit()
        db.refresh(portfolio)
        logger.info(f"Created new portfolio '{portfolio.name}' with ${settings.INITIAL_CAPITAL:.2f} initial capital")
    return portfolio


def reset_daily_stats(db: Session, portfolio: Portfolio):
    """Reset daily P&L tracking at start of each trading day."""
    portfolio.daily_pnl = 0.0
    portfolio.daily_pnl_start_value = portfolio.total_value
    portfolio.trading_halted = False
    db.add(portfolio)
    db.commit()


def check_and_apply_auto_market_switch() -> tuple:
    """
    Checks if Indian or US markets are open and switches trading mode & symbols dynamically.
    Returns: (switched: bool, target_mode: str)
    """
    import datetime
    import pytz
    from app.notifications import send_telegram_message

    now_utc = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
    
    # Check Indian Market Hours (9:15 AM to 3:30 PM IST / UTC+5:30)
    ist_tz = pytz.timezone('Asia/Kolkata')
    now_ist = now_utc.astimezone(ist_tz)
    
    # Check US Market Hours (9:30 AM to 4:00 PM EST/EDT)
    us_tz = pytz.timezone('America/New_York')
    now_us = now_utc.astimezone(us_tz)

    is_india_open = False
    is_us_open = False

    # IST Trading: Monday-Friday 09:15 to 15:30
    if now_ist.weekday() < 5:
        ist_time = now_ist.time()
        if datetime.time(9, 15) <= ist_time <= datetime.time(15, 30):
            is_india_open = True

    # US Trading: Monday-Friday 09:30 to 16:00
    if now_us.weekday() < 5:
        us_time = now_us.time()
        if datetime.time(9, 30) <= us_time <= datetime.time(16, 0):
            is_us_open = True

    if is_india_open:
        target_mode = "paper_sim"
        target_symbols = "RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,^NSEI"
        market_label = "🇮🇳 Indian NSE Market"
    elif is_us_open:
        target_mode = "paper_alpaca"
        target_symbols = "AAPL,MSFT,NVDA,SPY,QQQ,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,V,DIS"
        market_label = "🇺🇸 US Market"
    else:
        return False, "closed"

    if settings.TRADING_MODE != target_mode or settings.DEFAULT_SYMBOLS != target_symbols:
        logger.info(f"Auto-switching market to {market_label}: Mode {settings.TRADING_MODE} -> {target_mode}, Symbols -> {target_symbols}")
        settings.TRADING_MODE = target_mode
        settings.DEFAULT_SYMBOLS = target_symbols
        
        # Send update to Telegram
        msg = (
            f"🔄 *Market Auto-Switch Triggered*\n"
            f"───────────────────\n"
            f"☀️ Active Market: *{market_label}* (Open)\n"
            f"🔌 Trading Mode: `{target_mode}`\n"
            f"📈 Symbols: `{target_symbols}`\n"
            f"🚀 QuantAI is now active in this market!"
        )
        try:
            send_telegram_message(msg)
        except Exception as e:
            logger.error(f"Error sending auto-switch Telegram update: {e}")
        return True, target_mode

    return False, target_mode


def run_trading_cycle(symbols: List[str] = None) -> dict:
    """
    Execute a full trading cycle for all configured symbols.
    Called by the scheduler on each tick.
    """
    # Check and apply auto-switch logic before resolving settings
    try:
        switched, active_mode = check_and_apply_auto_market_switch()
        if switched:
            logger.info(f"Market context auto-switched to mode: {active_mode}")
    except Exception as e:
        logger.error(f"Failed to check/apply auto market switch: {e}")

    symbols = symbols or settings.symbols_list
    db = SessionLocal()
    broker = get_broker()
    results = {}

    try:
        portfolio = get_or_create_portfolio(db)

        # Sync account balance and positions first to ensure sizing and limits are based on live broker data
        try:
            account = broker.get_account_info(db, portfolio)
            results["account_initial"] = account
            broker.get_positions(db, portfolio)
        except Exception as e:
            logger.error(f"Error syncing account info and positions from broker at start of cycle: {e}")

        # Check if trading is halted
        is_allowed, halt_reason = risk_manager.check_daily_loss_limit(portfolio)
        if not is_allowed:
            logger.warning(f"Trading halted: {halt_reason}")
            portfolio.trading_halted = True
            db.add(portfolio)
            db.commit()
            return {"halted": True, "reason": halt_reason}

        for symbol in symbols:
            try:
                result = _process_symbol(db, portfolio, broker, symbol)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = {"error": str(e)}

        # Check stop-losses on all open positions
        _check_stop_losses(db, portfolio, broker)

        # Update account info again at end of cycle
        account = broker.get_account_info(db, portfolio)
        results["account"] = account

    except Exception as e:
        logger.error(f"Trading cycle error: {e}")
        results["error"] = str(e)
    finally:
        db.close()

    return results


def _process_symbol(db: Session, portfolio: Portfolio, broker, symbol: str) -> dict:
    """Process a single symbol: fetch data, compute indicators, get signal, check risk, execute."""

    # 1. Fetch data and compute indicators
    df = get_ohlcv(symbol, "1d", lookback_days=settings.LOOKBACK_DAYS)
    if df.empty or len(df) < 30:
        return {"status": "no_data", "symbol": symbol}

    df = compute_all_indicators(df)
    df.dropna(inplace=True)

    last = df.iloc[-1]
    current_price = float(last["close"])
    atr = float(last.get("atr_14", current_price * 0.02))
    avg_atr = float(df["atr_14"].tail(20).mean()) if "atr_14" in df.columns else atr

    # 2. AI Prediction
    ai_pred = None
    if settings.STRATEGY_AI_ENABLED:
        predictor = get_predictor(symbol)
        if predictor.is_trained:
            accuracy = predictor.model_metrics.get("accuracy", 0.0)
            if accuracy < 0.50:
                logger.info(f"AI Model accuracy for {symbol} too low ({accuracy:.1%}). Skipping AI input, relying on technicals/news.")
            else:
                ai_pred = predictor.predict(df)

    # 3. Strategy signal
    sentiment_score = 0.0
    if settings.STRATEGY_NEWS_SENTIMENT_ENABLED:
        sentiment_score = get_news_sentiment(symbol)

    signal_result = strategy_engine.evaluate(df, symbol, ai_pred, news_sentiment=sentiment_score)
    consensus = signal_result.get("consensus", {})
    action = consensus.get("action", "hold")
    confidence = consensus.get("consensus_confidence", 0.5)

    # 4. Record signal in DB
    indicators_dict = {k: v for k, v in last.items() if pd.notna(v)} if hasattr(last, "items") else {}
    indicators_dict["news_sentiment"] = sentiment_score

    signal_db = Signal(
        symbol=symbol,
        action=action,
        strategy="consensus",
        confidence=confidence,
        price_at_signal=current_price,
        indicators=indicators_dict,
        ai_prediction=ai_pred.get("direction") if ai_pred else None,
        ai_confidence=ai_pred.get("confidence") if ai_pred else None,
        executed=False,
    )
    db.add(signal_db)

    # Record AI prediction
    if ai_pred:
        pred_db = AIPrediction(
            signal=signal_db,
            symbol=symbol,
            model_type=ai_pred.get("model_type", "unknown"),
            predicted_direction=ai_pred.get("direction", "NEUTRAL"),
            confidence=ai_pred.get("confidence", 0.5),
        )
        db.add(pred_db)

    db.commit()
    db.refresh(signal_db)

    if action == "hold":
        return {"status": "hold", "symbol": symbol, "confidence": confidence}

    # 5. Risk checks & position sizing
    sizing = risk_manager.calculate_position_size(portfolio.total_value, current_price, atr, "long")
    qty = sizing["qty"]

    if action == "buy":
        # Cap qty to available cash if it exceeds it to utilize remaining cash
        max_affordable_qty = (portfolio.cash * 0.99) / current_price
        if qty > max_affordable_qty:
            qty = round(max_affordable_qty, 4)
            if qty <= 0.0001:
                return {"status": "insufficient_cash_skipped", "symbol": symbol}

    # Skip if position already open for this symbol
    existing_position = db.query(Position).filter(
        Position.portfolio_id == portfolio.id,
        Position.symbol == symbol
    ).first()

    if action == "buy" and existing_position:
        return {"status": "already_open", "symbol": symbol}

    if action == "sell" and not existing_position:
        return {"status": "no_position_to_close", "symbol": symbol}

    is_valid, reason = risk_manager.validate_trade(
        portfolio, symbol, current_price, qty, atr, avg_atr
    )

    if not is_valid:
        risk_manager.log_risk_event(
            db, "trade_rejected", symbol=symbol, detail=reason, portfolio_value=portfolio.total_value
        )
        return {"status": "risk_rejected", "symbol": symbol, "reason": reason}

    # 6. Execute trade
    if action == "buy":
        result = broker.place_order(
            db=db, portfolio=portfolio, symbol=symbol, side="buy", qty=qty,
            stop_loss=sizing["stop_loss"], take_profit=sizing["take_profit"],
            strategy="consensus", signal_id=signal_db.id,
        )
    elif action == "sell" and existing_position:
        result = broker.close_position(db, portfolio, existing_position, current_price)
    else:
        result = {"success": False, "error": "Unknown action"}

    if result.get("success"):
        signal_db.executed = True
        db.add(signal_db)
        db.commit()

    return {
        "status": "executed" if result.get("success") else "failed",
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "price": current_price,
        "qty": qty,
        "broker_result": result,
    }


def _check_stop_losses(db: Session, portfolio: Portfolio, broker):
    """Check all open positions for stop-loss and take-profit triggers."""
    positions = db.query(Position).filter(Position.portfolio_id == portfolio.id).all()

    for pos in positions:
        current_price = get_latest_price(pos.symbol)
        if current_price <= 0:
            continue

        # Check stop-loss
        sl_hit, sl_reason = risk_manager.check_position_stop_loss(pos, current_price)
        if sl_hit:
            logger.info(f"Stop-loss hit for {pos.symbol}: {sl_reason}")
            broker.close_position(db, portfolio, pos, current_price, sl_reason)
            risk_manager.log_risk_event(
                db, "stop_loss_hit", symbol=pos.symbol, detail=sl_reason, portfolio_value=portfolio.total_value
            )
            continue

        # Check take-profit
        tp_hit, tp_reason = risk_manager.check_position_take_profit(pos, current_price)
        if tp_hit:
            logger.info(f"Take-profit hit for {pos.symbol}: {tp_reason}")
            broker.close_position(db, portfolio, pos, current_price, tp_reason)


def trigger_model_training(symbols: List[str] = None) -> dict:
    """Trigger AI model (re)training for all configured symbols."""
    symbols = symbols or settings.symbols_list
    symbol_data = {}

    for symbol in symbols:
        df = get_ohlcv(symbol, "1d", lookback_days=settings.LOOKBACK_DAYS * 2)
        if not df.empty:
            df = compute_all_indicators(df)
            df.dropna(inplace=True)
            symbol_data[symbol] = df

    results = train_all_predictors(symbol_data)
    logger.info(f"Model training complete for {len(results)} symbols")
    return results

