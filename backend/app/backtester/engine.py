"""
Backtesting Engine
Simulates trading strategies on historical data.
Computes performance metrics: Sharpe, Drawdown, Win Rate, Profit Factor, CAGR.
"""
import logging
from typing import Optional, List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.data.data_loader import get_ohlcv
from app.data.indicators import compute_all_indicators
from app.strategies.engine import StrategyEngine
from app.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class BacktestResult:
    """Container for backtest performance metrics."""

    def __init__(
        self,
        symbol: str,
        initial_capital: float,
        equity_curve: List[float],
        trades: List[dict],
        start_date: str,
        end_date: str,
        dates: List[str] = None,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.equity_curve = equity_curve
        self.trades = trades
        self.start_date = start_date
        self.end_date = end_date
        self.dates = dates or []
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute performance statistics from equity curve and trade list."""
        eq = np.array(self.equity_curve)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])

        final_value = eq[-1] if len(eq) > 0 else self.initial_capital
        self.total_return = (final_value - self.initial_capital) / self.initial_capital
        self.total_return_dollars = final_value - self.initial_capital

        # Sharpe Ratio (annualized, assume daily data = ~252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            self.sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        else:
            self.sharpe_ratio = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / np.where(peak > 0, peak, 1)
        self.max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # CAGR
        n_days = max(len(eq), 1)
        years = n_days / 252.0
        if years > 0 and self.initial_capital > 0:
            self.cagr = float((final_value / self.initial_capital) ** (1 / years) - 1)
        else:
            self.cagr = 0.0

        # Trade stats
        completed_trades = [t for t in self.trades if t.get("pnl") is not None]
        winning_trades = [t for t in completed_trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in completed_trades if t.get("pnl", 0) <= 0]

        self.total_trades = len(completed_trades)
        self.winning_trades = len(winning_trades)
        self.losing_trades = len(losing_trades)
        self.win_rate = self.winning_trades / max(self.total_trades, 1)

        gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
        gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
        self.profit_factor = gross_profit / max(gross_loss, 0.01)

        self.avg_win = gross_profit / max(self.winning_trades, 1)
        self.avg_loss = -gross_loss / max(self.losing_trades, 1)

    def to_dict(self) -> dict:
        data = {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_value": self.initial_capital + self.total_return_dollars,
            "total_return_pct": round(self.total_return * 100, 2),
            "total_return_dollars": round(self.total_return_dollars, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "cagr_pct": round(self.cagr * 100, 2),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 2),
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "equity_curve": [round(v, 2) for v in self.equity_curve],
            "trades": self.trades[-50:],  # Return last 50 trades
        }

        if hasattr(self, "benchmark_curve") and self.benchmark_curve:
            data["benchmark_curve"] = [round(v, 2) for v in self.benchmark_curve]
            data["benchmark_symbol"] = getattr(self, "benchmark_symbol", "SPY")

        if hasattr(self, "individual_results") and self.individual_results:
            data["individual_results"] = self.individual_results

        return data



class BacktestEngine:
    """
    Walk-forward backtesting engine.
    
    Replays historical OHLCV data bar-by-bar,
    applies strategy signals and risk management,
    and simulates paper trade execution.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        commission_pct: float = 0.001,  # 0.1%
        slippage_pct: float = 0.0001,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.strategy_engine = StrategyEngine()
        self.risk_manager = RiskManager()

    def run(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        timeframe: str = "1d",
        lookback_days: int = 365,
        use_ai: bool = False,
    ) -> BacktestResult:
        """
        Run the backtest for a given symbol and date range.
        Returns a BacktestResult with full metrics.
        """
        logger.info(f"Starting backtest: {symbol} ({timeframe}, {lookback_days}d)")

        symbols = [s.strip().upper() for s in symbol.split(",") if s.strip()]
        if len(symbols) > 1:
            individual_results = []
            symbol_capital = self.initial_capital / len(symbols)
            for sym in symbols:
                sym_engine = BacktestEngine(
                    initial_capital=symbol_capital,
                    commission_pct=self.commission_pct,
                    slippage_pct=self.slippage_pct
                )
                sym_res = sym_engine.run(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    lookback_days=lookback_days,
                    use_ai=use_ai
                )
                individual_results.append(sym_res)

            valid_results = [r for r in individual_results if len(r.dates) > 0]
            if not valid_results:
                logger.error("All symbols in portfolio backtest failed to return data.")
                return BacktestResult(
                    symbol=symbol,
                    initial_capital=self.initial_capital,
                    equity_curve=[self.initial_capital],
                    trades=[],
                    start_date=start_date or "",
                    end_date=end_date or "",
                    dates=[]
                )

            # Align equity curves of all symbols
            series_list = []
            for r in valid_results:
                ser = pd.Series(r.equity_curve, index=pd.to_datetime(r.dates))
                ser = ser[~ser.index.duplicated(keep="first")]
                series_list.append(ser)

            # Create combined DataFrame
            portfolio_df = pd.concat(series_list, axis=1)
            portfolio_df.columns = [r.symbol for r in valid_results]
            portfolio_df = portfolio_df.ffill().bfill()

            # Combined equity curve (sum across symbols)
            combined_equity = portfolio_df.sum(axis=1)
            combined_equity_values = combined_equity.tolist()
            combined_dates = [d.isoformat() for d in combined_equity.index]

            # Combine and sort trades
            combined_trades = []
            for r in valid_results:
                combined_trades.extend(r.trades)
            combined_trades.sort(key=lambda t: t.get("date", ""))

            actual_start = combined_dates[0] if combined_dates else (start_date or "")
            actual_end = combined_dates[-1] if combined_dates else (end_date or "")

            portfolio_result = BacktestResult(
                symbol=symbol,
                initial_capital=self.initial_capital,
                equity_curve=combined_equity_values,
                trades=combined_trades,
                start_date=actual_start,
                end_date=actual_end,
                dates=combined_dates
            )

            # Attach individual metrics
            portfolio_result.individual_results = {
                r.symbol: {
                    "total_return_pct": round(r.total_return * 100, 2),
                    "total_return_dollars": round(r.total_return_dollars, 2),
                    "sharpe_ratio": round(r.sharpe_ratio, 3),
                    "max_drawdown_pct": round(r.max_drawdown * 100, 2),
                    "win_rate_pct": round(r.win_rate * 100, 2),
                    "total_trades": r.total_trades
                }
                for r in valid_results
            }
        else:
            # Single symbol path
            fetch_lookback = lookback_days + 250
            df = get_ohlcv(symbol, timeframe, fetch_lookback)
            if df.empty:
                logger.error(f"No data for {symbol}")
                return BacktestResult(
                    symbol, self.initial_capital, [self.initial_capital], [], start_date or "", end_date or "", []
                )

            df = compute_all_indicators(df)
            df.dropna(inplace=True)

            # Filter by date range or keep the last lookback_days
            if start_date:
                df = df[df.index >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]
            elif not start_date and not end_date:
                df = df.iloc[-lookback_days:]

            if len(df) < 30:
                logger.warning(f"Insufficient data after filtering: {len(df)} rows")
                return BacktestResult(
                    symbol, self.initial_capital, [self.initial_capital], [], start_date or "", end_date or "", []
                )

            # Optionally train AI model
            ai_predictor = None
            if use_ai:
                from app.models.predictor import get_predictor
                ai_predictor = get_predictor(symbol)
                if not ai_predictor.is_trained:
                    logger.info(f"Training AI model for {symbol} before backtest...")
                    ai_predictor.train(df)

            # Initialize state
            cash = self.initial_capital
            position = None  # {qty, entry_price, stop_loss, take_profit}
            equity_curve = [self.initial_capital]
            trades = []

            # Walk-forward simulation
            for i in range(30, len(df)):
                bar = df.iloc[i]
                hist = df.iloc[:i]

                close_price = float(bar["close"])
                atr = float(bar.get("atr_14", close_price * 0.02))

                # Check stop-loss / take-profit on existing position
                if position:
                    pnl = 0.0
                    exit_reason = None

                    if position["side"] == "long":
                        if close_price <= position["stop_loss"]:
                            exit_reason = "stop_loss"
                        elif close_price >= position["take_profit"]:
                            exit_reason = "take_profit"
                        elif i == len(df) - 1:
                            exit_reason = "end_of_data"

                    if exit_reason:
                        exec_price = close_price * (1 - self.slippage_pct)
                        pnl = (exec_price - position["entry_price"]) * position["qty"]
                        commission = exec_price * position["qty"] * self.commission_pct
                        proceeds = exec_price * position["qty"] - commission
                        cash += proceeds

                        trades.append({
                            "symbol": symbol,
                            "side": "sell",
                            "qty": position["qty"],
                            "entry_price": position["entry_price"],
                            "exit_price": exec_price,
                            "pnl": round(pnl - commission, 2),
                            "reason": exit_reason,
                            "bar_index": i,
                            "date": df.index[i].isoformat(),
                            "strategy": position.get("strategy", "N/A"),
                        })
                        position = None

                # Generate new signal if no position open
                if position is None:
                    ai_pred = None
                    if ai_predictor and ai_predictor.is_trained:
                        ai_pred = ai_predictor.predict(hist)

                    signal = self.strategy_engine.evaluate(hist, symbol, ai_pred)
                    consensus = signal.get("consensus", {})
                    action = consensus.get("action", "hold")
                    confidence = consensus.get("consensus_confidence", 0.5)

                    if action == "buy" and confidence > 0.55:
                        # Calculate position sizing
                        sizing = self.risk_manager.calculate_position_size(
                            cash, close_price, atr, "long"
                        )
                        qty = sizing["qty"]

                        if qty > 0:
                            exec_price = close_price * (1 + self.slippage_pct)
                            trade_value = exec_price * qty
                            commission = trade_value * self.commission_pct

                            if trade_value + commission <= cash:
                                cash -= (trade_value + commission)
                                position = {
                                    "qty": qty,
                                    "entry_price": exec_price,
                                    "stop_loss": sizing["stop_loss"],
                                    "take_profit": sizing["take_profit"],
                                    "side": "long",
                                    "strategy": consensus.get("votes", {}).get("buy", [{}])[0].get("strategy", "N/A") if consensus.get("votes", {}).get("buy") else "N/A",
                                    "bar_open_index": i,
                                }

                                trades.append({
                                    "symbol": symbol,
                                    "side": "buy",
                                    "qty": qty,
                                    "entry_price": exec_price,
                                    "exit_price": None,
                                    "pnl": None,
                                    "reason": "entry",
                                    "bar_index": i,
                                    "date": df.index[i].isoformat(),
                                    "strategy": position["strategy"],
                                })

                # Calculate equity
                position_value = (close_price * position["qty"]) if position else 0.0
                equity_curve.append(round(cash + position_value, 2))

            actual_start = df.index[0].isoformat() if len(df) > 0 else (start_date or "")
            actual_end = df.index[-1].isoformat() if len(df) > 0 else (end_date or "")

            dates = [df.index[29].isoformat()] + [df.index[idx].isoformat() for idx in range(30, len(df))]

            portfolio_result = BacktestResult(
                symbol=symbol,
                initial_capital=self.initial_capital,
                equity_curve=equity_curve,
                trades=trades,
                start_date=actual_start,
                end_date=actual_end,
                dates=dates
            )

        # Calculate benchmark equity curve
        is_indian = any(".NS" in sym or sym == "^NSEI" for sym in symbols)
        benchmark_symbol = "^NSEI" if is_indian else "SPY"

        try:
            bench_df = get_ohlcv(benchmark_symbol, timeframe, lookback_days + 250)
            if not bench_df.empty:
                target_index = pd.to_datetime(portfolio_result.dates)
                bench_series = bench_df["close"]

                # Handle timezone differences
                if target_index.tz is not None and bench_series.index.tz is None:
                    bench_series.index = bench_series.index.tz_localize(target_index.tz)
                elif target_index.tz is None and bench_series.index.tz is not None:
                    bench_series.index = bench_series.index.tz_convert(None)

                bench_aligned = bench_series.reindex(target_index).ffill().bfill()

                if len(bench_aligned) > 0 and bench_aligned.iloc[0] > 0:
                    initial_price = bench_aligned.iloc[0]
                    benchmark_equity = [
                        round(self.initial_capital * (p / initial_price), 2)
                        for p in bench_aligned
                    ]
                    portfolio_result.benchmark_curve = benchmark_equity
                    portfolio_result.benchmark_symbol = benchmark_symbol
                else:
                    portfolio_result.benchmark_curve = [self.initial_capital] * len(portfolio_result.dates)
                    portfolio_result.benchmark_symbol = benchmark_symbol
            else:
                portfolio_result.benchmark_curve = [self.initial_capital] * len(portfolio_result.dates)
                portfolio_result.benchmark_symbol = benchmark_symbol
        except Exception as e:
            logger.error(f"Failed to calculate benchmark for {benchmark_symbol}: {e}")
            portfolio_result.benchmark_curve = [self.initial_capital] * len(portfolio_result.dates)
            portfolio_result.benchmark_symbol = benchmark_symbol

        logger.info(
            f"Backtest complete: {portfolio_result.total_return:.2%} return, "
            f"{portfolio_result.sharpe_ratio:.2f} Sharpe, {portfolio_result.win_rate:.2%} win rate"
        )
        return portfolio_result

