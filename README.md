# QuantAI — Autonomous AI-Powered Algorithmic Trading System

> A production-grade, modular, and fully autonomous AI-based stock trading platform.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Copy environment config
copy .env.example .env

# Start the backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 3. One-Click Start (Windows)

```bat
start.bat
```

Open **http://localhost:5173** in your browser.

---

## 📐 Architecture

```
autonomous-trading-system/
├── backend/
│   ├── app/
│   │   ├── config.py          # Settings & environment
│   │   ├── main.py            # FastAPI app + REST + WebSocket
│   │   ├── db/
│   │   │   ├── database.py    # SQLAlchemy engine
│   │   │   └── models.py      # All ORM table models
│   │   ├── data/
│   │   │   ├── data_loader.py # yfinance + caching
│   │   │   └── indicators.py  # Technical indicators
│   │   ├── models/
│   │   │   └── predictor.py   # XGBoost/RandomForest AI engine
│   │   ├── strategies/
│   │   │   └── engine.py      # 4 strategy rules + consensus
│   │   ├── risk/
│   │   │   └── manager.py     # Position sizing, SL/TP, limits
│   │   ├── execution/
│   │   │   ├── broker.py      # SimulatedBroker + AlpacaBroker
│   │   │   └── orchestrator.py # Main trading loop
│   │   └── backtester/
│   │       └── engine.py      # Walk-forward backtesting
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── App.jsx            # Main app with page routing
│       ├── index.css          # Dark terminal design system
│       ├── components/
│       │   ├── Header.jsx
│       │   ├── PortfolioSummary.jsx
│       │   ├── PriceChart.jsx
│       │   ├── ActivePositions.jsx
│       │   ├── EquityChart.jsx
│       │   ├── SignalsPanel.jsx
│       │   ├── AIModelPanel.jsx
│       │   ├── BacktestPanel.jsx
│       │   └── SettingsPanel.jsx
│       ├── hooks/
│       │   └── useWebSocket.js
│       └── utils/
│           └── api.js
│
├── docker-compose.yml
└── start.bat
```

---

## ⚙️ Configuration

Edit `backend/.env`:

| Variable | Default | Description |
|---|---|---|
| `TRADING_MODE` | `paper_sim` | `paper_sim` (local) \| `paper_alpaca` \| `live` |
| `INITIAL_CAPITAL` | `1000.0` | Starting capital in USD |
| `ALPACA_API_KEY` | (blank) | Required only for Alpaca modes |
| `ALPACA_SECRET_KEY` | (blank) | Required only for Alpaca modes |
| `AI_CONFIDENCE_THRESHOLD` | `0.65` | Min AI confidence to trade |
| `MAX_RISK_PER_TRADE_PCT` | `0.01` | Max 1% capital per trade |
| `MAX_DAILY_LOSS_PCT` | `0.03` | Halt trading after 3% daily loss |
| `MAX_OPEN_POSITIONS` | `5` | Maximum concurrent positions |
| `DEFAULT_SYMBOLS` | `AAPL,MSFT,NVDA,SPY,QQQ` | Symbols to track and trade |

---

## 🧠 AI Prediction Engine

The system uses a **Phase 1 ML ensemble** trained on historical OHLCV + indicators:
- **XGBoost** (primary, when available)
- **Random Forest** (fallback)

**Features used:** RSI, MACD, EMA, Stochastic RSI, Bollinger Bands %, ATR, OBV, VWAP, momentum score, volume ratios, 5/10-day returns.

**Training:** Triggered via `POST /api/ai/train` or the "Train All Models" button.  
**Retraining:** Auto-scheduled every `MODEL_RETRAIN_INTERVAL_HOURS` hours.

---

## 📊 Trading Strategies

| Strategy | Signal Logic |
|---|---|
| **Momentum** | EMA9 > EMA21 + RSI > 50 + MACD crossover |
| **Mean Reversion** | RSI < 30 + price near BB lower band |
| **Breakout** | Price closes above 20-day resistance w/ high volume |
| **AI Confidence** | AI predicts UP/DOWN with ≥ 65% confidence |

All strategies vote and are **aggregated by weighted confidence scoring**.

---

## 🛡️ Risk Management

- **Position sizing:** `Risk Amount = Capital × 1%`, `Qty = Risk / (ATR × 2)`
- **Stop Loss:** ATR × 2 below entry (dynamic)
- **Take Profit:** ATR × 4 above entry (2:1 R/R ratio)
- **Max Drawdown:** Trading halts if daily loss exceeds 3%
- **Max Positions:** Hard limit of 5 simultaneous positions
- **Volatility Filter:** Skips trades when ATR > 2.5× average

---

## 📈 Backtesting

The walk-forward backtesting engine:
1. Replays historical bars sequentially
2. Applies strategy rules and risk manager
3. Simulates fills with slippage (0.01%) and commission (0.1%)
4. Computes: **Total Return, Sharpe Ratio, Max Drawdown, CAGR, Win Rate, Profit Factor**

Run via the **Backtesting** page in the dashboard or:
```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"symbol":"SPY","lookback_days":365,"initial_capital":1000}'
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/portfolio` | Portfolio balance & metrics |
| `GET` | `/api/positions` | Open positions with current P&L |
| `GET` | `/api/trades` | Trade history |
| `POST` | `/api/trades/manual` | Place a manual trade |
| `DELETE` | `/api/positions/{symbol}` | Close a position |
| `GET` | `/api/market/{symbol}/chart` | OHLCV + indicators |
| `GET` | `/api/signals` | Recent trading signals |
| `POST` | `/api/signals/generate` | Trigger trading cycle |
| `GET` | `/api/ai/status` | AI model metrics per symbol |
| `POST` | `/api/ai/train` | Retrain all AI models |
| `POST` | `/api/backtest` | Run backtest |
| `GET` | `/api/risk/logs` | Risk management event log |
| `WS` | `/ws` | Real-time portfolio & signal feed |
| `GET` | `/docs` | Swagger UI |

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

Runs:
- Backend on port `8000`
- Frontend on port `5173` (mapped to Nginx :80)

---

## 🗺️ Roadmap

### Phase 2 (Next)
- [ ] LSTM time-series models
- [ ] FinBERT sentiment analysis from news
- [ ] Reinforcement Learning (PPO agent)
- [ ] Multi-broker support (Interactive Brokers)
- [ ] Crypto trading via CCXT

### Phase 3 (Future)
- [ ] Multi-agent AI with strategy specialization
- [ ] Portfolio optimization (Markowitz)
- [ ] High-frequency optimization
- [ ] Multi-user SaaS model
- [ ] Cloud deployment (AWS ECS + RDS + ElastiCache)

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**.  
**Do NOT trade real money without thoroughly testing and understanding the system.**  
Past backtesting performance does not guarantee future results.  
Always start with paper trading mode (`TRADING_MODE=paper_sim`).
