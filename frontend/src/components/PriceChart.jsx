import { useState, useMemo } from 'react';
import {
  ComposedChart,
  AreaChart,
  BarChart,
  LineChart,
  Area,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  ChevronDown,
  Activity,
  BarChart2,
  Loader,
} from 'lucide-react';

/* ─────────────────────────────────────────────
   Constants
───────────────────────────────────────────── */
const TIMEFRAMES = ['1D', '5D', '1M', '3M', '1Y'];
const SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD', 'NFLX', 'JPM', 'V', 'DIS', 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', '^NSEI'];

/* ─────────────────────────────────────────────
   Custom Tooltip
───────────────────────────────────────────── */
function PriceTooltip({ active, payload, label }) {
  if (!active || !payload || !payload.length) return null;

  const d = payload[0]?.payload ?? {};
  const fmt = (v, digits = 2) =>
    v !== undefined && v !== null ? Number(v).toFixed(digits) : '—';
  const fmtVol = (v) =>
    v !== undefined ? (v >= 1_000_000 ? `${(v / 1_000_000).toFixed(2)}M` : `${(v / 1_000).toFixed(0)}K`) : '—';

  return (
    <div className="price-tooltip">
      <div className="price-tooltip__date">{label}</div>
      <div className="price-tooltip__grid">
        <span className="price-tooltip__label">O</span>
        <span className="price-tooltip__value">{fmt(d.open)}</span>
        <span className="price-tooltip__label">H</span>
        <span className="price-tooltip__value text-positive">{fmt(d.high)}</span>
        <span className="price-tooltip__label">L</span>
        <span className="price-tooltip__value text-negative">{fmt(d.low)}</span>
        <span className="price-tooltip__label">C</span>
        <span className="price-tooltip__value text-accent">{fmt(d.close)}</span>
        <span className="price-tooltip__label">Vol</span>
        <span className="price-tooltip__value">{fmtVol(d.volume)}</span>
      </div>
      <div className="price-tooltip__divider" />
      <div className="price-tooltip__grid">
        <span className="price-tooltip__label">EMA9</span>
        <span className="price-tooltip__value" style={{ color: '#f59e0b' }}>{fmt(d.ema_9)}</span>
        <span className="price-tooltip__label">EMA21</span>
        <span className="price-tooltip__value" style={{ color: '#a855f7' }}>{fmt(d.ema_21)}</span>
        <span className="price-tooltip__label">RSI</span>
        <span
          className="price-tooltip__value"
          style={{
            color:
              d.rsi_14 >= 70 ? 'var(--accent-red)' :
              d.rsi_14 <= 30 ? 'var(--accent-green)' :
              'var(--text-primary)',
          }}
        >
          {fmt(d.rsi_14)}
        </span>
        <span className="price-tooltip__label">MACD</span>
        <span
          className="price-tooltip__value"
          style={{ color: d.macd >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}
        >
          {fmt(d.macd)}
        </span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Volume Tooltip
───────────────────────────────────────────── */
function VolumeTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const vol = payload[0]?.value;
  const fmtVol = (v) =>
    v >= 1_000_000 ? `${(v / 1_000_000).toFixed(2)}M` : `${(v / 1_000).toFixed(0)}K`;
  return (
    <div className="price-tooltip">
      <span className="price-tooltip__label">Volume </span>
      <span className="price-tooltip__value text-accent">{fmtVol(vol)}</span>
    </div>
  );
}

/* ─────────────────────────────────────────────
   RSI Tooltip
───────────────────────────────────────────── */
function RsiTooltip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const rsi = payload[0]?.value;
  return (
    <div className="price-tooltip">
      <span className="price-tooltip__label">RSI(14) </span>
      <span
        className="price-tooltip__value"
        style={{
          color: rsi >= 70 ? 'var(--accent-red)' : rsi <= 30 ? 'var(--accent-green)' : 'var(--text-primary)',
        }}
      >
        {Number(rsi).toFixed(2)}
      </span>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Skeleton
───────────────────────────────────────────── */
function ChartSkeleton() {
  return (
    <div className="chart-skeleton">
      <div className="skeleton chart-skeleton__header" />
      <div className="skeleton chart-skeleton__main" />
      <div className="skeleton chart-skeleton__sub" />
      <div className="skeleton chart-skeleton__rsi" />
    </div>
  );
}

/* ─────────────────────────────────────────────
   Main Component
───────────────────────────────────────────── */
export default function PriceChart({ symbol, onSymbolChange, chartData = [], loading = false, livePrice, symbols = SYMBOLS }) {
  const [activeTimeframe, setActiveTimeframe] = useState('1D');
  const [symbolOpen, setSymbolOpen] = useState(false);

  /* Derived price info */
  const currentPrice = useMemo(() => {
    if (livePrice !== undefined && livePrice !== null) return livePrice;
    if (!chartData.length) return null;
    return chartData[chartData.length - 1]?.close ?? null;
  }, [chartData, livePrice]);

  const priceChange = useMemo(() => {
    if (chartData.length < 2) return { value: 0, pct: 0 };
    const first = chartData[0]?.close ?? 0;
    const last = livePrice !== undefined && livePrice !== null ? livePrice : (chartData[chartData.length - 1]?.close ?? 0);
    const value = last - first;
    const pct = first !== 0 ? (value / first) * 100 : 0;
    return { value, pct };
  }, [chartData, livePrice]);

  const isPositive = priceChange.value >= 0;

  /* Y-axis domain for price */
  const priceDomain = useMemo(() => {
    if (!chartData.length) return ['auto', 'auto'];
    const lows = chartData.map((d) => d.low).filter(Boolean);
    const highs = chartData.map((d) => d.high).filter(Boolean);
    const min = Math.min(...lows);
    const max = Math.max(...highs);
    const pad = (max - min) * 0.05;
    return [min - pad, max + pad];
  }, [chartData]);

  const handleSymbolSelect = (sym) => {
    onSymbolChange?.(sym);
    setSymbolOpen(false);
  };

  const tickFormatter = (val) => `$${Number(val).toFixed(0)}`;
  const rsiFormatter = (val) => Number(val).toFixed(0);

  return (
    <div className="card price-chart-card">
      {/* ── Header row ── */}
      <div className="price-chart__header">
        {/* Symbol selector */}
        <div className="price-chart__symbol-wrap">
          <button
            className="price-chart__symbol-btn"
            onClick={() => setSymbolOpen((p) => !p)}
            aria-haspopup="listbox"
            aria-expanded={symbolOpen}
          >
            <span className="price-chart__symbol-text">{symbol ?? 'AAPL'}</span>
            <ChevronDown size={14} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          </button>
          {symbolOpen && (
            <ul className="price-chart__symbol-dropdown" role="listbox">
              {symbols.map((sym) => (
                <li
                  key={sym}
                  role="option"
                  aria-selected={sym === symbol}
                  className={`price-chart__symbol-option${sym === symbol ? ' active' : ''}`}
                  onClick={() => handleSymbolSelect(sym)}
                >
                  {sym}
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Price + change */}
        <div className="price-chart__price-block">
          {loading ? (
            <div className="skeleton" style={{ width: 160, height: 36, borderRadius: 6 }} />
          ) : currentPrice !== null ? (
            <>
              <span className="price-chart__current-price">
                ${Number(currentPrice).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span className={`price-chart__change-badge ${isPositive ? 'positive' : 'negative'}`}>
                {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                {isPositive ? '+' : ''}{priceChange.value.toFixed(2)} ({isPositive ? '+' : ''}{priceChange.pct.toFixed(2)}%)
              </span>
            </>
          ) : (
            <span className="price-chart__no-data">No Data</span>
          )}
        </div>

        {/* Timeframe tabs */}
        <div className="price-chart__tf-wrap">
          <div className="tabs">
            {TIMEFRAMES.map((tf) => (
              <button
                key={tf}
                className={`tab${activeTimeframe === tf ? ' active' : ''}`}
                onClick={() => setActiveTimeframe(tf)}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Charts ── */}
      {loading ? (
        <ChartSkeleton />
      ) : !chartData.length ? (
        <div className="price-chart__empty">
          <Activity size={40} style={{ color: 'var(--text-muted)', marginBottom: 12 }} />
          <p style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            No chart data available
          </p>
        </div>
      ) : (
        <div className="price-chart__charts-wrap">

          {/* SVG gradient defs shared via hidden element */}
          <svg width={0} height={0} style={{ position: 'absolute' }}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.35} />
                <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0.0} />
              </linearGradient>
              <linearGradient id="rsiGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--accent-purple)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--accent-purple)" stopOpacity={0.0} />
              </linearGradient>
            </defs>
          </svg>

          {/* ── Price + EMA Chart ── */}
          <div className="price-chart__section">
            <div className="price-chart__section-label">
              <BarChart2 size={12} />
              PRICE &amp; EMA
            </div>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" vertical={false} />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickLine={false}
                  axisLine={false}
                  minTickGap={40}
                />
                <YAxis
                  domain={priceDomain}
                  tickFormatter={tickFormatter}
                  tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                  tickLine={false}
                  axisLine={false}
                  width={62}
                  orientation="right"
                />
                <Tooltip content={<PriceTooltip />} />
                {/* Area: close price */}
                <Area
                  type="monotone"
                  dataKey="close"
                  stroke="var(--accent-primary)"
                  strokeWidth={2}
                  fill="url(#priceGradient)"
                  dot={false}
                  activeDot={{ r: 4, fill: 'var(--accent-primary)', stroke: 'var(--bg-card)', strokeWidth: 2 }}
                  name="Close"
                />
                {/* EMA 9 */}
                <Line
                  type="monotone"
                  dataKey="ema_9"
                  stroke="#f59e0b"
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  dot={false}
                  name="EMA9"
                />
                {/* EMA 21 */}
                <Line
                  type="monotone"
                  dataKey="ema_21"
                  stroke="#a855f7"
                  strokeWidth={1.5}
                  strokeDasharray="8 4"
                  dot={false}
                  name="EMA21"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* ── Volume Bar Chart ── */}
          <div className="price-chart__section">
            <div className="price-chart__section-label">
              <BarChart2 size={12} />
              VOLUME
            </div>
            <ResponsiveContainer width="100%" height={80}>
              <BarChart data={chartData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }} barSize={3}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.04)" vertical={false} />
                <XAxis dataKey="timestamp" hide />
                <YAxis
                  tickFormatter={(v) => v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}K`}
                  tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                  tickLine={false}
                  axisLine={false}
                  width={44}
                  orientation="right"
                />
                <Tooltip content={<VolumeTooltip />} />
                <Bar
                  dataKey="volume"
                  fill="var(--accent-primary)"
                  fillOpacity={0.4}
                  radius={[2, 2, 0, 0]}
                  name="Volume"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* ── RSI Chart ── */}
          <div className="price-chart__section">
            <div className="price-chart__section-label">
              <Activity size={12} />
              RSI (14)
            </div>
            <ResponsiveContainer width="100%" height={90}>
              <ComposedChart data={chartData} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.04)" vertical={false} />
                <XAxis dataKey="timestamp" hide />
                <YAxis
                  domain={[0, 100]}
                  ticks={[0, 30, 50, 70, 100]}
                  tickFormatter={rsiFormatter}
                  tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'var(--font-mono)' }}
                  tickLine={false}
                  axisLine={false}
                  width={28}
                  orientation="right"
                />
                <Tooltip content={<RsiTooltip />} />
                <ReferenceLine y={70} stroke="var(--accent-red)" strokeDasharray="4 3" strokeOpacity={0.6} />
                <ReferenceLine y={30} stroke="var(--accent-green)" strokeDasharray="4 3" strokeOpacity={0.6} />
                <ReferenceLine y={50} stroke="var(--text-muted)" strokeDasharray="2 4" strokeOpacity={0.3} />
                <Area
                  type="monotone"
                  dataKey="rsi_14"
                  stroke="var(--accent-purple)"
                  strokeWidth={1.5}
                  fill="url(#rsiGradient)"
                  dot={false}
                  name="RSI"
                />
              </ComposedChart>
            </ResponsiveContainer>
            {/* RSI zone labels */}
            <div className="price-chart__rsi-labels">
              <span style={{ color: 'var(--accent-red)', fontSize: 9 }}>Overbought 70</span>
              <span style={{ color: 'var(--text-muted)', fontSize: 9 }}>Neutral 50</span>
              <span style={{ color: 'var(--accent-green)', fontSize: 9 }}>Oversold 30</span>
            </div>
          </div>
        </div>
      )}

      {/* Inline scoped styles */}
      <style>{`
        /* ── Card ── */
        .price-chart-card {
          padding: var(--space-4) var(--space-5);
          display: flex;
          flex-direction: column;
          gap: var(--space-3);
        }

        /* ── Header ── */
        .price-chart__header {
          display: flex;
          align-items: center;
          flex-wrap: wrap;
          gap: var(--space-3);
        }

        /* Symbol selector */
        .price-chart__symbol-wrap {
          position: relative;
          flex-shrink: 0;
        }
        .price-chart__symbol-btn {
          display: inline-flex;
          align-items: center;
          gap: var(--space-2);
          background: var(--bg-surface);
          border: 1px solid var(--border-dim);
          border-radius: var(--radius-md);
          color: var(--text-primary);
          font-family: var(--font-mono);
          font-size: 15px;
          font-weight: 700;
          padding: var(--space-2) var(--space-3);
          cursor: pointer;
          transition: border-color var(--transition-fast);
          letter-spacing: 0.5px;
        }
        .price-chart__symbol-btn:hover {
          border-color: var(--accent-primary);
        }
        .price-chart__symbol-text {
          color: var(--accent-primary);
        }
        .price-chart__symbol-dropdown {
          position: absolute;
          top: calc(100% + 6px);
          left: 0;
          z-index: 50;
          background: var(--bg-elevated);
          border: 1px solid var(--border-dim);
          border-radius: var(--radius-md);
          list-style: none;
          min-width: 120px;
          overflow: hidden;
          box-shadow: var(--shadow-card);
        }
        .price-chart__symbol-option {
          padding: var(--space-2) var(--space-3);
          font-family: var(--font-mono);
          font-size: 13px;
          font-weight: 600;
          color: var(--text-secondary);
          cursor: pointer;
          transition: background var(--transition-fast), color var(--transition-fast);
        }
        .price-chart__symbol-option:hover,
        .price-chart__symbol-option.active {
          background: var(--accent-primary-dim);
          color: var(--accent-primary);
        }

        /* Price block */
        .price-chart__price-block {
          display: flex;
          align-items: baseline;
          gap: var(--space-3);
          flex: 1;
        }
        .price-chart__current-price {
          font-family: var(--font-mono);
          font-size: 28px;
          font-weight: 700;
          color: var(--text-primary);
          letter-spacing: -0.5px;
          line-height: 1;
        }
        .price-chart__change-badge {
          display: inline-flex;
          align-items: center;
          gap: 4px;
          padding: 4px 10px;
          border-radius: var(--radius-full);
          font-family: var(--font-mono);
          font-size: 12px;
          font-weight: 600;
        }
        .price-chart__change-badge.positive {
          background: var(--accent-green-dim);
          color: var(--accent-green);
          border: 1px solid rgba(0,255,148,0.2);
        }
        .price-chart__change-badge.negative {
          background: var(--accent-red-dim);
          color: var(--accent-red);
          border: 1px solid rgba(255,51,102,0.2);
        }
        .price-chart__no-data {
          color: var(--text-muted);
          font-family: var(--font-mono);
          font-size: 14px;
        }

        /* Timeframe tabs */
        .price-chart__tf-wrap {
          margin-left: auto;
          flex-shrink: 0;
        }
        .price-chart__tf-wrap .tabs {
          gap: 2px;
        }
        .price-chart__tf-wrap .tab {
          flex: unset;
          padding: 5px 12px;
          font-size: 11px;
          font-family: var(--font-mono);
          letter-spacing: 0.5px;
        }

        /* Charts wrap */
        .price-chart__charts-wrap {
          display: flex;
          flex-direction: column;
          gap: 0;
        }
        .price-chart__section {
          position: relative;
        }
        .price-chart__section + .price-chart__section {
          border-top: 1px solid var(--border-subtle);
          padding-top: var(--space-2);
          margin-top: var(--space-2);
        }
        .price-chart__section-label {
          display: flex;
          align-items: center;
          gap: 5px;
          font-size: 10px;
          font-weight: 600;
          letter-spacing: 1.2px;
          text-transform: uppercase;
          color: var(--text-muted);
          margin-bottom: var(--space-1);
          font-family: var(--font-mono);
        }

        /* RSI labels */
        .price-chart__rsi-labels {
          display: flex;
          justify-content: space-between;
          padding: 2px 6px 0;
          font-family: var(--font-mono);
        }

        /* Empty state */
        .price-chart__empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 320px;
          gap: 0;
        }

        /* Tooltip */
        .price-tooltip {
          background: var(--bg-elevated);
          border: 1px solid var(--border-dim);
          border-radius: var(--radius-md);
          padding: var(--space-3) var(--space-3);
          font-family: var(--font-mono);
          font-size: 11px;
          box-shadow: var(--shadow-card);
          min-width: 140px;
        }
        .price-tooltip__date {
          color: var(--text-muted);
          font-size: 10px;
          margin-bottom: var(--space-2);
          letter-spacing: 0.5px;
        }
        .price-tooltip__grid {
          display: grid;
          grid-template-columns: auto 1fr;
          gap: 2px 10px;
        }
        .price-tooltip__label {
          color: var(--text-muted);
          font-size: 10px;
          letter-spacing: 0.5px;
        }
        .price-tooltip__value {
          color: var(--text-primary);
          font-weight: 600;
        }
        .price-tooltip__divider {
          border-top: 1px solid var(--border-subtle);
          margin: var(--space-2) 0;
        }

        /* Skeleton */
        .chart-skeleton {
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          padding: var(--space-2) 0;
        }
        .chart-skeleton__header {
          height: 20px;
          width: 60%;
        }
        .chart-skeleton__main {
          height: 280px;
        }
        .chart-skeleton__sub {
          height: 80px;
        }
        .chart-skeleton__rsi {
          height: 90px;
        }
      `}</style>
    </div>
  );
}
