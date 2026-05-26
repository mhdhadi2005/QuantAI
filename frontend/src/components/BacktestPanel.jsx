import { useState } from 'react'
import { BarChart2, Play, TrendingUp, Award } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from 'recharts'
import { runBacktest } from '../utils/api'

const PRESETS = [
  { name: 'US Tech Trio', value: 'AAPL,MSFT,NVDA' },
  { name: 'US Benchmarks', value: 'SPY,QQQ' },
  { name: 'Indian Bluechips', value: 'RELIANCE.NS,TCS.NS,HDFCBANK.NS' },
  { name: 'Cross-Border Mix', value: 'AAPL,NVDA,RELIANCE.NS' },
]

const MetricBox = ({ label, value, unit = '', positive = null, large = false }) => {
  const color = positive === null ? 'var(--text-primary)' : positive ? 'var(--accent-green)' : 'var(--accent-red)'
  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-md)', padding: '12px 16px',
    }}>
      <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: large ? 22 : 18, fontWeight: 700, fontFamily: 'var(--font-mono)', color }}>
        {value}{unit}
      </div>
    </div>
  )
}

export default function BacktestPanel() {
  const [symbol, setSymbol] = useState('AAPL,MSFT,NVDA')
  const [lookback, setLookback] = useState(365)
  const [capital, setCapital] = useState(1000)
  const [useAI, setUseAI] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleRun = async () => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await runBacktest({
        symbol, lookback_days: lookback, initial_capital: capital, use_ai: useAI,
      })
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Backtest failed')
    } finally {
      setLoading(false)
    }
  }

  const equityData = result?.equity_curve?.map((v, i) => ({
    bar: i,
    'Portfolio Value': v,
    Benchmark: result.benchmark_curve ? result.benchmark_curve[i] : null
  })) || []
  const isPositive = result ? result.total_return_pct >= 0 : true

  return (
    <div className="card col-12">
      <div className="card-header">
        <span className="card-title">
          <BarChart2 size={14} className="card-title-icon" />
          Strategy Backtester
        </span>
        <span className="badge badge-ai">Walk-Forward Simulation</span>
      </div>

      {/* Controls */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 20 }}>
        <div className="input-group" style={{ gridColumn: 'span 2' }}>
          <label className="input-label">Symbols (comma-separated)</label>
          <input
            className="input"
            type="text"
            value={symbol}
            onChange={e => setSymbol(e.target.value)}
            placeholder="e.g. AAPL, MSFT, RELIANCE.NS"
            style={{ width: '100%' }}
          />
          <div style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap', alignItems: 'center' }}>
            <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Presets:</span>
            {PRESETS.map(p => (
              <button
                key={p.name}
                type="button"
                onClick={() => setSymbol(p.value)}
                style={{
                  background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
                  borderRadius: 4, padding: '2px 6px', fontSize: 10, cursor: 'pointer',
                  color: 'var(--text-secondary)', transition: 'all 0.2s'
                }}
                onMouseOver={e => e.target.style.borderColor = 'var(--accent-primary)'}
                onMouseOut={e => e.target.style.borderColor = 'var(--border-subtle)'}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>
        <div className="input-group">
          <label className="input-label">Lookback (days)</label>
          <select className="select" value={lookback} onChange={e => setLookback(Number(e.target.value))}>
            <option value={90}>90 Days</option>
            <option value={180}>180 Days</option>
            <option value={365}>1 Year</option>
            <option value={730}>2 Years</option>
          </select>
        </div>
        <div className="input-group">
          <label className="input-label">Initial Capital ($)</label>
          <input className="input" type="number" value={capital} onChange={e => setCapital(Number(e.target.value))} min={100} />
        </div>
        <div className="input-group">
          <label className="input-label">Use AI Model</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingTop: 8 }}>
            <div
              onClick={() => setUseAI(!useAI)}
              style={{
                width: 40, height: 22, borderRadius: 11,
                background: useAI ? 'var(--accent-purple)' : 'var(--bg-surface)',
                border: '1px solid var(--border-dim)', cursor: 'pointer', position: 'relative',
                transition: 'background 0.2s',
              }}
            >
              <div style={{
                width: 16, height: 16, borderRadius: '50%', background: 'white',
                position: 'absolute', top: 2, left: useAI ? 20 : 2,
                transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
              }} />
            </div>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{useAI ? 'Enabled' : 'Disabled'}</span>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'flex-end' }}>
          <button className="btn btn-primary btn-lg w-full" onClick={handleRun} disabled={loading} style={{ height: 42 }}>
            {loading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Running...</> : <><Play size={16} /> Run Backtest</>}
          </button>
        </div>
      </div>

      {error && (
        <div style={{
          background: 'var(--accent-red-dim)', border: '1px solid rgba(255,51,102,0.3)',
          borderRadius: 'var(--radius-md)', padding: '12px 16px', color: 'var(--accent-red)', fontSize: 13, marginBottom: 16
        }}>
          ⚠️ {error}
        </div>
      )}

      {result && (
        <>
          {/* Summary metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 10, marginBottom: 20 }}>
            <MetricBox label="Total Return" value={`${result.total_return_pct >= 0 ? '+' : ''}${result.total_return_pct}%`}
              positive={result.total_return_pct >= 0} large />
            <MetricBox label="Final Value" value={`$${result.final_value?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`} />
            <MetricBox label="Sharpe Ratio" value={result.sharpe_ratio} positive={result.sharpe_ratio >= 1} />
            <MetricBox label="Max Drawdown" value={`${result.max_drawdown_pct}%`} positive={false} />
            <MetricBox label="Win Rate" value={`${result.win_rate_pct}%`} positive={result.win_rate_pct >= 50} />
            <MetricBox label="Total Trades" value={result.total_trades} />
            <MetricBox label="Profit Factor" value={result.profit_factor} positive={result.profit_factor >= 1} />
            <MetricBox label="CAGR" value={`${result.cagr_pct}%`} positive={result.cagr_pct >= 0} />
          </div>

          {/* Equity Curve */}
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
              Equity Curve vs Benchmark — {result.symbol} ({result.start_date?.split('T')[0]} → {result.end_date?.split('T')[0]})
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={equityData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" />
                <XAxis dataKey="bar" hide />
                <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }} tickFormatter={v => `$${v}`} axisLine={false} tickLine={false} />
                <Tooltip
                  formatter={(v, name) => [`$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, name]}
                  contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-dim)', borderRadius: 8, fontFamily: 'var(--font-mono)', fontSize: 12 }}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 12, fontFamily: 'var(--font-sans)', color: 'var(--text-secondary)' }} />
                <ReferenceLine y={capital} stroke="rgba(0,212,255,0.3)" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="Portfolio Value" stroke={isPositive ? '#00ff94' : '#ff3366'} strokeWidth={2} dot={false} />
                {result.benchmark_curve && (
                  <Line type="monotone" dataKey="Benchmark" stroke="var(--text-muted)" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name={`Benchmark (${result.benchmark_symbol || 'SPY'})`} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Individual Assets performance grid */}
          {result.individual_results && (
            <div style={{ marginTop: 24, marginBottom: 16 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Individual Asset Performance Summary
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 12 }}>
                {Object.entries(result.individual_results).map(([sym, data]) => {
                  const assetPos = data.total_return_pct >= 0;
                  return (
                    <div key={sym} style={{
                      background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-md)', padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 6,
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: 14 }}>{sym}</span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: 14, color: assetPos ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                          {assetPos ? '+' : ''}{data.total_return_pct}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', marginTop: 4 }}>
                        <span>DD: {data.max_drawdown_pct}%</span>
                        <span>Sharpe: {data.sharpe_ratio}</span>
                        <span>Trades: {data.total_trades}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Trade stats */}
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 12 }}>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
              <span style={{ color: 'var(--accent-green)' }}>✓ {result.winning_trades} wins</span>
              {' · '}
              <span style={{ color: 'var(--accent-red)' }}>✗ {result.losing_trades} losses</span>
              {' · '}
              Avg Win: <span style={{ color: 'var(--accent-green)' }}>${result.avg_win?.toFixed(2)}</span>
              {' · '}
              Avg Loss: <span style={{ color: 'var(--accent-red)' }}>${Math.abs(result.avg_loss || 0).toFixed(2)}</span>
            </div>
          </div>
        </>
      )}

      {!result && !loading && !error && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200,
          color: 'var(--text-muted)', flexDirection: 'column', gap: 12,
        }}>
          <BarChart2 size={40} style={{ opacity: 0.3 }} />
          <div style={{ fontSize: 13 }}>Configure parameters above and run a backtest</div>
        </div>
      )}
    </div>
  )
}
