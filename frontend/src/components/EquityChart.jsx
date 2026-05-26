import { useState } from 'react'
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { TrendingUp, TrendingDown } from 'lucide-react'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload || {}
  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border-dim)',
      borderRadius: 'var(--radius-md)', padding: '10px 14px', fontSize: 12,
      fontFamily: 'var(--font-mono)',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>
        {label ? new Date(label).toLocaleDateString() : ''}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? `$${p.value.toFixed(2)}` : p.value}</strong>
        </div>
      ))}
    </div>
  )
}

export default function EquityChart({ equityCurve = [], loading }) {
  const [period, setPeriod] = useState('all')

  const filtered = (() => {
    if (!equityCurve.length) return []
    const now = Date.now()
    const cutoffs = { '1w': 7, '1m': 30, '3m': 90, 'all': 99999 }
    const days = cutoffs[period] || 99999
    return equityCurve.filter(d => {
      const daysAgo = (now - new Date(d.timestamp).getTime()) / (1000 * 86400)
      return daysAgo <= days
    })
  })()

  const lastValue = filtered[filtered.length - 1]?.total_value || 0
  const firstValue = filtered[0]?.total_value || lastValue
  const change = lastValue - firstValue
  const changePct = firstValue > 0 ? (change / firstValue) * 100 : 0
  const isPositive = change >= 0

  const gradientColor = isPositive ? '#00ff94' : '#ff3366'

  return (
    <div className="card col-12">
      <div className="card-header">
        <span className="card-title">
          <TrendingUp size={14} className="card-title-icon" />
          Equity Curve
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span className={`metric-change ${isPositive ? 'up' : 'down'}`}>
            {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
            {isPositive ? '+' : ''}{changePct.toFixed(2)}%
          </span>
          <div className="tabs" style={{ padding: 2 }}>
            {['1w', '1m', '3m', 'all'].map(p => (
              <button key={p} className={`tab ${period === p ? 'active' : ''}`}
                onClick={() => setPeriod(p)} style={{ padding: '4px 10px', fontSize: 11 }}>
                {p.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {loading ? (
        <div className="skeleton" style={{ height: 220 }} />
      ) : filtered.length === 0 ? (
        <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
          No equity data yet. Start trading to see your performance.
        </div>
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={filtered} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={gradientColor} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={gradientColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,212,255,0.05)" />
              <XAxis
                dataKey="timestamp"
                tickFormatter={v => new Date(v).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}
                axisLine={false} tickLine={false}
                tickFormatter={v => `$${v.toFixed(0)}`}
                domain={['auto', 'auto']}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={firstValue} stroke="rgba(0,212,255,0.2)" strokeDasharray="4 4" />
              <Area
                type="monotone" dataKey="total_value" name="Portfolio Value"
                stroke={gradientColor} strokeWidth={2}
                fill="url(#equityGrad)"
                dot={false} activeDot={{ r: 4, fill: gradientColor }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}
