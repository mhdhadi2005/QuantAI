import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Activity,
  Wallet,
  ShieldAlert,
  BarChart2,
  AlertTriangle,
} from 'lucide-react';

// ─── Formatting helpers ───────────────────────────────────────────────────────
function fmtCurrency(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  return '$' + Number(value).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtPct(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(decimals)}%`;
}

function fmtNum(value, decimals = 2) {
  if (value == null || isNaN(value)) return '—';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${Number(value).toFixed(decimals)}`;
}

function signClass(value) {
  if (value == null || isNaN(value)) return '';
  return value >= 0 ? 'positive' : 'negative';
}

function clamp(val, min, max) {
  return Math.min(Math.max(val, min), max);
}

// ─── Skeleton loader ──────────────────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div className="card" style={{ minHeight: '120px' }}>
      <div className="skeleton" style={{ height: '12px', width: '50%', marginBottom: '16px' }} />
      <div className="skeleton" style={{ height: '32px', width: '75%', marginBottom: '8px' }} />
      <div className="skeleton" style={{ height: '10px', width: '40%' }} />
    </div>
  );
}

// ─── Individual metric card ───────────────────────────────────────────────────
function MetricCard({ icon: Icon, iconColor, title, primary, primaryClass, secondary, secondaryClass, accentBorder }) {
  return (
    <div
      className="card"
      style={accentBorder ? { borderColor: accentBorder } : undefined}
    >
      {/* Card header */}
      <div className="card-header">
        <span className="card-title">
          {Icon && (
            <Icon
              size={16}
              className="card-title-icon"
              style={{ color: iconColor || 'var(--accent-primary)' }}
              aria-hidden="true"
            />
          )}
          {title}
        </span>
      </div>

      {/* Primary value */}
      <div className={`metric-value${primaryClass ? ' ' + primaryClass : ''}`}>
        {primary}
      </div>

      {/* Secondary row */}
      {secondary && (
        <div
          className={`metric-change${secondaryClass ? ' ' + secondaryClass : ''}`}
          style={{ marginTop: '6px' }}
        >
          {secondary}
        </div>
      )}
    </div>
  );
}

// ─── Capital utilization bar ──────────────────────────────────────────────────
function UtilizationBar({ positionsValue, totalValue }) {
  const pct = totalValue > 0
    ? clamp((positionsValue / totalValue) * 100, 0, 100)
    : 0;

  const fillClass =
    pct >= 80 ? 'high' :
    pct >= 50 ? 'medium' :
    'low';

  return (
    <div style={{ flex: 1 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginBottom: '6px',
        }}
      >
        <span
          style={{
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '1px',
            color: 'var(--text-muted)',
            fontWeight: 600,
          }}
        >
          Capital Utilization
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            color: 'var(--text-secondary)',
          }}
        >
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="confidence-bar" style={{ height: '6px' }}>
        <div
          className={`confidence-fill ${fillClass}`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`Capital utilization: ${pct.toFixed(1)}%`}
        />
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '4px',
        }}
      >
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          In Positions: {fmtCurrency(positionsValue)}
        </span>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          Total: {fmtCurrency(totalValue)}
        </span>
      </div>
    </div>
  );
}

// ─── Trading mode badge (top-bar) ─────────────────────────────────────────────
function ModeBadge({ mode, trading_halted }) {
  if (trading_halted) {
    return (
      <span
        className="badge badge-sell"
        style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}
        aria-label="Trading halted"
      >
        <AlertTriangle size={11} aria-hidden="true" />
        HALTED
      </span>
    );
  }

  const modeUpper = (mode || 'UNKNOWN').toUpperCase();
  const badgeClass =
    modeUpper === 'LIVE'  ? 'badge badge-buy badge-live' :
    modeUpper === 'PAPER' ? 'badge badge-hold' :
    'badge badge-hold';

  return (
    <span className={badgeClass} aria-label={`Trading mode: ${modeUpper}`}>
      {modeUpper}
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function PortfolioSummary({ portfolio }) {
  // ── Loading state ────────────────────────────────────────────────────────
  if (!portfolio) {
    return (
      <section aria-label="Portfolio summary loading" aria-busy="true">
        {/* 6 skeleton cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
            gap: '16px',
            marginBottom: '16px',
          }}
        >
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
        {/* Skeleton top bar */}
        <div className="card">
          <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
            <div className="skeleton" style={{ height: '22px', width: '80px', borderRadius: '999px' }} />
            <div className="skeleton" style={{ height: '14px', flex: 1 }} />
            <div className="skeleton" style={{ height: '22px', width: '80px' }} />
          </div>
        </div>
      </section>
    );
  }

  // ── Destructure props ────────────────────────────────────────────────────
  const {
    cash            = 0,
    total_value     = 0,
    total_pnl       = 0,
    total_pnl_pct   = 0,
    daily_pnl       = 0,
    max_drawdown    = 0,
    initial_capital = 0,
    positions_value = 0,
    trading_halted  = false,
    mode            = 'paper',
    open_positions,
    sharpe_ratio,
  } = portfolio;

  // Derive open_positions count if not directly provided
  const openPositionsCount =
    open_positions != null
      ? open_positions
      : (positions_value > 0 ? '—' : 0);

  // ── Metric card definitions ──────────────────────────────────────────────
  const metrics = [
    {
      id: 'total-value',
      Icon: DollarSign,
      iconColor: 'var(--accent-primary)',
      title: 'Total Value',
      primary: fmtCurrency(total_value),
      primaryClass: total_value >= (initial_capital || 0) ? 'accent' : 'negative',
      secondary: initial_capital
        ? `Initial: ${fmtCurrency(initial_capital)}`
        : undefined,
      secondaryClass: '',
    },
    {
      id: 'total-pnl',
      Icon: total_pnl >= 0 ? TrendingUp : TrendingDown,
      iconColor: total_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
      title: 'Total P&L',
      primary: fmtCurrency(total_pnl),
      primaryClass: signClass(total_pnl),
      secondary: fmtPct(total_pnl_pct),
      secondaryClass: total_pnl_pct >= 0 ? 'up' : 'down',
      accentBorder: total_pnl >= 0 ? 'rgba(0,255,148,0.2)' : 'rgba(255,51,102,0.2)',
    },
    {
      id: 'daily-pnl',
      Icon: Activity,
      iconColor: daily_pnl >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
      title: 'Daily P&L',
      primary: fmtCurrency(daily_pnl),
      primaryClass: signClass(daily_pnl),
      secondary: daily_pnl >= 0 ? '▲ Today' : '▼ Today',
      secondaryClass: daily_pnl >= 0 ? 'up' : 'down',
    },
    {
      id: 'cash',
      Icon: Wallet,
      iconColor: 'var(--accent-primary)',
      title: 'Cash Available',
      primary: fmtCurrency(cash),
      primaryClass: 'accent',
      secondary: total_value > 0
        ? `${((cash / total_value) * 100).toFixed(1)}% of portfolio`
        : undefined,
      secondaryClass: '',
    },
    {
      id: 'drawdown',
      Icon: ShieldAlert,
      iconColor: 'var(--accent-red)',
      title: 'Max Drawdown',
      primary: fmtPct(max_drawdown),
      primaryClass: 'negative',
      secondary: 'Peak → Trough',
      secondaryClass: 'down',
      accentBorder: 'rgba(255,51,102,0.2)',
    },
    {
      id: 'open-positions',
      Icon: BarChart2,
      iconColor: 'var(--accent-purple)',
      title: 'Open Positions',
      primary: String(openPositionsCount),
      primaryClass: 'accent',
      secondary: openPositionsCount === 1 ? '1 active trade' : `${openPositionsCount} active trades`,
      secondaryClass: '',
    },
  ];

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <section aria-label="Portfolio summary">

      {/* 6-card grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))',
          gap: '16px',
          marginBottom: '16px',
        }}
      >
        {metrics.map(({ id, ...props }) => (
          <MetricCard key={id} {...props} />
        ))}
      </div>

      {/* Top bar */}
      <div
        className="card"
        style={{ padding: '14px 20px' }}
        aria-label="Portfolio overview bar"
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '24px',
            flexWrap: 'wrap',
          }}
        >
          {/* Trading mode */}
          <ModeBadge mode={mode} trading_halted={trading_halted} />

          {/* Divider */}
          <div
            aria-hidden="true"
            style={{ width: '1px', height: '28px', background: 'var(--border-subtle)', flexShrink: 0 }}
          />

          {/* Utilization bar */}
          <UtilizationBar
            positionsValue={positions_value}
            totalValue={total_value}
          />

          {/* Sharpe ratio (conditional) */}
          {sharpe_ratio != null && !isNaN(sharpe_ratio) && (
            <>
              <div
                aria-hidden="true"
                style={{ width: '1px', height: '28px', background: 'var(--border-subtle)', flexShrink: 0 }}
              />
              <div style={{ flexShrink: 0, textAlign: 'right' }}>
                <div
                  style={{
                    fontSize: '10px',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    color: 'var(--text-muted)',
                    fontWeight: 600,
                    marginBottom: '2px',
                  }}
                >
                  Sharpe Ratio
                </div>
                <div
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '18px',
                    fontWeight: 700,
                    color:
                      sharpe_ratio >= 2   ? 'var(--accent-green)'   :
                      sharpe_ratio >= 1   ? 'var(--accent-primary)' :
                      sharpe_ratio >= 0   ? 'var(--text-secondary)' :
                      'var(--accent-red)',
                    lineHeight: 1,
                  }}
                >
                  {fmtNum(sharpe_ratio)}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
