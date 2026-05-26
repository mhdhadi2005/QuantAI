import { useState, useMemo } from 'react';
import {
  Zap,
  RefreshCw,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle2,
  Loader2,
  Radio,
  Filter,
  AlertCircle,
} from 'lucide-react';

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Returns a human-readable relative time string from a date string or Date object.
 */
function relativeTime(dateInput) {
  if (!dateInput) return '—';
  const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
  if (isNaN(date.getTime())) return '—';

  const diffMs = Date.now() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 10) return 'just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffMin < 60) return diffMin === 1 ? '1 min ago' : `${diffMin} mins ago`;
  if (diffHour < 24) return diffHour === 1 ? '1 hr ago' : `${diffHour} hrs ago`;
  return diffDay === 1 ? 'Yesterday' : `${diffDay} days ago`;
}

function formatPrice(price) {
  if (price == null) return '—';
  return `$${parseFloat(price).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function confidenceClass(value) {
  const pct = parseFloat(value) || 0;
  const display = pct <= 1 ? pct * 100 : pct;
  if (display >= 70) return 'high';
  if (display >= 40) return 'medium';
  return 'low';
}

function confidenceDisplay(value) {
  const pct = parseFloat(value) || 0;
  return pct <= 1 ? (pct * 100).toFixed(1) : pct.toFixed(1);
}

// ─── Action badge ─────────────────────────────────────────────────────────────

function ActionBadge({ action }) {
  const a = (action || '').toUpperCase();

  if (a === 'BUY') {
    return (
      <span className="badge badge-buy" style={{ gap: '4px' }}>
        <TrendingUp size={10} />
        Buy
      </span>
    );
  }
  if (a === 'SELL') {
    return (
      <span className="badge badge-sell" style={{ gap: '4px' }}>
        <TrendingDown size={10} />
        Sell
      </span>
    );
  }
  return (
    <span className="badge badge-hold" style={{ gap: '4px' }}>
      <Minus size={10} />
      Hold
    </span>
  );
}

// ─── AI direction badge ───────────────────────────────────────────────────────

function AIPredictionBadge({ prediction, aiConfidence }) {
  if (!prediction) return null;
  const p = prediction.toUpperCase();

  const icon =
    p === 'UP' || p === 'BUY' ? (
      <TrendingUp size={10} />
    ) : p === 'DOWN' || p === 'SELL' ? (
      <TrendingDown size={10} />
    ) : (
      <Minus size={10} />
    );

  const label =
    p === 'UP' || p === 'BUY'
      ? 'AI ↑'
      : p === 'DOWN' || p === 'SELL'
        ? 'AI ↓'
        : 'AI ~';

  return (
    <span
      className="badge badge-ai"
      title={`AI confidence: ${aiConfidence ? confidenceDisplay(aiConfidence) + '%' : 'N/A'}`}
      style={{ gap: '4px' }}
    >
      {icon}
      {label}
      {aiConfidence != null && (
        <span style={{ opacity: 0.8 }}>{confidenceDisplay(aiConfidence)}%</span>
      )}
    </span>
  );
}

// ─── Single signal card ───────────────────────────────────────────────────────

function SignalCard({ signal }) {
  const {
    symbol,
    action,
    strategy,
    confidence,
    price_at_signal,
    ai_prediction,
    ai_confidence,
    executed,
    created_at,
  } = signal;

  const confPct = confidenceDisplay(confidence);
  const confCls = confidenceClass(confidence);

  const actionKey = (action || '').toUpperCase();
  const accentColor =
    actionKey === 'BUY'
      ? 'var(--accent-green)'
      : actionKey === 'SELL'
        ? 'var(--accent-red)'
        : 'var(--text-muted)';

  return (
    <div
      className="card"
      style={{
        padding: '14px 16px',
        marginBottom: '10px',
        borderLeft: `3px solid ${accentColor}`,
        borderRadius: 'var(--radius-md)',
        background: 'var(--bg-elevated)',
        transition: 'transform var(--transition-fast), border-color var(--transition-fast)',
        cursor: 'default',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateX(2px)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateX(0)';
      }}
    >
      {/* Row 1: symbol + badges + price */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '14px',
              fontWeight: '700',
              color: 'var(--text-primary)',
              letterSpacing: '0.5px',
            }}
          >
            {symbol}
          </span>
          <ActionBadge action={action} />
          {ai_prediction && (
            <AIPredictionBadge
              prediction={ai_prediction}
              aiConfidence={ai_confidence}
            />
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              fontWeight: '600',
              color: 'var(--text-primary)',
            }}
          >
            {formatPrice(price_at_signal)}
          </span>
          {actionKey === 'HOLD' ? (
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                color: 'var(--text-muted)',
                fontWeight: '600',
              }}
            >
              <CheckCircle2 size={13} />
              No Action
            </span>
          ) : executed ? (
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                color: 'var(--accent-green)',
                fontWeight: '600',
              }}
            >
              <CheckCircle2 size={13} />
              Executed
            </span>
          ) : (
            <span
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                fontSize: '11px',
                color: 'var(--accent-gold)',
                fontWeight: '600',
              }}
            >
              <Loader2 size={12} style={{ animation: 'spin 1.5s linear infinite' }} />
              Pending
            </span>
          )}
        </div>
      </div>

      {/* Row 2: strategy + confidence bar */}
      <div style={{ marginTop: '10px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '4px',
          }}
        >
          <span
            style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
              textTransform: 'uppercase',
              letterSpacing: '0.6px',
            }}
          >
            {strategy || 'Unknown Strategy'}
          </span>
          <span
            style={{
              fontSize: '11px',
              fontFamily: 'var(--font-mono)',
              fontWeight: '700',
              color:
                confCls === 'high'
                  ? 'var(--accent-green)'
                  : confCls === 'medium'
                    ? 'var(--accent-primary)'
                    : 'var(--accent-red)',
            }}
          >
            {confPct}%
          </span>
        </div>
        <div className="confidence-bar" style={{ height: '4px' }}>
          <div
            className={`confidence-fill ${confCls}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Row 3: timestamp */}
      <div
        style={{
          marginTop: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          fontSize: '11px',
          color: 'var(--text-muted)',
        }}
      >
        <Clock size={11} />
        {relativeTime(created_at)}
      </div>
    </div>
  );
}

// ─── Filter buttons ───────────────────────────────────────────────────────────

const FILTERS = ['All', 'Buy', 'Sell', 'Hold'];

function FilterBar({ active, onChange, counts }) {
  return (
    <div
      style={{
        display: 'flex',
        gap: '6px',
        padding: '0 20px 14px 20px',
        borderBottom: '1px solid var(--border-subtle)',
        flexWrap: 'wrap',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          marginRight: '4px',
          color: 'var(--text-muted)',
        }}
      >
        <Filter size={12} />
      </div>
      {FILTERS.map((f) => {
        const isActive = active === f;
        const count = counts[f] ?? 0;
        return (
          <button
            key={f}
            onClick={() => onChange(f)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '5px',
              padding: '4px 12px',
              borderRadius: 'var(--radius-full)',
              fontSize: '12px',
              fontWeight: '600',
              cursor: 'pointer',
              border: isActive
                ? '1px solid var(--border-active)'
                : '1px solid var(--border-subtle)',
              background: isActive ? 'var(--accent-primary-dim)' : 'transparent',
              color: isActive ? 'var(--accent-primary)' : 'var(--text-muted)',
              transition: 'all var(--transition-fast)',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {f}
            <span
              style={{
                background: isActive
                  ? 'rgba(0,212,255,0.2)'
                  : 'var(--bg-surface)',
                borderRadius: 'var(--radius-full)',
                padding: '0px 5px',
                fontSize: '10px',
                minWidth: '18px',
                textAlign: 'center',
              }}
            >
              {f === 'All' ? counts.__total : count}
            </span>
          </button>
        );
      })}
    </div>
  );
}

// ─── Empty state ──────────────────────────────────────────────────────────────

function EmptyState({ activeFilter, onGenerateSignals, loading }) {
  const isFiltered = activeFilter !== 'All';
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '48px 24px',
        gap: '16px',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          width: '52px',
          height: '52px',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--accent-primary-dim)',
          border: '1px solid var(--border-dim)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        {isFiltered ? (
          <AlertCircle size={24} color="var(--text-muted)" />
        ) : (
          <Radio size={24} color="var(--accent-primary)" />
        )}
      </div>
      <div>
        <div
          style={{
            fontSize: '14px',
            fontWeight: '600',
            color: 'var(--text-secondary)',
            marginBottom: '6px',
          }}
        >
          {isFiltered
            ? `No ${activeFilter.toUpperCase()} signals`
            : 'No signals yet'}
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          {isFiltered
            ? 'Try a different filter or generate new signals.'
            : 'Generate signals to start seeing trade recommendations.'}
        </div>
      </div>
      {!isFiltered && (
        <button
          className="btn btn-primary btn-sm"
          onClick={onGenerateSignals}
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner" style={{ width: '13px', height: '13px' }} />
              Generating…
            </>
          ) : (
            <>
              <Zap size={13} />
              Generate Signals
            </>
          )}
        </button>
      )}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

export default function SignalsPanel({
  signals = [],
  onGenerateSignals,
  loading = false,
}) {
  const [activeFilter, setActiveFilter] = useState('All');

  // Compute signal counts per filter
  const counts = useMemo(() => {
    const result = { All: signals.length, Buy: 0, Sell: 0, Hold: 0, __total: signals.length };
    signals.forEach((s) => {
      const a = (s.action || '').toUpperCase();
      if (a === 'BUY') result.Buy += 1;
      else if (a === 'SELL') result.Sell += 1;
      else result.Hold += 1;
    });
    return result;
  }, [signals]);

  const filteredSignals = useMemo(() => {
    if (activeFilter === 'All') return signals;
    return signals.filter(
      (s) => (s.action || '').toUpperCase() === activeFilter.toUpperCase(),
    );
  }, [signals, activeFilter]);

  // Find the most recent signal timestamp
  const lastUpdated = useMemo(() => {
    if (!signals.length) return null;
    const dates = signals
      .map((s) => (s.created_at ? new Date(s.created_at) : null))
      .filter(Boolean);
    if (!dates.length) return null;
    return new Date(Math.max(...dates.map((d) => d.getTime())));
  }, [signals]);

  return (
    <div
      className="card"
      style={{
        padding: 0,
        background: 'var(--bg-card)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: '16px 20px 14px 20px',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-md)',
              background: 'var(--accent-primary-dim)',
              border: '1px solid var(--border-dim)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Radio size={16} color="var(--accent-primary)" />
          </div>
          <div>
            <div
              style={{
                fontSize: '13px',
                fontWeight: '700',
                color: 'var(--text-primary)',
                letterSpacing: '0.3px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              Signal Feed
              {loading && (
                <span
                  className="badge badge-live"
                  style={{ fontSize: '9px', gap: '4px' }}
                >
                  <span
                    style={{
                      width: '5px',
                      height: '5px',
                      borderRadius: '50%',
                      background: 'var(--accent-green)',
                      display: 'inline-block',
                    }}
                  />
                  Live
                </span>
              )}
            </div>
            {lastUpdated ? (
              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  marginTop: '2px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                <Clock size={10} />
                Updated {relativeTime(lastUpdated)}
              </div>
            ) : (
              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--text-muted)',
                  marginTop: '2px',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                No data yet
              </div>
            )}
          </div>
        </div>

        <button
          className="btn btn-primary btn-sm"
          onClick={onGenerateSignals}
          disabled={loading}
        >
          {loading ? (
            <>
              <div className="spinner" style={{ width: '13px', height: '13px' }} />
              Generating…
            </>
          ) : (
            <>
              <Zap size={13} />
              Generate Signals
            </>
          )}
        </button>
      </div>

      {/* Filter bar */}
      <div style={{ padding: '12px 20px 0 20px' }}>
        <FilterBar
          active={activeFilter}
          onChange={setActiveFilter}
          counts={counts}
        />
      </div>

      {/* Signals list */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '14px 20px 20px 20px',
          maxHeight: '500px',
        }}
      >
        {filteredSignals.length === 0 ? (
          <EmptyState
            activeFilter={activeFilter}
            onGenerateSignals={onGenerateSignals}
            loading={loading}
          />
        ) : (
          filteredSignals.map((signal) => (
            <SignalCard key={signal.id ?? `${signal.symbol}-${signal.created_at}`} signal={signal} />
          ))
        )}
      </div>

      {/* Footer count */}
      {filteredSignals.length > 0 && (
        <div
          style={{
            padding: '10px 20px',
            borderTop: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span
            style={{
              fontSize: '11px',
              color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            Showing {filteredSignals.length} of {signals.length} signals
          </span>
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              fontSize: '11px',
              color: signals.filter((s) => s.executed).length > 0
                ? 'var(--accent-green)'
                : 'var(--text-muted)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <CheckCircle2 size={11} />
            {signals.filter((s) => s.executed).length} executed
          </span>
        </div>
      )}
    </div>
  );
}
