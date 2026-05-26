import { useMemo } from 'react';
import {
  X,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Layers,
  Clock,
  ChevronRight,
} from 'lucide-react';

/* ─────────────────────────────────────────────
   Helpers
───────────────────────────────────────────── */
const fmt2 = (v) =>
  v !== undefined && v !== null ? Number(v).toFixed(2) : '—';

const fmtPrice = (v) =>
  v !== undefined && v !== null
    ? Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : '—';

const fmtDate = (iso) => {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleString('en-US', {
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  } catch {
    return iso;
  }
};

/* ─────────────────────────────────────────────
   SL-to-TP progress bar
   progress = (currentPrice - sl) / (tp - sl) clamped [0,1]
───────────────────────────────────────────── */
function SlTpBar({ sl, tp, current, side }) {
  const pct = useMemo(() => {
    if (sl == null || tp == null || current == null) return null;
    const slN = Number(sl);
    const tpN = Number(tp);
    const curN = Number(current);
    const range = tpN - slN;
    if (range === 0) return null;
    const raw = ((curN - slN) / range) * 100;
    return Math.min(100, Math.max(0, raw));
  }, [sl, tp, current]);

  if (pct === null) return <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>—</span>;

  const isGood = side === 'LONG' ? pct >= 50 : pct <= 50;

  return (
    <div className="sltpbar">
      <div className="sltpbar__track">
        <div
          className="sltpbar__fill"
          style={{
            width: `${pct}%`,
            background: isGood ? 'var(--gradient-green)' : 'var(--gradient-primary)',
          }}
        />
        {/* Thumb marker at current position */}
        <div className="sltpbar__thumb" style={{ left: `${pct}%` }} />
      </div>
      <div className="sltpbar__labels">
        <span style={{ color: 'var(--accent-red)', fontSize: 9 }}>SL</span>
        <span style={{ color: 'var(--text-muted)', fontSize: 9 }}>{pct.toFixed(0)}%</span>
        <span style={{ color: 'var(--accent-green)', fontSize: 9 }}>TP</span>
      </div>
    </div>
  );
}

/* ─────────────────────────────────────────────
   Side Badge
───────────────────────────────────────────── */
function SideBadge({ side }) {
  const isLong = String(side).toUpperCase() === 'LONG' || String(side).toUpperCase() === 'BUY';
  return (
    <span className={`badge ${isLong ? 'badge-buy' : 'badge-sell'}`}>
      {isLong ? <TrendingUp size={9} /> : <TrendingDown size={9} />}
      {isLong ? 'LONG' : 'SHORT'}
    </span>
  );
}

/* ─────────────────────────────────────────────
   Skeleton row
───────────────────────────────────────────── */
function SkeletonRow() {
  return (
    <tr>
      {Array.from({ length: 12 }).map((_, i) => (
        <td key={i}>
          <div className="skeleton" style={{ height: 12, width: i === 0 ? 60 : i === 10 ? 90 : 50, borderRadius: 4 }} />
        </td>
      ))}
    </tr>
  );
}

/* ─────────────────────────────────────────────
   Empty state
───────────────────────────────────────────── */
function EmptyState() {
  return (
    <tr>
      <td colSpan={12}>
        <div className="positions-empty">
          <Layers size={40} style={{ color: 'var(--text-muted)' }} />
          <p className="positions-empty__title">No Open Positions</p>
          <p className="positions-empty__sub">
            Your active trades will appear here once you have open positions.
          </p>
        </div>
      </td>
    </tr>
  );
}

/* ─────────────────────────────────────────────
   Main Component
───────────────────────────────────────────── */
export default function ActivePositions({ positions = [], onClosePosition, loading = false }) {
  const totalUnrealizedPnl = useMemo(
    () => positions.reduce((acc, p) => acc + (Number(p.unrealized_pnl) || 0), 0),
    [positions]
  );

  const handleCloseAll = () => {
    if (!positions.length) return;
    positions.forEach((p) => onClosePosition?.(p.id ?? p.symbol));
  };

  const isAllPositive = totalUnrealizedPnl >= 0;

  return (
    <div className="card positions-card">
      {/* ── Card header ── */}
      <div className="card-header">
        <div className="positions-header__left">
          <span className="card-title">
            <Layers size={14} style={{ color: 'var(--accent-primary)' }} />
            Open Positions
          </span>
          <span className="badge positions-count-badge">
            {loading ? '…' : positions.length}
          </span>
          {!loading && positions.length > 0 && (
            <span
              className={`positions-pnl-summary ${isAllPositive ? 'positive' : 'negative'}`}
            >
              {isAllPositive ? '+' : ''}${fmt2(totalUnrealizedPnl)} total P&amp;L
            </span>
          )}
        </div>

        <div className="positions-header__right">
          {!loading && positions.length > 0 && (
            <button
              className="btn btn-danger btn-sm"
              onClick={handleCloseAll}
              title="Close all open positions"
            >
              <X size={12} />
              Close All
            </button>
          )}
        </div>
      </div>

      {/* ── Table ── */}
      <div className="positions-table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th>Qty</th>
              <th>Entry $</th>
              <th>Current $</th>
              <th>P&amp;L ($)</th>
              <th>P&amp;L (%)</th>
              <th>Stop Loss</th>
              <th>Take Profit</th>
              <th>Strategy</th>
              <th>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <Clock size={10} />
                  Opened At
                </span>
              </th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 4 }).map((_, i) => <SkeletonRow key={i} />)
            ) : positions.length === 0 ? (
              <EmptyState />
            ) : (
              positions.map((pos, idx) => {
                const pnlDollar = Number(pos.unrealized_pnl ?? 0);
                const pnlPct = Number(pos.unrealized_pnl_pct ?? 0);
                const isPnlPos = pnlDollar >= 0;
                const posId = pos.id ?? pos.symbol ?? idx;

                return (
                  <tr key={posId} className="positions-row">
                    {/* Symbol */}
                    <td>
                      <div className="positions-symbol-cell">
                        <span className="symbol-cell">{pos.symbol}</span>
                        {pos.exchange && (
                          <span className="positions-exchange">{pos.exchange}</span>
                        )}
                      </div>
                    </td>

                    {/* Side */}
                    <td>
                      <SideBadge side={pos.side} />
                    </td>

                    {/* Qty */}
                    <td style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                      {fmt2(pos.quantity ?? pos.qty)}
                    </td>

                    {/* Entry Price */}
                    <td style={{ color: 'var(--text-secondary)' }}>
                      ${fmtPrice(pos.entry_price)}
                    </td>

                    {/* Current Price */}
                    <td style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
                      ${fmtPrice(pos.current_price)}
                    </td>

                    {/* P&L Dollar */}
                    <td>
                      <span
                        className={`positions-pnl ${isPnlPos ? 'positive' : 'negative'}`}
                      >
                        {isPnlPos ? '+' : ''}${fmt2(pnlDollar)}
                      </span>
                    </td>

                    {/* P&L Percent */}
                    <td>
                      <span
                        className={`positions-pnl ${isPnlPos ? 'positive' : 'negative'}`}
                      >
                        {isPnlPos ? '+' : ''}{fmt2(pnlPct)}%
                      </span>
                    </td>

                    {/* Stop Loss */}
                    <td style={{ color: 'var(--accent-red)' }}>
                      {pos.stop_loss != null ? `$${fmtPrice(pos.stop_loss)}` : '—'}
                    </td>

                    {/* Take Profit + progress bar */}
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        <span style={{ color: 'var(--accent-green)' }}>
                          {pos.take_profit != null ? `$${fmtPrice(pos.take_profit)}` : '—'}
                        </span>
                        <SlTpBar
                          sl={pos.stop_loss}
                          tp={pos.take_profit}
                          current={pos.current_price}
                          side={pos.side}
                        />
                      </div>
                    </td>

                    {/* Strategy */}
                    <td>
                      {pos.strategy ? (
                        <span className="badge badge-ai" style={{ fontSize: 10 }}>
                          {pos.strategy}
                        </span>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>—</span>
                      )}
                    </td>

                    {/* Opened At */}
                    <td style={{ whiteSpace: 'nowrap', color: 'var(--text-muted)' }}>
                      {fmtDate(pos.opened_at)}
                    </td>

                    {/* Actions */}
                    <td>
                      <button
                        className="btn btn-danger btn-sm positions-close-btn"
                        onClick={() => onClosePosition?.(posId)}
                        title={`Close ${pos.symbol} position`}
                      >
                        <X size={11} />
                        Close
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Inline scoped styles */}
      <style>{`
        /* ── Card ── */
        .positions-card {
          padding: var(--space-4) var(--space-5);
        }

        /* ── Header ── */
        .positions-header__left {
          display: flex;
          align-items: center;
          gap: var(--space-3);
          flex-wrap: wrap;
        }
        .positions-header__right {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .positions-count-badge {
          background: var(--accent-primary-dim);
          color: var(--accent-primary);
          border: 1px solid var(--border-dim);
          font-family: var(--font-mono);
          font-size: 11px;
          font-weight: 700;
          padding: 2px 8px;
          border-radius: var(--radius-full);
        }

        .positions-pnl-summary {
          font-family: var(--font-mono);
          font-size: 12px;
          font-weight: 600;
          padding: 3px 8px;
          border-radius: var(--radius-md);
        }
        .positions-pnl-summary.positive {
          color: var(--accent-green);
          background: var(--accent-green-dim);
        }
        .positions-pnl-summary.negative {
          color: var(--accent-red);
          background: var(--accent-red-dim);
        }

        /* ── Table wrap ── */
        .positions-table-wrap {
          overflow-x: auto;
          margin-top: var(--space-2);
          border-radius: var(--radius-md);
        }
        .positions-table-wrap::-webkit-scrollbar {
          height: 4px;
        }

        /* ── Rows ── */
        .positions-row {
          transition: background var(--transition-fast);
        }

        /* Symbol cell */
        .positions-symbol-cell {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .positions-exchange {
          font-size: 9px;
          color: var(--text-muted);
          letter-spacing: 0.5px;
          text-transform: uppercase;
        }

        /* P&L colored values */
        .positions-pnl {
          font-family: var(--font-mono);
          font-weight: 700;
          font-size: 12px;
        }
        .positions-pnl.positive {
          color: var(--accent-green);
          text-shadow: 0 0 8px rgba(0,255,148,0.3);
        }
        .positions-pnl.negative {
          color: var(--accent-red);
          text-shadow: 0 0 8px rgba(255,51,102,0.3);
        }

        /* Close button */
        .positions-close-btn {
          white-space: nowrap;
          padding: 4px 10px;
          font-size: 11px;
        }

        /* ── SL-TP bar ── */
        .sltpbar {
          display: flex;
          flex-direction: column;
          gap: 2px;
          min-width: 80px;
        }
        .sltpbar__track {
          height: 4px;
          background: var(--bg-surface);
          border-radius: var(--radius-full);
          position: relative;
          overflow: visible;
        }
        .sltpbar__fill {
          height: 100%;
          border-radius: var(--radius-full);
          transition: width 0.4s ease;
        }
        .sltpbar__thumb {
          position: absolute;
          top: 50%;
          transform: translate(-50%, -50%);
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: var(--text-primary);
          border: 1.5px solid var(--bg-card);
          box-shadow: 0 0 4px rgba(0,212,255,0.5);
        }
        .sltpbar__labels {
          display: flex;
          justify-content: space-between;
          font-family: var(--font-mono);
          font-size: 9px;
        }

        /* ── Empty state ── */
        .positions-empty {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: var(--space-12) var(--space-6);
          gap: var(--space-3);
        }
        .positions-empty__title {
          font-size: 15px;
          font-weight: 700;
          color: var(--text-secondary);
          margin: 0;
        }
        .positions-empty__sub {
          font-size: 12px;
          color: var(--text-muted);
          text-align: center;
          max-width: 320px;
          margin: 0;
          font-family: var(--font-mono);
          line-height: 1.6;
        }
      `}</style>
    </div>
  );
}
