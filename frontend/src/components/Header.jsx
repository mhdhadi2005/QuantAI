import { useState, useEffect } from 'react';
import {
  LayoutDashboard,
  BriefcaseBusiness,
  Zap,
  BrainCircuit,
  FlaskConical,
  Settings,
  Wifi,
  WifiOff,
} from 'lucide-react';

// ─── Nav configuration ────────────────────────────────────────────────────────
const NAV_ITEMS = [
  { id: 'dashboard',   label: 'Dashboard',   Icon: LayoutDashboard },
  { id: 'portfolio',   label: 'Portfolio',   Icon: BriefcaseBusiness },
  { id: 'signals',     label: 'Signals',     Icon: Zap },
  { id: 'ai',          label: 'AI Engine',   Icon: BrainCircuit },
  { id: 'backtest',    label: 'Backtesting', Icon: FlaskConical },
  { id: 'settings',    label: 'Settings',    Icon: Settings },
];

// ─── Helpers ──────────────────────────────────────────────────────────────────
function useCurrentTime() {
  const [time, setTime] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return time;
}

function formatTime(date) {
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function getTradingModeMeta(mode) {
  switch ((mode || '').toLowerCase()) {
    case 'live':
      return { label: 'LIVE',  className: 'badge badge-buy'  };
    case 'paper':
      return { label: 'PAPER', className: 'badge badge-hold' };
    case 'halted':
      return { label: 'HALTED', className: 'badge badge-sell' };
    default:
      return { label: mode ? mode.toUpperCase() : 'UNKNOWN', className: 'badge badge-hold' };
  }
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function Header({ activePage, onNavigate, isConnected, tradingMode }) {
  const now = useCurrentTime();
  const modeMeta = getTradingModeMeta(tradingMode);

  return (
    <header className="header">
      <div className="header-inner">

        {/* ── Logo ── */}
        <div className="logo">
          <div className="logo-icon" aria-hidden="true">⚡</div>
          <span className="logo-text">QuantAI</span>
          <span className="logo-badge badge-live">LIVE</span>
        </div>

        {/* ── Navigation ── */}
        <nav className="nav" role="navigation" aria-label="Main navigation">
          {NAV_ITEMS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              className={`nav-item${activePage === id ? ' active' : ''}`}
              onClick={() => onNavigate?.(id)}
              aria-current={activePage === id ? 'page' : undefined}
            >
              <Icon size={14} aria-hidden="true" />
              {label}
            </button>
          ))}
        </nav>

        {/* ── Right side controls ── */}
        <div className="header-right">

          {/* Connection status */}
          <div
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
            title={isConnected ? 'Connected' : 'Disconnected'}
          >
            <span
              className={`status-dot${isConnected ? '' : ' offline'}`}
              role="status"
              aria-label={isConnected ? 'Connected' : 'Disconnected'}
            />
            {isConnected
              ? <Wifi size={13} style={{ color: 'var(--accent-green)', opacity: 0.85 }} aria-hidden="true" />
              : <WifiOff size={13} style={{ color: 'var(--accent-red)', opacity: 0.85 }} aria-hidden="true" />
            }
            <span
              style={{
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                color: isConnected ? 'var(--accent-green)' : 'var(--accent-red)',
                letterSpacing: '0.5px',
              }}
            >
              {isConnected ? 'CONNECTED' : 'OFFLINE'}
            </span>
          </div>

          {/* Divider */}
          <div
            aria-hidden="true"
            style={{
              width: '1px',
              height: '20px',
              background: 'var(--border-subtle)',
            }}
          />

          {/* Live clock */}
          <time
            dateTime={now.toISOString()}
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              color: 'var(--text-secondary)',
              letterSpacing: '0.5px',
              userSelect: 'none',
            }}
          >
            {formatTime(now)}
          </time>

          {/* Divider */}
          <div
            aria-hidden="true"
            style={{
              width: '1px',
              height: '20px',
              background: 'var(--border-subtle)',
            }}
          />

          {/* Trading mode badge */}
          <span className={modeMeta.className} aria-label={`Trading mode: ${modeMeta.label}`}>
            {modeMeta.label}
          </span>
        </div>
      </div>
    </header>
  );
}
