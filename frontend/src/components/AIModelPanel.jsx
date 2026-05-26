import {
  Brain,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  BarChart2,
  Cpu,
  Zap,
} from 'lucide-react';

// ─── Helpers ────────────────────────────────────────────────────────────────

function confidenceClass(value) {
  if (value >= 70) return 'high';
  if (value >= 40) return 'medium';
  return 'low';
}

function formatPct(value) {
  if (value == null) return '—';
  const num = typeof value === 'number' ? value : parseFloat(value);
  if (isNaN(num)) return '—';
  // Support both 0-1 fraction and 0-100 integer
  return num <= 1 ? `${(num * 100).toFixed(1)}%` : `${num.toFixed(1)}%`;
}

// Normalise feature importance values to 0-100 range for display
function normaliseImportance(importanceMap) {
  if (!importanceMap || typeof importanceMap !== 'object') return [];
  const entries = Object.entries(importanceMap)
    .map(([name, val]) => ({ name, val: parseFloat(val) || 0 }))
    .sort((a, b) => b.val - a.val)
    .slice(0, 5);

  const max = entries[0]?.val || 1;
  return entries.map((e) => ({
    name: e.name,
    pct: Math.round((e.val / max) * 100),
    raw: e.val,
  }));
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function DirectionBadge({ direction }) {
  if (!direction) return null;

  const dir = direction.toUpperCase();

  if (dir === 'UP' || dir === 'BUY') {
    return (
      <span className="badge badge-buy" style={{ gap: '4px' }}>
        <TrendingUp size={11} />
        UP
      </span>
    );
  }
  if (dir === 'DOWN' || dir === 'SELL') {
    return (
      <span className="badge badge-sell" style={{ gap: '4px' }}>
        <TrendingDown size={11} />
        DOWN
      </span>
    );
  }
  return (
    <span className="badge badge-hold" style={{ gap: '4px' }}>
      <Minus size={11} />
      NEUTRAL
    </span>
  );
}

function MetricsRow({ metrics }) {
  if (!metrics) return null;
  const items = [
    { label: 'Accuracy', value: formatPct(metrics.accuracy) },
    { label: 'F1', value: formatPct(metrics.f1) },
    { label: 'Precision', value: formatPct(metrics.precision) },
    { label: 'Recall', value: formatPct(metrics.recall) },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '8px',
        marginTop: '12px',
      }}
    >
      {items.map(({ label, value }) => (
        <div
          key={label}
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            padding: '6px 8px',
            textAlign: 'center',
          }}
        >
          <div
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '13px',
              fontWeight: '700',
              color: 'var(--accent-primary)',
            }}
          >
            {value}
          </div>
          <div
            style={{
              fontSize: '10px',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.8px',
              marginTop: '2px',
            }}
          >
            {label}
          </div>
        </div>
      ))}
    </div>
  );
}

function FeatureImportanceChart({ importanceMap }) {
  const features = normaliseImportance(importanceMap);

  if (!features.length) {
    return (
      <div
        style={{
          fontSize: '12px',
          color: 'var(--text-muted)',
          marginTop: '12px',
          fontStyle: 'italic',
        }}
      >
        No feature data available.
      </div>
    );
  }

  return (
    <div style={{ marginTop: '12px' }}>
      <div
        style={{
          fontSize: '10px',
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '1px',
          marginBottom: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        <BarChart2 size={11} />
        Feature Importance (Top 5)
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
        {features.map(({ name, pct }) => (
          <div key={name}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '3px',
              }}
            >
              <span
                style={{
                  fontSize: '11px',
                  color: 'var(--text-secondary)',
                  fontFamily: 'var(--font-mono)',
                  textOverflow: 'ellipsis',
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  maxWidth: '65%',
                }}
              >
                {name}
              </span>
              <span
                style={{
                  fontSize: '11px',
                  color: 'var(--accent-purple)',
                  fontFamily: 'var(--font-mono)',
                  fontWeight: '600',
                }}
              >
                {pct}%
              </span>
            </div>
            <div
              style={{
                height: '5px',
                background: 'var(--bg-surface)',
                borderRadius: 'var(--radius-full)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${pct}%`,
                  background:
                    'linear-gradient(90deg, var(--accent-purple) 0%, #6d28d9 100%)',
                  borderRadius: 'var(--radius-full)',
                  transition: 'width var(--transition-slow)',
                  boxShadow: '0 0 6px rgba(168,85,247,0.5)',
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfidenceGauge({ confidence }) {
  const pct = Math.min(100, Math.max(0, parseFloat(confidence) || 0));
  // Backend may return 0-1 fraction
  const display = pct <= 1 ? pct * 100 : pct;
  const cls = confidenceClass(display);

  return (
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
            fontSize: '10px',
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.8px',
          }}
        >
          Confidence
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '12px',
            fontWeight: '700',
            color:
              cls === 'high'
                ? 'var(--accent-green)'
                : cls === 'medium'
                  ? 'var(--accent-primary)'
                  : 'var(--accent-red)',
          }}
        >
          {display.toFixed(1)}%
        </span>
      </div>
      <div className="confidence-bar" style={{ height: '6px' }}>
        <div
          className={`confidence-fill ${cls}`}
          style={{ width: `${display}%` }}
        />
      </div>
    </div>
  );
}

function SymbolCard({ symbol, data }) {
  const isTrained = !!data?.is_trained;
  const confidence = data?.prediction_confidence != null
    ? parseFloat(data.prediction_confidence) * 100
    : (data?.metrics?.accuracy
        ? parseFloat(data.metrics.accuracy) * 100
        : 0);
  const direction = data?.prediction_direction || data?.direction || null;
  const modelType = data?.model_type || 'Unknown';

  return (
    <div
      className="card"
      style={{
        borderColor: isTrained ? 'var(--border-dim)' : 'var(--border-subtle)',
        background: isTrained
          ? 'var(--bg-card)'
          : 'rgba(13, 21, 32, 0.5)',
        marginBottom: '12px',
      }}
    >
      {/* Card top stripe for trained models */}
      {isTrained && (
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '2px',
            background:
              'linear-gradient(90deg, transparent, var(--accent-purple), transparent)',
          }}
        />
      )}

      {/* Symbol header row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '36px',
              height: '36px',
              borderRadius: 'var(--radius-md)',
              background: isTrained
                ? 'var(--accent-purple-dim)'
                : 'var(--bg-surface)',
              border: `1px solid ${isTrained ? 'rgba(168,85,247,0.3)' : 'var(--border-subtle)'}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              fontWeight: '700',
              color: isTrained ? 'var(--accent-purple)' : 'var(--text-muted)',
              flexShrink: 0,
            }}
          >
            {symbol.slice(0, 2)}
          </div>
          <div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '15px',
                fontWeight: '700',
                color: 'var(--text-primary)',
                letterSpacing: '0.5px',
              }}
            >
              {symbol}
            </div>
            <div
              style={{
                fontSize: '10px',
                color: 'var(--text-muted)',
                marginTop: '1px',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {modelType}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {direction && <DirectionBadge direction={direction} />}
          <span
            className={`badge ${isTrained ? 'badge-ai' : 'badge-hold'}`}
            style={{ display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            {isTrained ? (
              <>
                <CheckCircle size={10} />
                Trained
              </>
            ) : (
              <>
                <AlertCircle size={10} />
                Untrained
              </>
            )}
          </span>
        </div>
      </div>

      {/* Confidence gauge */}
      <ConfidenceGauge confidence={confidence} />

      {/* Metrics */}
      <MetricsRow metrics={data?.metrics} />

      {/* Feature importance chart */}
      <FeatureImportanceChart importanceMap={data?.feature_importance} />
    </div>
  );
}

// ─── Training overlay ────────────────────────────────────────────────────────

function TrainingOverlay() {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: 'rgba(8, 13, 20, 0.85)',
        backdropFilter: 'blur(4px)',
        borderRadius: 'var(--radius-lg)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '16px',
        zIndex: 10,
      }}
    >
      <div
        style={{
          width: '48px',
          height: '48px',
          border: '3px solid var(--border-dim)',
          borderTopColor: 'var(--accent-purple)',
          borderRadius: '50%',
          animation: 'spin 0.7s linear infinite',
        }}
      />
      <div style={{ textAlign: 'center' }}>
        <div
          style={{
            fontSize: '14px',
            fontWeight: '600',
            color: 'var(--accent-purple)',
            marginBottom: '4px',
          }}
        >
          Training Models…
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          This may take a few moments
        </div>
      </div>
    </div>
  );
}

// ─── Empty state ─────────────────────────────────────────────────────────────

function EmptyState({ onTrainModels, isTraining }) {
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
          width: '56px',
          height: '56px',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--accent-purple-dim)',
          border: '1px solid rgba(168,85,247,0.2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Brain size={26} color="var(--accent-purple)" />
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
          No AI models loaded
        </div>
        <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          Train models to see predictions and metrics.
        </div>
      </div>
      <button
        className="btn btn-primary btn-sm"
        onClick={onTrainModels}
        disabled={isTraining}
      >
        {isTraining ? (
          <>
            <div className="spinner" style={{ width: '14px', height: '14px' }} />
            Training…
          </>
        ) : (
          <>
            <Zap size={14} />
            Train All Models
          </>
        )}
      </button>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function AIModelPanel({ aiStatus = {}, onTrainModels, isTraining = false }) {
  const symbols = Object.keys(aiStatus);
  const trainedCount = symbols.filter((s) => aiStatus[s]?.is_trained).length;

  // Detect a dominant model type across all symbols
  const modelTypes = symbols
    .map((s) => aiStatus[s]?.model_type)
    .filter(Boolean);
  const dominantModel = modelTypes.length
    ? modelTypes.sort(
        (a, b) =>
          modelTypes.filter((v) => v === b).length -
          modelTypes.filter((v) => v === a).length,
      )[0]
    : null;

  return (
    <div
      className="card"
      style={{
        position: 'relative',
        padding: 0,
        overflow: 'visible',
        background: 'var(--bg-card)',
      }}
    >
      {/* Panel header */}
      <div
        style={{
          padding: '16px 20px 0 20px',
          borderBottom: '1px solid var(--border-subtle)',
          paddingBottom: '14px',
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
              background: 'var(--accent-purple-dim)',
              border: '1px solid rgba(168,85,247,0.25)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Brain size={16} color="var(--accent-purple)" />
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
              AI Prediction Engine
              {dominantModel && (
                <span className="badge badge-ai" style={{ fontSize: '10px' }}>
                  <Cpu size={9} />
                  {dominantModel}
                </span>
              )}
            </div>
            <div
              style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                marginTop: '2px',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {symbols.length > 0
                ? `${trainedCount}/${symbols.length} models trained`
                : 'No models loaded'}
            </div>
          </div>
        </div>

        <button
          className="btn btn-primary btn-sm"
          onClick={onTrainModels}
          disabled={isTraining}
        >
          {isTraining ? (
            <>
              <div className="spinner" style={{ width: '13px', height: '13px' }} />
              Training…
            </>
          ) : (
            <>
              <RefreshCw size={13} />
              Train All Models
            </>
          )}
        </button>
      </div>

      {/* Body */}
      <div
        style={{
          padding: '16px 20px 20px 20px',
          position: 'relative',
          maxHeight: '560px',
          overflowY: 'auto',
        }}
      >
        {isTraining && <TrainingOverlay />}

        {symbols.length === 0 ? (
          <EmptyState onTrainModels={onTrainModels} isTraining={isTraining} />
        ) : (
          symbols.map((symbol) => (
            <SymbolCard key={symbol} symbol={symbol} data={aiStatus[symbol]} />
          ))
        )}
      </div>
    </div>
  );
}
