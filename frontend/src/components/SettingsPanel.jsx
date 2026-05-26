import { useState } from 'react'
import { 
  Settings, Save, RefreshCw, AlertTriangle, CheckCircle, 
  Lock, Eye, EyeOff, ShieldCheck, AlertCircle, Key, Globe
} from 'lucide-react'
import { testAlpaca } from '../utils/api'

export default function SettingsPanel({ settings, onSave, onSaveAlpaca }) {
  const [form, setForm] = useState({
    symbols: settings?.symbols?.join(',') || 'AAPL,MSFT,NVDA,SPY,QQQ',
    ai_confidence_threshold: settings?.ai_confidence_threshold || 0.65,
    max_risk_per_trade_pct: settings?.max_risk_per_trade_pct || 0.01,
    max_daily_loss_pct: settings?.max_daily_loss_pct || 0.03,
    max_open_positions: settings?.max_open_positions || 5,
    stop_loss_atr_multiplier: settings?.stop_loss_atr_multiplier || 2.0,
    take_profit_atr_multiplier: settings?.take_profit_atr_multiplier || 4.0,
    momentum: settings?.strategies?.momentum ?? true,
    mean_reversion: settings?.strategies?.mean_reversion ?? true,
    breakout: settings?.strategies?.breakout ?? true,
    ai_confidence: settings?.strategies?.ai_confidence ?? true,
    
    // Simulated Broker updates
    sim_commission: settings?.sim_commission ?? 0.0,
    sim_slippage_pct: settings?.sim_slippage_pct ?? 0.0001,

    // Alpaca Wizard updates
    alpaca_api_key: settings?.alpaca_api_key || '',
    alpaca_secret_key: settings?.alpaca_secret_key || '',
    alpaca_base_url: settings?.alpaca_base_url || 'https://paper-api.alpaca.markets',
    trading_mode: settings?.trading_mode || 'paper_sim',
  })

  const [saved, setSaved] = useState(false)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null) // null | { success: true } | { success: false, error: string }
  const [savingAlpaca, setSavingAlpaca] = useState(false)
  const [showSecret, setShowSecret] = useState(false)

  const handleChange = (key, value) => setForm(f => ({ ...f, [key]: value }))

  const handleSave = () => {
    onSave?.(form)
    setSaved(true)
    setTimeout(() => setSaved(false), 2500)
  }

  const handleTestAlpaca = async () => {
    setTesting(true)
    setTestResult(null)
    try {
      const res = await testAlpaca({
        api_key: form.alpaca_api_key,
        secret_key: form.alpaca_secret_key,
        base_url: form.alpaca_base_url
      })
      if (res.data && res.data.success) {
        setTestResult({ success: true, message: res.data.message })
      } else {
        setTestResult({ success: false, error: res.data?.error || 'Validation failed' })
      }
    } catch (e) {
      setTestResult({ success: false, error: e.response?.data?.detail || e.message || 'Request failed' })
    } finally {
      setTesting(false)
    }
  }

  const handleSaveAlpacaCredentials = async () => {
    setSavingAlpaca(true)
    try {
      await onSaveAlpaca?.({
        api_key: form.alpaca_api_key,
        secret_key: form.alpaca_secret_key,
        base_url: form.alpaca_base_url,
        trading_mode: form.trading_mode
      })
    } catch (e) {
      console.error(e)
    } finally {
      setSavingAlpaca(false)
    }
  }

  const ToggleSwitch = ({ checked, onChange }) => (
    <div onClick={() => onChange(!checked)} style={{
      width: 44, height: 24, borderRadius: 12, cursor: 'pointer',
      background: checked ? 'var(--accent-primary)' : 'var(--bg-surface)',
      border: '1px solid var(--border-dim)', position: 'relative', transition: 'background 0.2s',
    }}>
      <div style={{
        width: 18, height: 18, borderRadius: '50%', background: 'white',
        position: 'absolute', top: 2, left: checked ? 22 : 2,
        transition: 'left 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
      }} />
    </div>
  )

  const Section = ({ title, children }) => (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-lg)', padding: '16px 20px', marginBottom: 16
    }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
        {title}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
        {children}
      </div>
    </div>
  )

  return (
    <div className="card col-12">
      <div className="card-header">
        <span className="card-title">
          <Settings size={14} className="card-title-icon" />
          System Configuration
        </span>
        <div style={{ display: 'flex', gap: 8 }}>
          {saved && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--accent-green)' }}>
              <CheckCircle size={14} /> Saved!
            </div>
          )}
          <button className="btn btn-primary btn-sm" onClick={handleSave}>
            <Save size={13} /> Save Settings
          </button>
        </div>
      </div>

      <div style={{
        background: 'var(--accent-primary-dim)', border: '1px solid var(--border-dim)',
        borderRadius: 'var(--radius-md)', padding: '10px 14px', marginBottom: 16,
        display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, color: 'var(--text-primary)',
      }}>
        <CheckCircle size={14} style={{ color: 'var(--accent-primary)' }} />
        Settings updated here will be persisted to the <code style={{ fontFamily: 'var(--font-mono)', background: 'rgba(0,0,0,0.2)', padding: '1px 4px', borderRadius: 3 }}>.env</code> file on disk and applied in-memory immediately.
      </div>

      <Section title="📈 Symbols & Data">
        <div className="input-group" style={{ gridColumn: '1 / -1' }}>
          <label className="input-label">Tracked Symbols (comma-separated)</label>
          <input className="input" value={form.symbols} onChange={e => handleChange('symbols', e.target.value)}
            placeholder="AAPL,MSFT,NVDA,SPY,QQQ" />
        </div>
      </Section>

      <Section title="🤖 AI Model">
        <div className="input-group">
          <label className="input-label">AI Confidence Threshold</label>
          <input className="input" type="number" step={0.05} min={0.5} max={0.99}
            value={form.ai_confidence_threshold}
            onChange={e => handleChange('ai_confidence_threshold', parseFloat(e.target.value))} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Only trade when AI confidence ≥ {(form.ai_confidence_threshold * 100).toFixed(0)}%
          </div>
        </div>
      </Section>

      <Section title="⚠️ Risk Management & Fees">
        <div className="input-group">
          <label className="input-label">Max Risk Per Trade (%)</label>
          <input className="input" type="number" step={0.005} min={0.005} max={0.05}
            value={form.max_risk_per_trade_pct}
            onChange={e => handleChange('max_risk_per_trade_pct', parseFloat(e.target.value))} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Risk {(form.max_risk_per_trade_pct * 100).toFixed(1)}% of capital per trade
          </div>
        </div>
        <div className="input-group">
          <label className="input-label">Max Daily Loss (%)</label>
          <input className="input" type="number" step={0.005} min={0.01} max={0.1}
            value={form.max_daily_loss_pct}
            onChange={e => handleChange('max_daily_loss_pct', parseFloat(e.target.value))} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Halt trading after {(form.max_daily_loss_pct * 100).toFixed(1)}% daily loss
          </div>
        </div>
        <div className="input-group">
          <label className="input-label">Max Open Positions</label>
          <input className="input" type="number" min={1} max={20}
            value={form.max_open_positions}
            onChange={e => handleChange('max_open_positions', parseInt(e.target.value))} />
        </div>
        <div className="input-group">
          <label className="input-label">Stop Loss ATR Multiplier</label>
          <input className="input" type="number" step={0.5} min={0.5} max={10}
            value={form.stop_loss_atr_multiplier}
            onChange={e => handleChange('stop_loss_atr_multiplier', parseFloat(e.target.value))} />
        </div>
        <div className="input-group">
          <label className="input-label">Take Profit ATR Multiplier</label>
          <input className="input" type="number" step={0.5} min={1} max={20}
            value={form.take_profit_atr_multiplier}
            onChange={e => handleChange('take_profit_atr_multiplier', parseFloat(e.target.value))} />
        </div>
        <div className="input-group">
          <label className="input-label">Simulated Commission ($)</label>
          <input className="input" type="number" step={0.1} min={0}
            value={form.sim_commission}
            onChange={e => handleChange('sim_commission', parseFloat(e.target.value) || 0.0)} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Fixed fee per simulated order
          </div>
        </div>
        <div className="input-group">
          <label className="input-label">Simulated Slippage (%)</label>
          <input className="input" type="number" step={0.005} min={0} max={1}
            value={parseFloat((form.sim_slippage_pct * 100).toFixed(4))}
            onChange={e => handleChange('sim_slippage_pct', (parseFloat(e.target.value) || 0.0) / 100)} />
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
            Applied slippage per order execution
          </div>
        </div>
      </Section>

      <Section title="🔌 Alpaca API Setup Wizard">
        <div className="input-group" style={{ gridColumn: '1 / -1' }}>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 8 }}>
            Configure API access to Alpaca Markets. Switches trading execution between the local simulator and live/paper brokerage environments.
          </div>
        </div>

        <div className="input-group">
          <label className="input-label">Active Broker Mode</label>
          <select className="select" value={form.trading_mode} onChange={e => handleChange('trading_mode', e.target.value)}>
            <option value="paper_sim">Local Simulator (Virtual Paper)</option>
            <option value="paper_alpaca">Alpaca Paper Trading</option>
            <option value="live_alpaca">Alpaca Live Brokerage (Real Money)</option>
          </select>
        </div>

        {form.trading_mode !== 'paper_sim' && (
          <>
            <div className="input-group">
              <label className="input-label">Alpaca API Key ID</label>
              <div style={{ position: 'relative' }}>
                <Key size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
                <input className="input" style={{ paddingLeft: 30 }} value={form.alpaca_api_key} onChange={e => handleChange('alpaca_api_key', e.target.value)} placeholder="API Key ID" />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Alpaca Secret Key</label>
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <Lock size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
                <input 
                  className="input" 
                  style={{ paddingLeft: 30, paddingRight: 40 }} 
                  type={showSecret ? "text" : "password"} 
                  value={form.alpaca_secret_key} 
                  onChange={e => handleChange('alpaca_secret_key', e.target.value)} 
                  placeholder="Secret Key" 
                />
                <button 
                  type="button" 
                  onClick={() => setShowSecret(!showSecret)}
                  style={{
                    position: 'absolute', right: 5, background: 'transparent', border: 'none', 
                    color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', height: '100%'
                  }}
                >
                  {showSecret ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Alpaca Base URL</label>
              <div style={{ position: 'relative' }}>
                <Globe size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-muted)' }} />
                <input className="input" style={{ paddingLeft: 30 }} value={form.alpaca_base_url} onChange={e => handleChange('alpaca_base_url', e.target.value)} placeholder="https://paper-api.alpaca.markets" />
              </div>
              <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                <button 
                  className="btn btn-ghost btn-sm" 
                  type="button" 
                  onClick={() => handleChange('alpaca_base_url', 'https://paper-api.alpaca.markets')}
                  style={{ fontSize: 10, padding: '2px 8px' }}
                >
                  Use Paper URL
                </button>
                <button 
                  className="btn btn-ghost btn-sm" 
                  type="button" 
                  onClick={() => handleChange('alpaca_base_url', 'https://api.alpaca.markets')}
                  style={{ fontSize: 10, padding: '2px 8px' }}
                >
                  Use Live URL
                </button>
              </div>
            </div>
          </>
        )}

        {form.trading_mode !== 'paper_sim' && (
          <div className="input-group" style={{ gridColumn: '1 / -1', marginTop: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', gap: 10 }}>
              <button 
                type="button" 
                className="btn btn-ghost" 
                onClick={handleTestAlpaca} 
                disabled={testing || !form.alpaca_api_key || !form.alpaca_secret_key}
              >
                {testing ? (
                  <RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <RefreshCw size={13} />
                )}
                Test Connection
              </button>
              <button 
                type="button" 
                className="btn btn-success" 
                onClick={handleSaveAlpacaCredentials} 
                disabled={savingAlpaca || !form.alpaca_api_key || !form.alpaca_secret_key}
              >
                {savingAlpaca ? (
                  <RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Save size={13} />
                )}
                Save & Activate Alpaca
              </button>
            </div>

            {testResult && (
              <div style={{
                background: testResult.success ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)',
                border: `1px solid ${testResult.success ? 'rgba(0, 255, 148, 0.2)' : 'rgba(255, 51, 102, 0.2)'}`,
                borderRadius: 'var(--radius-md)', padding: '10px 14px', fontSize: 12,
                display: 'flex', alignItems: 'center', gap: 8, color: 'var(--text-primary)'
              }}>
                {testResult.success ? (
                  <>
                    <ShieldCheck size={14} style={{ color: 'var(--accent-green)' }} />
                    <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>Connection Success!</span> {testResult.message}
                  </>
                ) : (
                  <>
                    <AlertCircle size={14} style={{ color: 'var(--accent-red)' }} />
                    <span style={{ color: 'var(--accent-red)', fontWeight: 600 }}>Connection Failed!</span> {testResult.error}
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {form.trading_mode === 'paper_sim' && (
          <div className="input-group" style={{ gridColumn: '1 / -1', marginTop: 10 }}>
            <button 
              type="button" 
              className="btn btn-success" 
              onClick={handleSaveAlpacaCredentials} 
              disabled={savingAlpaca}
              style={{ width: 'fit-content' }}
            >
              {savingAlpaca ? (
                <RefreshCw size={13} style={{ animation: 'spin 1s linear infinite' }} />
              ) : (
                <Save size={13} />
              )}
              Save & Activate Local Simulator
            </button>
          </div>
        )}
      </Section>

      <Section title="📊 Active Strategies">
        {[
          { key: 'momentum', label: 'Momentum Strategy', desc: 'EMA crossovers, RSI, MACD signals' },
          { key: 'mean_reversion', label: 'Mean Reversion', desc: 'RSI oversold/overbought + Bollinger Bands' },
          { key: 'breakout', label: 'Breakout Strategy', desc: 'Price breaks above/below S&R levels' },
          { key: 'ai_confidence', label: 'AI Confidence Strategy', desc: `Trade when AI confidence ≥ ${(form.ai_confidence_threshold * 100).toFixed(0)}%` },
        ].map(({ key, label, desc }) => (
          <div key={key} style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 12 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: form[key] ? 'var(--text-primary)' : 'var(--text-muted)', marginBottom: 2 }}>{label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{desc}</div>
            </div>
            <ToggleSwitch checked={form[key]} onChange={v => handleChange(key, v)} />
          </div>
        ))}
      </Section>
    </div>
  )
}
