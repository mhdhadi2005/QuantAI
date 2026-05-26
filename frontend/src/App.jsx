import { useState, useEffect, useCallback, useMemo } from 'react'
import Header from './components/Header'
import NewsTicker from './components/NewsTicker'
import PortfolioSummary from './components/PortfolioSummary'
import PriceChart from './components/PriceChart'
import ActivePositions from './components/ActivePositions'
import EquityChart from './components/EquityChart'
import SignalsPanel from './components/SignalsPanel'
import AIModelPanel from './components/AIModelPanel'
import BacktestPanel from './components/BacktestPanel'
import SettingsPanel from './components/SettingsPanel'
import { useWebSocket } from './hooks/useWebSocket'
import {
  getPortfolio, getPositions, getTrades, getSignals,
  getEquityCurve, getAIStatus, getSettings, generateSignals,
  trainModels, closePosition, getChartData, getNews,
  updateSettings, saveAlpaca,
} from './utils/api'

export default function App() {
  const [activePage, setActivePage] = useState('dashboard')
  const [portfolio, setPortfolio] = useState(null)
  const [positions, setPositions] = useState([])
  const [trades, setTrades] = useState([])
  const [signals, setSignals] = useState([])
  const [equityCurve, setEquityCurve] = useState([])
  const [aiStatus, setAiStatus] = useState({})
  const [settings, setSettings] = useState(null)
  const [chartData, setChartData] = useState([])
  const [selectedSymbol, setSelectedSymbol] = useState('SPY')
  const [chartTimeframe, setChartTimeframe] = useState('1d')
  const [isTraining, setIsTraining] = useState(false)
  const [loadingChart, setLoadingChart] = useState(false)
  const [livePrices, setLivePrices] = useState({})
  const [toasts, setToasts] = useState([])
  const [news, setNews] = useState([])

  const { isConnected, lastMessage } = useWebSocket()

  const addToast = useCallback((msg, type = 'info') => {
    const id = Date.now()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }, [])

  // Initial data load
  useEffect(() => {
    const load = async () => {
      try {
        const [port, pos, tr, sig, eq, ai, stg, nws] = await Promise.allSettled([
          getPortfolio(), getPositions(), getTrades(), getSignals(),
          getEquityCurve(), getAIStatus(), getSettings(), getNews(),
        ])
        if (port.status === 'fulfilled') setPortfolio(port.value.data)
        if (pos.status === 'fulfilled') setPositions(pos.value.data.positions || [])
        if (tr.status === 'fulfilled') setTrades(tr.value.data.trades || [])
        if (sig.status === 'fulfilled') setSignals(sig.value.data.signals || [])
        if (eq.status === 'fulfilled') setEquityCurve(eq.value.data.equity_curve || [])
        if (ai.status === 'fulfilled') setAiStatus(ai.value.data || {})
        if (stg.status === 'fulfilled') setSettings(stg.value.data)
        if (nws.status === 'fulfilled') setNews(nws.value.data.news || [])
      } catch (e) {
        console.error('Initial load error:', e)
      }
    }
    load()
  }, [])

  // Poll data every 60s
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [port, pos, sig, nws] = await Promise.allSettled([
          getPortfolio(), getPositions(), getSignals(), getNews(),
        ])
        if (port.status === 'fulfilled') setPortfolio(port.value.data)
        if (pos.status === 'fulfilled') setPositions(pos.value.data.positions || [])
        if (sig.status === 'fulfilled') setSignals(sig.value.data.signals || [])
        if (nws.status === 'fulfilled') setNews(nws.value.data.news || [])
      } catch (e) {}
    }, 60000)
    return () => clearInterval(interval)
  }, [])

  // WebSocket updates
  useEffect(() => {
    if (!lastMessage) return
    if (lastMessage.type === 'live_prices') {
      setLivePrices(prev => ({
        ...prev,
        ...lastMessage.prices
      }))
    }
    if (lastMessage.type === 'initial_state' || lastMessage.type === 'portfolio_update') {
      if (lastMessage.portfolio) setPortfolio(lastMessage.portfolio)
      if (lastMessage.positions) setPositions(lastMessage.positions)
    }
    if (lastMessage.type === 'trading_cycle') {
      addToast('Trading cycle complete', 'success')
      getPortfolio().then(r => setPortfolio(r.data)).catch(() => {})
      getPositions().then(r => setPositions(r.data.positions || [])).catch(() => {})
      getSignals().then(r => setSignals(r.data.signals || [])).catch(() => {})
    }
  }, [lastMessage, addToast])

  const livePositions = useMemo(() => {
    return positions.map(pos => {
      const livePrice = livePrices[pos.symbol]
      if (livePrice === undefined) return pos
      const unrealized_pnl = (livePrice - pos.entry_price) * pos.qty
      const unrealized_pnl_pct = (livePrice - pos.entry_price) / pos.entry_price
      return {
        ...pos,
        current_price: livePrice,
        unrealized_pnl,
        unrealized_pnl_pct,
      }
    })
  }, [positions, livePrices])

  const livePortfolio = useMemo(() => {
    if (!portfolio) return null
    let positions_value = 0
    positions.forEach(pos => {
      const livePrice = livePrices[pos.symbol] ?? pos.current_price
      positions_value += livePrice * pos.qty
    })
    const total_value = portfolio.cash + positions_value
    const total_pnl = total_value - portfolio.initial_capital
    const total_pnl_pct = portfolio.initial_capital > 0 ? (total_pnl / portfolio.initial_capital) * 100 : 0
    const daily_pnl_start = portfolio.daily_pnl_start_value ?? portfolio.initial_capital
    const daily_pnl = total_value - daily_pnl_start
    return {
      ...portfolio,
      positions_value,
      total_value,
      total_pnl,
      total_pnl_pct,
      daily_pnl,
      open_positions: positions.length,
    }
  }, [portfolio, positions, livePrices])

  // Load chart data when symbol/timeframe changes
  useEffect(() => {
    const loadChart = async () => {
      setLoadingChart(true)
      try {
        const res = await getChartData(selectedSymbol, chartTimeframe, 180)
        setChartData(res.data.bars || [])
      } catch (e) {
        console.error('Chart load error:', e)
      } finally {
        setLoadingChart(false)
      }
    }
    loadChart()
  }, [selectedSymbol, chartTimeframe])

  const handleClosePosition = async (symbol) => {
    try {
      await closePosition(symbol)
      addToast(`Position ${symbol} closed`, 'success')
      const res = await getPositions()
      setPositions(res.data.positions || [])
      const port = await getPortfolio()
      setPortfolio(port.data)
    } catch (e) {
      addToast(`Failed to close ${symbol}: ${e.message}`, 'error')
    }
  }

  const handleGenerateSignals = async () => {
    try {
      await generateSignals()
      addToast('Signal generation triggered', 'info')
      setTimeout(async () => {
        const res = await getSignals()
        setSignals(res.data.signals || [])
      }, 3000)
    } catch (e) {
      addToast('Failed to generate signals', 'error')
    }
  }

  const handleTrainModels = async () => {
    setIsTraining(true)
    try {
      await trainModels()
      addToast('Model training started in background', 'info')
      setTimeout(async () => {
        const res = await getAIStatus()
        setAiStatus(res.data || {})
        setIsTraining(false)
      }, 30000)
    } catch (e) {
      addToast('Training failed: ' + e.message, 'error')
      setIsTraining(false)
    }
  }

  const handleSaveSettings = async (formData) => {
    try {
      const res = await updateSettings(formData)
      if (res.data && res.data.success) {
        addToast('Settings saved and applied successfully!', 'success')
        const updated = await getSettings()
        setSettings(updated.data)
      } else {
        addToast(res.data?.error || 'Failed to save settings', 'error')
      }
    } catch (e) {
      addToast(`Error saving settings: ${e.message}`, 'error')
    }
  }

  const handleSaveAlpaca = async (alpacaData) => {
    try {
      const res = await saveAlpaca(alpacaData)
      if (res.data && res.data.success) {
        addToast('Alpaca credentials saved and broker mode activated!', 'success')
        const updated = await getSettings()
        setSettings(updated.data)
        const port = await getPortfolio()
        setPortfolio(port.data)
        const pos = await getPositions()
        setPositions(pos.data.positions || [])
      } else {
        addToast(res.data?.error || 'Failed to save Alpaca settings', 'error')
      }
    } catch (e) {
      addToast(`Error saving Alpaca credentials: ${e.message}`, 'error')
    }
  }

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              <PortfolioSummary portfolio={livePortfolio} />
            </div>
            <div className="col-8">
              <PriceChart
                symbol={selectedSymbol}
                onSymbolChange={setSelectedSymbol}
                chartData={chartData}
                loading={loadingChart}
                timeframe={chartTimeframe}
                onTimeframeChange={setChartTimeframe}
                livePrice={livePrices[selectedSymbol]}
              />
            </div>
            <div className="col-4">
              <SignalsPanel signals={signals} onGenerateSignals={handleGenerateSignals} loading={false} />
            </div>
            <div className="col-12">
              <EquityChart equityCurve={equityCurve} loading={!livePortfolio} />
            </div>
          </div>
        )
      case 'portfolio':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              <PortfolioSummary portfolio={livePortfolio} />
            </div>
            <div className="col-12">
              <ActivePositions positions={livePositions} onClosePosition={handleClosePosition} loading={false} />
            </div>
            <div className="col-12">
              <EquityChart equityCurve={equityCurve} loading={!livePortfolio} />
            </div>
          </div>
        )
      case 'signals':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              <SignalsPanel signals={signals} onGenerateSignals={handleGenerateSignals} loading={false} />
            </div>
          </div>
        )
      case 'ai':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              <AIModelPanel aiStatus={aiStatus} onTrainModels={handleTrainModels} isTraining={isTraining} />
            </div>
          </div>
        )
      case 'backtest':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              <BacktestPanel />
            </div>
          </div>
        )
      case 'settings':
        return (
          <div className="dashboard-grid">
            <div className="col-12">
              {settings ? (
                <SettingsPanel 
                  settings={settings} 
                  onSave={handleSaveSettings} 
                  onSaveAlpaca={handleSaveAlpaca} 
                />
              ) : (
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '40px', gap: '16px' }}>
                  <RefreshCw size={24} className="spin" style={{ color: 'var(--accent-primary)' }} />
                  <span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Loading configuration settings...</span>
                </div>
              )}
            </div>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="app">
      <Header
        activePage={activePage}
        onNavigate={setActivePage}
        isConnected={isConnected}
        tradingMode={settings?.trading_mode || 'paper_sim'}
      />
      <NewsTicker news={news} />
      <main className="main-content">
        {renderPage()}
      </main>

      {/* Toast notifications */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast ${t.type}`}>
            <span>{t.type === 'success' ? '✓' : t.type === 'error' ? '✗' : 'ℹ'}</span>
            <span style={{ fontSize: 13 }}>{t.msg}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
