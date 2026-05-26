import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export const getPortfolio = () => api.get('/api/portfolio')
export const getEquityCurve = (limit = 200) => api.get(`/api/portfolio/equity-curve?limit=${limit}`)
export const getPositions = () => api.get('/api/positions')
export const closePosition = (symbol) => api.delete(`/api/positions/${symbol}`)
export const getTrades = (limit = 50) => api.get(`/api/trades?limit=${limit}`)
export const placeManualTrade = (data) => api.post('/api/trades/manual', data)
export const getSignals = (limit = 50) => api.get(`/api/signals?limit=${limit}`)
export const generateSignals = () => api.post('/api/signals/generate')
export const getChartData = (symbol, timeframe = '1d', lookback = 180) =>
  api.get(`/api/market/${symbol}/chart?timeframe=${timeframe}&lookback_days=${lookback}`)
export const getCurrentPrice = (symbol) => api.get(`/api/market/${symbol}/price`)
export const getFundamentals = (symbol) => api.get(`/api/market/${symbol}/fundamentals`)
export const getAIStatus = () => api.get('/api/ai/status')
export const trainModels = () => api.post('/api/ai/train')
export const getPrediction = (symbol) => api.get(`/api/ai/predict/${symbol}`)
export const runBacktest = (data) => api.post('/api/backtest', data)
export const getRiskLogs = (limit = 50) => api.get(`/api/risk/logs?limit=${limit}`)
export const getSettings = () => api.get('/api/settings')
export const updateSettings = (data) => api.put('/api/settings', data)
export const testAlpaca = (data) => api.post('/api/settings/alpaca/test', data)
export const saveAlpaca = (data) => api.post('/api/settings/alpaca/save', data)
export const getStatus = () => api.get('/api/status')
export const getHealth = () => api.get('/api/health')
export const getNews = () => api.get('/api/news')

export default api

