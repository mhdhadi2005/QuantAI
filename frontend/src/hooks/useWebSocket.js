import { useState, useEffect, useRef, useCallback } from 'react'

// Dynamically build WS URL from current page location (works on any port)
const getWsUrl = () => {
  if (typeof window === 'undefined') return 'ws://localhost:8000/ws'
  const env = import.meta.env.VITE_WS_URL
  if (env) return env
  // In dev, proxy routes /ws → backend, so use same host/port as frontend
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState(null)
  const ws = useRef(null)
  const reconnectTimer = useRef(null)

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(getWsUrl())

      ws.current.onopen = () => {
        setIsConnected(true)
        console.log('WebSocket connected')
        // Start heartbeat
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'ping' }))
        }
      }

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
        } catch (e) {
          console.error('WebSocket parse error:', e)
        }
      }

      ws.current.onclose = () => {
        setIsConnected(false)
        console.log('WebSocket disconnected. Reconnecting in 3s...')
        reconnectTimer.current = setTimeout(connect, 3000)
      }

      ws.current.onerror = (err) => {
        console.error('WebSocket error:', err)
        ws.current?.close()
      }
    } catch (e) {
      console.error('WebSocket connection failed:', e)
      reconnectTimer.current = setTimeout(connect, 5000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      ws.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((data) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    }
  }, [])

  const requestUpdate = useCallback(() => {
    sendMessage({ type: 'request_update' })
  }, [sendMessage])

  return { isConnected, lastMessage, sendMessage, requestUpdate }
}
