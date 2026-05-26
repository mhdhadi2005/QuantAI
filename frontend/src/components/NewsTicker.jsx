import { useRef } from 'react'

export default function NewsTicker({ news }) {
  const scrollRef = useRef(null)

  if (!news || news.length === 0) {
    return (
      <div className="news-ticker-container">
        <div className="news-ticker-label">
          <span className="pulse-dot"></span>
          <span>LIVE MARKET NEWS</span>
        </div>
        <div className="news-ticker-content">
          <div className="news-ticker-scroll">
            <span className="news-ticker-item">Loading live finance news feeds...</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="news-ticker-container">
      <div className="news-ticker-label">
        <span className="pulse-dot"></span>
        <span>LIVE MARKET NEWS</span>
      </div>
      <div className="news-ticker-content">
        <div className="news-ticker-scroll" ref={scrollRef}>
          {[...news, ...news].map((item, idx) => (
            <span key={item.id + '-' + idx} className="news-ticker-item">
              <span className="news-ticker-symbol">[{item.symbol}]</span>
              {item.url ? (
                <a href={item.url} target="_blank" rel="noopener noreferrer" className="news-ticker-link">
                  {item.title}
                </a>
              ) : (
                <span className="news-ticker-title">{item.title}</span>
              )}
              <span className="news-ticker-provider">({item.provider || 'Yahoo Finance'})</span>
              <span className="news-ticker-divider">•</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}
