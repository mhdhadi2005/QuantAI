from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    # Broker
    TRADING_MODE: str = "paper_sim"
    INITIAL_CAPITAL: float = 1000.0
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"

    # Database
    DATABASE_URL: str = "sqlite:///./trading_system.db"

    # AI Model
    AI_CONFIDENCE_THRESHOLD: float = 0.65
    MODEL_RETRAIN_INTERVAL_HOURS: int = 24
    LOOKBACK_DAYS: int = 365

    # Risk Management
    MAX_RISK_PER_TRADE_PCT: float = 0.01
    MAX_DAILY_LOSS_PCT: float = 0.03
    MAX_OPEN_POSITIONS: int = 5
    STOP_LOSS_ATR_MULTIPLIER: float = 2.0
    TAKE_PROFIT_ATR_MULTIPLIER: float = 4.0
    SIM_COMMISSION: float = 0.0
    SIM_SLIPPAGE_PCT: float = 0.0001

    # Strategies
    DEFAULT_SYMBOLS: str = "AAPL,MSFT,NVDA,SPY,QQQ,TSLA,AMZN,GOOGL,META,AMD,NFLX,JPM,V,DIS,RELIANCE.NS,TCS.NS,HDFCBANK.NS,INFY.NS,ICICIBANK.NS,^NSEI"
    STRATEGY_MOMENTUM_ENABLED: bool = True
    STRATEGY_MEAN_REVERSION_ENABLED: bool = True
    STRATEGY_BREAKOUT_ENABLED: bool = True
    STRATEGY_AI_ENABLED: bool = True
    STRATEGY_NEWS_SENTIMENT_ENABLED: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    LOG_LEVEL: str = "info"

    # Telegram Notifications
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    @property
    def symbols_list(self) -> List[str]:
        return [s.strip() for s in self.DEFAULT_SYMBOLS.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return [s.strip() for s in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def update_env_file(updates: dict) -> bool:
    """Updates key-value pairs in the .env file while preserving comments and structure."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Try multiple common paths for .env
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),  # root/backend/.env from app/config.py
        os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"),                  # backend/.env from app/config.py
        ".env",
    ]
    
    env_path = None
    for path in env_paths:
        if os.path.exists(path):
            env_path = path
            break
            
    if not env_path:
        # Default to root/backend/.env
        env_path = env_paths[1]
        logger.warning(f".env file not found, defaulting to creating/writing at: {env_path}")

    try:
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        new_lines = []
        keys_to_update = set(updates.keys())

        for line in lines:
            stripped = line.strip()
            # Preserve comments and empty lines
            if not stripped or stripped.startswith("#"):
                new_lines.append(line)
                continue

            parts = line.split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                if key in keys_to_update:
                    val = str(updates[key])
                    # Wrap strings with spaces/commas in double quotes
                    if " " in val or "," in val:
                        val = f'"{val}"'
                    new_lines.append(f"{key}={val}\n")
                    keys_to_update.remove(key)
                    continue
            new_lines.append(line)

        # Append any keys that weren't found in the existing .env file
        for key in keys_to_update:
            val = str(updates[key])
            if " " in val or "," in val:
                val = f'"{val}"'
            new_lines.append(f"{key}={val}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        logger.info(f"Successfully updated .env file at {env_path} with keys: {list(updates.keys())}")
        return True
    except Exception as e:
        logger.error(f"Failed to update .env file: {e}")
        return False
