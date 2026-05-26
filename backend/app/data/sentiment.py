"""
Sentiment Analysis Module: Fetches news from yfinance and evaluates sentiment scores.
Uses Hugging Face's FinBERT classifier by default, falling back to NLTK VADER if FinBERT fails.
"""
import logging
import yfinance as yf
import nltk

logger = logging.getLogger(__name__)

# Pre-download vader lexicon as fallback
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
        logger.info("Successfully downloaded NLTK vader_lexicon.")
    except Exception as e:
        logger.error(f"Failed to download NLTK vader_lexicon: {e}")

# Try to initialize VADER SentimentIntensityAnalyzer as a fallback
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _vader_sia = SentimentIntensityAnalyzer()
except Exception as e:
    logger.error(f"Error initializing SentimentIntensityAnalyzer: {e}")
    _vader_sia = None

# FinBERT pipeline container (lazy loaded)
_finbert_pipeline = None


def _get_finbert_pipeline():
    """Lazy loader for Hugging Face FinBERT pipeline to prevent blocking FastAPI startup."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        logger.info("Initializing FinBERT model (ProsusAI/finbert)...")
        # Load the model. Hugging Face downloads the model (approx 440MB) on the first call.
        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        logger.info("FinBERT successfully initialized.")
        return _finbert_pipeline
    except Exception as e:
        logger.warning(f"Failed to initialize FinBERT pipeline: {e}. Falling back to VADER.")
        _finbert_pipeline = False  # Mark as False to prevent repeated loading attempts
        return None


def get_news_sentiment(symbol: str) -> float:
    """
    Fetch recent news headlines and summaries for a symbol and calculate the average compound sentiment score.
    Uses FinBERT by default. Falls back to NLTK VADER if FinBERT is unavailable.
    Returns a score between -1.0 (bearish) and +1.0 (bullish).
    Returns 0.0 if no news is found or if parsing fails.
    """
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            logger.debug(f"No news found for {symbol}")
            return 0.0

        # Extract articles text context (title + summary)
        articles_text = []
        for article in news[:5]:
            content = article.get("content") or {}
            title = content.get("title") or ""
            summary = content.get("summary") or ""
            text = f"{title}. {summary}" if summary else title
            if text.strip():
                articles_text.append(text)

        if not articles_text:
            return 0.0

        # 1. Attempt FinBERT sentiment classification
        finbert = _get_finbert_pipeline()
        if finbert:
            try:
                # Batch predict to run efficiently
                # We ask for all scores (positive, negative, neutral)
                predictions = finbert(articles_text, top_k=None)
                
                scores = []
                for pred in predictions:
                    pos_score = 0.0
                    neg_score = 0.0
                    for label_obj in pred:
                        label = label_obj.get("label", "").lower()
                        score = label_obj.get("score", 0.0)
                        if label == "positive":
                            pos_score = score
                        elif label == "negative":
                            neg_score = score
                    # Sentiment score is Prob(positive) - Prob(negative)
                    scores.append(pos_score - neg_score)
                
                if scores:
                    avg_sentiment = sum(scores) / len(scores)
                    logger.info(f"FinBERT sentiment calculated for {symbol}: {avg_sentiment:.3f} (based on {len(scores)} articles)")
                    return round(avg_sentiment, 4)
            except Exception as e:
                logger.error(f"Error during FinBERT execution: {e}. Falling back to VADER.")

        # 2. Fallback to NLTK VADER compound sentiment analysis
        if _vader_sia is None:
            logger.warning("Sentiment Intensity Analyzer fallback not initialized, returning 0.0 sentiment.")
            return 0.0

        scores = []
        for text in articles_text:
            pol = _vader_sia.polarity_scores(text)
            scores.append(pol["compound"])

        if not scores:
            return 0.0

        avg_sentiment = sum(scores) / len(scores)
        logger.info(f"VADER fallback sentiment calculated for {symbol}: {avg_sentiment:.3f} (based on {len(scores)} articles)")
        return round(avg_sentiment, 4)

    except Exception as e:
        logger.error(f"Error calculating news sentiment for {symbol}: {e}")
        return 0.0
