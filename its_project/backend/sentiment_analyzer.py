"""
Sentiment Analysis for Cryptocurrency Markets
Uses FinBERT model for financial text sentiment analysis
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT model.
    
    FinBERT is a pre-trained NLP model for financial sentiment analysis
    that can classify text as positive, negative, or neutral.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading FinBERT model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise
    
    def analyze_sentiment(self, text: Union[str, List[str]]) -> Dict[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Single text string or list of text strings
            
        Returns:
            Dictionary with sentiment scores:
            - positive: probability of positive sentiment
            - negative: probability of negative sentiment
            - neutral: probability of neutral sentiment
            - label: predicted label (positive, negative, neutral)
        """
        if isinstance(text, str):
            text = [text]
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
            
            # Get results
            probs_np = probs.cpu().numpy()
            
            # FinBERT labels: 0=negative, 1=neutral, 2=positive
            result = {
                "negative": float(probs_np[0, 0]),
                "neutral": float(probs_np[0, 1]),
                "positive": float(probs_np[0, 2]),
                "label": self._get_label(probs_np[0])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                "negative": 0.0,
                "neutral": 1.0,
                "positive": 0.0,
                "label": "neutral"
            }
    
    def _get_label(self, probs: np.ndarray) -> str:
        """Get label from probabilities."""
        idx = np.argmax(probs)
        if idx == 0:
            return "negative"
        elif idx == 1:
            return "neutral"
        else:
            return "positive"
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of sentiment dictionaries
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
        return results
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get single sentiment score (-1 to 1).
        
        Args:
            text: Text string
            
        Returns:
            Sentiment score: -1 (negative) to 1 (positive)
        """
        sentiment = self.analyze_sentiment(text)
        score = sentiment["positive"] - sentiment["negative"]
        return score


# Mock implementation for testing without internet
class MockSentimentAnalyzer:
    """
    Mock sentiment analyzer for testing without internet access.
    Uses simple keyword-based sentiment analysis.
    """
    
    def __init__(self):
        self.positive_keywords = [
            "bull", "bullish", "buy", "rise", "increase", "growth", "up",
            "positive", "good", "great", "excellent", "strong", "gain"
        ]
        self.negative_keywords = [
            "bear", "bearish", "sell", "fall", "decrease", "drop", "down",
            "negative", "bad", "poor", "weak", "loss", "crash", "dump"
        ]
    
    def analyze_sentiment(self, text: Union[str, List[str]]) -> Dict[str, float]:
        """Analyze sentiment using keyword matching."""
        if isinstance(text, str):
            text = [text]
        
        text_lower = text[0].lower()
        
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        
        total = positive_count + negative_count + 1  # +1 for neutral baseline
        
        positive = positive_count / total
        negative = negative_count / total
        neutral = 1 - positive - negative
        
        if positive > negative and positive > neutral:
            label = "positive"
        elif negative > positive and negative > neutral:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "label": label
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_sentiment_score(self, text: str) -> float:
        """Get single sentiment score (-1 to 1)."""
        sentiment = self.analyze_sentiment(text)
        return sentiment["positive"] - sentiment["negative"]


def get_sentiment_analyzer(use_mock: bool = False) -> Union[SentimentAnalyzer, MockSentimentAnalyzer]:
    """
    Get sentiment analyzer instance.
    
    Args:
        use_mock: If True, use mock analyzer for testing
        
    Returns:
        Sentiment analyzer instance
    """
    if use_mock:
        logger.info("Using mock sentiment analyzer")
        return MockSentimentAnalyzer()
    else:
        try:
            return SentimentAnalyzer()
        except Exception as e:
            logger.warning(f"Failed to load FinBERT, using mock: {e}")
            return MockSentimentAnalyzer()
