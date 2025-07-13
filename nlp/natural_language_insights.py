"""
Natural Language Insights Module

Advanced NLP capabilities for financial text analysis including:
- Analyst commentary parsing
- Earnings call summarization
- Ticker extraction and sentiment analysis
- News sentiment classification
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# Import NLP libraries with fallback handling
try:
    from transformers import (
        pipeline,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - some features will be disabled")

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available - some features will be disabled")

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available - some features will be disabled")

logger = logging.getLogger(__name__)


class NaturalLanguageInsights:
    """Advanced NLP processor for financial text analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NLP processor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize NLP models
        self._initialize_models()

        # Ticker patterns for extraction
        self.ticker_patterns = [
            r"\$[A-Z]{1,5}",  # $AAPL, $TSLA
            r"\b[A-Z]{1,5}\b",  # AAPL, TSLA (context dependent)
            r"\b[A-Z]{1,5}\.[A-Z]{2,3}\b",  # AAPL.US, TSLA.NASDAQ
        ]

        # Financial keywords for context
        self.financial_keywords = {
            "positive": ["bullish", "buy", "strong", "growth", "positive", "up", "rise", "gain"],
            "negative": ["bearish", "sell", "weak", "decline", "negative", "down", "fall", "loss"],
            "neutral": ["hold", "maintain", "stable", "steady", "unchanged"],
        }

        logger.info("Natural Language Insights initialized successfully")

    def _initialize_models(self):
        """Initialize NLP models with error handling."""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Initialize sentiment analysis pipeline
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True,
                )

                # Initialize summarization pipeline
                self.summarizer_pipeline = pipeline(
                    "summarization", model="facebook/bart-large-cnn", max_length=130, min_length=30
                )

                # Initialize text classification for financial sentiment
                self.financial_sentiment_pipeline = pipeline(
                    "text-classification", model="ProsusAI/finbert", return_all_scores=True
                )

                logger.info("Transformers models loaded successfully")
            else:
                self.sentiment_pipeline = None
                self.summarizer_pipeline = None
                self.financial_sentiment_pipeline = None
                logger.warning("Transformers models not available")

            if NLTK_AVAILABLE:
                # Download required NLTK data
                try:
                    nltk.data.find("tokenizers/punkt")
                except LookupError:
                    nltk.download("punkt")

                try:
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords")

                try:
                    nltk.data.find("corpora/wordnet")
                except LookupError:
                    nltk.download("wordnet")

                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
                logger.info("NLTK components initialized successfully")
            else:
                self.stop_words = set()
                self.lemmatizer = None
                logger.warning("NLTK components not available")

            if SPACY_AVAILABLE:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("spaCy model loaded successfully")
                except OSError:
                    logger.warning("spaCy model not found, using basic text processing")
                    self.nlp = None
            else:
                self.nlp = None
                logger.warning("spaCy not available")

        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
            # Set fallback values
            self.sentiment_pipeline = None
            self.summarizer_pipeline = None
            self.financial_sentiment_pipeline = None
            self.stop_words = set()
            self.lemmatizer = None
            self.nlp = None

    def extract_tickers(self, text: str) -> List[Dict[str, Any]]:
        """Extract stock tickers from text with context.

        Args:
            text: Input text to analyze

        Returns:
            List of dictionaries with ticker info
        """
        try:
            tickers = []

            # Extract tickers using patterns
            for pattern in self.ticker_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    ticker = match.group()

                    # Get context around ticker
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]

                    # Analyze sentiment in context
                    sentiment = self._analyze_context_sentiment(context)

                    tickers.append(
                        {
                            "ticker": ticker,
                            "position": match.start(),
                            "context": context,
                            "sentiment": sentiment,
                            "confidence": self._calculate_confidence(context),
                        }
                    )

            # Remove duplicates while preserving order
            seen = set()
            unique_tickers = []
            for ticker in tickers:
                if ticker["ticker"] not in seen:
                    seen.add(ticker["ticker"])
                    unique_tickers.append(ticker)

            logger.info(f"Extracted {len(unique_tickers)} unique tickers")
            return unique_tickers

        except Exception as e:
            logger.error(f"Error extracting tickers: {e}")
            return []

    def _analyze_context_sentiment(self, context: str) -> Dict[str, Any]:
        """Analyze sentiment of text context.

        Args:
            context: Text context to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if self.financial_sentiment_pipeline:
                # Use FinBERT for financial sentiment
                result = self.financial_sentiment_pipeline(context[:512])  # Limit length
                return {"label": result[0]["label"], "score": result[0]["score"], "method": "finbert"}
            elif self.sentiment_pipeline:
                # Use general sentiment analysis
                result = self.sentiment_pipeline(context[:512])
                return {"label": result[0]["label"], "score": result[0]["score"], "method": "general"}
            else:
                # Fallback to keyword-based analysis
                return self._keyword_sentiment_analysis(context)

        except Exception as e:
            logger.error(f"Error analyzing context sentiment: {e}")
            return {"label": "neutral", "score": 0.5, "method": "fallback"}

    def _keyword_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Keyword-based sentiment analysis fallback.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            text_lower = text.lower()

            positive_count = sum(1 for word in self.financial_keywords["positive"] if word in text_lower)
            negative_count = sum(1 for word in self.financial_keywords["negative"] if word in text_lower)
            neutral_count = sum(1 for word in self.financial_keywords["neutral"] if word in text_lower)

            total = positive_count + negative_count + neutral_count

            if total == 0:
                return {"label": "neutral", "score": 0.5, "method": "keyword"}

            if positive_count > negative_count:
                return {"label": "positive", "score": positive_count / total, "method": "keyword"}
            elif negative_count > positive_count:
                return {"label": "negative", "score": negative_count / total, "method": "keyword"}
            else:
                return {"label": "neutral", "score": 0.5, "method": "keyword"}

        except Exception as e:
            logger.error(f"Error in keyword sentiment analysis: {e}")
            return {"label": "neutral", "score": 0.5, "method": "keyword"}

    def _calculate_confidence(self, context: str) -> float:
        """Calculate confidence score for ticker extraction.

        Args:
            context: Text context around ticker

        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence = 0.5  # Base confidence

            # Increase confidence for financial context
            financial_terms = ["stock", "shares", "market", "trading", "price", "earnings"]
            if any(term in context.lower() for term in financial_terms):
                confidence += 0.2

            # Increase confidence for proper formatting
            if re.search(r"\$[A-Z]{1,5}", context):
                confidence += 0.2

            # Increase confidence for company names
            if self.nlp:
                doc = self.nlp(context)
                org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                if org_entities:
                    confidence += 0.1

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def summarize_earnings_call(self, transcript: str, max_length: int = 500) -> Dict[str, Any]:
        """Summarize earnings call transcript.

        Args:
            transcript: Earnings call transcript text
            max_length: Maximum summary length

        Returns:
            Dictionary with summary and key insights
        """
        try:
            if not transcript or len(transcript.strip()) < 100:
                return {
                    "summary": "Insufficient transcript data",
                    "key_points": [],
                    "sentiment": "neutral",
                    "confidence": 0.0,
                }

            # Clean transcript
            cleaned_transcript = self._clean_transcript(transcript)

            # Generate summary
            if self.summarizer_pipeline:
                summary_result = self.summarizer_pipeline(
                    cleaned_transcript, max_length=max_length, min_length=100, do_sample=False
                )
                summary = summary_result[0]["summary_text"]
            else:
                # Fallback to extractive summarization
                summary = self._extractive_summarization(cleaned_transcript, max_length)

            # Extract key points
            key_points = self._extract_key_points(cleaned_transcript)

            # Analyze overall sentiment
            sentiment = self._analyze_context_sentiment(cleaned_transcript)

            return {
                "summary": summary,
                "key_points": key_points,
                "sentiment": sentiment,
                "confidence": self._calculate_summary_confidence(summary, key_points),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error summarizing earnings call: {e}")
            return {
                "summary": "Error processing transcript",
                "key_points": [],
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e),
            }

    def _clean_transcript(self, transcript: str) -> str:
        """Clean earnings call transcript.

        Args:
            transcript: Raw transcript text

        Returns:
            Cleaned transcript text
        """
        try:
            # Remove speaker labels
            cleaned = re.sub(r"^[A-Z\s]+:", "", transcript, flags=re.MULTILINE)

            # Remove timestamps
            cleaned = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", cleaned)

            # Remove extra whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)

            # Remove special characters but keep punctuation
            cleaned = re.sub(r"[^\w\s\.\,\!\?\-\$\%]", "", cleaned)

            return cleaned.strip()

        except Exception as e:
            logger.error(f"Error cleaning transcript: {e}")
            return transcript

    def _extractive_summarization(self, text: str, max_length: int) -> str:
        """Extractive summarization fallback.

        Args:
            text: Text to summarize
            max_length: Maximum summary length

        Returns:
            Extractive summary
        """
        try:
            if NLTK_AVAILABLE:
                # Tokenize into sentences
                sentences = sent_tokenize(text)

                # Score sentences based on word frequency
                word_freq = {}
                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    for word in words:
                        if word not in self.stop_words and word.isalnum():
                            word_freq[word] = word_freq.get(word, 0) + 1

                # Score sentences
                sentence_scores = {}
                for sentence in sentences:
                    words = word_tokenize(sentence.lower())
                    score = sum(
                        word_freq.get(word, 0) for word in words if word not in self.stop_words and word.isalnum()
                    )
                    sentence_scores[sentence] = score

                # Select top sentences
                sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

                summary_sentences = []
                current_length = 0

                for sentence, score in sorted_sentences:
                    if current_length + len(sentence) <= max_length:
                        summary_sentences.append(sentence)
                        current_length += len(sentence)
                    else:
                        break

                return " ".join(summary_sentences)
            else:
                # Simple fallback
                return text[:max_length] + "..." if len(text) > max_length else text

        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text

    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text.

        Args:
            text: Text to analyze

        Returns:
            List of key points
        """
        try:
            key_points = []

            # Look for financial metrics
            metric_patterns = [
                r"revenue.*?\$[\d,]+",  # Revenue mentions
                r"earnings.*?\$[\d,]+",  # Earnings mentions
                r"growth.*?\d+%",  # Growth percentages
                r"profit.*?\$[\d,]+",  # Profit mentions
            ]

            for pattern in metric_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    key_points.append(match.group())

            # Look for forward-looking statements
            forward_patterns = [
                r"we expect.*?",
                r"outlook.*?",
                r"guidance.*?",
                r"forecast.*?",
            ]

            for pattern in forward_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    key_points.append(match.group())

            # Limit to top 5 key points
            return key_points[:5]

        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return []

    def _calculate_summary_confidence(self, summary: str, key_points: List[str]) -> float:
        """Calculate confidence score for summary.

        Args:
            summary: Generated summary
            key_points: Extracted key points

        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence = 0.5  # Base confidence

            # Increase confidence for longer summary
            if len(summary) > 200:
                confidence += 0.2

            # Increase confidence for key points
            if len(key_points) > 0:
                confidence += 0.2

            # Increase confidence for financial terms
            financial_terms = ["revenue", "earnings", "growth", "profit", "guidance"]
            if any(term in summary.lower() for term in financial_terms):
                confidence += 0.1

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating summary confidence: {e}")
            return 0.5

    def analyze_analyst_commentary(self, commentary: str) -> Dict[str, Any]:
        """Analyze analyst commentary for insights.

        Args:
            commentary: Analyst commentary text

        Returns:
            Dictionary with analysis results
        """
        try:
            if not commentary or len(commentary.strip()) < 50:
                return {"tickers": [], "sentiment": "neutral", "recommendations": [], "confidence": 0.0}

            # Extract tickers
            tickers = self.extract_tickers(commentary)

            # Analyze sentiment
            sentiment = self._analyze_context_sentiment(commentary)

            # Extract recommendations
            recommendations = self._extract_recommendations(commentary)

            return {
                "tickers": tickers,
                "sentiment": sentiment,
                "recommendations": recommendations,
                "confidence": self._calculate_commentary_confidence(commentary, tickers, recommendations),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error analyzing analyst commentary: {e}")
            return {"tickers": [], "sentiment": "neutral", "recommendations": [], "confidence": 0.0, "error": str(e)}

    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract investment recommendations from text.

        Args:
            text: Text to analyze

        Returns:
            List of recommendations
        """
        try:
            recommendations = []

            # Look for recommendation patterns
            recommendation_patterns = [
                r"buy.*?",
                r"sell.*?",
                r"hold.*?",
                r"strong buy.*?",
                r"strong sell.*?",
                r"outperform.*?",
                r"underperform.*?",
                r"neutral.*?",
            ]

            for pattern in recommendation_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    recommendations.append(match.group())

            return recommendations[:5]  # Limit to top 5

        except Exception as e:
            logger.error(f"Error extracting recommendations: {e}")
            return []

    def _calculate_commentary_confidence(
        self, commentary: str, tickers: List[Dict], recommendations: List[str]
    ) -> float:
        """Calculate confidence score for commentary analysis.

        Args:
            commentary: Original commentary text
            tickers: Extracted tickers
            recommendations: Extracted recommendations

        Returns:
            Confidence score between 0 and 1
        """
        try:
            confidence = 0.3  # Base confidence

            # Increase confidence for tickers found
            if len(tickers) > 0:
                confidence += 0.2

            # Increase confidence for recommendations found
            if len(recommendations) > 0:
                confidence += 0.2

            # Increase confidence for longer commentary
            if len(commentary) > 200:
                confidence += 0.1

            # Increase confidence for financial terms
            financial_terms = ["price target", "earnings", "revenue", "growth", "valuation"]
            if any(term in commentary.lower() for term in financial_terms):
                confidence += 0.2

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating commentary confidence: {e}")
            return 0.3

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of NLP components.

        Returns:
            Dictionary with component status
        """
        return {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "nltk_available": NLTK_AVAILABLE,
            "spacy_available": SPACY_AVAILABLE,
            "models_loaded": {
                "sentiment_pipeline": self.sentiment_pipeline is not None,
                "summarizer_pipeline": self.summarizer_pipeline is not None,
                "financial_sentiment_pipeline": self.financial_sentiment_pipeline is not None,
                "spacy_model": self.nlp is not None,
            },
            "timestamp": datetime.now().isoformat(),
        }


# Global instance
_nlp_insights = None


def get_nlp_insights() -> NaturalLanguageInsights:
    """Get global NLP insights instance.

    Returns:
        NaturalLanguageInsights instance
    """
    global _nlp_insights
    if _nlp_insights is None:
        _nlp_insights = NaturalLanguageInsights()
    return _nlp_insights
