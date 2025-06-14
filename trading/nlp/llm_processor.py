import logging
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    pipeline,
    Pipeline
)

class LLMProcessor:
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the LLM processor with PyTorch-based models.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize models with PyTorch backend
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Initialize pipelines with PyTorch backend and framework="pt"
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            self.summarizer_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                device=0 if torch.cuda.is_available() else -1,
                framework="pt"
            )
            
            self.logger.info(f"LLM Processor initialized using device: {self.device}")
        except Exception as e:
            self.logger.error(f"Error initializing LLM Processor: {str(e)}")
            raise

    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze the sentiment of the given text using PyTorch model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        try:
            result = self.sentiment_pipeline(text)[0]
            self.logger.info(f"Sentiment analysis result: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "ERROR", "score": 0.0}

    def summarize_text(self, text: str, max_length: int = 50) -> str:
        """Summarize the given text using PyTorch model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Summarized text
        """
        try:
            summary = self.summarizer_pipeline(
                text,
                max_length=max_length,
                min_length=10,
                do_sample=False
            )[0]['summary_text']
            
            self.logger.info(f"Text summarized: {summary}")
            return summary
        except Exception as e:
            self.logger.error(f"Error in text summarization: {str(e)}")
            return "Error generating summary"

    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from the given text using PyTorch model.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of dictionaries containing entity information
        """
        try:
            entities = self.ner_pipeline(text)
            self.logger.info(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            return [] 