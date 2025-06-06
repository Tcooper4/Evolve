import logging
from typing import Dict, List, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification
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
            # Sentiment analysis
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased-finetuned-sst-2-english'
            ).to(self.device)
            
            # Text summarization
            self.summarizer_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                'facebook/bart-large-cnn'
            ).to(self.device)
            
            # Named entity recognition
            self.ner_tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
            self.ner_model = AutoModelForTokenClassification.from_pretrained(
                'dbmdz/bert-large-cased-finetuned-conll03-english'
            ).to(self.device)
            
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
            inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            sentiment = "POSITIVE" if predictions[0][1] > predictions[0][0] else "NEGATIVE"
            score = float(predictions[0][1] if sentiment == "POSITIVE" else predictions[0][0])
            
            result = {
                "label": sentiment,
                "score": score
            }
            
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
            inputs = self.summarizer_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                summary_ids = self.summarizer_model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=10,
                    num_beams=4,
                    no_repeat_ngram_size=2
                )
                
            summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
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
            inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.ner_model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entities = []
            current_entity = {"word": "", "entity": "", "score": 0.0}
            
            for token, pred in zip(tokens, predictions[0]):
                if token.startswith("##"):
                    current_entity["word"] += token[2:]
                else:
                    if current_entity["word"]:
                        entities.append(current_entity)
                    current_entity = {
                        "word": token,
                        "entity": self.ner_model.config.id2label[pred.item()],
                        "score": float(torch.nn.functional.softmax(outputs.logits[0][0], dim=-1)[pred].item())
                    }
            
            if current_entity["word"]:
                entities.append(current_entity)
                
            self.logger.info(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            self.logger.error(f"Error in entity extraction: {str(e)}")
            return [] 