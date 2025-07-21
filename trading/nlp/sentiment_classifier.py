"""
Fine-tuned Sentiment Classifier with Readability Scoring

This module provides a fine-tuned classifier for pre-processing sentiment
from financial news headlines with Flesch-Kincaid readability scoring
as a quality filter.
"""

import json
import logging
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from textstat import textstat

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ PyTorch not available. Disabling transformer models.")
    print(f"   Missing: {e}")
    torch = None
    nn = None
    Dataset = None
    DataLoader = None
    TORCH_AVAILABLE = False

# Try to import transformers
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ transformers not available. Disabling transformer models.")
    print(f"   Missing: {e}")
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    TrainingArguments = None
    Trainer = None
    TRANSFORMERS_AVAILABLE = False

# Try to import scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report

    SKLEARN_AVAILABLE = True
except ImportError as e:
    print("âš ï¸ scikit-learn not available. Disabling traditional ML models.")
    print(f"   Missing: {e}")
    TfidfVectorizer = None
    LogisticRegression = None
    classification_report = None
    accuracy_score = None
    SKLEARN_AVAILABLE = False

# Overall ML availability
ML_AVAILABLE = TORCH_AVAILABLE or SKLEARN_AVAILABLE

logger = logging.getLogger(__name__)


class SentimentLabel(Enum):
    """Sentiment labels for classification."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ReadabilityLevel(Enum):
    """Readability levels based on Flesch-Kincaid scores."""

    VERY_EASY = "very_easy"  # 90-100
    EASY = "easy"  # 80-89
    FAIRLY_EASY = "fairly_easy"  # 70-79
    STANDARD = "standard"  # 60-69
    FAIRLY_DIFFICULT = "fairly_difficult"  # 50-59
    DIFFICULT = "difficult"  # 30-49
    VERY_DIFFICULT = "very_difficult"  # 0-29


@dataclass
class ClassificationResult:
    """Result of sentiment classification."""

    text: str
    sentiment_label: SentimentLabel
    confidence: float
    readability_score: float
    readability_level: ReadabilityLevel
    quality_score: float
    passes_quality_filter: bool
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TrainingExample:
    """Training example for the classifier."""

    text: str
    label: SentimentLabel
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifierConfig:
    """Configuration for the sentiment classifier."""

    model_type: str = "tfidf_logistic"  # "tfidf_logistic", "transformer", "ensemble"
    min_readability_score: float = 30.0
    max_readability_score: float = 100.0
    quality_threshold: float = 0.6
    confidence_threshold: float = 0.7
    max_text_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    enable_fine_tuning: bool = True
    use_gpu: bool = False


class FinancialNewsDataset(Dataset):
    """Dataset for financial news sentiment classification."""

    def __init__(
        self, texts: List[str], labels: List[str], tokenizer=None, max_length: int = 512
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Cannot create FinancialNewsDataset."
            )
        """
        Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: Tokenizer for transformer models
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create label mapping
        self.label_map = {
            SentimentLabel.POSITIVE.value: 0,
            SentimentLabel.NEGATIVE.value: 1,
            SentimentLabel.NEUTRAL.value: 2,
            SentimentLabel.MIXED.value: 3,
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.tokenizer:
            # Tokenize for transformer models
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(self.label_map[label], dtype=torch.long),
            }
        else:
            # For non-transformer models
            return {"text": text, "labels": self.label_map[label]}


class ReadabilityAnalyzer:
    """Analyzes text readability using Flesch-Kincaid scoring."""

    def __init__(self, min_score: float = 30.0, max_score: float = 100.0):
        """
        Initialize readability analyzer.

        Args:
            min_score: Minimum acceptable readability score
            max_score: Maximum acceptable readability score
        """
        self.min_score = min_score
        self.max_score = max_score

    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid readability score.

        Args:
            text: Text to analyze

        Returns:
            Readability score (0-100, higher is more readable)
        """
        try:
            # Clean text for analysis
            clean_text = re.sub(r"[^\w\s\.]", "", text)
            if len(clean_text.split()) < 10:  # Too short for reliable scoring
                return 50.0

            score = textstat.flesch_reading_ease(clean_text)
            return max(0.0, min(100.0, score))
        except Exception as e:
            logger.warning(f"Error calculating readability score: {e}")
            return 50.0  # Default score

    def get_readability_level(self, score: float) -> ReadabilityLevel:
        """
        Get readability level from score.

        Args:
            score: Readability score

        Returns:
            Readability level
        """
        if score >= 90:
            return ReadabilityLevel.VERY_EASY
        elif score >= 80:
            return ReadabilityLevel.EASY
        elif score >= 70:
            return ReadabilityLevel.FAIRLY_EASY
        elif score >= 60:
            return ReadabilityLevel.STANDARD
        elif score >= 50:
            return ReadabilityLevel.FAIRLY_DIFFICULT
        elif score >= 30:
            return ReadabilityLevel.DIFFICULT
        else:
            return ReadabilityLevel.VERY_DIFFICULT

    def is_acceptable_readability(self, score: float) -> bool:
        """
        Check if readability score is acceptable.

        Args:
            score: Readability score

        Returns:
            True if score is acceptable
        """
        return self.min_score <= score <= self.max_score


class TfidfLogisticClassifier:
    """TF-IDF + Logistic Regression classifier."""

    def __init__(self, config: ClassifierConfig):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is not available. Cannot create TfidfLogisticClassifier."
            )
        """
        Initialize TF-IDF + Logistic Regression classifier.

        Args:
            config: Classifier configuration
        """
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        """
        Train the classifier.

        Args:
            texts: Training texts
            labels: Training labels
        """
        try:
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts)

            # Train classifier
            self.classifier.fit(X, labels)
            self.is_trained = True

            logger.info("TF-IDF + Logistic Regression classifier trained successfully")

        except Exception as e:
            logger.error(f"Error training TF-IDF classifier: {e}")
            raise

    def predict(self, text: str) -> Tuple[SentimentLabel, float]:
        """
        Predict sentiment for a text.

        Args:
            text: Text to classify

        Returns:
            Tuple of (sentiment_label, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        try:
            # Vectorize text
            X = self.vectorizer.transform([text])

            # Predict
            prediction = self.classifier.predict(X)[0]
            confidence = np.max(self.classifier.predict_proba(X))

            return SentimentLabel(prediction), confidence

        except Exception as e:
            logger.error(f"Error predicting with TF-IDF classifier: {e}")
            return SentimentLabel.NEUTRAL, 0.5


class TransformerClassifier:
    """Transformer-based classifier using pre-trained models."""

    def __init__(
        self, config: ClassifierConfig, model_name: str = "distilbert-base-uncased"
    ):
        """
        Initialize transformer classifier.

        Args:
            config: Classifier configuration
            model_name: Pre-trained model name
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not available. Cannot create TransformerClassifier."
            )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not available. Cannot create TransformerClassifier."
            )

        self.config = config
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_trained = False

        self._load_model()

    def _load_model(self):
        """Load pre-trained model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, num_labels=4  # positive, negative, neutral, mixed
            )

            if self.config.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()

            logger.info(f"Transformer model loaded: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise

    def train(self, texts: List[str], labels: List[str]):
        """
        Train the transformer classifier.

        Args:
            texts: Training texts
            labels: Training labels
        """
        try:
            # Create dataset
            dataset = FinancialNewsDataset(
                texts, labels, self.tokenizer, self.config.max_text_length
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir="./sentiment_classifier",
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                save_steps=500,
                save_total_limit=2,
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=500,
                load_best_model_at_end=True,
            )

            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
            )

            # Train
            trainer.train()
            self.is_trained = True

            logger.info("Transformer classifier trained successfully")

        except Exception as e:
            logger.error(f"Error training transformer classifier: {e}")
            raise

    def predict(self, text: str) -> Tuple[SentimentLabel, float]:
        """
        Predict sentiment for a text.

        Args:
            text: Text to classify

        Returns:
            Tuple of (sentiment_label, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained")

        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.max_text_length,
                return_tensors="pt",
            )

            if self.config.use_gpu and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()

            # Map prediction to sentiment label
            label_map = {
                0: SentimentLabel.POSITIVE,
                1: SentimentLabel.NEGATIVE,
                2: SentimentLabel.NEUTRAL,
                3: SentimentLabel.MIXED,
            }

            return label_map[prediction], confidence

        except Exception as e:
            logger.error(f"Error predicting with transformer classifier: {e}")
            return SentimentLabel.NEUTRAL, 0.5


class SentimentClassifier:
    """
    Fine-tuned sentiment classifier with readability scoring.
    """

    def __init__(
        self,
        config: Optional[ClassifierConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize sentiment classifier.

        Args:
            config: Classifier configuration
            model_path: Path to saved model
        """
        self.config = config or ClassifierConfig()
        self.readability_analyzer = ReadabilityAnalyzer(
            self.config.min_readability_score, self.config.max_readability_score
        )

        # Initialize classifier based on type
        if self.config.model_type == "tfidf_logistic":
            self.classifier = TfidfLogisticClassifier(self.config)
        elif self.config.model_type == "transformer":
            self.classifier = TransformerClassifier(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

        logger.info(
            f"SentimentClassifier initialized with {self.config.model_type} model"
        )

    def train(self, training_data: List[TrainingExample]):
        """
        Train the classifier.

        Args:
            training_data: List of training examples
        """
        try:
            texts = [example.text for example in training_data]
            labels = [example.label.value for example in training_data]

            # Train classifier
            self.classifier.train(texts, labels)

            logger.info(f"Classifier trained on {len(training_data)} examples")

        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            raise

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify sentiment of text with readability analysis.

        Args:
            text: Text to classify

        Returns:
            ClassificationResult with sentiment and readability analysis
        """
        try:
            # Calculate readability score
            readability_score = self.readability_analyzer.calculate_readability_score(
                text
            )
            readability_level = self.readability_analyzer.get_readability_level(
                readability_score
            )

            # Predict sentiment
            sentiment_label, confidence = self.classifier.predict(text)

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                confidence, readability_score, text
            )

            # Check if passes quality filter
            passes_quality_filter = (
                quality_score >= self.config.quality_threshold
                and confidence >= self.config.confidence_threshold
                and self.readability_analyzer.is_acceptable_readability(
                    readability_score
                )
            )

            # Extract features
            features = self._extract_features(text)

            return ClassificationResult(
                text=text,
                sentiment_label=sentiment_label,
                confidence=confidence,
                readability_score=readability_score,
                readability_level=readability_level,
                quality_score=quality_score,
                passes_quality_filter=passes_quality_filter,
                features=features,
                metadata={
                    "model_type": self.config.model_type,
                    "text_length": len(text),
                    "word_count": len(text.split()),
                },
            )

        except Exception as e:
            logger.error(f"Error classifying text: {e}")
            return ClassificationResult(
                text=text,
                sentiment_label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                readability_score=50.0,
                readability_level=ReadabilityLevel.STANDARD,
                quality_score=0.0,
                passes_quality_filter=False,
                metadata={"error": str(e)},
            )

    def _calculate_quality_score(
        self, confidence: float, readability_score: float, text: str
    ) -> float:
        """
        Calculate overall quality score.

        Args:
            confidence: Classification confidence
            readability_score: Readability score
            text: Input text

        Returns:
            Quality score (0.0 to 1.0)
        """
        # Normalize readability score to 0-1
        normalized_readability = readability_score / 100.0

        # Text length factor (prefer medium length texts)
        word_count = len(text.split())
        length_factor = 1.0
        if word_count < 5:
            length_factor = 0.5
        elif word_count > 100:
            length_factor = 0.8

        # Calculate weighted quality score
        quality_score = (
            confidence * 0.5 + normalized_readability * 0.3 + length_factor * 0.2
        )

        return min(1.0, max(0.0, quality_score))

    def _extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract features from text.

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        features = {}

        # Basic text features
        features["word_count"] = len(text.split())
        features["char_count"] = len(text)
        features["avg_word_length"] = (
            np.mean([len(word) for word in text.split()]) if text.split() else 0
        )

        # Sentiment indicators
        positive_words = [
            "bullish",
            "surge",
            "rally",
            "gain",
            "profit",
            "growth",
            "positive",
            "strong",
            "beat",
        ]
        negative_words = [
            "bearish",
            "plunge",
            "crash",
            "loss",
            "decline",
            "negative",
            "weak",
            "miss",
        ]

        text_lower = text.lower()
        features["positive_word_count"] = sum(
            1 for word in positive_words if word in text_lower
        )
        features["negative_word_count"] = sum(
            1 for word in negative_words if word in text_lower
        )

        # Punctuation features
        features["exclamation_count"] = text.count("!")
        features["question_count"] = text.count("?")
        features["capital_letter_ratio"] = (
            sum(1 for c in text if c.isupper()) / len(text) if text else 0
        )

        return features

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of classification results
        """
        results = []

        for text in texts:
            result = self.classify(text)
            results.append(result)

        return results

    def save_model(self, model_path: str):
        """
        Save the trained model.

        Args:
            model_path: Path to save model
        """
        try:
            model_dir = Path(model_path)
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save classifier
            if isinstance(self.classifier, TfidfLogisticClassifier):
                with open(model_dir / "vectorizer.pkl", "wb") as f:
                    pickle.dump(self.classifier.vectorizer, f)

                with open(model_dir / "classifier.pkl", "wb") as f:
                    pickle.dump(self.classifier.classifier, f)
            else:
                # Save transformer model
                self.classifier.model.save_pretrained(model_dir)
                self.classifier.tokenizer.save_pretrained(model_dir)

            # Save configuration
            with open(model_dir / "config.json", "w") as f:
                json.dump(self.config.__dict__, f, indent=2)

            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model.

        Args:
            model_path: Path to saved model
        """
        try:
            model_dir = Path(model_path)

            # Load configuration
            with open(model_dir / "config.json", "r") as f:
                config_dict = json.load(f)
                self.config = ClassifierConfig(**config_dict)

            # Load classifier
            if self.config.model_type == "tfidf_logistic":
                with open(model_dir / "vectorizer.pkl", "rb") as f:
                    self.classifier.vectorizer = pickle.load(f)

                with open(model_dir / "classifier.pkl", "rb") as f:
                    self.classifier.classifier = pickle.load(f)

                self.classifier.is_trained = True
            else:
                # Load transformer model
                self.classifier.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.classifier.model = (
                    AutoModelForSequenceClassification.from_pretrained(model_path)
                )
                self.classifier.is_trained = True

            logger.info(f"Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_classification_summary(
        self, results: List[ClassificationResult]
    ) -> Dict[str, Any]:
        """
        Get summary of classification results.

        Args:
            results: List of classification results

        Returns:
            Summary statistics
        """
        if not results:
            return {}

        # Filter results that pass quality filter
        quality_results = [r for r in results if r.passes_quality_filter]

        summary = {
            "total_classified": len(results),
            "quality_passed": len(quality_results),
            "quality_rate": len(quality_results) / len(results) if results else 0,
            "sentiment_distribution": {},
            "readability_distribution": {},
            "avg_confidence": np.mean([r.confidence for r in results]),
            "avg_readability": np.mean([r.readability_score for r in results]),
            "avg_quality_score": np.mean([r.quality_score for r in results]),
        }

        # Sentiment distribution
        for result in results:
            sentiment = result.sentiment_label.value
            summary["sentiment_distribution"][sentiment] = (
                summary["sentiment_distribution"].get(sentiment, 0) + 1
            )

        # Readability distribution
        for result in results:
            readability = result.readability_level.value
            summary["readability_distribution"][readability] = (
                summary["readability_distribution"].get(readability, 0) + 1
            )

        return summary


def create_sentiment_classifier(
    config: Optional[ClassifierConfig] = None, model_path: Optional[str] = None
) -> SentimentClassifier:
    """
    Create a sentiment classifier instance.

    Args:
        config: Classifier configuration
        model_path: Path to saved model

    Returns:
        SentimentClassifier instance
    """
    return SentimentClassifier(config, model_path)
