"""
Advanced Prompt Parser

This module provides intelligent prompt parsing using spaCy NLP and GPT-based classification
to extract structured action plans from natural language prompts.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


@dataclass
class ActionPlan:
    """
    Structured action plan extracted from natural language prompts.
    
    Attributes:
        model: Target model for the action
        strategy: Trading strategy to apply
        backtest_flag: Whether to run backtesting
        export_type: Type of export (csv, json, pdf, etc.)
        confidence: Confidence score of the classification
        raw_prompt: Original prompt text
        extracted_entities: Named entities found in the prompt
    """
    model: str
    strategy: str
    backtest_flag: bool
    export_type: str
    confidence: float
    raw_prompt: str
    extracted_entities: Dict[str, List[str]]

    def __post_init__(self):
        """Validate and normalize action plan attributes."""
        # Normalize model names
        model_mapping = {
            "lstm": "LSTM",
            "transformer": "Transformer",
            "xgboost": "XGBoost",
            "arima": "ARIMA",
            "prophet": "Prophet",
            "ensemble": "Ensemble",
            "hybrid": "Hybrid",
        }
        self.model = model_mapping.get(self.model.lower(), self.model)
        
        # Normalize strategy names
        strategy_mapping = {
            "bollinger": "BollingerBands",
            "macd": "MACD",
            "rsi": "RSI",
            "moving_average": "MovingAverage",
            "mean_reversion": "MeanReversion",
            "momentum": "Momentum",
            "breakout": "Breakout",
        }
        self.strategy = strategy_mapping.get(self.strategy.lower(), self.strategy)
        
        # Normalize export types
        export_mapping = {
            "csv": "csv",
            "json": "json",
            "pdf": "pdf",
            "excel": "xlsx",
            "html": "html",
            "report": "pdf",
        }
        self.export_type = export_mapping.get(self.export_type.lower(), self.export_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert action plan to dictionary."""
        return {
            "model": self.model,
            "strategy": self.strategy,
            "backtest_flag": self.backtest_flag,
            "export_type": self.export_type,
            "confidence": self.confidence,
            "raw_prompt": self.raw_prompt,
            "extracted_entities": self.extracted_entities,
        }

    def __str__(self) -> str:
        """String representation of action plan."""
        return (f"ActionPlan(model='{self.model}', strategy='{self.strategy}', "
                f"backtest={self.backtest_flag}, export='{self.export_type}', "
                f"confidence={self.confidence:.2f})")


class PromptParser:
    """
    Advanced prompt parser using spaCy NLP and pattern matching.
    
    Features:
    - Named entity recognition for models, strategies, and actions
    - Pattern matching for common trading scenarios
    - Confidence scoring for classifications
    - Support for complex multi-step instructions
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the prompt parser.

        Args:
            model_name: spaCy model to use for NLP processing
        """
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found, using basic tokenization")
            self.nlp = spacy.blank("en")
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        
        # Model and strategy keywords
        self.model_keywords = {
            "LSTM": ["lstm", "long short term memory", "neural network", "deep learning"],
            "Transformer": ["transformer", "attention", "bert", "gpt"],
            "XGBoost": ["xgboost", "gradient boosting", "tree", "ml"],
            "ARIMA": ["arima", "autoregressive", "time series", "statistical"],
            "Prophet": ["prophet", "facebook", "seasonal", "trend"],
            "Ensemble": ["ensemble", "combination", "multiple", "hybrid"],
            "Hybrid": ["hybrid", "combined", "mixed", "ensemble"],
        }
        
        self.strategy_keywords = {
            "BollingerBands": ["bollinger", "bands", "volatility", "std"],
            "MACD": ["macd", "moving average convergence", "momentum"],
            "RSI": ["rsi", "relative strength", "oscillator"],
            "MovingAverage": ["moving average", "ma", "sma", "ema"],
            "MeanReversion": ["mean reversion", "reversion", "regression"],
            "Momentum": ["momentum", "trend", "directional"],
            "Breakout": ["breakout", "break", "resistance", "support"],
        }
        
        self.action_keywords = {
            "backtest": ["backtest", "back testing", "historical", "test"],
            "export": ["export", "save", "download", "generate"],
            "train": ["train", "fit", "learn", "optimize"],
            "predict": ["predict", "forecast", "estimate", "project"],
        }

    def _setup_patterns(self) -> None:
        """Setup spaCy patterns for entity recognition."""
        # Model patterns
        model_patterns = [
            [{"LOWER": {"IN": ["lstm", "transformer", "xgboost", "arima", "prophet"]}}],
            [{"LOWER": "use"}, {"LOWER": {"IN": ["lstm", "transformer", "xgboost", "arima", "prophet"]}}],
            [{"LOWER": "with"}, {"LOWER": {"IN": ["lstm", "transformer", "xgboost", "arima", "prophet"]}}],
        ]
        
        # Strategy patterns
        strategy_patterns = [
            [{"LOWER": {"IN": ["bollinger", "macd", "rsi", "moving", "mean", "momentum", "breakout"]}}],
            [{"LOWER": "strategy"}, {"LOWER": {"IN": ["bollinger", "macd", "rsi", "moving", "mean", "momentum", "breakout"]}}],
        ]
        
        # Action patterns
        action_patterns = [
            [{"LOWER": {"IN": ["backtest", "test", "validate"]}}],
            [{"LOWER": {"IN": ["export", "save", "download"]}}],
            [{"LOWER": {"IN": ["train", "fit", "optimize"]}}],
        ]
        
        # Add patterns to matcher
        self.matcher.add("MODEL", model_patterns)
        self.matcher.add("STRATEGY", strategy_patterns)
        self.matcher.add("ACTION", action_patterns)

    def parse_prompt(self, prompt: str) -> ActionPlan:
        """
        Parse natural language prompt into structured ActionPlan.

        Args:
            prompt: Natural language prompt to parse

        Returns:
            ActionPlan object with extracted information
        """
        logger.info(f"Parsing prompt: {prompt[:100]}...")
        
        # Process with spaCy
        doc = self.nlp(prompt.lower())
        
        # Extract entities and patterns
        entities = self._extract_entities(doc)
        patterns = self._extract_patterns(doc)
        
        # Classify components
        model = self._classify_model(doc, entities, patterns)
        strategy = self._classify_strategy(doc, entities, patterns)
        backtest_flag = self._detect_backtest(doc, entities, patterns)
        export_type = self._detect_export_type(doc, entities, patterns)
        
        # Calculate confidence
        confidence = self._calculate_confidence(doc, entities, patterns)
        
        # Create action plan
        action_plan = ActionPlan(
            model=model,
            strategy=strategy,
            backtest_flag=backtest_flag,
            export_type=export_type,
            confidence=confidence,
            raw_prompt=prompt,
            extracted_entities=entities
        )
        
        logger.info(f"Parsed action plan: {action_plan}")
        return action_plan

    def _extract_entities(self, doc: Doc) -> Dict[str, List[str]]:
        """
        Extract named entities and custom entities from document.

        Args:
            doc: spaCy document

        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            "models": [],
            "strategies": [],
            "actions": [],
            "parameters": [],
            "timeframes": [],
        }
        
        # Extract spaCy named entities
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"]:
                entities["models"].append(ent.text)
            elif ent.label_ in ["QUANTITY", "PERCENT"]:
                entities["parameters"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                entities["timeframes"].append(ent.text)
        
        # Extract custom entities using matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            
            if label == "MODEL":
                entities["models"].append(span.text)
            elif label == "STRATEGY":
                entities["strategies"].append(span.text)
            elif label == "ACTION":
                entities["actions"].append(span.text)
        
        return entities

    def _extract_patterns(self, doc: Doc) -> Dict[str, List[str]]:
        """
        Extract patterns and keywords from document.

        Args:
            doc: spaCy document

        Returns:
            Dictionary of pattern types and their matches
        """
        patterns = {
            "model_keywords": [],
            "strategy_keywords": [],
            "action_keywords": [],
        }
        
        # Extract model keywords
        for token in doc:
            for model, keywords in self.model_keywords.items():
                if token.text.lower() in keywords:
                    patterns["model_keywords"].append(model)
        
        # Extract strategy keywords
        for token in doc:
            for strategy, keywords in self.strategy_keywords.items():
                if token.text.lower() in keywords:
                    patterns["strategy_keywords"].append(strategy)
        
        # Extract action keywords
        for token in doc:
            for action, keywords in self.action_keywords.items():
                if token.text.lower() in keywords:
                    patterns["action_keywords"].append(action)
        
        return patterns

    def _classify_model(self, doc: Doc, entities: Dict[str, List[str]], patterns: Dict[str, List[str]]) -> str:
        """
        Classify the target model from the prompt.

        Args:
            doc: spaCy document
            entities: Extracted entities
            patterns: Extracted patterns

        Returns:
            Classified model name
        """
        # Check entities first
        if entities["models"]:
            return entities["models"][0]
        
        # Check pattern matches
        if patterns["model_keywords"]:
            return patterns["model_keywords"][0]
        
        # Check for model mentions in text
        for token in doc:
            for model, keywords in self.model_keywords.items():
                if token.text.lower() in keywords:
                    return model
        
        # Default to ensemble if no specific model mentioned
        return "Ensemble"

    def _classify_strategy(self, doc: Doc, entities: Dict[str, List[str]], patterns: Dict[str, List[str]]) -> str:
        """
        Classify the trading strategy from the prompt.

        Args:
            doc: spaCy document
            entities: Extracted entities
            patterns: Extracted patterns

        Returns:
            Classified strategy name
        """
        # Check entities first
        if entities["strategies"]:
            return entities["strategies"][0]
        
        # Check pattern matches
        if patterns["strategy_keywords"]:
            return patterns["strategy_keywords"][0]
        
        # Check for strategy mentions in text
        for token in doc:
            for strategy, keywords in self.strategy_keywords.items():
                if token.text.lower() in keywords:
                    return strategy
        
        # Default to momentum if no specific strategy mentioned
        return "Momentum"

    def _detect_backtest(self, doc: Doc, entities: Dict[str, List[str]], patterns: Dict[str, List[str]]) -> bool:
        """
        Detect if backtesting is requested.

        Args:
            doc: spaCy document
            entities: Extracted entities
            patterns: Extracted patterns

        Returns:
            True if backtesting is requested
        """
        # Check for backtest keywords
        backtest_keywords = ["backtest", "back testing", "historical", "test", "validate"]
        
        # Check entities
        for action in entities["actions"]:
            if any(keyword in action.lower() for keyword in backtest_keywords):
                return True
        
        # Check patterns
        if "backtest" in patterns["action_keywords"]:
            return True
        
        # Check text
        for token in doc:
            if token.text.lower() in backtest_keywords:
                return True
        
        return False

    def _detect_export_type(self, doc: Doc, entities: Dict[str, List[str]], patterns: Dict[str, List[str]]) -> str:
        """
        Detect the export type from the prompt.

        Args:
            doc: spaCy document
            entities: Extracted entities
            patterns: Extracted patterns

        Returns:
            Export type (csv, json, pdf, etc.)
        """
        export_keywords = {
            "csv": ["csv", "excel", "spreadsheet"],
            "json": ["json", "api", "data"],
            "pdf": ["pdf", "report", "document"],
            "html": ["html", "web", "dashboard"],
        }
        
        # Check text for export keywords
        for token in doc:
            for export_type, keywords in export_keywords.items():
                if token.text.lower() in keywords:
                    return export_type
        
        # Default to csv
        return "csv"

    def _calculate_confidence(self, doc: Doc, entities: Dict[str, List[str]], patterns: Dict[str, List[str]]) -> float:
        """
        Calculate confidence score for the classification.

        Args:
            doc: spaCy document
            entities: Extracted entities
            patterns: Extracted patterns

        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Base confidence from entities
        if entities["models"]:
            confidence += 0.3
        if entities["strategies"]:
            confidence += 0.3
        if entities["actions"]:
            confidence += 0.2
        
        # Pattern confidence
        if patterns["model_keywords"]:
            confidence += 0.2
        if patterns["strategy_keywords"]:
            confidence += 0.2
        if patterns["action_keywords"]:
            confidence += 0.1
        
        # Text length confidence (longer prompts are usually more specific)
        if len(doc) > 20:
            confidence += 0.1
        
        return min(1.0, confidence)

    def parse_batch(self, prompts: List[str]) -> List[ActionPlan]:
        """
        Parse multiple prompts in batch.

        Args:
            prompts: List of prompts to parse

        Returns:
            List of ActionPlan objects
        """
        action_plans = []
        
        for prompt in prompts:
            try:
                action_plan = self.parse_prompt(prompt)
                action_plans.append(action_plan)
            except Exception as e:
                logger.error(f"Failed to parse prompt: {e}")
                # Create default action plan for failed parsing
                default_plan = ActionPlan(
                    model="Ensemble",
                    strategy="Momentum",
                    backtest_flag=False,
                    export_type="csv",
                    confidence=0.0,
                    raw_prompt=prompt,
                    extracted_entities={}
                )
                action_plans.append(default_plan)
        
        return action_plans

    def get_parsing_statistics(self, action_plans: List[ActionPlan]) -> Dict[str, Any]:
        """
        Get statistics about parsed action plans.

        Args:
            action_plans: List of parsed action plans

        Returns:
            Dictionary with parsing statistics
        """
        if not action_plans:
            return {}
        
        stats = {
            "total_plans": len(action_plans),
            "average_confidence": sum(ap.confidence for ap in action_plans) / len(action_plans),
            "model_distribution": {},
            "strategy_distribution": {},
            "backtest_requests": sum(1 for ap in action_plans if ap.backtest_flag),
            "export_distribution": {},
        }
        
        # Count distributions
        for plan in action_plans:
            stats["model_distribution"][plan.model] = stats["model_distribution"].get(plan.model, 0) + 1
            stats["strategy_distribution"][plan.strategy] = stats["strategy_distribution"].get(plan.strategy, 0) + 1
            stats["export_distribution"][plan.export_type] = stats["export_distribution"].get(plan.export_type, 0) + 1
        
        return stats


# Convenience functions for backward compatibility
def parse_prompt(prompt: str) -> ActionPlan:
    """
    Parse a single prompt using default parser.

    Args:
        prompt: Natural language prompt

    Returns:
        ActionPlan object
    """
    parser = PromptParser()
    return parser.parse_prompt(prompt)


def parse_prompts(prompts: List[str]) -> List[ActionPlan]:
    """
    Parse multiple prompts using default parser.

    Args:
        prompts: List of natural language prompts

    Returns:
        List of ActionPlan objects
    """
    parser = PromptParser()
    return parser.parse_batch(prompts)
