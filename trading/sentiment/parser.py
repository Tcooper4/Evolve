"""
Sentiment Parser - Batch 20
Enhanced text cleaning with financial context preservation
"""

import logging
import re
import string
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class FinancialSymbol(Enum):
    """Types of financial symbols."""
    TICKER = "ticker"
    HASHTAG = "hashtag"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"
    PRICE = "price"

@dataclass
class ParsedText:
    """Parsed text with preserved financial context."""
    original_text: str
    cleaned_text: str
    preserved_symbols: List[Dict[str, Any]] = field(default_factory=list)
    financial_context: Dict[str, Any] = field(default_factory=dict)
    parse_time: datetime = field(default_factory=datetime.now)

@dataclass
class FinancialContext:
    """Extracted financial context from text."""
    tickers: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    percentages: List[float] = field(default_factory=list)
    currencies: List[str] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)

class SentimentParser:
    """
    Enhanced sentiment parser with financial context preservation.
    
    Features:
    - Preserves $, #, % symbols for financial context
    - Ticker symbol detection and preservation
    - Hashtag extraction for trending topics
    - Percentage and currency value preservation
    - Financial context extraction and analysis
    """
    
    def __init__(self, 
                 preserve_financial_symbols: bool = True,
                 enable_ticker_detection: bool = True,
                 enable_hashtag_extraction: bool = True):
        """
        Initialize sentiment parser.
        
        Args:
            preserve_financial_symbols: Preserve financial symbols in cleaned text
            enable_ticker_detection: Enable ticker symbol detection
            enable_hashtag_extraction: Enable hashtag extraction
        """
        self.preserve_financial_symbols = preserve_financial_symbols
        self.enable_ticker_detection = enable_ticker_detection
        self.enable_hashtag_extraction = enable_hashtag_extraction
        
        # Financial symbol patterns
        self.ticker_pattern = r'\$[A-Z]{1,5}'
        self.hashtag_pattern = r'#[A-Za-z0-9_]+'
        self.percentage_pattern = r'\d+(?:\.\d+)?%'
        self.currency_pattern = r'\$[\d,]+(?:\.\d{2})?'
        self.price_pattern = r'\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)'
        
        # Common financial hashtags
        self.financial_hashtags = {
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'bullish', 'bearish', 'rally', 'crash', 'volatility', 'trend',
            'dividend', 'buyback', 'merger', 'acquisition', 'ipo', 'filing',
            'guidance', 'forecast', 'analyst', 'rating', 'upgrade', 'downgrade'
        }
        
        # Parse statistics
        self.parse_stats = {
            'total_parses': 0,
            'tickers_found': 0,
            'hashtags_found': 0,
            'percentages_found': 0,
            'currencies_found': 0,
            'prices_found': 0
        }
        
        logger.info(f"SentimentParser initialized with financial symbol preservation: {preserve_financial_symbols}")
    
    def clean_text(self, text: str) -> ParsedText:
        """
        Clean text while preserving financial context.
        
        Args:
            text: Raw text to clean
            
        Returns:
            ParsedText with cleaned text and preserved context
        """
        if not text or not isinstance(text, str):
            return ParsedText(original_text=text, cleaned_text="")
        
        original_text = text
        preserved_symbols = []
        financial_context = FinancialContext()
        
        # Extract financial symbols before cleaning
        if self.preserve_financial_symbols:
            preserved_symbols, financial_context = self._extract_financial_symbols(text)
        
        # Clean text while preserving important symbols
        cleaned_text = self._clean_text_preserving_symbols(text, preserved_symbols)
        
        # Create parsed text object
        parsed_text = ParsedText(
            original_text=original_text,
            cleaned_text=cleaned_text,
            preserved_symbols=preserved_symbols,
            financial_context=financial_context.__dict__
        )
        
        # Update statistics
        self._update_parse_stats(financial_context)
        
        logger.debug(f"Cleaned text: {len(original_text)} -> {len(cleaned_text)} chars")
        return parsed_text
    
    def _extract_financial_symbols(self, text: str) -> Tuple[List[Dict[str, Any]], FinancialContext]:
        """
        Extract financial symbols from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (preserved_symbols, financial_context)
        """
        preserved_symbols = []
        financial_context = FinancialContext()
        
        # Extract ticker symbols
        if self.enable_ticker_detection:
            ticker_matches = re.finditer(self.ticker_pattern, text, re.IGNORECASE)
            for match in ticker_matches:
                ticker = match.group()
                symbol_info = {
                    'type': FinancialSymbol.TICKER.value,
                    'symbol': ticker,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'context': self._get_context_window(text, match.start(), match.end())
                }
                preserved_symbols.append(symbol_info)
                financial_context.tickers.append(ticker)
        
        # Extract hashtags
        if self.enable_hashtag_extraction:
            hashtag_matches = re.finditer(self.hashtag_pattern, text, re.IGNORECASE)
            for match in hashtag_matches:
                hashtag = match.group()
                symbol_info = {
                    'type': FinancialSymbol.HASHTAG.value,
                    'symbol': hashtag,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'context': self._get_context_window(text, match.start(), match.end()),
                    'is_financial': hashtag.lower() in self.financial_hashtags
                }
                preserved_symbols.append(symbol_info)
                financial_context.hashtags.append(hashtag)
        
        # Extract percentages
        percentage_matches = re.finditer(self.percentage_pattern, text)
        for match in percentage_matches:
            percentage_str = match.group()
            percentage_value = float(percentage_str.rstrip('%'))
            symbol_info = {
                'type': FinancialSymbol.PERCENTAGE.value,
                'symbol': percentage_str,
                'value': percentage_value,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'context': self._get_context_window(text, match.start(), match.end())
            }
            preserved_symbols.append(symbol_info)
            financial_context.percentages.append(percentage_value)
        
        # Extract currency amounts
        currency_matches = re.finditer(self.currency_pattern, text)
        for match in currency_matches:
            currency_str = match.group()
            currency_value = self._parse_currency(currency_str)
            symbol_info = {
                'type': FinancialSymbol.CURRENCY.value,
                'symbol': currency_str,
                'value': currency_value,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'context': self._get_context_window(text, match.start(), match.end())
            }
            preserved_symbols.append(symbol_info)
            financial_context.currencies.append(currency_str)
        
        # Extract prices with currency
        price_matches = re.finditer(self.price_pattern, text, re.IGNORECASE)
        for match in price_matches:
            price_str = match.group()
            price_value = self._parse_price(price_str)
            symbol_info = {
                'type': FinancialSymbol.PRICE.value,
                'symbol': price_str,
                'value': price_value,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'context': self._get_context_window(text, match.start(), match.end())
            }
            preserved_symbols.append(symbol_info)
            financial_context.prices.append(price_value)
        
        return preserved_symbols, financial_context
    
    def _clean_text_preserving_symbols(self, text: str, preserved_symbols: List[Dict[str, Any]]) -> str:
        """
        Clean text while preserving important financial symbols.
        
        Args:
            text: Original text
            preserved_symbols: List of symbols to preserve
            
        Returns:
            Cleaned text with preserved symbols
        """
        # Sort symbols by position (descending) to avoid index shifting
        sorted_symbols = sorted(preserved_symbols, key=lambda x: x['start_pos'], reverse=True)
        
        cleaned_text = text
        
        # Replace preserved symbols with placeholders
        symbol_placeholders = {}
        for i, symbol_info in enumerate(sorted_symbols):
            placeholder = f"__SYMBOL_{i}__"
            symbol_placeholders[placeholder] = symbol_info['symbol']
            
            start_pos = symbol_info['start_pos']
            end_pos = symbol_info['end_pos']
            cleaned_text = cleaned_text[:start_pos] + placeholder + cleaned_text[end_pos:]
        
        # Apply standard text cleaning
        cleaned_text = self._apply_standard_cleaning(cleaned_text)
        
        # Restore preserved symbols
        for placeholder, symbol in symbol_placeholders.items():
            cleaned_text = cleaned_text.replace(placeholder, symbol)
        
        return cleaned_text
    
    def _apply_standard_cleaning(self, text: str) -> str:
        """
        Apply standard text cleaning operations.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove user mentions (but keep @ for context)
        text = re.sub(r'@\w+', '@user', text)
        
        # Remove special characters except preserved ones
        # Keep: $, #, %, @, ., ,, !, ?, -, _, +, =, <, >, (, ), [, ]
        text = re.sub(r'[^\w\s$#%@.,!?\-_+=<>()\[\]]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _get_context_window(self, text: str, start_pos: int, end_pos: int, window_size: int = 20) -> str:
        """
        Get context window around a symbol.
        
        Args:
            text: Full text
            start_pos: Start position of symbol
            end_pos: End position of symbol
            window_size: Size of context window
            
        Returns:
            Context window text
        """
        context_start = max(0, start_pos - window_size)
        context_end = min(len(text), end_pos + window_size)
        return text[context_start:context_end]
    
    def _parse_currency(self, currency_str: str) -> float:
        """Parse currency string to float value."""
        try:
            # Remove $ and commas
            clean_str = currency_str.replace('$', '').replace(',', '')
            return float(clean_str)
        except ValueError:
            return 0.0
    
    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float value."""
        try:
            # Extract numeric part
            numeric_match = re.search(r'(\d+(?:\.\d{2})?)', price_str)
            if numeric_match:
                return float(numeric_match.group(1))
            return 0.0
        except ValueError:
            return 0.0
    
    def _update_parse_stats(self, financial_context: FinancialContext):
        """Update parsing statistics."""
        self.parse_stats['total_parses'] += 1
        self.parse_stats['tickers_found'] += len(financial_context.tickers)
        self.parse_stats['hashtags_found'] += len(financial_context.hashtags)
        self.parse_stats['percentages_found'] += len(financial_context.percentages)
        self.parse_stats['currencies_found'] += len(financial_context.currencies)
        self.parse_stats['prices_found'] += len(financial_context.prices)
    
    def analyze_financial_context(self, parsed_text: ParsedText) -> Dict[str, Any]:
        """
        Analyze financial context from parsed text.
        
        Args:
            parsed_text: Parsed text object
            
        Returns:
            Financial context analysis
        """
        context = parsed_text.financial_context
        analysis = {
            'has_tickers': len(context['tickers']) > 0,
            'has_financial_hashtags': any(
                hashtag.lower() in self.financial_hashtags 
                for hashtag in context['hashtags']
            ),
            'has_percentages': len(context['percentages']) > 0,
            'has_currencies': len(context['currencies']) > 0,
            'ticker_count': len(context['tickers']),
            'hashtag_count': len(context['hashtags']),
            'percentage_count': len(context['percentages']),
            'currency_count': len(context['currencies']),
            'price_count': len(context['prices']),
            'financial_relevance_score': 0.0
        }
        
        # Calculate financial relevance score
        score = 0.0
        if analysis['has_tickers']:
            score += 0.3
        if analysis['has_financial_hashtags']:
            score += 0.3
        if analysis['has_percentages']:
            score += 0.2
        if analysis['has_currencies']:
            score += 0.2
        
        analysis['financial_relevance_score'] = min(score, 1.0)
        
        return analysis
    
    def get_parse_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics."""
        stats = self.parse_stats.copy()
        
        if stats['total_parses'] > 0:
            stats['avg_tickers_per_parse'] = stats['tickers_found'] / stats['total_parses']
            stats['avg_hashtags_per_parse'] = stats['hashtags_found'] / stats['total_parses']
            stats['avg_percentages_per_parse'] = stats['percentages_found'] / stats['total_parses']
            stats['avg_currencies_per_parse'] = stats['currencies_found'] / stats['total_parses']
            stats['avg_prices_per_parse'] = stats['prices_found'] / stats['total_parses']
        
        return stats
    
    def enable_financial_symbol_preservation(self, enable: bool = True):
        """Enable or disable financial symbol preservation."""
        self.preserve_financial_symbols = enable
        logger.info(f"Financial symbol preservation {'enabled' if enable else 'disabled'}")

def create_sentiment_parser(preserve_financial_symbols: bool = True) -> SentimentParser:
    """Factory function to create sentiment parser."""
    return SentimentParser(preserve_financial_symbols=preserve_financial_symbols)
