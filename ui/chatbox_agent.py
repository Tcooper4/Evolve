"""Voice & Chat-Driven Interface for Evolve Trading Platform.

This module provides voice input support and natural language processing
for trading commands and interactions.
"""

import speech_recognition as sr
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
import threading
from queue import Queue
import re
import os
from pathlib import Path

# OpenAI imports for Whisper
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenAI not available: {e}")
    OPENAI_AVAILABLE = False

# Text-to-speech imports
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"pyttsx3 not available: {e}")
    TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Represents a chat message."""
    id: str
    content: str
    sender: str  # "user" or "assistant"
    timestamp: datetime
    message_type: str  # "text", "voice", "command"
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TradingCommand:
    """Represents a parsed trading command."""
    action: str  # "buy", "sell", "analyze", "backtest", "explain"
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    timeframe: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.0
    raw_text: str = ""

@dataclass
class VoiceInput:
    """Represents voice input data."""
    audio_data: bytes
    duration: float
    sample_rate: int
    language: str = "en-US"
    timestamp: datetime = None

class SpeechRecognizer:
    """Handles speech recognition and transcription."""
    
    def __init__(self, 
                 language: str = "en-US",
                 use_whisper: bool = True,
                 whisper_api_key: Optional[str] = None):
        """Initialize speech recognizer.
        
        Args:
            language: Language for recognition
            use_whisper: Use OpenAI Whisper for transcription
            whisper_api_key: OpenAI API key for Whisper
        """
        self.language = language
        self.use_whisper = use_whisper and OPENAI_AVAILABLE
        self.whisper_api_key = whisper_api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        
        # OpenAI client for Whisper
        if self.use_whisper and self.whisper_api_key:
            openai.api_key = self.whisper_api_key
        
        logger.info(f"Initialized Speech Recognizer (Whisper: {self.use_whisper})")
    
    def listen_for_speech(self, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[VoiceInput]:
        """Listen for speech input.
        
        Args:
            timeout: Timeout in seconds
            phrase_time_limit: Maximum phrase length
            
        Returns:
            Voice input data or None
        """
        try:
            with sr.Microphone() as source:
                logger.info("Listening for speech...")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
                
                # Create voice input
                voice_input = VoiceInput(
                    audio_data=audio.get_wav_data(),
                    duration=len(audio.frame_data) / audio.sample_rate,
                    sample_rate=audio.sample_rate,
                    language=self.language,
                    timestamp=datetime.now()
                )
                
                logger.info(f"Captured {voice_input.duration:.2f}s of audio")
                return voice_input
                
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            logger.warning("Speech was unintelligible")
            return None
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            return None
    
    def transcribe_audio(self, voice_input: VoiceInput) -> Optional[str]:
        """Transcribe audio to text.
        
        Args:
            voice_input: Voice input data
            
        Returns:
            Transcribed text or None
        """
        try:
            if self.use_whisper and self.whisper_api_key:
                return self._transcribe_with_whisper(voice_input)
            else:
                return self._transcribe_with_sphinx(voice_input)
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def _transcribe_with_whisper(self, voice_input: VoiceInput) -> Optional[str]:
        """Transcribe using OpenAI Whisper."""
        try:
            # Save audio to temporary file
            temp_file = Path("temp_audio.wav")
            with open(temp_file, "wb") as f:
                f.write(voice_input.audio_data)
            
            # Transcribe with Whisper
            with open(temp_file, "rb") as f:
                transcript = openai.Audio.transcribe(
                    "whisper-1",
                    f,
                    language=self.language
                )
            
            # Clean up
            temp_file.unlink()
            
            return transcript["text"]
            
        except Exception as e:
            logger.error(f"Error with Whisper transcription: {e}")
            return None
    
    def _transcribe_with_sphinx(self, voice_input: VoiceInput) -> Optional[str]:
        """Transcribe using Sphinx (offline)."""
        try:
            # Convert audio data to AudioData object
            audio = sr.AudioData(
                voice_input.audio_data,
                voice_input.sample_rate,
                2  # Sample width
            )
            
            # Transcribe
            text = self.recognizer.recognize_sphinx(audio, language=self.language)
            return text
            
        except Exception as e:
            logger.error(f"Error with Sphinx transcription: {e}")
            return None

class TextToSpeech:
    """Handles text-to-speech output."""
    
    def __init__(self, voice_rate: int = 150, voice_volume: float = 0.9):
        """Initialize text-to-speech.
        
        Args:
            voice_rate: Speech rate
            voice_volume: Voice volume
        """
        if not TTS_AVAILABLE:
            logger.warning("Text-to-speech not available")
            return
        
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', voice_rate)
        self.engine.setProperty('volume', voice_volume)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        
        logger.info("Initialized Text-to-Speech")
    
    def speak(self, text: str):
        """Convert text to speech.
        
        Args:
            text: Text to speak
        """
        if not TTS_AVAILABLE:
            logger.warning("Text-to-speech not available")
            return
        
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            logger.info(f"Spoke: {text}")
            return
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return

class CommandParser:
    """Parses natural language into trading commands."""
    
    def __init__(self):
        """Initialize command parser."""
        self.action_patterns = {
            "buy": r"\b(buy|purchase|long|go long|take long)\b",
            "sell": r"\b(sell|short|go short|take short|exit)\b",
            "analyze": r"\b(analyze|analysis|examine|study|look at)\b",
            "backtest": r"\b(backtest|test|simulate|paper trade)\b",
            "explain": r"\b(explain|why|reason|justify|rationale)\b",
            "stop": r"\b(stop|stop loss|stop loss at)\b",
            "target": r"\b(target|take profit|profit target|exit at)\b"
        }
        
        self.symbol_patterns = [
            r"\b([A-Z]{1,5})\b",  # Stock symbols
            r"\b(BTC|ETH|ADA|DOT|LINK)\b",  # Crypto symbols
            r"\b(SPY|QQQ|IWM|GLD|SLV)\b"  # ETF symbols
        ]
        
        self.quantity_patterns = [
            r"\b(\d+(?:\.\d+)?)\s*(shares?|units?|contracts?)\b",
            r"\b(\d+(?:\.\d+)?)\s*%\b",  # Percentage
            r"\b(\d+(?:\.\d+)?)\s*of\s*portfolio\b"
        ]
        
        self.price_patterns = [
            r"\bat\s*\$?(\d+(?:\.\d+)?)\b",
            r"\bprice\s*\$?(\d+(?:\.\d+)?)\b",
            r"\bwhen\s*it\s*hits\s*\$?(\d+(?:\.\d+)?)\b"
        ]
        
        self.timeframe_patterns = [
            r"\b(next\s+week|this\s+week|tomorrow|today)\b",
            r"\b(\d+\s+(?:days?|weeks?|months?))\b",
            r"\b(short\s+term|medium\s+term|long\s+term)\b"
        ]
        
        self.strategy_patterns = [
            r"\b(breakout|momentum|mean\s+reversion|trend\s+following)\b",
            r"\b(bollinger|rsi|macd|moving\s+average)\b",
            r"\b(swing|day\s+trading|scalping|position\s+trading)\b"
        ]
        
        logger.info("Initialized Command Parser")
    
    def parse_command(self, text: str) -> TradingCommand:
        """Parse natural language text into trading command.
        
        Args:
            text: Natural language text
            
        Returns:
            Parsed trading command
        """
        text_lower = text.lower()
        command = TradingCommand(raw_text=text)
        
        # Parse action
        command.action = self._parse_action(text_lower)
        
        # Parse symbol
        command.symbol = self._parse_symbol(text)
        
        # Parse quantity
        command.quantity = self._parse_quantity(text_lower)
        
        # Parse price
        command.price = self._parse_price(text_lower)
        
        # Parse timeframe
        command.timeframe = self._parse_timeframe(text_lower)
        
        # Parse strategy
        command.strategy = self._parse_strategy(text_lower)
        
        # Parse stop loss and take profit
        command.stop_loss = self._parse_stop_loss(text_lower)
        command.take_profit = self._parse_take_profit(text_lower)
        
        # Calculate confidence
        command.confidence = self._calculate_confidence(command)
        
        return command
    
    def _parse_action(self, text: str) -> str:
        """Parse trading action."""
        for action, pattern in self.action_patterns.items():
            if re.search(pattern, text):
                return action
        return "analyze"  # Default action
    
    def _parse_symbol(self, text: str) -> Optional[str]:
        """Parse stock/crypto symbol."""
        for pattern in self.symbol_patterns:
            matches = re.findall(pattern, text.upper())
            if matches:
                # Return the first match that looks like a symbol
                for match in matches:
                    if len(match) >= 1 and len(match) <= 5:
                        return match
        return None
    
    def _parse_quantity(self, text: str) -> Optional[float]:
        """Parse quantity."""
        for pattern in self.quantity_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _parse_price(self, text: str) -> Optional[float]:
        """Parse price."""
        for pattern in self.price_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _parse_timeframe(self, text: str) -> Optional[str]:
        """Parse timeframe."""
        for pattern in self.timeframe_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _parse_strategy(self, text: str) -> Optional[str]:
        """Parse strategy."""
        for pattern in self.strategy_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None
    
    def _parse_stop_loss(self, text: str) -> Optional[float]:
        """Parse stop loss."""
        stop_patterns = [
            r"stop\s+loss\s*at\s*\$?(\d+(?:\.\d+)?)",
            r"stop\s*at\s*\$?(\d+(?:\.\d+)?)",
            r"risk\s*\$?(\d+(?:\.\d+)?)"
        ]
        
        for pattern in stop_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _parse_take_profit(self, text: str) -> Optional[float]:
        """Parse take profit."""
        profit_patterns = [
            r"target\s*\$?(\d+(?:\.\d+)?)",
            r"take\s+profit\s*at\s*\$?(\d+(?:\.\d+)?)",
            r"exit\s*at\s*\$?(\d+(?:\.\d+)?)"
        ]
        
        for pattern in profit_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _calculate_confidence(self, command: TradingCommand) -> float:
        """Calculate confidence in parsed command."""
        confidence = 0.0
        
        # Base confidence for having an action
        if command.action:
            confidence += 0.3
        
        # Symbol confidence
        if command.symbol:
            confidence += 0.2
        
        # Quantity confidence
        if command.quantity:
            confidence += 0.15
        
        # Price confidence
        if command.price:
            confidence += 0.15
        
        # Strategy confidence
        if command.strategy:
            confidence += 0.1
        
        # Timeframe confidence
        if command.timeframe:
            confidence += 0.1
        
        return min(confidence, 1.0)

class ChatboxAgent:
    """Main chatbox agent for voice and text interactions."""
    
    def __init__(self, 
                 enable_voice: bool = True,
                 enable_tts: bool = True,
                 whisper_api_key: Optional[str] = None):
        """Initialize chatbox agent.
        
        Args:
            enable_voice: Enable voice input
            enable_tts: Enable text-to-speech
            whisper_api_key: OpenAI API key for Whisper
        """
        self.enable_voice = enable_voice
        self.enable_tts = enable_tts
        
        # Initialize components
        self.speech_recognizer = SpeechRecognizer(whisper_api_key=whisper_api_key) if enable_voice else None
        self.tts = TextToSpeech() if enable_tts else None
        self.command_parser = CommandParser()
        
        # Chat state
        self.messages = []
        self.conversation_context = {}
        
        # Trading interface (to be set externally)
        self.trading_interface = None
        self.strategy_engine = None
        self.analysis_engine = None
        
        # Response templates
        self.response_templates = self._load_response_templates()
        
        logger.info("Initialized Chatbox Agent")
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates."""
        return {
            "order_confirmation": "I'll {action} {quantity} shares of {symbol} at ${price}. Would you like me to proceed?",
            "analysis_request": "I'll analyze {symbol} using {strategy} strategy. This may take a moment.",
            "backtest_request": "I'll backtest the {strategy} strategy on {symbol} for {timeframe}.",
            "explanation_request": "Let me explain the reasoning behind this trade recommendation.",
            "error": "I'm sorry, I couldn't understand that. Could you please repeat?",
            "success": "Great! The {action} order for {symbol} has been executed successfully.",
            "insufficient_funds": "I'm sorry, but there are insufficient funds for this trade.",
            "invalid_symbol": "I'm sorry, but {symbol} is not a valid trading symbol."
        }
    
    def add_message(self, content: str, sender: str = "user", message_type: str = "text") -> str:
        """Add a message to the chat.
        
        Args:
            content: Message content
            sender: Message sender
            message_type: Type of message
            
        Returns:
            Message ID
        """
        message_id = f"msg_{len(self.messages)}_{datetime.now().strftime('%H%M%S')}"
        
        message = ChatMessage(
            id=message_id,
            content=content,
            sender=sender,
            timestamp=datetime.now(),
            message_type=message_type
        )
        
        self.messages.append(message)
        return message_id
    
    def process_text_input(self, text: str) -> str:
        """Process text input and generate response.
        
        Args:
            text: Input text
            
        Returns:
            Response text
        """
        # Add user message
        self.add_message(text, "user", "text")
        
        try:
            # Parse command
            command = self.command_parser.parse_command(text)
            
            # Generate response based on command
            response = self._generate_response(command)
            
            # Add assistant response
            self.add_message(response, "assistant", "text")
            
            # Speak response if TTS is enabled
            if self.enable_tts and self.tts:
                self.tts.speak(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            error_response = self.response_templates["error"]
            self.add_message(error_response, "assistant", "text")
            return error_response
    
    def process_voice_input(self) -> Optional[str]:
        """Process voice input and generate response.
        
        Args:
            None (listens for voice input)
            
        Returns:
            Response text or None
        """
        if not self.enable_voice or not self.speech_recognizer:
            return "Voice input is not enabled"
        
        try:
            # Listen for speech
            voice_input = self.speech_recognizer.listen_for_speech()
            
            if not voice_input:
                return "No speech detected"
            
            # Transcribe audio
            text = self.speech_recognizer.transcribe_audio(voice_input)
            
            if not text:
                return "Could not transcribe speech"
            
            # Add voice message
            self.add_message(text, "user", "voice")
            
            # Process as text
            response = self.process_text_input(text)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing voice input: {e}")
            return "Error processing voice input"
    
    def _generate_response(self, command: TradingCommand) -> str:
        """Generate response based on parsed command.
        
        Args:
            command: Parsed trading command
            
        Returns:
            Response text
        """
        if command.confidence < 0.3:
            return self.response_templates["error"]
        
        if command.action == "buy" or command.action == "sell":
            return self._handle_trade_command(command)
        elif command.action == "analyze":
            return self._handle_analysis_command(command)
        elif command.action == "backtest":
            return self._handle_backtest_command(command)
        elif command.action == "explain":
            return self._handle_explanation_command(command)
        else:
            return "I understand you want to " + command.action + ". How can I help you with that?"
    
    def _handle_trade_command(self, command: TradingCommand) -> str:
        """Handle trade commands."""
        if not command.symbol:
            return "Please specify which symbol you'd like to trade."
        
        if not command.quantity:
            return f"Please specify how many shares of {command.symbol} you'd like to {command.action}."
        
        # Validate symbol
        if not self._is_valid_symbol(command.symbol):
            return self.response_templates["invalid_symbol"].format(symbol=command.symbol)
        
        # Check if trading interface is available
        if not self.trading_interface:
            return "Trading interface is not available at the moment."
        
        # Create order request
        order_request = self._create_order_request(command)
        
        try:
            # Place order
            order_status = self.trading_interface.place_order(order_request)
            
            if order_status.status == "filled":
                return self.response_templates["success"].format(
                    action=command.action,
                    symbol=command.symbol
                )
            elif order_status.status == "rejected":
                return "The order was rejected. Please check your account balance and try again."
            else:
                return f"Order placed with status: {order_status.status}"
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return "There was an error placing your order. Please try again."
    
    def _handle_analysis_command(self, command: TradingCommand) -> str:
        """Handle analysis commands."""
        if not command.symbol:
            return "Please specify which symbol you'd like me to analyze."
        
        if not self.analysis_engine:
            return "Analysis engine is not available at the moment."
        
        # Perform analysis
        try:
            analysis_result = self.analysis_engine.analyze_symbol(
                command.symbol,
                strategy=command.strategy,
                timeframe=command.timeframe
            )
            
            return f"Analysis of {command.symbol}: {analysis_result.get('summary', 'Analysis completed')}"
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return "There was an error performing the analysis. Please try again."
    
    def _handle_backtest_command(self, command: TradingCommand) -> str:
        """Handle backtest commands."""
        if not command.symbol:
            return "Please specify which symbol you'd like to backtest."
        
        if not command.strategy:
            return "Please specify which strategy you'd like to backtest."
        
        if not self.strategy_engine:
            return "Strategy engine is not available at the moment."
        
        # Perform backtest
        try:
            backtest_result = self.strategy_engine.backtest_strategy(
                command.strategy,
                command.symbol,
                timeframe=command.timeframe
            )
            
            return f"Backtest results for {command.strategy} on {command.symbol}: {backtest_result.get('summary', 'Backtest completed')}"
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return "There was an error performing the backtest. Please try again."
    
    def _handle_explanation_command(self, command: TradingCommand) -> str:
        """Handle explanation commands."""
        return self.response_templates["explanation_request"]
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid."""
        # Basic validation - in practice, check against available symbols
        valid_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC", "ETH", "SPY", "QQQ"]
        return symbol.upper() in valid_symbols
    
    def _create_order_request(self, command: TradingCommand):
        """Create order request from command."""
        from execution.live_trading_interface import OrderRequest
        
        return OrderRequest(
            symbol=command.symbol,
            side=command.action,
            quantity=command.quantity,
            order_type="market",  # Default to market order
            limit_price=command.price,
            strategy_id="chatbox_agent"
        )
    
    def get_conversation_history(self) -> List[ChatMessage]:
        """Get conversation history."""
        return self.messages.copy()
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.messages = []
        self.conversation_context = {}
        return {"status": "conversation_cleared"}
    
    def set_trading_interface(self, trading_interface):
        """Set trading interface."""
        self.trading_interface = trading_interface
        return {"status": "trading_interface_set"}
    
    def set_strategy_engine(self, strategy_engine):
        """Set strategy engine."""
        self.strategy_engine = strategy_engine
        return {"status": "strategy_engine_set"}
    
    def set_analysis_engine(self, analysis_engine):
        """Set analysis engine."""
        self.analysis_engine = analysis_engine
        return {"status": "analysis_engine_set"}

def create_chatbox_agent(enable_voice: bool = True,
                        enable_tts: bool = True,
                        whisper_api_key: Optional[str] = None) -> ChatboxAgent:
    """Create chatbox agent.
    
    Args:
        enable_voice: Enable voice input
        enable_tts: Enable text-to-speech
        whisper_api_key: OpenAI API key for Whisper
        
    Returns:
        Chatbox agent
    """
    try:
        agent = ChatboxAgent(
            enable_voice=enable_voice,
            enable_tts=enable_tts,
            whisper_api_key=whisper_api_key
        )
        return agent
    except Exception as e:
        logger.error(f"Error creating chatbox agent: {e}")
        return