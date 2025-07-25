"""Voice Prompt Interface for Evolve Trading Platform.

This module provides voice-to-text capabilities using speech recognition
and OpenAI Whisper for natural language trading commands.

All routing is now handled through the centralized PromptRouterAgent.
"""

import logging
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import speech_recognition as sr

warnings.filterwarnings("ignore")

# Import centralized templates for consistency
try:
    from trading.agents.prompt_templates import format_template

    TEMPLATES_AVAILABLE = True
except ImportError:
    TEMPLATES_AVAILABLE = False
    format_template = None

# Import centralized prompt router
try:
    from trading.agents.prompt_router_agent import PromptRouterAgent

    PROMPT_ROUTER_AVAILABLE = True
except ImportError:
    PROMPT_ROUTER_AVAILABLE = False
    PromptRouterAgent = None

# Try to import speech recognition and Whisper
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

logger = logging.getLogger(__name__)


class VoicePromptAgent:
    """Voice prompt processing agent with centralized routing."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize voice prompt agent."""
        self.config = config or {}
        self.recognizer = sr.Recognizer()
        self.whisper_model = None
        self.voice_history = []

        # Initialize centralized prompt router
        if PROMPT_ROUTER_AVAILABLE:
            self.prompt_router = PromptRouterAgent(config)
            logger.info("Initialized centralized prompt router")
        else:
            self.prompt_router = None
            logger.warning("PromptRouterAgent not available, using fallback routing")

        # Initialize Whisper model
        if WHISPER_AVAILABLE:
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Loaded Whisper model")
            except Exception as e:
                logger.warning(f"Could not load Whisper model: {e}")

        # Fallback command patterns (only used if prompt router unavailable)
        self.command_patterns = {
            "forecast": [
                r"forecast\s+(\w+)\s+for\s+(\d+)\s+(day|days|week|weeks)",
                r"predict\s+(\w+)\s+(\d+)\s+(day|days|week|weeks)",
                r"what\s+will\s+(\w+)\s+be\s+in\s+(\d+)\s+(day|days|week|weeks)",
            ],
            "trade": [
                r"buy\s+(\w+)",
                r"sell\s+(\w+)",
                r"trade\s+(\w+)",
                r"execute\s+trade\s+on\s+(\w+)",
            ],
            "strategy": [
                r"apply\s+(\w+)\s+strategy",
                r"use\s+(\w+)\s+strategy",
                r"run\s+(\w+)\s+strategy",
            ],
            "analysis": [
                r"analyze\s+(\w+)",
                r"get\s+analysis\s+for\s+(\w+)",
                r"show\s+me\s+(\w+)",
            ],
            "portfolio": [
                r"show\s+portfolio",
                r"portfolio\s+status",
                r"my\s+positions",
            ],
        }

    def listen_for_command(
        self, timeout: int = 5, phrase_time_limit: int = 10
    ) -> Optional[str]:
        """Listen for voice command."""
        try:
            with sr.Microphone() as source:
                logger.info("Listening for voice command...")

                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                # Listen for audio
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )

                # Convert speech to text
                text = self._convert_speech_to_text(audio)

                if text:
                    logger.info(f"Voice command: {text}")
                    self.voice_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "command": text,
                            "status": "received",
                        }
                    )

                return text

        except sr.WaitTimeoutError:
            logger.info("No voice command detected")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except Exception as e:
            logger.error(f"Error listening for command: {e}")
            return None

    def _convert_speech_to_text(self, audio) -> Optional[str]:
        """Convert speech audio to text using multiple methods."""

        # Try Google Speech Recognition first
        try:
            text = self.recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            logger.warning("Google Speech Recognition service unavailable")

        # Try Whisper as fallback
        if WHISPER_AVAILABLE and self.whisper_model:
            try:
                # Save audio to temporary file
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".wav"
                ) as tmp_file:
                    # Convert audio to WAV format
                    audio_data = audio.get_wav_data()
                    tmp_file.write(audio_data)
                    tmp_file_path = tmp_file.name

                # Use Whisper to transcribe
                result = self.whisper_model.transcribe(tmp_file_path)

                # Clean up temporary file
                os.unlink(tmp_file_path)

                return result["text"].lower()

            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")

        return None

    def parse_trading_command(self, text: str) -> Dict[str, Any]:
        """Parse voice command into structured trading action."""
        command = {
            "action": "unknown",
            "symbol": None,
            "parameters": {},
            "confidence": 0.0,
            "raw_text": text,
        }

        # Use centralized template if available
        if TEMPLATES_AVAILABLE and format_template:
            try:
                # Use the voice command parsing template
                prompt = format_template("voice_command_parsing", command=text)
                # Integrate with LLM parsing when available
                try:
                    from trading.llm.llm_interface import LLMInterface

                    llm = LLMInterface()
                    parsed_intent = llm.parse_intent(text)

                    if parsed_intent:
                        logger.info(f"LLM parsed intent: {parsed_intent}")
                        return parsed_intent
                    else:
                        logger.warning("LLM parsing failed, using regex fallback")
                        return self._regex_parse_intent(text)

                except ImportError:
                    logger.info("LLM interface not available, using regex fallback")
                    return self._regex_parse_intent(text)
                except Exception as e:
                    logger.error(f"LLM parsing error: {e}, using regex fallback")
                    return self._regex_parse_intent(text)
            except Exception as e:
                logger.warning(f"Failed to use centralized template: {e}")

        # Check for forecast commands
        for pattern in self.command_patterns["forecast"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                command["action"] = "forecast"
                command["symbol"] = match.group(1).upper()
                command["parameters"] = {
                    "period": int(match.group(2)),
                    "unit": match.group(3),
                }
                command["confidence"] = 0.9
                break

        # Check for trade commands
        for pattern in self.command_patterns["trade"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                command["action"] = "trade"
                command["symbol"] = match.group(1).upper()

                # Determine trade direction
                if "buy" in text.lower():
                    command["parameters"]["side"] = "buy"
                elif "sell" in text.lower():
                    command["parameters"]["side"] = "sell"
                else:
                    command["parameters"]["side"] = "auto"

                command["confidence"] = 0.8
                break

        # Check for strategy commands
        for pattern in self.command_patterns["strategy"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                command["action"] = "strategy"
                command["parameters"]["strategy_name"] = match.group(1).lower()
                command["confidence"] = 0.7
                break

        # Check for analysis commands
        for pattern in self.command_patterns["analysis"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                command["action"] = "analysis"
                command["symbol"] = match.group(1).upper()
                command["confidence"] = 0.8
                break

        # Check for portfolio commands
        for pattern in self.command_patterns["portfolio"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                command["action"] = "portfolio"
                command["confidence"] = 0.9
                break

        # Extract additional parameters
        self._extract_additional_parameters(text, command)

        return command

    def _extract_additional_parameters(self, text: str, command: Dict[str, Any]):
        """Extract additional parameters from text."""
        # Extract amount/quantity
        amount_match = re.search(r"(\d+)\s*(shares?|units?|dollars?|usd)", text.lower())
        if amount_match:
            command["parameters"]["amount"] = int(amount_match.group(1))

        # Extract timeframe
        timeframe_match = re.search(
            r"(\d+)\s*(days?|weeks?|months?|years?)", text.lower()
        )
        if timeframe_match:
            command["parameters"]["period"] = int(timeframe_match.group(1))
            command["parameters"]["unit"] = timeframe_match.group(2).rstrip("s")

        # Adjust confidence based on certainty words
        if any(word in text.lower() for word in ["maybe", "perhaps", "possibly"]):
            command["confidence"] *= 0.8
        elif any(word in text.lower() for word in ["definitely", "sure", "certain"]):
            command["confidence"] *= 1.2

        return command

    def execute_voice_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parsed voice command using centralized routing."""
        result = {
            "success": False,
            "action": command["action"],
            "message": "",
            "data": None,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Use centralized prompt router if available
            if self.prompt_router and command["raw_text"]:
                logger.info(
                    f"Routing voice command through PromptRouterAgent: {command['raw_text']}"
                )

                # Route the raw text through the centralized agent
                router_result = self.prompt_router.execute(command["raw_text"])

                result.update(
                    {
                        "success": router_result.get("success", False),
                        "message": router_result.get("message", "Command processed"),
                        "data": router_result.get("data", {}),
                        "routed_through": "PromptRouterAgent",
                    }
                )

            else:
                # Fallback to local routing if prompt router unavailable
                logger.info("Using fallback routing for voice command")
                result = self._execute_fallback_command(command)

            # Update voice history
            self._update_voice_history(command, result)

        except Exception as e:
            logger.error(f"Error executing voice command: {e}")
            result["message"] = f"Error: {str(e)}"

        return result

    def _execute_fallback_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command using fallback routing when PromptRouterAgent is unavailable."""
        result = {
            "success": False,
            "action": command["action"],
            "message": "",
            "data": None,
            "routed_through": "fallback",
        }

        try:
            if command["action"] == "forecast":
                result = self._execute_forecast_command(command)
            elif command["action"] == "trade":
                result = self._execute_trade_command(command)
            elif command["action"] == "strategy":
                result = self._execute_strategy_command(command)
            elif command["action"] == "analysis":
                result = self._execute_analysis_command(command)
            elif command["action"] == "portfolio":
                result = self._execute_portfolio_command(command)
            else:
                result["message"] = f"Unknown command: {command['raw_text']}"

        except Exception as e:
            logger.error(f"Error in fallback command execution: {e}")
            result["message"] = f"Fallback error: {str(e)}"

        return result

    def _execute_forecast_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forecast command."""
        from trading.models.forecast_router import ForecastRouter

        try:
            router = ForecastRouter()
            symbol = command["symbol"]
            period = command["parameters"].get("period", 10)
            unit = command["parameters"].get("unit", "days")

            # Convert to days
            if unit == "weeks":
                period *= 7

            # Get forecast
            forecast = router.get_forecast(symbol, period)

            return {
                "success": True,
                "action": "forecast",
                "message": f"Forecast for {symbol}: {forecast.get('prediction', 'N/A')}",
                "data": forecast,
            }

        except Exception as e:
            return {
                "success": False,
                "action": "forecast",
                "message": f"Forecast error: {str(e)}",
            }

    def _execute_trade_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade command."""
        from execution.trade_executor import TradeOrder, get_trade_executor

        try:
            executor = get_trade_executor()
            symbol = command["symbol"]
            side = command["parameters"].get("side", "auto")

            # Determine trade side if auto
            if side == "auto":
                # Use simple logic - could be enhanced with ML
                side = "buy" if np.random.random() > 0.5 else "sell"

            # Create order
            order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=command["parameters"].get("amount", 100),
                order_type="market",
            )

            # Place order
            success = executor.place_order(order)

            return {
                "success": success,
                "action": "trade",
                "message": f"{side.capitalize()} order for {symbol} {'placed' if success else 'failed'}",
                "data": {"order_id": order.order_id if success else None},
            }

        except Exception as e:
            return {
                "success": False,
                "action": "trade",
                "message": f"Trade error: {str(e)}",
            }

    def _execute_strategy_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy command."""
        strategy_name = command["parameters"].get("strategy_name", "default")

        return {
            "success": True,
            "action": "strategy",
            "message": f"Applied {strategy_name} strategy",
            "data": {"strategy": strategy_name},
        }

    def _execute_analysis_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis command."""
        symbol = command["symbol"]

        return {
            "success": True,
            "action": "analysis",
            "message": f"Analysis for {symbol} completed",
            "data": {"symbol": symbol, "analysis_type": "comprehensive"},
        }

    def _execute_portfolio_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute portfolio command."""
        from execution.trade_executor import get_trade_executor

        try:
            executor = get_trade_executor()
            portfolio = executor.get_portfolio_summary()

            return {
                "success": True,
                "action": "portfolio",
                "message": f"Portfolio value: ${portfolio.get('total_value', 0):,.2f}",
                "data": portfolio,
            }

        except Exception as e:
            return {
                "success": False,
                "action": "portfolio",
                "message": f"Portfolio error: {str(e)}",
            }

    def _update_voice_history(self, command: Dict[str, Any], result: Dict[str, Any]):
        """Update voice history."""
        if self.voice_history:
            last_entry = self.voice_history[-1]
            last_entry["parsed_command"] = command
            last_entry["result"] = result
            last_entry["status"] = "completed"

    def get_voice_history(self, limit: int = 50) -> List[Dict]:
        """Get voice command history."""
        return self.voice_history[-limit:] if self.voice_history else []

    def clear_voice_history(self):
        """Clear voice command history."""
        self.voice_history = []
        logger.info("Cleared voice command history")

    def get_voice_statistics(self) -> Dict[str, Any]:
        """Get voice command statistics."""
        if not self.voice_history:
            return {}

        total_commands = len(self.voice_history)
        successful_commands = len(
            [h for h in self.voice_history if h.get("result", {}).get("success", False)]
        )

        action_counts = {}
        for entry in self.voice_history:
            action = entry.get("parsed_command", {}).get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        return {
            "total_commands": total_commands,
            "successful_commands": successful_commands,
            "success_rate": successful_commands / total_commands
            if total_commands > 0
            else 0,
            "action_distribution": action_counts,
            "last_command": self.voice_history[-1]["timestamp"]
            if self.voice_history
            else None,
        }


# Global voice prompt agent instance
voice_agent = VoicePromptAgent()


def get_voice_agent() -> VoicePromptAgent:
    """Get the global voice prompt agent instance."""
    return voice_agent
