import logging
from typing import Any, Dict, Callable

from ..llm.llm_interface import LLMInterface


class AgentRouter:
    """Route user prompts to the appropriate trading agent based on intent."""

    def __init__(
        self,
        llm: LLMInterface,
        forecaster: Any,
        strategy_manager: Any,
        backtester: Any,
        commentary_agent: Any | None = None,
    ) -> None:
        self.llm = llm
        self.forecaster = forecaster
        self.strategy_manager = strategy_manager
        self.backtester = backtester
        self.commentary_agent = commentary_agent or llm
        self.logger = logging.getLogger(self.__class__.__name__)

        self.intent_map: Dict[str, Callable[[str, Dict], Any]] = {
            "forecast": self._handle_forecast,
            "recommend": self._handle_strategy,
            "strategy": self._handle_strategy,
            "backtest": self._handle_backtest,
            "analyze": self._handle_commentary,
            "explain": self._handle_commentary,
        }

    def route(self, prompt: str, **kwargs) -> Any:
        """Route the prompt to the correct agent based on intent."""
        intent = self.llm.prompt_processor.extract_intent(prompt)
        handler = self.intent_map.get(intent, self._handle_commentary)
        self.logger.debug("Routing intent '%s'", intent)
        return handler(prompt, **kwargs)

    # Handlers -----------------------------------------------------------------
    def _handle_forecast(self, prompt: str, **kwargs) -> Any:
        data = kwargs.get("data")
        return self.forecaster.predict(data)

    def _handle_strategy(self, prompt: str, **kwargs) -> Any:
        data = kwargs.get("data")
        return self.strategy_manager.generate_signals(data)

    def _handle_backtest(self, prompt: str, **kwargs) -> Any:
        params = kwargs.get("params", {})
        return self.backtester.run_backtest(**params)

    def _handle_commentary(self, prompt: str, **kwargs) -> Any:
        return self.commentary_agent.process_prompt(prompt)
