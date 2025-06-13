import pytest
from unittest.mock import Mock

from trading.agents.router import AgentRouter
from trading.llm.llm_interface import LLMInterface


@pytest.fixture
def router_components():
    llm = Mock(spec=LLMInterface)
    llm.prompt_processor.extract_intent.return_value = "forecast"
    llm.process_prompt.return_value = {"content": "ok"}

    forecaster = Mock()
    strategy = Mock()
    backtester = Mock()

    router = AgentRouter(llm, forecaster, strategy, backtester)
    return router, llm, forecaster, strategy, backtester


def test_forecast_routing(router_components):
    router, llm, forecaster, strategy, backtester = router_components
    llm.prompt_processor.extract_intent.return_value = "forecast"
    router.route("Forecast next week", data="df")
    forecaster.predict.assert_called_once_with("df")
    strategy.generate_signals.assert_not_called()
    backtester.run_backtest.assert_not_called()


def test_strategy_routing(router_components):
    router, llm, forecaster, strategy, backtester = router_components
    llm.prompt_processor.extract_intent.return_value = "recommend"
    router.route("Recommend a trade", data="df")
    strategy.generate_signals.assert_called_once_with("df")
    forecaster.predict.assert_not_called()
    backtester.run_backtest.assert_not_called()


def test_backtest_routing(router_components):
    router, llm, forecaster, strategy, backtester = router_components
    llm.prompt_processor.extract_intent.return_value = "backtest"
    router.route("backtest strategy", params={"strategy": "s", "params": {}})
    backtester.run_backtest.assert_called_once_with(strategy="s", params={})
    forecaster.predict.assert_not_called()
    strategy.generate_signals.assert_not_called()


def test_commentary_routing(router_components):
    router, llm, forecaster, strategy, backtester = router_components
    llm.prompt_processor.extract_intent.return_value = "explain"
    router.route("Explain RSI")
    llm.process_prompt.assert_called_once_with("Explain RSI")
    forecaster.predict.assert_not_called()
    strategy.generate_signals.assert_not_called()
    backtester.run_backtest.assert_not_called()

