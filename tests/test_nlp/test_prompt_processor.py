import pytest
from pathlib import Path
from trading.nlp.prompt_processor import PromptProcessor, ProcessedPrompt, EntityMatch

@pytest.fixture
def processor():
    """Create a PromptProcessor instance for testing."""
    config_dir = Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"
    return PromptProcessor(config_dir)

def test_process_prompt_forecast(processor):
    """Test processing a forecast-related prompt."""
    prompt = "What's the price forecast for BTC in the next 7 days?"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "forecast"
    assert result.confidence > 0.5
    
    # Check for required entities
    asset = processor.get_entity_by_type(result, "asset")
    timeframe = processor.get_entity_by_type(result, "timeframe")
    
    assert asset is not None
    assert asset.value == "BTC"
    assert timeframe is not None
    assert "7" in timeframe.value

def test_process_prompt_analysis(processor):
    """Test processing an analysis-related prompt."""
    prompt = "Analyze the current market conditions for ETH"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "analyze"
    assert result.confidence > 0.5
    
    # Check for required entities
    asset = processor.get_entity_by_type(result, "asset")
    assert asset is not None
    assert asset.value == "ETH"

def test_process_prompt_recommendation(processor):
    """Test processing a recommendation-related prompt."""
    prompt = "Should I buy SOL now?"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "recommend"
    assert result.confidence > 0.5
    
    # Check for required entities
    asset = processor.get_entity_by_type(result, "asset")
    action = processor.get_entity_by_type(result, "action")
    
    assert asset is not None
    assert asset.value == "SOL"
    assert action is not None
    assert action.value == "buy"

def test_process_prompt_explanation(processor):
    """Test processing an explanation-related prompt."""
    prompt = "Explain the recent price movement of BTC"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "explain"
    assert result.confidence > 0.5
    
    # Check for required entities
    asset = processor.get_entity_by_type(result, "asset")
    assert asset is not None
    assert asset.value == "BTC"

def test_process_prompt_comparison(processor):
    """Test processing a comparison-related prompt."""
    prompt = "Compare the performance of BTC and ETH"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "compare"
    assert result.confidence > 0.5
    
    # Check for required entities
    assets = processor.get_entity_values(result, "asset")
    assert len(assets) >= 2
    assert "BTC" in assets
    assert "ETH" in assets

def test_process_prompt_optimization(processor):
    """Test processing an optimization-related prompt."""
    prompt = "Optimize the MACD strategy for BTC"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "optimize"
    assert result.confidence > 0.5
    
    # Check for required entities
    strategy = processor.get_entity_by_type(result, "strategy")
    asset = processor.get_entity_by_type(result, "asset")
    
    assert strategy is not None
    assert strategy.value == "MACD"
    assert asset is not None
    assert asset.value == "BTC"

def test_process_prompt_validation(processor):
    """Test processing a validation-related prompt."""
    prompt = "Validate the performance of the RSI strategy"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "validate"
    assert result.confidence > 0.5
    
    # Check for required entities
    strategy = processor.get_entity_by_type(result, "strategy")
    assert strategy is not None
    assert strategy.value == "RSI"

def test_process_prompt_monitoring(processor):
    """Test processing a monitoring-related prompt."""
    prompt = "Monitor the portfolio performance"
    result = processor.process_prompt(prompt)
    
    assert isinstance(result, ProcessedPrompt)
    assert result.intent == "monitor"
    assert result.confidence > 0.5

def test_validate_prompt(processor):
    """Test prompt validation."""
    # Test valid forecast prompt
    forecast_prompt = ProcessedPrompt(
        text="What's the price forecast for BTC in the next 7 days?",
        intent="forecast",
        confidence=0.8,
        entities=[
            EntityMatch(type="asset", value="BTC", start=28, end=31),
            EntityMatch(type="timeframe", value="7 days", start=35, end=42)
        ]
    )
    is_valid, missing = processor.validate_prompt(forecast_prompt)
    assert is_valid
    assert not missing
    
    # Test invalid forecast prompt (missing timeframe)
    invalid_prompt = ProcessedPrompt(
        text="What's the price forecast for BTC?",
        intent="forecast",
        confidence=0.8,
        entities=[
            EntityMatch(type="asset", value="BTC", start=28, end=31)
        ]
    )
    is_valid, missing = processor.validate_prompt(invalid_prompt)
    assert not is_valid
    assert "timeframe" in missing

def test_get_entity_by_type(processor):
    """Test getting entity by type."""
    prompt = ProcessedPrompt(
        text="What's the price forecast for BTC in the next 7 days?",
        intent="forecast",
        confidence=0.8,
        entities=[
            EntityMatch(type="asset", value="BTC", start=28, end=31),
            EntityMatch(type="timeframe", value="7 days", start=35, end=42)
        ]
    )
    
    asset = processor.get_entity_by_type(prompt, "asset")
    assert asset is not None
    assert asset.value == "BTC"
    
    timeframe = processor.get_entity_by_type(prompt, "timeframe")
    assert timeframe is not None
    assert timeframe.value == "7 days"
    
    # Test non-existent entity type
    non_existent = processor.get_entity_by_type(prompt, "non_existent")
    assert non_existent is None

def test_get_entity_values(processor):
    """Test getting entity values."""
    prompt = ProcessedPrompt(
        text="Compare BTC and ETH performance",
        intent="compare",
        confidence=0.8,
        entities=[
            EntityMatch(type="asset", value="BTC", start=8, end=11),
            EntityMatch(type="asset", value="ETH", start=16, end=19)
        ]
    )
    
    assets = processor.get_entity_values(prompt, "asset")
    assert len(assets) == 2
    assert "BTC" in assets
    assert "ETH" in assets
    
    # Test non-existent entity type
    non_existent = processor.get_entity_values(prompt, "non_existent")
    assert len(non_existent) == 0

def test_has_entity_type(processor):
    """Test checking for entity type presence."""
    prompt = ProcessedPrompt(
        text="What's the price forecast for BTC in the next 7 days?",
        intent="forecast",
        confidence=0.8,
        entities=[
            EntityMatch(type="asset", value="BTC", start=28, end=31),
            EntityMatch(type="timeframe", value="7 days", start=35, end=42)
        ]
    )
    
    assert processor.has_entity_type(prompt, "asset")
    assert processor.has_entity_type(prompt, "timeframe")
    assert not processor.has_entity_type(prompt, "non_existent") 