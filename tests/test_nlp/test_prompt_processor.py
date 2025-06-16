import pytest
from pathlib import Path
import random
import string
from trading.nlp.prompt_processor import PromptProcessor, ProcessedPrompt, EntityMatch, PromptProcessingError

@pytest.fixture
def processor():
    """Create a PromptProcessor instance for testing."""
    config_dir = Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"
    return PromptProcessor(config_dir)

def generate_random_string(length=50):
    """Generate a random string of specified length."""
    return ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=length))

def test_fuzz_random_characters(processor):
    """Test processing prompts with random characters."""
    # Test with completely random strings
    for _ in range(10):
        random_prompt = generate_random_string()
        result = processor.process_prompt(random_prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0  # Should not crash
        assert result.confidence <= 1  # Should not crash
    
    # Test with random characters mixed with valid words
    valid_words = ["forecast", "BTC", "price", "analysis", "ETH"]
    for _ in range(10):
        mixed_prompt = ' '.join(random.choices(valid_words, k=3) + [generate_random_string(10)])
        result = processor.process_prompt(mixed_prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0
        assert result.confidence <= 1

def test_fuzz_partial_sentences(processor):
    """Test processing incomplete or partial sentences."""
    partial_prompts = [
        "forecast",  # Just intent
        "BTC",  # Just asset
        "next 7 days",  # Just timeframe
        "should I",  # Incomplete question
        "analyze the",  # Incomplete command
        "compare",  # Incomplete comparison
        "optimize",  # Incomplete optimization
        "validate",  # Incomplete validation
        "monitor",  # Incomplete monitoring
        "explain",  # Incomplete explanation
    ]
    
    for prompt in partial_prompts:
        result = processor.process_prompt(prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0
        assert result.confidence <= 1
        # Should not crash, but confidence should be low
        assert result.confidence < 0.5

def test_fuzz_contradictory_instructions(processor):
    """Test processing prompts with contradictory instructions."""
    contradictory_prompts = [
        "Buy and sell BTC now",
        "Forecast the past price of ETH",
        "Compare BTC with itself",
        "Optimize the strategy without parameters",
        "Validate the performance without data",
        "Monitor the portfolio without assets",
        "Explain the future price movement",
        "Analyze the market without indicators",
        "Recommend both buying and selling",
        "Forecast the price without a timeframe"
    ]
    
    for prompt in contradictory_prompts:
        result = processor.process_prompt(prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0
        assert result.confidence <= 1
        # Should handle contradictions gracefully
        assert result.confidence < 0.8

def test_fuzz_edge_cases(processor):
    """Test processing prompts with edge cases."""
    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "!@#$%^&*()",  # Special characters only
        "1234567890",  # Numbers only
        "a" * 1000,  # Very long string
        "forecast" * 100,  # Repeated words
        "BTC" + " " * 50 + "ETH",  # Large gaps
        "forecast" + "\n" * 10 + "BTC",  # Multiple newlines
        "forecast" + "\t" * 10 + "BTC",  # Multiple tabs
        "forecast" + " " * 10 + "BTC" + " " * 10 + "7 days"  # Multiple large gaps
    ]
    
    for prompt in edge_cases:
        result = processor.process_prompt(prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0
        assert result.confidence <= 1

def test_fuzz_mixed_languages(processor):
    """Test processing prompts with mixed languages and scripts."""
    mixed_prompts = [
        "forecast BTC 预测",  # English + Chinese
        "analyze ETH анализ",  # English + Russian
        "compare BTCとETH",  # English + Japanese
        "optimize strategy استراتيجية",  # English + Arabic
        "validate performance प्रदर्शन",  # English + Hindi
        "monitor portfolio 포트폴리오",  # English + Korean
        "explain price 가격",  # English + Korean
        "recommend buy 買う",  # English + Japanese
        "analyze market 시장",  # English + Korean
        "forecast price 价格"  # English + Chinese
    ]
    
    for prompt in mixed_prompts:
        result = processor.process_prompt(prompt)
        assert isinstance(result, ProcessedPrompt)
        assert result.confidence >= 0
        assert result.confidence <= 1

def test_fuzz_error_handling(processor):
    """Test error handling for malformed prompts."""
    # Test with None
    with pytest.raises(PromptProcessingError):
        processor.process_prompt(None)
    
    # Test with non-string input
    with pytest.raises(PromptProcessingError):
        processor.process_prompt(123)
    
    # Test with very large input
    with pytest.raises(PromptProcessingError):
        processor.process_prompt("a" * 10000)
    
    # Test with invalid UTF-8
    with pytest.raises(PromptProcessingError):
        processor.process_prompt(b"forecast BTC \xff".decode('utf-8', errors='ignore'))
    
    # Test with control characters
    with pytest.raises(PromptProcessingError):
        processor.process_prompt("forecast BTC \x00\x01\x02")

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