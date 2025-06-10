import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from trading.nlp.response_formatter import ResponseFormatter, ResponseData

@pytest.fixture
def formatter():
    """Create a ResponseFormatter instance for testing."""
    config_dir = Path(__file__).parent.parent.parent / "trading" / "nlp" / "config"
    return ResponseFormatter(config_dir)

@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    historical = pd.Series(np.random.randn(10), index=dates)
    forecast = pd.Series(np.random.randn(5), index=dates[-5:])
    confidence_intervals = {
        'lower': forecast - 0.1,
        'upper': forecast + 0.1
    }
    
    return ResponseData(
        content={
            "timeframe": "daily",
            "asset": "BTC",
            "prediction": forecast.mean(),
            "confidence": 85.5,
            "factors": {"factor1": 0.6, "factor2": 0.4},
            "historical_dates": historical.index,
            "historical_values": historical.values,
            "forecast_dates": forecast.index,
            "forecast_values": forecast.values,
            "confidence_intervals": confidence_intervals
        },
        type="forecast",
        confidence=0.855,
        metadata={
            "timeframe": "daily",
            "asset": "BTC",
            "model": "LSTM"
        }
    )

@pytest.fixture
def sample_analysis_data():
    """Create sample analysis data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(10),
        'high': np.random.randn(10),
        'low': np.random.randn(10),
        'close': np.random.randn(10),
        'volume': np.random.randint(1000, 2000, 10)
    }, index=dates)
    
    return ResponseData(
        content={
            "asset": "ETH",
            "timeframe": "daily",
            "state": "bullish",
            "indicators": {
                "RSI": 65.5,
                "MACD": 0.5,
                "BB": {"upper": 1.1, "middle": 1.0, "lower": 0.9}
            },
            "confidence": 75.0,
            "dates": data.index,
            "open": data['open'],
            "high": data['high'],
            "low": data['low'],
            "close": data['close'],
            "volume": data['volume']
        },
        type="analysis",
        confidence=0.75,
        metadata={
            "asset": "ETH",
            "timeframe": "daily",
            "indicators": ["RSI", "MACD", "BB"]
        }
    )

@pytest.fixture
def sample_recommendation_data():
    """Create sample recommendation data for testing."""
    return ResponseData(
        content={
            "asset": "SOL",
            "action": "buy",
            "entry": 100.0,
            "stop_loss": 95.0,
            "take_profit": 110.0,
            "confidence": 80.0,
            "rationale": "Strong technical indicators and positive market sentiment",
            "dates": pd.date_range(start='2024-01-01', periods=5, freq='D'),
            "prices": np.random.randn(5),
            "entry_date": pd.Timestamp('2024-01-05'),
            "entry_price": 100.0
        },
        type="recommendation",
        confidence=0.8,
        metadata={
            "asset": "SOL",
            "action": "buy",
            "strategy": "MACD"
        }
    )

def test_format_forecast(formatter, sample_forecast_data):
    """Test formatting forecast response."""
    formatted = formatter.format_response(sample_forecast_data)
    
    assert isinstance(formatted, str)
    assert "BTC" in formatted
    assert "daily" in formatted
    assert str(sample_forecast_data.content["prediction"]) in formatted
    assert str(sample_forecast_data.content["confidence"]) in formatted

def test_format_analysis(formatter, sample_analysis_data):
    """Test formatting analysis response."""
    formatted = formatter.format_response(sample_analysis_data)
    
    assert isinstance(formatted, str)
    assert "ETH" in formatted
    assert "bullish" in formatted
    assert "RSI" in formatted
    assert "MACD" in formatted
    assert str(sample_analysis_data.content["confidence"]) in formatted

def test_format_recommendation(formatter, sample_recommendation_data):
    """Test formatting recommendation response."""
    formatted = formatter.format_response(sample_recommendation_data)
    
    assert isinstance(formatted, str)
    assert "SOL" in formatted
    assert "buy" in formatted
    assert str(sample_recommendation_data.content["entry"]) in formatted
    assert str(sample_recommendation_data.content["stop_loss"]) in formatted
    assert str(sample_recommendation_data.content["take_profit"]) in formatted

def test_create_forecast_viz(formatter, sample_forecast_data):
    """Test creating forecast visualization."""
    viz = formatter.create_visualization(sample_forecast_data)
    
    assert viz is not None
    # Check if the visualization contains the required traces
    assert len(viz.data) >= 2  # Historical and forecast lines
    assert any("Historical" in trace.name for trace in viz.data)
    assert any("Forecast" in trace.name for trace in viz.data)

def test_create_analysis_viz(formatter, sample_analysis_data):
    """Test creating analysis visualization."""
    viz = formatter.create_visualization(sample_analysis_data)
    
    assert viz is not None
    # Check if the visualization contains candlestick chart
    assert any("Candlestick" in trace.name for trace in viz.data)
    # Check if indicators are plotted
    assert any("RSI" in trace.name for trace in viz.data)
    assert any("MACD" in trace.name for trace in viz.data)

def test_create_recommendation_viz(formatter, sample_recommendation_data):
    """Test creating recommendation visualization."""
    viz = formatter.create_visualization(sample_recommendation_data)
    
    assert viz is not None
    # Check if the visualization contains price line and entry/exit points
    assert any("Price" in trace.name for trace in viz.data)
    assert any("Entry" in trace.name for trace in viz.data)
    assert any("Stop Loss" in trace.name for trace in viz.data)
    assert any("Take Profit" in trace.name for trace in viz.data)

def test_load_templates(formatter):
    """Test loading response templates."""
    templates = formatter._load_templates()
    
    assert isinstance(templates, dict)
    assert "forecast" in templates
    assert "analysis" in templates
    assert "recommendation" in templates
    assert "explanation" in templates

def test_load_viz_settings(formatter):
    """Test loading visualization settings."""
    settings = formatter._load_viz_settings()
    
    assert isinstance(settings, dict)
    assert "line" in settings
    assert "candlestick" in settings
    assert "scatter" in settings
    assert "bar" in settings
    assert "heatmap" in settings
    assert "layout" in settings

def test_error_handling(formatter):
    """Test error handling in response formatting."""
    # Test with invalid response type
    invalid_data = ResponseData(
        content={"error": "Test error"},
        type="invalid_type",
        confidence=0.0,
        metadata={}
    )
    
    formatted = formatter.format_response(invalid_data)
    assert "Error" in formatted
    
    # Test with missing required fields
    incomplete_data = ResponseData(
        content={},
        type="forecast",
        confidence=0.0,
        metadata={}
    )
    
    formatted = formatter.format_response(incomplete_data)
    assert "Error" in formatted 