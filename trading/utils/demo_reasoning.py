"""
Reasoning Demo

Demonstrates the reasoning logger and display components with sample decisions.
"""

import time
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Configure logging
logger = logging.getLogger(__name__)

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from utils.reasoning_logger import ReasoningLogger, DecisionType, ConfidenceLevel
from utils.reasoning_display import ReasoningDisplay

def create_sample_forecast_decision():
    """Create a sample forecast decision."""
    return {
        'agent_name': 'LSTMForecaster',
        'decision_type': DecisionType.FORECAST,
        'action_taken': 'Predicted AAPL will reach $185.50 in 7 days',
        'context': {
            'symbol': 'AAPL',
            'timeframe': '1h',
            'market_conditions': {
                'trend': 'bullish',
                'volatility': 'medium',
                'volume': 'high',
                'rsi': 65,
                'macd': 'positive'
            },
            'available_data': ['price', 'volume', 'rsi', 'macd', 'bollinger_bands'],
            'constraints': {'max_forecast_days': 30},
            'user_preferences': {'risk_tolerance': 'medium'}
        },
        'reasoning': {
            'primary_reason': 'Strong technical indicators showing bullish momentum with RSI at 65 and MACD positive',
            'supporting_factors': [
                'RSI indicates bullish momentum (65)',
                'MACD shows positive crossover',
                'Volume is above average',
                'Price above 50-day moving average',
                'Bollinger Bands show upward expansion'
            ],
            'alternatives_considered': [
                'Conservative forecast of $180.00',
                'Aggressive forecast of $190.00',
                'Neutral forecast of $182.50'
            ],
            'risks_assessed': [
                'Market volatility could increase',
                'Earnings announcement next week',
                'Fed policy changes',
                'Technical resistance at $185.00'
            ],
            'confidence_explanation': 'High confidence due to strong technical signals and consistent model performance',
            'expected_outcome': 'AAPL expected to continue bullish trend with 70% probability of reaching target'
        },
        'confidence_level': ConfidenceLevel.HIGH,
        'metadata': {
            'model_name': 'LSTM_v2',
            'forecast_value': 185.50,
            'confidence_score': 0.85,
            'prediction_horizon': 7
        }
    }

def create_sample_strategy_decision():
    """Create a sample strategy decision."""
    return {
        'agent_name': 'RSIStrategy',
        'decision_type': DecisionType.STRATEGY,
        'action_taken': 'Executed BUY signal for AAPL with 100 shares at $182.30',
        'context': {
            'symbol': 'AAPL',
            'timeframe': '1h',
            'market_conditions': {
                'trend': 'bullish',
                'volatility': 'low',
                'volume': 'normal',
                'rsi': 35,
                'support_level': 180.00
            },
            'available_data': ['price', 'rsi', 'volume', 'support_resistance'],
            'constraints': {'max_position_size': 1000, 'stop_loss_pct': 0.02},
            'user_preferences': {'aggressive_trading': False}
        },
        'reasoning': {
            'primary_reason': 'RSI oversold condition (35) with strong support at $180.00 indicating buying opportunity',
            'supporting_factors': [
                'RSI below 40 indicates oversold condition',
                'Price near strong support level',
                'Low volatility reduces risk',
                'Volume confirms price action',
                'Risk-reward ratio favorable (2:1)'
            ],
            'alternatives_considered': [
                'Wait for RSI to drop further',
                'Buy with smaller position size',
                'Use different entry strategy',
                'Wait for confirmation signal'
            ],
            'risks_assessed': [
                'Support level could break',
                'Market sentiment could worsen',
                'Position size risk manageable',
                'Stop loss at $178.65'
            ],
            'confidence_explanation': 'Medium confidence due to clear technical setup but market uncertainty',
            'expected_outcome': 'Expect 3-5% upside with stop loss protection'
        },
        'confidence_level': ConfidenceLevel.MEDIUM,
        'metadata': {
            'strategy_name': 'RSI_Mean_Reversion',
            'position_size': 100,
            'entry_price': 182.30,
            'stop_loss': 178.65,
            'target_price': 187.50
        }
    }

def create_sample_model_selection_decision():
    """Create a sample model selection decision."""
    return {
        'agent_name': 'ModelSelector',
        'decision_type': DecisionType.MODEL_SELECTION,
        'action_taken': 'Selected LSTM model over XGBoost for AAPL forecasting',
        'context': {
            'symbol': 'AAPL',
            'timeframe': '1h',
            'market_conditions': {
                'trend': 'mixed',
                'volatility': 'high',
                'data_quality': 'excellent'
            },
            'available_data': ['price', 'volume', 'technical_indicators', 'sentiment'],
            'constraints': {'max_training_time': 3600, 'min_accuracy': 0.75},
            'user_preferences': {'prefer_interpretable': False}
        },
        'reasoning': {
            'primary_reason': 'LSTM outperformed XGBoost in recent backtests with 2% higher accuracy',
            'supporting_factors': [
                'LSTM accuracy: 78.5% vs XGBoost: 76.3%',
                'LSTM better at capturing temporal patterns',
                'Lower overfitting on validation set',
                'Consistent performance across timeframes',
                'Better handling of market regime changes'
            ],
            'alternatives_considered': [
                'XGBoost with feature engineering',
                'Ensemble of both models',
                'Transformer model (too slow)',
                'Simple moving average (too basic)'
            ],
            'risks_assessed': [
                'LSTM training time longer',
                'Black box interpretability',
                'Potential overfitting',
                'Computational resource usage'
            ],
            'confidence_explanation': 'High confidence based on comprehensive backtesting and validation',
            'expected_outcome': 'LSTM expected to provide 2-3% better forecasting accuracy'
        },
        'confidence_level': ConfidenceLevel.HIGH,
        'metadata': {
            'models_evaluated': ['LSTM', 'XGBoost', 'Transformer'],
            'best_model': 'LSTM',
            'accuracy_improvement': 0.022,
            'training_time': 1800
        }
    }

def demo_basic_logging():
    """Demonstrate basic decision logging."""
    logger.info("=== Basic Decision Logging Demo ===")
    
    # Initialize reasoning logger
    reasoning_logger = ReasoningLogger()
    
    # Create sample decisions
    decisions = [
        create_sample_forecast_decision(),
        create_sample_strategy_decision(),
        create_sample_model_selection_decision()
    ]
    
    # Log decisions
    decision_ids = []
    for decision_data in decisions:
        decision_id = reasoning_logger.log_decision(**decision_data)
        decision_ids.append(decision_id)
        logger.info(f"✅ Logged decision: {decision_id}")
    
    return {'success': True, 'result': reasoning_logger, decision_ids, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

def demo_display_components():
    """Demonstrate display components."""
    logger.info("\n=== Display Components Demo ===")
    
    reasoning_logger = ReasoningLogger()
    display = ReasoningDisplay(reasoning_logger)
    
    # Log a sample decision
    decision_data = create_sample_forecast_decision()
    decision_id = reasoning_logger.log_decision(**decision_data)
    
    # Display statistics
    logger.info("\n📊 Statistics:")
    stats = reasoning_logger.get_statistics()
    logger.info(f"Total decisions: {stats['total_decisions']}")
    
    # Display recent decisions
    logger.info("\n📋 Recent Decisions:")
    display.display_recent_decisions_terminal(limit=3)
    
    # Display specific decision
    decision = reasoning_logger.get_decision(decision_id)
    if decision:
        logger.info(f"\n📄 Specific Decision ({decision_ids[0]}):")
        display.display_decision_terminal(decision)

def demo_real_time_updates():
    """Demonstrate real-time decision logging."""
    logger.info("\n=== Real-time Updates Demo ===")
    
    reasoning_logger = ReasoningLogger()
    
    # Simulate real-time decision logging
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    agents = ['LSTMForecaster', 'RSIStrategy', 'ModelSelector']
    
    logger.info("Simulating real-time decision logging...")
    
    for i in range(5):
        # Create random decision
        symbol = random.choice(symbols)
        agent = random.choice(agents)
        
        decision_data = {
            'agent_name': agent,
            'decision_type': random.choice(list(DecisionType)),
            'action_taken': f'Action for {symbol} at {datetime.now().strftime("%H:%M:%S")}',
            'context': {
                'symbol': symbol,
                'timeframe': '1h',
                'market_conditions': {'trend': random.choice(['bullish', 'bearish', 'neutral'])}
            },
            'reasoning': {
                'primary_reason': f'Random decision for {symbol}',
                'supporting_factors': ['Factor 1', 'Factor 2'],
                'alternatives_considered': ['Alternative 1'],
                'risks_assessed': ['Risk 1'],
                'expected_outcome': 'Expected outcome'
            },
            'confidence_level': random.choice(list(ConfidenceLevel)),
            'metadata': {'iteration': i + 1}
        }
        
        decision_id = reasoning_logger.log_decision(**decision_data)
        logger.info(f"🔄 Logged real-time decision {i+1}: {decision_id}")
        time.sleep(0.5)
    
    # Show final statistics
    stats = reasoning_logger.get_statistics()
    logger.info("\n📊 Final Statistics:")
    logger.info(f"Total decisions: {stats['total_decisions']}")
    logger.info(f"Active agents: {len(stats['decisions_by_agent'])}")

def demo_streamlit_components():
    """Demonstrate Streamlit components."""
    logger.info("\n=== Streamlit Components Demo ===")
    
    reasoning_logger = ReasoningLogger()
    display = ReasoningDisplay(reasoning_logger)
    
    # Log some sample decisions
    for _ in range(3):
        decision_data = create_sample_forecast_decision()
        reasoning_logger.log_decision(**decision_data)
    
    logger.info("Streamlit components would display:")
    logger.info("1. 📊 Statistics dashboard with charts")
    logger.info("2. 📋 Recent decisions table")
    logger.info("3. 🔴 Live decision feed")
    logger.info("4. 📄 Detailed decision viewer")
    logger.info("5. 🤖 Sidebar controls for filtering")
    
    logger.info("\nTo run Streamlit dashboard:")
    logger.info("streamlit run trading/utils/reasoning_display.py")

def demo_search_and_filter():
    """Demonstrate search and filter functionality."""
    logger.info("\n=== Search and Filter Demo ===")
    
    reasoning_logger = ReasoningLogger()
    
    # Log decisions from different agents
    agents = ['LSTMForecaster', 'RSIStrategy', 'ModelSelector']
    for agent in agents:
        for _ in range(2):
            decision_data = create_sample_forecast_decision()
            decision_data['agent_name'] = agent
            reasoning_logger.log_decision(**decision_data)
    
    # Test filtering by agent
    logger.info("Filtering by agent:")
    lstm_decisions = reasoning_logger.get_agent_decisions('LSTMForecaster')
    logger.info(f"LSTMForecaster decisions: {len(lstm_decisions)}")
    
    # Test filtering by decision type
    logger.info("\nFiltering by decision type:")
    forecast_decisions = reasoning_logger.get_decisions_by_type(DecisionType.FORECAST)
    logger.info(f"Forecast decisions: {len(forecast_decisions)}")
    
    # Test getting explanations
    logger.info("\nGetting explanations:")
    if forecast_decisions:
        decision_id = forecast_decisions[0].decision_id
        explanation = reasoning_logger.get_explanation(decision_id)
        logger.info(f"Explanation preview: {explanation[:100]}...")

def main():
    """Run the complete demo."""
    logger.info("🤖 Reasoning Logger and Display Demo")
    logger.info("=" * 50)
    
    try:
        # Run all demos
        demo_basic_logging()
        demo_display_components()
        demo_real_time_updates()
        demo_streamlit_components()
        demo_search_and_filter()
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ Demo completed!")
        logger.info("\n📚 Next steps:")
        logger.info("1. Integrate with your trading agents")
        logger.info("2. Start the reasoning service: python trading/utils/launch_reasoning_service.py")
        logger.info("3. Run Streamlit dashboard: streamlit run trading/utils/reasoning_display.py")
        logger.info("4. Monitor decisions in real-time")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

if __name__ == "__main__":
    main() 