"""
Centralized Prompt Templates

This module serves as the single source of truth for all prompt templates
used throughout the trading system. All hardcoded prompt strings should
be moved here and referenced by name.

Usage:
    from trading.agents.prompt_templates import PROMPT_TEMPLATES

    prompt = PROMPT_TEMPLATES["forecast_request"].format(
        symbol="AAPL",
        timeframe="1 week"
    )
"""

from typing import Dict, List

# Main prompt templates dictionary
PROMPT_TEMPLATES = {
    # === FORECASTING TEMPLATES ===
    "forecast_request": """
Please provide a forecast for {symbol} over the next {timeframe}.

Requirements:
- Use the most appropriate model for this timeframe
- Include confidence intervals
- Provide key factors influencing the prediction
- Consider market conditions and volatility

Symbol: {symbol}
Timeframe: {timeframe}
Model Type: {model_type}
""",
    "forecast_analysis": """
Forecast Analysis Results:

Asset: {asset}
Timeframe: {timeframe}
Prediction: {prediction}
Confidence: {confidence}%
Key Factors: {factors}

Technical Analysis:
- Trend: {trend}
- Support/Resistance: {support_resistance}
- Volume Analysis: {volume_analysis}

Market Context:
- Economic Indicators: {economic_indicators}
- Sector Performance: {sector_performance}
- Risk Factors: {risk_factors}
""",
    # === INTENT PARSING TEMPLATES ===
    "intent_classification": """
You are an intent classifier for a trading system.
Classify the user's intent and extract arguments as JSON.

Available intents: forecasting, backtesting, tuning, research, portfolio, risk, sentiment, analysis, comparison, optimization

User Query: {query}

Return format: {{"intent": "intent_name", "confidence": 0.95, "args": {{"key": "value"}}}}
""",
    "intent_extraction": """
Analyze the following query and determine the user's intent:

Query: {query}

Extract the following information:
1. Primary intent (forecasting, backtesting, tuning, research, portfolio, risk, sentiment)
2. Secondary intent (if any)
3. Key entities (symbols, timeframes, models, strategies)
4. Parameters (amounts, percentages, dates)
5. Confidence level (0-1)

Return as JSON:
{{
    "primary_intent": "intent_name",
    "secondary_intent": "intent_name",
    "entities": {{"symbols": [], "timeframes": [], "models": []}},
    "parameters": {{"amounts": [], "percentages": [], "dates": []}},
    "confidence": 0.95
}}
""",
    # === RESPONSE GENERATION TEMPLATES ===
    "response_generation": """
Generate a response to the following query:

Query: {query}
Intent: {intent}
Entities: {entities}

Please respond with a JSON object containing:
1. response: The main response text
2. visualizations: List of visualizations to include (if any)
3. confidence: A float between 0 and 1 indicating confidence in the response
4. metadata: Additional information about the response

Example response:
{{
    "response": "Based on the analysis, I predict...",
    "visualizations": [
        {{
            "type": "forecast_plot",
            "data": {{...}},
            "narrative": "The forecast shows..."
        }}
    ],
    "confidence": 0.9,
    "metadata": {{
        "timeframe": "1 week",
        "model_used": "LSTM"
    }}
}}
""",
    # === BACKTESTING TEMPLATES ===
    "backtest_request": """
Please run a backtest for the following configuration:

Strategy: {strategy}
Symbol: {symbol}
Timeframe: {timeframe}
Start Date: {start_date}
End Date: {end_date}
Parameters: {parameters}

Requirements:
- Calculate key performance metrics (Sharpe ratio, max drawdown, etc.)
- Provide equity curve visualization
- Include trade analysis and statistics
- Consider transaction costs and slippage
""",
    "backtest_results": """
Backtest Results Summary:

Strategy: {strategy}
Symbol: {symbol}
Period: {start_date} to {end_date}

Performance Metrics:
- Total Return: {total_return}%
- Annualized Return: {annualized_return}%
- Sharpe Ratio: {sharpe_ratio}
- Maximum Drawdown: {max_drawdown}%
- Win Rate: {win_rate}%

Trade Statistics:
- Total Trades: {total_trades}
- Winning Trades: {winning_trades}
- Losing Trades: {losing_trades}
- Average Win: {avg_win}%
- Average Loss: {avg_loss}%

Risk Metrics:
- Volatility: {volatility}%
- VaR (95%): {var_95}%
- CVaR (95%): {cvar_95}%
""",
    # === OPTIMIZATION TEMPLATES ===
    "optimization_request": """
Please optimize the following strategy parameters:

Strategy: {strategy}
Symbol: {symbol}
Timeframe: {timeframe}
Objective: {objective}
Parameters to Optimize: {parameters}

Constraints:
- Parameter ranges: {parameter_ranges}
- Risk limits: {risk_limits}
- Performance targets: {performance_targets}

Optimization Method: {method}
Number of Trials: {trials}
""",
    "optimization_results": """
Optimization Results:

Strategy: {strategy}
Symbol: {symbol}
Objective: {objective}

Best Parameters:
{best_parameters}

Performance:
- Objective Value: {objective_value}
- Sharpe Ratio: {sharpe_ratio}
- Total Return: {total_return}%
- Maximum Drawdown: {max_drawdown}%

Parameter Sensitivity:
{parameter_sensitivity}

Recommendations:
{recommendations}
""",
    # === RESEARCH TEMPLATES ===
    "research_request": """
Please research the following topic for trading insights:

Topic: {topic}
Scope: {scope}
Timeframe: {timeframe}
Sources: {sources}

Requirements:
- Find relevant academic papers, articles, and reports
- Extract key insights and findings
- Identify practical applications for trading
- Assess the credibility and relevance of sources
- Provide actionable recommendations
""",
    "research_summary": """
Research Summary:

Topic: {topic}
Sources Reviewed: {source_count}
Key Findings: {key_findings}

Main Insights:
{main_insights}

Practical Applications:
{practical_applications}

Recommendations:
{recommendations}

Confidence Level: {confidence}%
""",
    # === PORTFOLIO TEMPLATES ===
    "portfolio_analysis": """
Please analyze the current portfolio:

Portfolio Composition: {composition}
Current Value: {current_value}
Target Allocation: {target_allocation}

Analysis Requirements:
- Performance attribution
- Risk analysis
- Rebalancing recommendations
- Diversification assessment
- Correlation analysis
""",
    "portfolio_recommendations": """
Portfolio Recommendations:

Current Portfolio:
{current_portfolio}

Analysis Results:
- Performance: {performance}
- Risk Metrics: {risk_metrics}
- Diversification Score: {diversification_score}

Recommendations:
{recommendations}

Action Items:
{action_items}

Expected Impact:
{expected_impact}
""",
    # === RISK ANALYSIS TEMPLATES ===
    "risk_analysis": """
Please perform a comprehensive risk analysis:

Asset/Portfolio: {asset}
Timeframe: {timeframe}
Risk Metrics: {risk_metrics}

Analysis Requirements:
- Value at Risk (VaR) calculation
- Expected Shortfall (CVaR)
- Stress testing scenarios
- Correlation analysis
- Volatility forecasting
- Tail risk assessment
""",
    "risk_report": """
Risk Analysis Report:

Asset/Portfolio: {asset}
Analysis Date: {date}

Risk Metrics:
- VaR (95%): {var_95}
- VaR (99%): {var_99}
- CVaR (95%): {cvar_95}
- Expected Shortfall: {expected_shortfall}
- Volatility: {volatility}%

Stress Test Results:
{stress_test_results}

Risk Factors:
{risk_factors}

Recommendations:
{recommendations}
""",
    # === SENTIMENT ANALYSIS TEMPLATES ===
    "sentiment_analysis": """
Please analyze sentiment for the following:

Symbol: {symbol}
Timeframe: {timeframe}
Data Sources: {sources}

Analysis Requirements:
- News sentiment analysis
- Social media sentiment
- Market sentiment indicators
- Sentiment trends over time
- Impact on price movement
""",
    "sentiment_report": """
Sentiment Analysis Report:

Symbol: {symbol}
Analysis Period: {period}

Overall Sentiment: {overall_sentiment}
Sentiment Score: {sentiment_score}

Breakdown by Source:
{source_breakdown}

Key Sentiment Drivers:
{key_drivers}

Trend Analysis:
{trend_analysis}

Impact Assessment:
{impact_assessment}
""",
    # === VOICE COMMAND TEMPLATES ===
    "voice_command_parsing": """
Parse the following voice command into structured trading action:

Voice Command: {command}

Extract:
1. Action type (forecast, trade, analysis, portfolio, strategy)
2. Symbol/ticker
3. Parameters (timeframe, amount, side)
4. Confidence level
5. Additional context

Return as JSON:
{{
    "action": "action_type",
    "symbol": "SYMBOL",
    "parameters": {{"timeframe": "1d", "amount": 100}},
    "confidence": 0.9,
    "raw_text": "original command"
}}
""",
    # === ERROR HANDLING TEMPLATES ===
    "error_response": """
I encountered an error while processing your request:

Error Type: {error_type}
Error Message: {error_message}
Request: {request}

What I can do to help:
1. {suggestion_1}
2. {suggestion_2}
3. {suggestion_3}

Please try again or provide more specific information.
""",
    "fallback_response": """
I'm having trouble understanding your request. Let me help you get what you need:

Your request: {request}

Here are some things I can help you with:
- Forecast stock prices: "Forecast AAPL for 1 week"
- Run backtests: "Backtest momentum strategy on SPY"
- Analyze portfolio: "Analyze my portfolio performance"
- Optimize strategies: "Optimize my trading strategy"
- Research topics: "Research machine learning in trading"

Could you please rephrase your request or choose from the options above?
""",
    # === SYSTEM TEMPLATES ===
    "system_status": """
System Status Report:

Available Services:
{available_services}

System Health:
- API Status: {api_status}
- Database Status: {db_status}
- Model Status: {model_status}
- Agent Status: {agent_status}

Performance Metrics:
- Response Time: {response_time}ms
- Success Rate: {success_rate}%
- Error Rate: {error_rate}%

Recent Activity:
{recent_activity}
""",
    "help_message": """
Welcome to the Trading AI Assistant!

I can help you with:

📈 Forecasting:
  - "Forecast AAPL for 1 week"
  - "Predict SPY price movement"

📊 Analysis:
  - "Analyze TSLA technical indicators"
  - "Compare AAPL vs GOOGL performance"

🔄 Backtesting:
  - "Backtest momentum strategy on SPY"
  - "Test my trading strategy"

⚙️ Optimization:
  - "Optimize my strategy parameters"
  - "Find best parameters for mean reversion"

📚 Research:
  - "Research machine learning in trading"
  - "Find papers on portfolio optimization"

💼 Portfolio:
  - "Analyze my portfolio"
  - "Get portfolio recommendations"

🎤 Voice Commands:
  - "Forecast AAPL for 1 week"
  - "Show my portfolio"

Just ask me anything about trading, analysis, or portfolio management!
""",
    # === RESEARCH AGENT TEMPLATES ===
    "research_summarize": """
Summarize this for a quant trading engineer:

{text}

Focus on:
- Key insights and findings
- Practical applications for trading
- Technical details and methodologies
- Potential implementation considerations
""",
    "research_code_suggestion": """
Suggest Python code for a quant trading engineer based on this description:

{description}

Provide:
- Clean, well-commented code
- Best practices for trading systems
- Error handling and validation
- Performance considerations
""",
    # === MULTIMODAL AGENT TEMPLATES ===
    "vision_chart_analysis": """
{vision_prompt}

Analyze this trading chart and provide insights for a quant trading engineer.
Focus on:
- Key patterns and trends
- Technical indicators visible
- Potential trading signals
- Risk considerations
- Market context and implications
""",
    "vision_equity_curve": """
Describe the equity curve shape, trends, and any notable features for a quant trading engineer.

Key aspects to analyze:
- Overall trend direction
- Volatility patterns
- Drawdown periods
- Recovery characteristics
- Performance consistency
- Risk-adjusted metrics
""",
    "vision_drawdown_analysis": """
Describe drawdown spikes and risk periods in this trading equity curve.

Focus on:
- Magnitude and duration of drawdowns
- Recovery patterns
- Risk management implications
- Volatility clustering
- Potential stress testing scenarios
""",
    "vision_performance_analysis": """
Describe the strategy's performance over time and any patterns in the returns.

Analyze:
- Return distribution
- Performance consistency
- Seasonal patterns
- Risk-adjusted metrics
- Drawdown characteristics
- Sharpe ratio implications
""",
    # === ENHANCED PROMPT ROUTER TEMPLATES ===
    "enhanced_intent_classification": """
You are an intent classifier for a trading system.
Classify the user's intent and extract arguments as JSON.
Available intents: forecasting, backtesting, tuning, research, portfolio, risk, sentiment
Return format: {{"intent": "intent_name", "confidence": 0.95, "args": {{"key": "value"}}}}

User Query: {query}
""",
    "enhanced_intent_extraction": """
Task: Classify trading intent
User input: {user_input}
Intent:
""",
    # === STRATEGY SELECTOR TEMPLATES ===
    "strategy_selection_reasoning": """
Generate reasoning for strategy selection based on market conditions and analysis.

Market Context: {market_context}
Available Strategies: {available_strategies}
Historical Performance: {historical_performance}

Provide:
- Strategy recommendation with confidence level
- Reasoning based on market conditions
- Risk considerations
- Expected performance characteristics
- Implementation recommendations
""",
    # === PERFORMANCE CRITIC TEMPLATES ===
    "performance_evaluation": """
Generate predictions using the model.

Model Type: {model_type}
Input Data: {input_data}
Prediction Horizon: {prediction_horizon}

Requirements:
- Generate accurate predictions
- Include confidence intervals
- Consider model limitations
- Provide uncertainty estimates
""",
    "trading_signal_generation": """
Generate trading signals from predictions.

Predictions: {predictions}
Market Context: {market_context}
Risk Parameters: {risk_parameters}

Generate:
- Buy/sell signals with confidence
- Position sizing recommendations
- Risk management rules
- Entry/exit criteria
""",
    "performance_recommendations": """
Generate recommendations based on evaluation results.

Evaluation Results: {evaluation_results}
Performance Metrics: {performance_metrics}
Model Characteristics: {model_characteristics}

Provide:
- Improvement recommendations
- Parameter optimization suggestions
- Risk management advice
- Implementation guidance
""",
    # === OPTIMIZER AGENT TEMPLATES ===
    "optimization_summary": """
Generate optimization summary.

Strategy: {strategy}
Parameters Tested: {parameters_tested}
Best Results: {best_results}
Performance Metrics: {performance_metrics}

Summary should include:
- Best parameter combinations
- Performance improvements
- Risk considerations
- Implementation recommendations
- Next steps for optimization
""",
    # === META RESEARCH AGENT TEMPLATES ===
    "model_implementation_code": """
Generate model implementation code.

Model Type: {model_type}
Requirements: {requirements}
Framework: {framework}

Provide:
- Complete implementation code
- Documentation and comments
- Error handling
- Performance optimizations
- Testing considerations
""",
    "transformer_model_code": """
Generate transformer model code for time series forecasting.

Requirements: {requirements}
Input Features: {input_features}
Output Horizon: {output_horizon}

Include:
- Model architecture
- Training loop
- Data preprocessing
- Evaluation metrics
- Deployment considerations
""",
    "lstm_model_code": """
Generate LSTM model code for financial time series.

Requirements: {requirements}
Sequence Length: {sequence_length}
Prediction Horizon: {prediction_horizon}

Include:
- LSTM architecture
- Data preparation
- Training configuration
- Validation methods
- Performance monitoring
""",
    "ensemble_model_code": """
Generate ensemble model code combining multiple approaches.

Base Models: {base_models}
Ensemble Method: {ensemble_method}
Weighting Strategy: {weighting_strategy}

Include:
- Individual model implementations
- Ensemble combination logic
- Weight optimization
- Performance evaluation
- Risk management
""",
    # === DATA QUALITY AGENT TEMPLATES ===
    "data_quality_recommendations": """
Generate recommendations for data quality improvement.

Issues Found: {issues_found}
Data Characteristics: {data_characteristics}
Impact Assessment: {impact_assessment}

Provide:
- Specific improvement actions
- Data cleaning procedures
- Quality monitoring setup
- Validation protocols
- Implementation timeline
""",
    # === EXECUTION AGENT TEMPLATES ===
    "execution_summary": """
Generate execution summary.

Trades Executed: {trades_executed}
Performance Metrics: {performance_metrics}
Risk Measures: {risk_measures}

Include:
- Execution quality assessment
- Performance analysis
- Risk management effectiveness
- Recommendations for improvement
- Next steps
""",
}

# Template categories for organization
TEMPLATE_CATEGORIES = {
    "forecasting": ["forecast_request", "forecast_analysis"],
    "intent_parsing": [
        "intent_classification",
        "intent_extraction",
        "enhanced_intent_classification",
        "enhanced_intent_extraction",
    ],
    "response_generation": ["response_generation"],
    "backtesting": ["backtest_request", "backtest_results"],
    "optimization": [
        "optimization_request",
        "optimization_results",
        "optimization_summary",
    ],
    "research": [
        "research_request",
        "research_summary",
        "research_summarize",
        "research_code_suggestion",
    ],
    "portfolio": ["portfolio_analysis", "portfolio_recommendations"],
    "risk": ["risk_analysis", "risk_report"],
    "sentiment": ["sentiment_analysis", "sentiment_report"],
    "voice": ["voice_command_parsing"],
    "error_handling": ["error_response", "fallback_response"],
    "system": ["system_status", "help_message"],
    "multimodal": [
        "vision_chart_analysis",
        "vision_equity_curve",
        "vision_drawdown_analysis",
        "vision_performance_analysis",
    ],
    "strategy": ["strategy_selection_reasoning"],
    "performance": [
        "performance_evaluation",
        "trading_signal_generation",
        "performance_recommendations",
    ],
    "model_development": [
        "model_implementation_code",
        "transformer_model_code",
        "lstm_model_code",
        "ensemble_model_code",
    ],
    "data_quality": ["data_quality_recommendations"],
    "execution": ["execution_summary"],
}


def get_template(name: str) -> str:
    """
    Get a prompt template by name.

    Args:
        name: Name of the template

    Returns:
        Template string

    Raises:
        KeyError: If template not found
    """
    if name not in PROMPT_TEMPLATES:
        raise KeyError(
            f"Template '{name}' not found. Available templates: {list(PROMPT_TEMPLATES.keys())}"
        )

    return PROMPT_TEMPLATES[name]


def get_templates_by_category(category: str) -> Dict[str, str]:
    """
    Get all templates in a specific category.

    Args:
        category: Category name

    Returns:
        Dictionary of template names to template strings
    """
    if category not in TEMPLATE_CATEGORIES:
        raise KeyError(
            f"Category '{category}' not found. Available categories: {list(TEMPLATE_CATEGORIES.keys())}"
        )

    template_names = TEMPLATE_CATEGORIES[category]
    return {name: PROMPT_TEMPLATES[name] for name in template_names}


def list_templates() -> List[str]:
    """Get list of all available template names."""
    return list(PROMPT_TEMPLATES.keys())


def list_categories() -> List[str]:
    """Get list of all available template categories."""
    return list(TEMPLATE_CATEGORIES.keys())


def format_template(name: str, **kwargs) -> str:
    """
    Get and format a template with the provided arguments.

    Args:
        name: Template name
        **kwargs: Arguments to format the template

    Returns:
        Formatted template string
    """
    template = get_template(name)
    return template.format(**kwargs)


# Legacy template mappings for backward compatibility
LEGACY_TEMPLATES = {
    "forecast": "forecast_request",
    "analysis": "forecast_analysis",
    "backtest": "backtest_request",
    "optimize": "optimization_request",
    "research": "research_request",
    "portfolio": "portfolio_analysis",
    "risk": "risk_analysis",
    "sentiment": "sentiment_analysis",
    "voice": "voice_command_parsing",
    "error": "error_response",
    "help": "help_message",
}


def get_legacy_template(name: str) -> str:
    """
    Get a template using legacy naming for backward compatibility.

    Args:
        name: Legacy template name

    Returns:
        Template string
    """
    if name in LEGACY_TEMPLATES:
        return PROMPT_TEMPLATES[LEGACY_TEMPLATES[name]]
    else:
        return get_template(name)


# Add support for multilingual templates
TEMPLATES = {
    "en": PROMPT_TEMPLATES,
    "es": {
        "forecast": "¿Qué acciones deberían comprarse hoy?",
        "analysis": "Analiza el rendimiento de {symbol}",
        "strategy": "Genera una estrategia de trading para {symbol}",
        "optimization": "Optimiza los parámetros de la estrategia",
        "portfolio": "Gestiona el portafolio de inversiones",
        "risk": "Evalúa el riesgo de la inversión",
        "backtest": "Ejecuta un backtest de la estrategia",
        "research": "Investiga las últimas tendencias del mercado",
        "sentiment": "Analiza el sentimiento del mercado",
        "system": "Verifica el estado del sistema",
    },
    "fr": {
        "forecast": "Quelles actions devraient être achetées aujourd'hui?",
        "analysis": "Analyser la performance de {symbol}",
        "strategy": "Générer une stratégie de trading pour {symbol}",
        "optimization": "Optimiser les paramètres de la stratégie",
        "portfolio": "Gérer le portefeuille d'investissement",
        "risk": "Évaluer le risque de l'investissement",
        "backtest": "Exécuter un backtest de la stratégie",
        "research": "Rechercher les dernières tendances du marché",
        "sentiment": "Analyser le sentiment du marché",
        "system": "Vérifier l'état du système",
    },
    "de": {
        "forecast": "Welche Aktien sollten heute gekauft werden?",
        "analysis": "Analysieren Sie die Performance von {symbol}",
        "strategy": "Generieren Sie eine Trading-Strategie für {symbol}",
        "optimization": "Optimieren Sie die Strategieparameter",
        "portfolio": "Verwalten Sie das Anlageportfolio",
        "risk": "Bewerten Sie das Anlagerisiko",
        "backtest": "Führen Sie einen Backtest der Strategie durch",
        "research": "Erforschen Sie die neuesten Markttrends",
        "sentiment": "Analysieren Sie die Marktstimmung",
        "system": "Überprüfen Sie den Systemstatus",
    },
}
