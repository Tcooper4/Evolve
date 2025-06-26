"""
PromptRouterAgent: Smart prompt router for agent orchestration.
- Detects user intent (forecasting, backtesting, tuning, research)
- Parses arguments using OpenAI or regex fallback
- Routes to the correct agent automatically
"""

import re
import logging
from typing import Dict, Any, Optional, Tuple

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

class PromptRouterAgent:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or (openai.api_key if openai else None)
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key
        self.intent_keywords = {
            'forecasting': ['forecast', 'predict', 'projection', 'future'],
            'backtesting': ['backtest', 'historical', 'simulate', 'performance'],
            'tuning': ['tune', 'optimize', 'hyperparameter', 'search', 'bayesian'],
            'research': ['research', 'find', 'paper', 'github', 'arxiv', 'summarize']
        }

    def parse_intent(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Detect intent and parse arguments from a prompt."""
        # Try OpenAI NLU if available
        if openai and self.openai_api_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Classify the user's intent as one of: forecasting, backtesting, tuning, research. Extract any arguments (symbols, dates, params, etc) as a JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200
                )
                content = response.choices[0].message['content']
                # Expecting: {"intent": "forecasting", "args": {"symbol": "AAPL", ...}}
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if match:
                    parsed = eval(match.group(0), {"__builtins__": {}})
                    return parsed.get('intent', 'unknown'), parsed.get('args', {})
            except Exception as e:
                logger.warning(f"OpenAI NLU failed: {e}")
        # Fallback: regex/keyword matching
        intent = 'unknown'
        args = {}
        prompt_lower = prompt.lower()
        for key, keywords in self.intent_keywords.items():
            if any(word in prompt_lower for word in keywords):
                intent = key
                break
        # Simple argument extraction
        symbol_match = re.search(r'([A-Z]{2,6})', prompt)
        if symbol_match:
            args['symbol'] = symbol_match.group(1)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', prompt)
        if date_match:
            args['date'] = date_match.group(1)
        return intent, args

    def route(self, prompt: str, agents: Dict[str, Any]) -> Any:
        """Route the prompt to the correct agent based on intent and arguments."""
        intent, args = self.parse_intent(prompt)
        logger.info(f"Routing intent: {intent}, args: {args}")
        if intent == 'forecasting' and 'forecasting' in agents:
            return agents['forecasting'].run_forecast(**args)
        elif intent == 'backtesting' and 'backtesting' in agents:
            return agents['backtesting'].run_backtest(**args)
        elif intent == 'tuning' and 'tuning' in agents:
            return agents['tuning'].run_tuning(**args)
        elif intent == 'research' and 'research' in agents:
            return agents['research'].research(**args)
        else:
            logger.warning(f"No agent found for intent: {intent}")
            return {'error': f'Unknown or unsupported intent: {intent}'} 