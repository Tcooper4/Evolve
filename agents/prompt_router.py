"""
Prompt Router Module

This module handles prompt processing and routing for the Evolve Trading Platform:
- Natural language prompt analysis
- Request classification and routing
- Prompt validation and preprocessing
- Context management and enhancement
- Compound prompt parsing and sub-task dispatch
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Types of user requests."""

    FORECAST = "forecast"
    STRATEGY = "strategy"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"
    PORTFOLIO = "portfolio"
    SYSTEM = "system"
    GENERAL = "general"
    INVESTMENT = "investment"  # New type for investment-related queries
    UNKNOWN = "unknown"


@dataclass
class PromptContext:
    """Context information for prompt processing."""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    previous_requests: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessedPrompt:
    """Processed prompt information."""

    original_prompt: str
    request_type: RequestType
    confidence: float
    extracted_parameters: Dict[str, Any]
    context: PromptContext
    routing_suggestions: List[str]
    processing_time: float


@dataclass
class SubTask:
    """Individual sub-task from compound prompt."""
    task_id: str
    original_prompt: str
    task_type: RequestType
    parameters: Dict[str, Any]
    priority: int
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class CompoundPromptResult:
    """Result of processing a compound prompt."""
    original_prompt: str
    sub_tasks: List[SubTask]
    execution_order: List[str]
    overall_status: str
    combined_result: Dict[str, Any]
    processing_time: float
    errors: List[str] = field(default_factory=list)


class CompoundPromptParser:
    """Parser for compound prompts that contain multiple sub-tasks."""
    
    def __init__(self):
        """Initialize compound prompt parser."""
        self.separator_patterns = [
            r"\s+and\s+",
            r"\s+then\s+",
            r"\s+also\s+",
            r"\s+additionally\s+",
            r"\s+moreover\s+",
            r"\s+furthermore\s+",
            r"\s+next\s+",
            r"\s+after\s+that\s+",
            r"\s+subsequently\s+",
            r"\s+meanwhile\s+",
            r"\s+while\s+",
            r"\s+during\s+",
            r"\s+at\s+the\s+same\s+time\s+",
            r"\s+concurrently\s+",
            r"\s+in\s+parallel\s+",
        ]
        
        self.dependency_indicators = [
            r"based\s+on\s+",
            r"using\s+the\s+result\s+of\s+",
            r"after\s+",
            r"following\s+",
            r"once\s+",
            r"when\s+",
            r"if\s+",
            r"provided\s+that\s+",
        ]
        
        self.priority_indicators = {
            "urgent": ["urgent", "immediately", "asap", "right now", "now"],
            "high": ["important", "critical", "priority", "first", "primary"],
            "medium": ["also", "additionally", "moreover", "furthermore"],
            "low": ["later", "eventually", "when possible", "if time permits"]
        }
    
    def is_compound_prompt(self, prompt: str) -> bool:
        """
        Check if a prompt is compound (contains multiple sub-tasks).
        
        Args:
            prompt: User prompt
            
        Returns:
            True if prompt is compound
        """
        normalized_prompt = prompt.lower()
        
        # Check for separator patterns
        for pattern in self.separator_patterns:
            if re.search(pattern, normalized_prompt):
                return True
        
        # Check for multiple action verbs
        action_verbs = [
            "forecast", "analyze", "optimize", "create", "build", "test",
            "evaluate", "compare", "select", "recommend", "investigate"
        ]
        
        verb_count = sum(1 for verb in action_verbs if verb in normalized_prompt)
        return verb_count > 1
    
    def parse_compound_prompt(self, prompt: str, processor: 'PromptProcessor') -> List[SubTask]:
        """
        Parse compound prompt into sub-tasks.
        
        Args:
            prompt: Compound prompt
            processor: Prompt processor for sub-task classification
            
        Returns:
            List of sub-tasks
        """
        sub_tasks = []
        
        # Split prompt into sub-tasks
        task_texts = self._split_prompt(prompt)
        
        for i, task_text in enumerate(task_texts):
            task_text = task_text.strip()
            if not task_text:
                continue
            
            # Process sub-task
            processed = processor.process_prompt(task_text)
            
            # Determine priority
            priority = self._determine_priority(task_text)
            
            # Find dependencies
            dependencies = self._find_dependencies(task_text, task_texts, i)
            
            sub_task = SubTask(
                task_id=f"task_{i}_{hash(task_text) % 10000}",
                original_prompt=task_text,
                task_type=processed.request_type,
                parameters=processed.extracted_parameters,
                priority=priority,
                dependencies=dependencies
            )
            
            sub_tasks.append(sub_task)
        
        return sub_tasks
    
    def _split_prompt(self, prompt: str) -> List[str]:
        """Split compound prompt into individual tasks."""
        # Try to split on separator patterns
        for pattern in self.separator_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                parts = re.split(pattern, prompt, flags=re.IGNORECASE)
                # Clean up parts
                parts = [part.strip() for part in parts if part.strip()]
                if len(parts) > 1:
                    return parts
        
        # If no clear separators, try to split on sentence boundaries
        sentences = re.split(r'[.!?]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If still only one part, try to identify natural breaks
        if len(sentences) == 1:
            # Look for natural task boundaries
            natural_breaks = [
                r"\s+and\s+",
                r"\s+then\s+",
                r"\s+also\s+",
                r"\s+additionally\s+"
            ]
            
            for break_pattern in natural_breaks:
                if re.search(break_pattern, sentences[0], re.IGNORECASE):
                    parts = re.split(break_pattern, sentences[0], flags=re.IGNORECASE)
                    parts = [part.strip() for part in parts if part.strip()]
                    if len(parts) > 1:
                        return parts
        
        return [prompt]  # Return as single task if can't split
    
    def _determine_priority(self, task_text: str) -> int:
        """Determine priority of a sub-task."""
        normalized_text = task_text.lower()
        
        for priority_level, indicators in self.priority_indicators.items():
            for indicator in indicators:
                if indicator in normalized_text:
                    priority_map = {
                        "urgent": 1,
                        "high": 2,
                        "medium": 3,
                        "low": 4
                    }
                    return priority_map.get(priority_level, 3)
        
        return 3  # Default medium priority
    
    def _find_dependencies(self, task_text: str, all_tasks: List[str], current_index: int) -> List[str]:
        """Find dependencies for a sub-task."""
        dependencies = []
        
        for i, other_task in enumerate(all_tasks):
            if i == current_index:
                continue
            
            # Check if current task depends on other task
            normalized_text = task_text.lower()
            normalized_other = other_task.lower()
            
            # Look for dependency indicators
            for pattern in self.dependency_indicators:
                if re.search(pattern, normalized_text):
                    # Check if other task is referenced
                    # Extract key terms from other task
                    other_terms = re.findall(r'\b\w+\b', normalized_other)
                    for term in other_terms:
                        if len(term) > 3 and term in normalized_text:
                            dependencies.append(f"task_{i}_{hash(other_task) % 10000}")
                            break
        
        return dependencies


class PromptProcessor:
    """Processes and analyzes user prompts."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the prompt processor.
        
        Args:
            config: Configuration dictionary with patterns and keywords
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Load patterns from config or use defaults
        self.classification_patterns = self.config.get("classification_patterns", self._get_default_classification_patterns())
        self.parameter_patterns = self.config.get("parameter_patterns", self._get_default_parameter_patterns())
        self.investment_keywords = self.config.get("investment_keywords", self._get_default_investment_keywords())
        
        # Initialize compound prompt parser
        self.compound_parser = CompoundPromptParser()
        
    def _get_default_classification_patterns(self) -> Dict[RequestType, List[str]]:
        """Get default classification patterns."""
        return {
            RequestType.FORECAST: [
                r"\b(forecast|predict|future|next|upcoming|tomorrow|next week|next month)\b",
                r"\b(price|stock|market|trend|movement|direction)\b",
                r"\b(how much|what will|when will|where will)\b",
            ],
            RequestType.STRATEGY: [
                r"\b(strategy|trading|signal|entry|exit|position)\b",
                r"\b(buy|sell|hold|long|short|trade)\b",
                r"\b(rsi|macd|bollinger|moving average|indicator)\b",
            ],
            RequestType.ANALYSIS: [
                r"\b(analyze|analysis|examine|study|review|assess|evaluate)\b",
                r"\b(performance|metrics|statistics|data|chart|graph)\b",
                r"\b(why|what caused|what happened|explain)\b",
            ],
            RequestType.OPTIMIZATION: [
                r"\b(optimize|tune|improve|enhance|better|best|optimal)\b",
                r"\b(parameters|settings|configuration|hyperparameters)\b",
                r"\b(performance|efficiency|accuracy|speed)\b",
            ],
            RequestType.PORTFOLIO: [
                r"\b(portfolio|allocation|diversification|risk|balance)\b",
                r"\b(asset|investment|holdings|positions|weights)\b",
                r"\b(rebalance|adjust|change|modify)\b",
            ],
            RequestType.SYSTEM: [
                r"\b(system|status|health|monitor|check|diagnose)\b",
                r"\b(error|problem|issue|bug|fix|repair)\b",
                r"\b(restart|stop|start|configure|setup)\b",
            ],
            RequestType.INVESTMENT: [
                r"\b(invest|investment|buy|purchase|acquire)\b",
                r"\b(top stocks|best stocks|recommended|suggest)\b",
                r"\b(what should|which stocks|what to buy|where to invest)\b",
                r"\b(opportunity|potential|growth|returns)\b",
                r"\b(today|now|current|market)\b",
            ],
        }
        
    def _get_default_parameter_patterns(self) -> Dict[str, str]:
        """Get default parameter extraction patterns."""
        return {
            "symbol": r"\b([A-Z]{1,5})\b",
            "timeframe": r"\b(1m|5m|15m|30m|1h|4h|1d|1w|1M)\b",
            "days": r"\b(\d+)\s*(days?|d)\b",
            "model": r"\b(lstm|arima|xgboost|prophet|ensemble|transformer)\b",
            "strategy": r"\b(rsi|macd|bollinger|sma|ema|custom)\b",
            "risk_level": r"\b(low|medium|high|conservative|aggressive)\b",
        }
        
    def _get_default_investment_keywords(self) -> List[str]:
        """Get default investment keywords."""
        return [
            "invest",
            "investment",
            "buy",
            "purchase",
            "acquire",
            "top stocks",
            "best stocks",
            "recommended",
            "suggest",
            "what should",
            "which stocks",
            "what to buy",
            "where to invest",
            "opportunity",
            "potential",
            "growth",
            "returns",
            "today",
            "now",
            "current",
            "market",
        ]
        
    def update_patterns(self, new_patterns: Dict[str, Any]):
        """Update patterns from external configuration.
        
        Args:
            new_patterns: Dictionary with new patterns to update
        """
        if "classification_patterns" in new_patterns:
            self.classification_patterns.update(new_patterns["classification_patterns"])
            
        if "parameter_patterns" in new_patterns:
            self.parameter_patterns.update(new_patterns["parameter_patterns"])
            
        if "investment_keywords" in new_patterns:
            self.investment_keywords.extend(new_patterns["investment_keywords"])
            
        self.logger.info("Patterns updated from configuration")
        
    def add_strategy_keywords(self, strategy_name: str, keywords: List[str]):
        """Add custom strategy keywords for pattern recognition.
        
        Args:
            strategy_name: Name of the strategy
            keywords: List of keywords for the strategy
        """
        # Add to strategy patterns
        if RequestType.STRATEGY not in self.classification_patterns:
            self.classification_patterns[RequestType.STRATEGY] = []
            
        # Create pattern for the strategy keywords
        keyword_pattern = r"\b(" + "|".join(keywords) + r")\b"
        self.classification_patterns[RequestType.STRATEGY].append(keyword_pattern)
        
        # Add to parameter patterns
        self.parameter_patterns["strategy"] = r"\b(" + "|".join([self.parameter_patterns.get("strategy", ""), strategy_name.lower()]) + r")\b"
        
        self.logger.info(f"Added keywords for strategy: {strategy_name}")
        
    def load_patterns_from_file(self, file_path: str):
        """Load patterns from a configuration file.
        
        Args:
            file_path: Path to the configuration file
        """
        try:
            import json
            with open(file_path, 'r') as f:
                config = json.load(f)
            self.update_patterns(config)
            self.logger.info(f"Patterns loaded from file: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to load patterns from file {file_path}: {e}")
            
    def get_current_patterns(self) -> Dict[str, Any]:
        """Get current patterns for debugging or export.
        
        Returns:
            Dictionary with current patterns
        """
        return {
            "classification_patterns": self.classification_patterns,
            "parameter_patterns": self.parameter_patterns,
            "investment_keywords": self.investment_keywords
        }

    def process_prompt(
        self, prompt: str, context: Optional[PromptContext] = None
    ) -> ProcessedPrompt:
        """
        Process a user prompt and extract information.

        Args:
            prompt: User's input prompt
            context: Optional context information

        Returns:
            ProcessedPrompt: Processed prompt information
        """
        start_time = datetime.now()

        if context is None:
            context = PromptContext()

        # Normalize prompt (lowercase, strip whitespace)
        normalized_prompt = self._normalize_prompt(prompt)

        # Check for investment-related queries first
        if self._is_investment_query(normalized_prompt):
            request_type = RequestType.INVESTMENT
            confidence = 0.9  # High confidence for investment queries
        else:
            # Classify request type
            request_type = self._classify_request(normalized_prompt)
            confidence = self._calculate_confidence(normalized_prompt, request_type)

        # Extract parameters
        extracted_parameters = self._extract_parameters(normalized_prompt)

        # Generate routing suggestions
        routing_suggestions = self._generate_routing_suggestions(
            request_type, extracted_parameters
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        processed_prompt = ProcessedPrompt(
            original_prompt=prompt,
            request_type=request_type,
            confidence=confidence,
            extracted_parameters=extracted_parameters,
            context=context,
            routing_suggestions=routing_suggestions,
            processing_time=processing_time,
        )

        self.logger.info(
            f"Processed prompt: {request_type.value} (confidence: {confidence:.2f})"
        )
        return processed_prompt

    def process_compound_prompt(self, prompt: str, context: Optional[PromptContext] = None) -> CompoundPromptResult:
        """
        Process compound prompt by parsing into sub-tasks and determining execution order.
        
        Args:
            prompt: Compound prompt
            context: Processing context
            
        Returns:
            CompoundPromptResult with sub-tasks and execution plan
        """
        start_time = datetime.now()
        
        try:
            # Check if prompt is compound
            if not self.compound_parser.is_compound_prompt(prompt):
                # Single task - create simple result
                processed = self.process_prompt(prompt, context)
                sub_task = SubTask(
                    task_id=f"task_0_{hash(prompt) % 10000}",
                    original_prompt=prompt,
                    task_type=processed.request_type,
                    parameters=processed.extracted_parameters,
                    priority=3
                )
                
                return CompoundPromptResult(
                    original_prompt=prompt,
                    sub_tasks=[sub_task],
                    execution_order=[sub_task.task_id],
                    overall_status="single_task",
                    combined_result={"type": "single_task", "task": sub_task},
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Parse compound prompt
            sub_tasks = self.compound_parser.parse_compound_prompt(prompt, self)
            
            # Determine execution order based on dependencies and priorities
            execution_order = self._determine_execution_order(sub_tasks)
            
            return CompoundPromptResult(
                original_prompt=prompt,
                sub_tasks=sub_tasks,
                execution_order=execution_order,
                overall_status="parsed",
                combined_result={"type": "compound", "task_count": len(sub_tasks)},
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            self.logger.error(f"Error processing compound prompt: {e}")
            return CompoundPromptResult(
                original_prompt=prompt,
                sub_tasks=[],
                execution_order=[],
                overall_status="error",
                combined_result={"error": str(e)},
                processing_time=(datetime.now() - start_time).total_seconds(),
                errors=[str(e)]
            )
    
    def _determine_execution_order(self, sub_tasks: List[SubTask]) -> List[str]:
        """
        Determine optimal execution order for sub-tasks.
        
        Args:
            sub_tasks: List of sub-tasks
            
        Returns:
            List of task IDs in execution order
        """
        # Create dependency graph
        task_map = {task.task_id: task for task in sub_tasks}
        dependencies = {task.task_id: set(task.dependencies) for task in sub_tasks}
        
        # Topological sort with priority consideration
        execution_order = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected: {task_id}")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            # Visit dependencies first
            for dep_id in dependencies.get(task_id, set()):
                if dep_id in task_map:
                    visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            execution_order.append(task_id)
        
        # Sort tasks by priority first, then visit
        sorted_tasks = sorted(sub_tasks, key=lambda t: t.priority)
        
        for task in sorted_tasks:
            if task.task_id not in visited:
                visit(task.task_id)
        
        return execution_order

    def _normalize_prompt(self, prompt: str) -> str:
        """
        Normalize prompt for consistent processing.

        Args:
            prompt: Original prompt

        Returns:
            str: Normalized prompt
        """
        # Convert to lowercase and strip whitespace
        normalized = prompt.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    def _is_investment_query(self, normalized_prompt: str) -> bool:
        """
        Check if the prompt is an investment-related query.

        Args:
            normalized_prompt: Normalized prompt

        Returns:
            bool: True if investment query, False otherwise
        """
        # Check for exact keyword matches
        for keyword in self.investment_keywords:
            if keyword in normalized_prompt:
                return True

        # Check for fuzzy matches
        for keyword in self.investment_keywords:
            if self._fuzzy_match(normalized_prompt, keyword, threshold=0.8):
                return True

        # Check for common investment question patterns
        investment_patterns = [
            r"\bwhat\s+(stocks?|should|to)\s+(invest|buy|purchase)\b",
            r"\bwhich\s+(stocks?|companies?)\s+(to|should)\s+(invest|buy)\b",
            r"\b(top|best)\s+(stocks?|investments?)\b",
            r"\b(recommend|suggest)\s+(stocks?|investments?)\b",
        ]

        for pattern in investment_patterns:
            if re.search(pattern, normalized_prompt):
                return True

        return False

    def _fuzzy_match(self, text: str, keyword: str, threshold: float = 0.8) -> bool:
        """
        Perform fuzzy matching between text and keyword.

        Args:
            text: Text to search in
            keyword: Keyword to search for
            threshold: Similarity threshold (0.0 to 1.0)

        Returns:
            bool: True if match found, False otherwise
        """
        words = text.split()
        for word in words:
            similarity = SequenceMatcher(None, word, keyword).ratio()
            if similarity >= threshold:
                return True
        return False

    def _classify_request(self, prompt: str) -> RequestType:
        """
        Classify the type of request.

        Args:
            prompt: User prompt

        Returns:
            RequestType: Classified request type
        """
        prompt_lower = prompt.lower()
        scores = {}

        for request_type, patterns in self.classification_patterns.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, prompt_lower)
                score += len(matches)
            scores[request_type] = score

        # Find the request type with the highest score
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]

        return RequestType.UNKNOWN

    def _extract_parameters(self, prompt: str) -> Dict[str, Any]:
        """
        Extract parameters from the prompt.

        Args:
            prompt: User prompt

        Returns:
            Dict: Extracted parameters
        """
        parameters = {}
        prompt_lower = prompt.lower()

        for param_name, pattern in self.parameter_patterns.items():
            matches = re.findall(pattern, prompt_lower)
            if matches:
                if param_name == "days":
                    # Extract the number
                    numbers = re.findall(r"\d+", str(matches[0]))
                    if numbers:
                        parameters[param_name] = int(numbers[0])
                else:
                    parameters[param_name] = (
                        matches[0] if len(matches) == 1 else matches
                    )

        return parameters

    def _calculate_confidence(self, prompt: str, request_type: RequestType) -> float:
        """
        Calculate confidence in the classification.

        Args:
            prompt: User prompt
            request_type: Classified request type

        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        if request_type == RequestType.UNKNOWN:
            return 0.0

        prompt_lower = prompt.lower()
        patterns = self.classification_patterns[request_type]

        total_matches = 0
        for pattern in patterns:
            matches = re.findall(pattern, prompt_lower)
            total_matches += len(matches)

        # Normalize by prompt length and pattern count
        confidence = min(1.0, total_matches / max(1, len(prompt.split())))

        return confidence

    def _generate_routing_suggestions(
        self, request_type: RequestType, parameters: Dict[str, Any]
    ) -> List[str]:
        """
        Generate routing suggestions based on request type and parameters.

        Args:
            request_type: Classified request type
            parameters: Extracted parameters

        Returns:
            List[str]: Suggested routing targets
        """
        suggestions = []

        if request_type == RequestType.FORECAST:
            suggestions.extend(["ModelSelectorAgent", "ForecastEngine"])
            if "model" in parameters:
                suggestions.append(f"{parameters['model'].title()}Model")

        elif request_type == RequestType.STRATEGY:
            suggestions.extend(["StrategySelectorAgent", "StrategyEngine"])
            if "strategy" in parameters:
                suggestions.append(f"{parameters['strategy'].title()}Strategy")

        elif request_type == RequestType.INVESTMENT:
            # Route investment queries to TopRankedForecastAgent
            suggestions.extend(
                ["TopRankedForecastAgent", "PortfolioManager", "RiskManager"]
            )

        elif request_type == RequestType.ANALYSIS:
            suggestions.extend(["AnalysisEngine", "DataAnalyzer"])

        elif request_type == RequestType.OPTIMIZATION:
            suggestions.extend(["OptimizationEngine", "HyperparameterTuner"])

        elif request_type == RequestType.PORTFOLIO:
            suggestions.extend(["PortfolioManager", "AssetAllocator"])

        elif request_type == RequestType.SYSTEM:
            suggestions.extend(["SystemMonitor", "HealthChecker"])

        else:
            # Fallback for unknown requests
            suggestions.extend(["GeneralAssistant", "HelpSystem"])

        return suggestions

    def validate_prompt(self, prompt: str) -> Tuple[bool, List[str]]:
        """
        Validate a prompt for processing.

        Args:
            prompt: User prompt

        Returns:
            Tuple[bool, List[str]]: (is_valid, error_messages)
        """
        errors = []

        if not prompt or not prompt.strip():
            errors.append("Prompt cannot be empty")

        if len(prompt) > 1000:
            errors.append("Prompt too long (max 1000 characters)")

        if len(prompt.split()) < 2:
            errors.append("Prompt too short (minimum 2 words)")

        # Check for potentially harmful content
        harmful_patterns = [
            r"\b(delete|remove|drop|truncate)\b",
            r"\b(password|secret|key)\b",
            r"\b(exec|eval|system)\b",
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, prompt.lower()):
                errors.append("Prompt contains potentially harmful content")
                break

        return len(errors) == 0, errors

    def enhance_prompt(self, prompt: str, context: PromptContext) -> str:
        """
        Enhance a prompt with context information.

        Args:
            prompt: Original prompt
            context: Context information

        Returns:
            str: Enhanced prompt
        """
        enhanced_prompt = prompt

        # Add user preferences if available
        if context.user_preferences:
            preferences_str = ", ".join(
                [f"{k}: {v}" for k, v in context.user_preferences.items()]
            )
            enhanced_prompt += f" [User preferences: {preferences_str}]"

        # Add system state if available
        if context.system_state:
            state_str = ", ".join(
                [f"{k}: {v}" for k, v in context.system_state.items()]
            )
            enhanced_prompt += f" [System state: {state_str}]"

        return enhanced_prompt


def get_prompt_processor() -> PromptProcessor:
    """Get a singleton instance of the prompt processor."""
    if not hasattr(get_prompt_processor, "_instance"):
        get_prompt_processor._instance = PromptProcessor()
    return get_prompt_processor._instance


class PromptRouterAgent:
    """Agent for routing prompts to appropriate handlers."""

    def __init__(self):
        """Initialize the prompt router agent."""
        self.processor = get_prompt_processor()
        self.logger = logging.getLogger(__name__)

    def handle_prompt(
        self, prompt: str, context: Optional[PromptContext] = None
    ) -> Dict[str, Any]:
        """
        Handle a user prompt and route to appropriate agent.

        Args:
            prompt: User's input prompt
            context: Optional context information

        Returns:
            Dict: Response with routing information and result
        """
        try:
            # Process the prompt
            processed = self.processor.process_prompt(prompt, context)

            # Route based on request type
            if processed.request_type == RequestType.INVESTMENT:
                return self._handle_investment_query(prompt, processed)
            elif processed.request_type == RequestType.FORECAST:
                return self._handle_forecast_query(prompt, processed)
            elif processed.request_type == RequestType.STRATEGY:
                return self._handle_strategy_query(prompt, processed)
            else:
                return self._handle_general_query(prompt, processed)

        except Exception as e:
            self.logger.error(f"Error handling prompt: {e}")
            return {
                "success": False,
                "message": f"Error processing prompt: {str(e)}",
                "routing_suggestions": ["GeneralAssistant"],
                "fallback_used": True,
            }

    def _handle_investment_query(
        self, prompt: str, processed: ProcessedPrompt
    ) -> Dict[str, Any]:
        """
        Handle investment-related queries with fallback to TopRankedForecastAgent.

        Args:
            prompt: Original prompt
            processed: Processed prompt information

        Returns:
            Dict: Response with investment recommendations
        """
        try:
            # Try to route to TopRankedForecastAgent
            from agents.top_ranked_forecast_agent import TopRankedForecastAgent

            agent = TopRankedForecastAgent()
            result = agent.run(prompt)

            return {
                "success": True,
                "message": f"Investment analysis completed. {result.get('message', '')}",
                "request_type": "investment",
                "confidence": processed.confidence,
                "routing_suggestions": processed.routing_suggestions,
                "result": result,
                "agent_used": "TopRankedForecastAgent",
            }

        except ImportError:
            self.logger.warning("TopRankedForecastAgent not available, using fallback")
            return self._fallback_investment_response(prompt, processed)
        except Exception as e:
            self.logger.error(f"Error in TopRankedForecastAgent: {e}")
            return self._fallback_investment_response(prompt, processed)

    def _fallback_investment_response(
        self, prompt: str, processed: ProcessedPrompt
    ) -> Dict[str, Any]:
        """
        Fallback response for investment queries when TopRankedForecastAgent is unavailable.

        Args:
            prompt: Original prompt
            processed: Processed prompt information

        Returns:
            Dict: Fallback response
        """
        return {
            "success": True,
            "message": "I can help you with investment decisions. Please try asking about specific stocks or use the forecast feature for detailed analysis.",
            "request_type": "investment",
            "confidence": processed.confidence,
            "routing_suggestions": processed.routing_suggestions,
            "fallback_used": True,
            "suggestions": [
                "Try: 'Forecast AAPL for next week'",
                "Try: 'Analyze TSLA performance'",
                "Try: 'What's the best strategy for tech stocks?'",
            ],
        }

    def _handle_forecast_query(
        self, prompt: str, processed: ProcessedPrompt
    ) -> Dict[str, Any]:
        """
        Handle forecast-related queries.

        Args:
            prompt: Original prompt
            processed: Processed prompt information

        Returns:
            Dict: Response with forecast information
        """
        return {
            "success": True,
            "message": f"Forecast request processed. Routing to: {', '.join(processed.routing_suggestions)}",
            "request_type": "forecast",
            "confidence": processed.confidence,
            "routing_suggestions": processed.routing_suggestions,
            "parameters": processed.extracted_parameters,
        }

    def _handle_strategy_query(
        self, prompt: str, processed: ProcessedPrompt
    ) -> Dict[str, Any]:
        """
        Handle strategy-related queries.

        Args:
            prompt: Original prompt
            processed: Processed prompt information

        Returns:
            Dict: Response with strategy information
        """
        return {
            "success": True,
            "message": f"Strategy request processed. Routing to: {', '.join(processed.routing_suggestions)}",
            "request_type": "strategy",
            "confidence": processed.confidence,
            "routing_suggestions": processed.routing_suggestions,
            "parameters": processed.extracted_parameters,
        }

    def _handle_general_query(
        self, prompt: str, processed: ProcessedPrompt
    ) -> Dict[str, Any]:
        """
        Handle general queries.

        Args:
            prompt: Original prompt
            processed: Processed prompt information

        Returns:
            Dict: Response with general information
        """
        return {
            "success": True,
            "message": f"General request processed. Routing to: {', '.join(processed.routing_suggestions)}",
            "request_type": processed.request_type.value,
            "confidence": processed.confidence,
            "routing_suggestions": processed.routing_suggestions,
            "parameters": processed.extracted_parameters,
        }

    def run(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Main run method for the agent.

        Args:
            prompt: User's input prompt
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Dict: Response from prompt handling
        """
        return self.handle_prompt(prompt)
