"""Code review agent for auditing and fixing forecast logic and strategies."""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from trading.base_agent import BaseMetaAgent
from trading.models import BaseModel
from trading.strategies import StrategyManager
from trading.memory.performance_memory import PerformanceMemory

class CodeReviewAgent(BaseMetaAgent):
    """Agent for auditing and fixing forecast logic and strategies."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the code review agent.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("code_review", config)
        
        # Initialize components
        self.strategy_manager = StrategyManager()
        self.performance_memory = PerformanceMemory()
        
        # Setup code analysis tools
        self.analysis_tools = {
            "complexity": self._analyze_complexity,
            "coverage": self._analyze_coverage,
            "performance": self._analyze_performance,
            "security": self._analyze_security
        }
    
    def run(self) -> Dict[str, Any]:
        """Run code review and collect results.
        
        Returns:
            Dict containing review results and suggested fixes
        """
        results = {
            "forecast_logic": self._review_forecast_logic(),
            "strategies": self._review_strategies(),
            "suggested_fixes": []
        }
        
        # Generate fixes based on review
        fixes = self._generate_fixes(results)
        results["suggested_fixes"] = fixes
        
        # Log results
        self.log_action("Code review completed", results)
        
        return results
    
    def _review_forecast_logic(self) -> Dict[str, Any]:
        """Review forecast logic in models.
        
        Returns:
            Dict containing review results
        """
        results = {}
        
        # Analyze each model
        for model_name in self._get_model_files():
            try:
                model_results = self._analyze_model(model_name)
                results[model_name] = model_results
            except Exception as e:
                self.logger.error(f"Error analyzing model {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        return results
    
    def _review_strategies(self) -> Dict[str, Any]:
        """Review trading strategies.
        
        Returns:
            Dict containing review results
        """
        results = {}
        
        # Analyze each strategy
        for strategy_name in self.strategy_manager.get_strategy_names():
            try:
                strategy_results = self._analyze_strategy(strategy_name)
                results[strategy_name] = strategy_results
            except Exception as e:
                self.logger.error(f"Error analyzing strategy {strategy_name}: {str(e)}")
                results[strategy_name] = {"error": str(e)}
        
        return results
    
    def _analyze_model(self, model_name: str) -> Dict[str, Any]:
        """Analyze a specific model.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dict containing analysis results
        """
        results = {}
        
        # Get model file
        model_file = self._get_model_file(model_name)
        if not model_file:
            return {"error": "Model file not found"}
        
        # Analyze code
        with open(model_file) as f:
            tree = ast.parse(f.read())
        
        # Run analysis tools
        for tool_name, tool_func in self.analysis_tools.items():
            try:
                tool_results = tool_func(tree)
                results[tool_name] = tool_results
            except Exception as e:
                self.logger.error(f"Error running {tool_name} analysis: {str(e)}")
                results[tool_name] = {"error": str(e)}
        
        return results
    
    def _analyze_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Analyze a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to analyze
            
        Returns:
            Dict containing analysis results
        """
        results = {}
        
        # Get strategy
        try:
            strategy = self.strategy_manager.get_strategy(strategy_name)
        except Exception as e:
            return {"error": f"Failed to get strategy: {str(e)}"}
        
        # Analyze performance
        metrics = self.performance_memory.get_metrics(strategy_name)
        if metrics:
            results["performance"] = {
                "accuracy": metrics.get("accuracy"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown")
            }
        
        # Analyze code
        strategy_file = self._get_strategy_file(strategy_name)
        if strategy_file:
            with open(strategy_file) as f:
                tree = ast.parse(f.read())
            
            # Run analysis tools
            for tool_name, tool_func in self.analysis_tools.items():
                try:
                    tool_results = tool_func(tree)
                    results[tool_name] = tool_results
                except Exception as e:
                    self.logger.error(f"Error running {tool_name} analysis: {str(e)}")
                    results[tool_name] = {"error": str(e)}
        
        return results
    
    def _generate_fixes(self, review_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate fixes based on review results.
        
        Args:
            review_results: Results from code review
            
        Returns:
            List of suggested fixes
        """
        fixes = []
        
        # Analyze forecast logic
        for model_name, results in review_results["forecast_logic"].items():
            if "error" in results:
                continue
            
            # Check complexity
            if results.get("complexity", {}).get("score", 0) > 0.8:
                fixes.append({
                    "type": "complexity",
                    "target": model_name,
                    "description": "High complexity detected",
                    "suggestion": "Consider refactoring into smaller functions"
                })
            
            # Check coverage
            if results.get("coverage", {}).get("score", 0) < 0.8:
                fixes.append({
                    "type": "coverage",
                    "target": model_name,
                    "description": "Low test coverage",
                    "suggestion": "Add more unit tests"
                })
        
        # Analyze strategies
        for strategy_name, results in review_results["strategies"].items():
            if "error" in results:
                continue
            
            # Check performance
            if results.get("performance", {}).get("accuracy", 0) < 0.6:
                fixes.append({
                    "type": "performance",
                    "target": strategy_name,
                    "description": "Low accuracy",
                    "suggestion": "Review and optimize strategy parameters"
                })
        
        return fixes
    
    def _get_model_files(self) -> List[str]:
        """Get list of model files.
        
        Returns:
            List of model file names
        """
        model_dir = Path("trading/models")
        return [f.stem for f in model_dir.glob("*.py") if f.stem != "__init__"]
    
    def _get_model_file(self, model_name: str) -> Optional[Path]:
        """Get path to model file.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to model file if found
        """
        model_file = Path(f"trading/models/{model_name}.py")
        return model_file if model_file.exists() else None
    
    def _get_strategy_file(self, strategy_name: str) -> Optional[Path]:
        """Get path to strategy file.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Path to strategy file if found
        """
        strategy_file = Path(f"trading/strategies/{strategy_name}.py")
        return strategy_file if strategy_file.exists() else None
    
    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dict containing complexity analysis results
        """
        # Implement complexity analysis
        return {"score": 0.5}  # Placeholder
    
    def _analyze_coverage(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze test coverage.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dict containing coverage analysis results
        """
        # Implement coverage analysis
        return {"score": 0.7}  # Placeholder
    
    def _analyze_performance(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code performance.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dict containing performance analysis results
        """
        # Implement performance analysis
        return {"score": 0.8}  # Placeholder
    
    def _analyze_security(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code security.
        
        Args:
            tree: AST of the code
            
        Returns:
            Dict containing security analysis results
        """
        # Implement security analysis
        return {"score": 0.9}  # Placeholder 