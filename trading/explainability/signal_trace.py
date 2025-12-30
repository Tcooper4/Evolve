"""
Signal Trace - Batch 21
Safe signal tracing with ast.literal_eval() instead of eval()
"""

import ast
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TraceLevel(Enum):
    """Trace levels for signal analysis."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SignalTrace:
    """Signal trace with metadata."""

    signal_id: str
    timestamp: datetime
    level: TraceLevel
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    safe_eval_result: Optional[Any] = None
    eval_error: Optional[str] = None


class SignalTracer:
    """
    Enhanced signal tracer with safe evaluation.

    Features:
    - Safe evaluation using ast.literal_eval() instead of eval()
    - Comprehensive error handling and logging
    - Signal trace history and analysis
    - Security-focused data processing
    """

    def __init__(
        self,
        enable_safe_eval: bool = True,
        max_trace_history: int = 1000,
        log_eval_errors: bool = True,
    ):
        """
        Initialize signal tracer.

        Args:
            enable_safe_eval: Enable safe evaluation mode
            max_trace_history: Maximum number of traces to keep
            log_eval_errors: Log evaluation errors
        """
        self.enable_safe_eval = enable_safe_eval
        self.max_trace_history = max_trace_history
        self.log_eval_errors = log_eval_errors

        # Trace history
        self.trace_history: List[SignalTrace] = []

        # Statistics
        self.stats = {
            "total_traces": 0,
            "safe_eval_success": 0,
            "safe_eval_failures": 0,
            "eval_errors": 0,
            "security_violations": 0,
        }

        logger.info(
            f"SignalTracer initialized with safe evaluation: {enable_safe_eval}"
        )

    def safe_eval(
        self, expression: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Optional[str]]:
        """
        Safely evaluate expression using ast.literal_eval().

        Args:
            expression: Expression to evaluate
            context: Optional context for evaluation

        Returns:
            Tuple of (result, error_message)
        """
        if not self.enable_safe_eval:
            logger.warning(
                "Safe evaluation disabled - using ast.literal_eval() anyway for security"
            )

        try:
            # First, try to parse the expression safely
            parsed_ast = ast.parse(expression, mode="eval")

            # Check if the AST contains only literals and basic operations
            if self._is_safe_ast(parsed_ast.body):
                # Use ast.literal_eval for safe evaluation
                result = ast.literal_eval(expression)
                self.stats["safe_eval_success"] += 1
                return result, None
            else:
                error_msg = f"Unsafe expression detected: {expression}"
                self.stats["security_violations"] += 1
                logger.warning(error_msg)
                return None, error_msg

        except (ValueError, SyntaxError) as e:
            error_msg = f"Syntax error in expression '{expression}': {str(e)}"
            self.stats["safe_eval_failures"] += 1
            if self.log_eval_errors:
                logger.error(error_msg)
            return None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error evaluating '{expression}': {str(e)}"
            self.stats["eval_errors"] += 1
            if self.log_eval_errors:
                logger.error(error_msg)
            return None, error_msg

    def _is_safe_ast(self, node: ast.AST) -> bool:
        """
        Check if AST node contains only safe operations.

        Args:
            node: AST node to check

        Returns:
            True if AST is safe for evaluation
        """
        if isinstance(node, ast.Constant):
            return True
        elif isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            # Check all elements in containers
            if isinstance(node, ast.List):
                return all(self._is_safe_ast(elt) for elt in node.elts)
            elif isinstance(node, ast.Tuple):
                return all(self._is_safe_ast(elt) for elt in node.elts)
            elif isinstance(node, ast.Dict):
                return all(self._is_safe_ast(k) for k in node.keys) and all(
                    self._is_safe_ast(v) for v in node.values
                )
            elif isinstance(node, ast.Set):
                return all(self._is_safe_ast(elt) for elt in node.elts)
        elif isinstance(node, ast.UnaryOp):
            return isinstance(
                node.op, (ast.UAdd, ast.USub, ast.Not)
            ) and self._is_safe_ast(node.operand)
        elif isinstance(node, ast.BinOp):
            return (
                isinstance(
                    node.op,
                    (
                        ast.Add,
                        ast.Sub,
                        ast.Mult,
                        ast.Div,
                        ast.Mod,
                        ast.Pow,
                        ast.LShift,
                        ast.RShift,
                        ast.BitOr,
                        ast.BitXor,
                        ast.BitAnd,
                        ast.FloorDiv,
                    ),
                )
                and self._is_safe_ast(node.left)
                and self._is_safe_ast(node.right)
            )
        elif isinstance(node, ast.Compare):
            return (
                all(
                    isinstance(
                        op,
                        (
                            ast.Eq,
                            ast.NotEq,
                            ast.Lt,
                            ast.LtE,
                            ast.Gt,
                            ast.GtE,
                            ast.Is,
                            ast.IsNot,
                            ast.In,
                            ast.NotIn,
                        ),
                    )
                    for op in node.ops
                )
                and self._is_safe_ast(node.left)
                and all(self._is_safe_ast(comp) for comp in node.comparators)
            )
        else:
            return False

    def trace_signal(
        self,
        signal_id: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        level: TraceLevel = TraceLevel.INFO,
    ) -> SignalTrace:
        """
        Create a signal trace entry.

        Args:
            signal_id: Unique signal identifier
            message: Trace message
            data: Associated data
            level: Trace level

        Returns:
            SignalTrace object
        """
        trace = SignalTrace(
            signal_id=signal_id,
            timestamp=datetime.now(),
            level=level,
            message=message,
            data=data or {},
        )

        self.trace_history.append(trace)
        self.stats["total_traces"] += 1

        # Limit trace history
        if len(self.trace_history) > self.max_trace_history:
            self.trace_history = self.trace_history[-self.max_trace_history :]

        logger.debug(f"Signal trace created: {signal_id} - {message}")
        return trace

    def trace_evaluation(
        self, signal_id: str, expression: str, context: Optional[Dict[str, Any]] = None
    ) -> SignalTrace:
        """
        Trace expression evaluation with safe eval.

        Args:
            signal_id: Signal identifier
            expression: Expression to evaluate
            context: Evaluation context

        Returns:
            SignalTrace with evaluation result
        """
        # Perform safe evaluation
        result, error = self.safe_eval(expression, context)

        # Create trace
        trace_data = {
            "expression": expression,
            "context": context,
            "result": result,
            "error": error,
        }

        level = TraceLevel.ERROR if error else TraceLevel.INFO
        message = f"Expression evaluation: {expression}"

        trace = self.trace_signal(signal_id, message, trace_data, level)
        trace.safe_eval_result = result
        trace.eval_error = error

        return trace

    def trace_signal_processing(
        self, signal_id: str, processing_steps: List[Dict[str, Any]]
    ) -> List[SignalTrace]:
        """
        Trace signal processing steps.

        Args:
            signal_id: Signal identifier
            processing_steps: List of processing steps

        Returns:
            List of SignalTrace objects
        """
        traces = []

        for i, step in enumerate(processing_steps):
            step_name = step.get("name", f"step_{i}")
            step_data = step.get("data", {})

            # Check for expressions that need evaluation
            if "expression" in step_data:
                trace = self.trace_evaluation(
                    signal_id, step_data["expression"], step_data.get("context")
                )
            else:
                trace = self.trace_signal(
                    signal_id, f"Processing step: {step_name}", step_data
                )

            traces.append(trace)

        return traces

    def get_signal_traces(
        self,
        signal_id: str,
        level: Optional[TraceLevel] = None,
        limit: Optional[int] = None,
    ) -> List[SignalTrace]:
        """
        Get traces for a specific signal.

        Args:
            signal_id: Signal identifier
            level: Filter by trace level
            limit: Maximum number of traces to return

        Returns:
            List of matching traces
        """
        traces = [t for t in self.trace_history if t.signal_id == signal_id]

        if level:
            traces = [t for t in traces if t.level == level]

        if limit:
            traces = traces[-limit:]

        return traces

    def analyze_signal_traces(self, signal_id: str) -> Dict[str, Any]:
        """
        Analyze traces for a signal.

        Args:
            signal_id: Signal identifier

        Returns:
            Analysis results
        """
        traces = self.get_signal_traces(signal_id)

        if not traces:
            return {"error": "No traces found for signal"}

        # Count by level
        level_counts = {}
        for level in TraceLevel:
            level_counts[level.value] = len([t for t in traces if t.level == level])

        # Find errors
        errors = [t for t in traces if t.level == TraceLevel.ERROR]

        # Find evaluation results
        eval_traces = [t for t in traces if t.safe_eval_result is not None]

        analysis = {
            "signal_id": signal_id,
            "total_traces": len(traces),
            "level_distribution": level_counts,
            "error_count": len(errors),
            "evaluation_count": len(eval_traces),
            "first_trace": traces[0].timestamp.isoformat() if traces else None,
            "last_trace": traces[-1].timestamp.isoformat() if traces else None,
            "recent_errors": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "message": t.message,
                    "error": t.eval_error,
                }
                for t in errors[-5:]  # Last 5 errors
            ],
        }

        return analysis

    def get_trace_statistics(self) -> Dict[str, Any]:
        """Get overall trace statistics."""
        stats = self.stats.copy()

        # Add trace level distribution
        level_counts = {}
        for level in TraceLevel:
            level_counts[level.value] = len(
                [t for t in self.trace_history if t.level == level]
            )

        stats["trace_level_distribution"] = level_counts
        stats["total_trace_history"] = len(self.trace_history)

        # Add recent activity
        if self.trace_history:
            stats["last_trace_time"] = self.trace_history[-1].timestamp.isoformat()
            stats["unique_signals"] = len(set(t.signal_id for t in self.trace_history))

        return stats

    def clear_trace_history(self, signal_id: Optional[str] = None):
        """
        Clear trace history.

        Args:
            signal_id: Clear traces for specific signal (all if None)
        """
        if signal_id:
            self.trace_history = [
                t for t in self.trace_history if t.signal_id != signal_id
            ]
            logger.info(f"Cleared trace history for signal: {signal_id}")
        else:
            self.trace_history.clear()
            logger.info("Cleared all trace history")

    def enable_safe_eval(self, enable: bool = True):
        """Enable or disable safe evaluation."""
        self.enable_safe_eval = enable
        logger.info(f"Safe evaluation {'enabled' if enable else 'disabled'}")


def create_signal_tracer(enable_safe_eval: bool = True) -> SignalTracer:
    """Factory function to create signal tracer."""
    return SignalTracer(enable_safe_eval=enable_safe_eval)
