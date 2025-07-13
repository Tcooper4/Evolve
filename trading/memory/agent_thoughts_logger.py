"""
Agent Thoughts Logger for recording decision rationale and enabling recall.
Enhanced with timestamping and task ID tagging for better traceability.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentThought:
    """Represents a single agent thought or decision rationale."""

    thought_id: str
    agent_name: str
    task_id: str  # Task identifier for traceability
    timestamp: str
    context: str
    decision: str
    rationale: str
    confidence: float
    alternatives_considered: List[str]
    factors: Dict[str, Any]
    outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    execution_time_ms: Optional[int] = None  # Execution time in milliseconds
    parent_thought_id: Optional[str] = None  # For thought chains
    session_id: Optional[str] = None  # Session identifier
    correlation_id: Optional[str] = None  # For tracing related thoughts


@dataclass
class TaskContext:
    """Context information for a task."""

    task_id: str
    task_type: str
    priority: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    parent_task_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentThoughtsLogger:
    """
    Logger for recording agent decision rationale and enabling recall.

    This system allows agents to log their thought processes, decisions,
    and rationale for future analysis and improvement.
    Enhanced with task tracking and improved timestamp management.
    """

    def __init__(self, log_file: str = "logs/agent_thoughts.json", task_log_file: str = "logs/agent_tasks.json"):
        """
        Initialize the agent thoughts logger.

        Args:
            log_file: Path to the thoughts log file
            task_log_file: Path to the task log file
        """
        self.log_file = Path(log_file)
        self.task_log_file = Path(task_log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()
        self._task_cache = {}  # Cache for active tasks

        # Initialize log files if they don't exist
        if not self.log_file.exists():
            self._initialize_log_file()

        if not self.task_log_file.exists():
            self._initialize_task_log_file()

        logger.info(f"Initialized AgentThoughtsLogger with log file: {self.log_file}")

    def _initialize_log_file(self) -> None:
        """Initialize the log file with basic structure."""
        try:
            initial_data = {
                "agent_thoughts": [],
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "version": "2.0",
                    "description": "Agent decision rationale and thought process logs with task tracking",
                },
            }

            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

            logger.info(f"Initialized agent thoughts log file: {self.log_file}")

        except Exception as e:
            logger.error(f"Error initializing log file: {str(e)}")

    def _initialize_task_log_file(self) -> None:
        """Initialize the task log file with basic structure."""
        try:
            initial_data = {
                "agent_tasks": [],
                "metadata": {
                    "created": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0",
                    "description": "Agent task tracking and context logs",
                },
            }

            with open(self.task_log_file, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

            logger.info(f"Initialized agent tasks log file: {self.task_log_file}")

        except Exception as e:
            logger.error(f"Error initializing task log file: {str(e)}")

    def create_task(
        self,
        task_type: str,
        agent_name: str,
        priority: int = 1,
        parent_task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new task for tracking.

        Args:
            task_type: Type of task
            agent_name: Name of the agent
            priority: Task priority (1-10, higher is more important)
            parent_task_id: ID of parent task if this is a subtask
            session_id: Session identifier
            metadata: Additional task metadata

        Returns:
            str: Task ID
        """
        try:
            task_id = str(uuid.uuid4())

            task_context = TaskContext(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                parent_task_id=parent_task_id,
                session_id=session_id,
                metadata=metadata or {},
            )

            # Add to cache
            with self._lock:
                self._task_cache[task_id] = task_context

            # Log to file
            tasks_data = self._load_tasks()
            tasks_data["agent_tasks"].append(asdict(task_context))

            # Keep only last 1000 tasks
            if len(tasks_data["agent_tasks"]) > 1000:
                tasks_data["agent_tasks"] = tasks_data["agent_tasks"][-1000:]

            self._save_tasks(tasks_data)

            logger.info(f"Created task {task_id} for {agent_name}: {task_type}")
            return task_id

        except Exception as e:
            logger.error(f"Error creating task: {str(e)}")
            return ""

    def start_task(self, task_id: str) -> bool:
        """
        Mark a task as started.

        Args:
            task_id: Task ID to start

        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                if task_id in self._task_cache:
                    self._task_cache[task_id].started_at = datetime.now(timezone.utc).isoformat()
                    self._task_cache[task_id].status = "running"

            # Update in file
            tasks_data = self._load_tasks()
            for task in tasks_data["agent_tasks"]:
                if task.get("task_id") == task_id:
                    task["started_at"] = datetime.now(timezone.utc).isoformat()
                    task["status"] = "running"
                    break

            self._save_tasks(tasks_data)
            return True

        except Exception as e:
            logger.error(f"Error starting task: {str(e)}")
            return False

    def complete_task(self, task_id: str, status: str = "completed") -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task ID to complete
            status: Completion status ("completed" or "failed")

        Returns:
            bool: True if successful
        """
        try:
            with self._lock:
                if task_id in self._task_cache:
                    self._task_cache[task_id].completed_at = datetime.now(timezone.utc).isoformat()
                    self._task_cache[task_id].status = status

            # Update in file
            tasks_data = self._load_tasks()
            for task in tasks_data["agent_tasks"]:
                if task.get("task_id") == task_id:
                    task["completed_at"] = datetime.now(timezone.utc).isoformat()
                    task["status"] = status
                    break

            self._save_tasks(tasks_data)
            return True

        except Exception as e:
            logger.error(f"Error completing task: {str(e)}")
            return False

    def log_thought(
        self,
        agent_name: str,
        task_id: str,
        context: str,
        decision: str,
        rationale: str,
        confidence: float,
        alternatives_considered: Optional[List[str]] = None,
        factors: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        execution_time_ms: Optional[int] = None,
        parent_thought_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Log an agent thought or decision with enhanced traceability.

        Args:
            agent_name: Name of the agent
            task_id: Task identifier for traceability
            context: Context of the decision
            decision: The decision made
            rationale: Reasoning behind the decision
            confidence: Confidence level (0.0 to 1.0)
            alternatives_considered: List of alternatives considered
            factors: Key factors that influenced the decision
            outcome: Outcome of the decision (if known)
            metadata: Additional metadata
            tags: List of tags for categorization and filtering
            execution_time_ms: Execution time in milliseconds
            parent_thought_id: ID of parent thought for thought chains
            session_id: Session identifier
            correlation_id: Correlation ID for tracing related thoughts

        Returns:
            str: ID of the logged thought
        """
        try:
            thought_id = str(uuid.uuid4())
            current_time = datetime.now(timezone.utc).isoformat()

            thought = AgentThought(
                thought_id=thought_id,
                agent_name=agent_name,
                task_id=task_id,
                timestamp=current_time,
                context=context,
                decision=decision,
                rationale=rationale,
                confidence=confidence,
                alternatives_considered=alternatives_considered or [],
                factors=factors or {},
                outcome=outcome,
                metadata=metadata or {},
                tags=tags or [],
                execution_time_ms=execution_time_ms,
                parent_thought_id=parent_thought_id,
                session_id=session_id,
                correlation_id=correlation_id,
            )

            # Load existing thoughts
            thoughts_data = self._load_thoughts()

            # Add new thought
            thoughts_data["agent_thoughts"].append(asdict(thought))

            # Keep only last 1000 thoughts to prevent file bloat
            if len(thoughts_data["agent_thoughts"]) > 1000:
                thoughts_data["agent_thoughts"] = thoughts_data["agent_thoughts"][-1000:]

            # Save thoughts
            self._save_thoughts(thoughts_data)

            logger.info(f"Logged thought {thought_id} for {agent_name} (task: {task_id}): {decision[:50]}...")

            return thought_id

        except Exception as e:
            logger.error(f"Error logging thought: {str(e)}")
            return ""

    def log_thought_with_timing(
        self,
        agent_name: str,
        task_id: str,
        context: str,
        decision: str,
        rationale: str,
        confidence: float,
        start_time: float,
        **kwargs,
    ) -> str:
        """
        Log a thought with automatic execution time calculation.

        Args:
            agent_name: Name of the agent
            task_id: Task identifier
            context: Context of the decision
            decision: The decision made
            rationale: Reasoning behind the decision
            confidence: Confidence level (0.0 to 1.0)
            start_time: Start time from time.time()
            **kwargs: Additional arguments for log_thought

        Returns:
            str: ID of the logged thought
        """
        execution_time_ms = int((time.time() - start_time) * 1000)
        return self.log_thought(
            agent_name=agent_name,
            task_id=task_id,
            context=context,
            decision=decision,
            rationale=rationale,
            confidence=confidence,
            execution_time_ms=execution_time_ms,
            **kwargs,
        )

    def add_tags_to_thought(self, thought_id: str, tags: List[str]) -> bool:
        """
        Add tags to an existing thought.

        Args:
            thought_id: ID of the thought to tag
            tags: List of tags to add

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            thoughts_data = self._load_thoughts()

            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("thought_id") == thought_id:
                    existing_tags = thought.get("tags", [])
                    # Add new tags without duplicates
                    for tag in tags:
                        if tag not in existing_tags:
                            existing_tags.append(tag)
                    thought["tags"] = existing_tags

                    self._save_thoughts(thoughts_data)
                    logger.info(f"Added tags {tags} to thought {thought_id}")
                    return True

            logger.warning(f"Thought {thought_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error adding tags to thought: {str(e)}")
            return False

    def remove_tags_from_thought(self, thought_id: str, tags: List[str]) -> bool:
        """
        Remove tags from an existing thought.

        Args:
            thought_id: ID of the thought to untag
            tags: List of tags to remove

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            thoughts_data = self._load_thoughts()

            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("thought_id") == thought_id:
                    existing_tags = thought.get("tags", [])
                    # Remove specified tags
                    thought["tags"] = [tag for tag in existing_tags if tag not in tags]

                    self._save_thoughts(thoughts_data)
                    logger.info(f"Removed tags {tags} from thought {thought_id}")
                    return True

            logger.warning(f"Thought {thought_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error removing tags from thought: {str(e)}")
            return False

    def search_thoughts_by_tags(
        self,
        tags: List[str],
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        match_all: bool = False,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search thoughts by tags with optional task filtering.

        Args:
            tags: List of tags to search for
            agent_name: Filter by agent name
            task_id: Filter by task ID
            match_all: If True, all tags must match; if False, any tag can match
            limit: Maximum number of results

        Returns:
            List of matching thoughts
        """
        try:
            thoughts_data = self._load_thoughts()
            matching_thoughts = []

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                thought_tags = thought.get("tags", [])

                # Check tag matching
                if match_all:
                    if not all(tag in thought_tags for tag in tags):
                        continue
                else:
                    if not any(tag in thought_tags for tag in tags):
                        continue

                matching_thoughts.append(thought)

                if len(matching_thoughts) >= limit:
                    break

            return matching_thoughts

        except Exception as e:
            logger.error(f"Error searching thoughts by tags: {str(e)}")
            return []

    def get_thoughts_by_task(self, task_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all thoughts for a specific task.

        Args:
            task_id: Task ID to filter by
            limit: Maximum number of results

        Returns:
            List of thoughts for the task
        """
        try:
            thoughts_data = self._load_thoughts()
            task_thoughts = []

            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("task_id") == task_id:
                    task_thoughts.append(thought)

                    if len(task_thoughts) >= limit:
                        break

            return task_thoughts

        except Exception as e:
            logger.error(f"Error getting thoughts by task: {str(e)}")
            return []

    def get_thought_chain(self, thought_id: str, max_depth: int = 5) -> List[Dict[str, Any]]:
        """
        Get a chain of related thoughts starting from a given thought.

        Args:
            thought_id: Starting thought ID
            max_depth: Maximum depth to traverse

        Returns:
            List of thoughts in the chain
        """
        try:
            thoughts_data = self._load_thoughts()
            thought_map = {t["thought_id"]: t for t in thoughts_data["agent_thoughts"]}

            chain = []
            current_thought_id = thought_id
            depth = 0

            while current_thought_id and depth < max_depth:
                if current_thought_id in thought_map:
                    thought = thought_map[current_thought_id]
                    chain.append(thought)
                    current_thought_id = thought.get("parent_thought_id")
                    depth += 1
                else:
                    break

            return chain

        except Exception as e:
            logger.error(f"Error getting thought chain: {str(e)}")
            return []

    def get_all_tags(self, agent_name: Optional[str] = None, task_id: Optional[str] = None) -> List[str]:
        """
        Get all unique tags used in thoughts.

        Args:
            agent_name: Filter by agent name
            task_id: Filter by task ID

        Returns:
            List of unique tags
        """
        try:
            thoughts_data = self._load_thoughts()
            all_tags = set()

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                tags = thought.get("tags", [])
                all_tags.update(tags)

            return sorted(list(all_tags))

        except Exception as e:
            logger.error(f"Error getting all tags: {str(e)}")
            return []

    def get_thoughts_by_confidence_range(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 1.0,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get thoughts within a confidence range.

        Args:
            min_confidence: Minimum confidence level
            max_confidence: Maximum confidence level
            agent_name: Filter by agent name
            task_id: Filter by task ID
            limit: Maximum number of results

        Returns:
            List of thoughts in confidence range
        """
        try:
            thoughts_data = self._load_thoughts()
            matching_thoughts = []

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                confidence = thought.get("confidence", 0.0)
                if min_confidence <= confidence <= max_confidence:
                    matching_thoughts.append(thought)

                    if len(matching_thoughts) >= limit:
                        break

            return matching_thoughts

        except Exception as e:
            logger.error(f"Error getting thoughts by confidence range: {str(e)}")
            return []

    def get_thoughts_by_context_keywords(
        self, keywords: List[str], agent_name: Optional[str] = None, task_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search thoughts by keywords in context.

        Args:
            keywords: List of keywords to search for
            agent_name: Filter by agent name
            task_id: Filter by task ID
            limit: Maximum number of results

        Returns:
            List of matching thoughts
        """
        try:
            thoughts_data = self._load_thoughts()
            matching_thoughts = []

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                context = thought.get("context", "").lower()
                if any(keyword.lower() in context for keyword in keywords):
                    matching_thoughts.append(thought)

                    if len(matching_thoughts) >= limit:
                        break

            return matching_thoughts

        except Exception as e:
            logger.error(f"Error searching thoughts by keywords: {str(e)}")
            return []

    def get_agent_thoughts(
        self,
        agent_name: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get agent thoughts with various filters.

        Args:
            agent_name: Filter by agent name
            task_id: Filter by task ID
            limit: Maximum number of results
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)

        Returns:
            List of filtered thoughts
        """
        try:
            thoughts_data = self._load_thoughts()
            filtered_thoughts = []

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                # Date filtering
                if start_date or end_date:
                    thought_timestamp = thought.get("timestamp", "")
                    if start_date and thought_timestamp < start_date:
                        continue
                    if end_date and thought_timestamp > end_date:
                        continue

                filtered_thoughts.append(thought)

                if len(filtered_thoughts) >= limit:
                    break

            return filtered_thoughts

        except Exception as e:
            logger.error(f"Error getting agent thoughts: {str(e)}")
            return []

    def get_thought_by_id(self, thought_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific thought by ID.

        Args:
            thought_id: ID of the thought to retrieve

        Returns:
            Thought data or None if not found
        """
        try:
            thoughts_data = self._load_thoughts()

            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("thought_id") == thought_id:
                    return thought

            return None

        except Exception as e:
            logger.error(f"Error getting thought by ID: {str(e)}")
            return None

    def update_thought_outcome(self, thought_id: str, outcome: str) -> bool:
        """
        Update the outcome of a thought.

        Args:
            thought_id: ID of the thought to update
            outcome: New outcome

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            thoughts_data = self._load_thoughts()

            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("thought_id") == thought_id:
                    thought["outcome"] = outcome
                    self._save_thoughts(thoughts_data)
                    logger.info(f"Updated outcome for thought {thought_id}")
                    return True

            logger.warning(f"Thought {thought_id} not found")
            return False

        except Exception as e:
            logger.error(f"Error updating thought outcome: {str(e)}")
            return False

    def get_decision_patterns(self, agent_name: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze decision patterns for an agent or task.

        Args:
            agent_name: Filter by agent name
            task_id: Filter by task ID

        Returns:
            Dictionary with pattern analysis
        """
        try:
            thoughts = self.get_agent_thoughts(agent_name=agent_name, task_id=task_id, limit=1000)

            if not thoughts:
                return {"message": "No thoughts found for analysis"}

            # Calculate patterns
            total_thoughts = len(thoughts)
            avg_confidence = sum(t.get("confidence", 0) for t in thoughts) / total_thoughts

            # Most common tags
            tag_counts = {}
            for thought in thoughts:
                for tag in thought.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            common_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Confidence distribution
            confidence_ranges = {
                "low": len([t for t in thoughts if t.get("confidence", 0) < 0.3]),
                "medium": len([t for t in thoughts if 0.3 <= t.get("confidence", 0) < 0.7]),
                "high": len([t for t in thoughts if t.get("confidence", 0) >= 0.7]),
            }

            return {
                "total_thoughts": total_thoughts,
                "average_confidence": avg_confidence,
                "common_tags": common_tags,
                "confidence_distribution": confidence_ranges,
                "agent_name": agent_name,
                "task_id": task_id,
            }

        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {str(e)}")
            return {"error": str(e)}

    def get_agent_performance_summary(self, agent_name: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance summary for an agent.

        Args:
            agent_name: Name of the agent
            task_id: Optional task ID filter

        Returns:
            Dictionary with performance summary
        """
        try:
            thoughts = self.get_agent_thoughts(agent_name=agent_name, task_id=task_id, limit=1000)

            if not thoughts:
                return {"message": f"No thoughts found for agent {agent_name}"}

            # Calculate metrics
            total_thoughts = len(thoughts)
            avg_confidence = sum(t.get("confidence", 0) for t in thoughts) / total_thoughts

            # Execution time analysis
            execution_times = [t.get("execution_time_ms", 0) for t in thoughts if t.get("execution_time_ms")]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

            # Success rate (thoughts with positive outcomes)
            successful_thoughts = len(
                [t for t in thoughts if t.get("outcome") and "success" in t.get("outcome", "").lower()]
            )
            success_rate = successful_thoughts / total_thoughts if total_thoughts > 0 else 0

            return {
                "agent_name": agent_name,
                "task_id": task_id,
                "total_thoughts": total_thoughts,
                "average_confidence": avg_confidence,
                "average_execution_time_ms": avg_execution_time,
                "success_rate": success_rate,
                "successful_thoughts": successful_thoughts,
            }

        except Exception as e:
            logger.error(f"Error getting agent performance summary: {str(e)}")
            return {"error": str(e)}

    def search_thoughts(
        self, query: str, agent_name: Optional[str] = None, task_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search thoughts by text query.

        Args:
            query: Search query
            agent_name: Filter by agent name
            task_id: Filter by task ID
            limit: Maximum number of results

        Returns:
            List of matching thoughts
        """
        try:
            thoughts_data = self._load_thoughts()
            matching_thoughts = []
            query_lower = query.lower()

            for thought in thoughts_data["agent_thoughts"]:
                # Apply filters
                if agent_name and thought.get("agent_name") != agent_name:
                    continue
                if task_id and thought.get("task_id") != task_id:
                    continue

                # Search in context, decision, and rationale
                context = thought.get("context", "").lower()
                decision = thought.get("decision", "").lower()
                rationale = thought.get("rationale", "").lower()

                if query_lower in context or query_lower in decision or query_lower in rationale:
                    matching_thoughts.append(thought)

                    if len(matching_thoughts) >= limit:
                        break

            return matching_thoughts

        except Exception as e:
            logger.error(f"Error searching thoughts: {str(e)}")
            return []

    def _load_thoughts(self) -> Dict[str, Any]:
        """Load thoughts from file."""
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading thoughts: {str(e)}")
            return {"agent_thoughts": [], "metadata": {}}

    def _save_thoughts(self, thoughts_data: Dict[str, Any]) -> None:
        """Save thoughts to file."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(thoughts_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving thoughts: {str(e)}")

    def _load_tasks(self) -> Dict[str, Any]:
        """Load tasks from file."""
        try:
            with open(self.task_log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading tasks: {str(e)}")
            return {"agent_tasks": [], "metadata": {}}

    def _save_tasks(self, tasks_data: Dict[str, Any]) -> None:
        """Save tasks to file."""
        try:
            with open(self.task_log_file, "w", encoding="utf-8") as f:
                json.dump(tasks_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tasks: {str(e)}")

    def clear_thoughts(self, agent_name: Optional[str] = None, task_id: Optional[str] = None) -> bool:
        """
        Clear thoughts, optionally filtered by agent or task.

        Args:
            agent_name: Clear thoughts for specific agent
            task_id: Clear thoughts for specific task

        Returns:
            bool: True if successful
        """
        try:
            thoughts_data = self._load_thoughts()

            if agent_name or task_id:
                # Filter out thoughts to keep
                filtered_thoughts = []
                for thought in thoughts_data["agent_thoughts"]:
                    if agent_name and thought.get("agent_name") == agent_name:
                        continue
                    if task_id and thought.get("task_id") == task_id:
                        continue
                    filtered_thoughts.append(thought)

                thoughts_data["agent_thoughts"] = filtered_thoughts
            else:
                # Clear all thoughts
                thoughts_data["agent_thoughts"] = []

            self._save_thoughts(thoughts_data)
            logger.info(f"Cleared thoughts for agent: {agent_name}, task: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing thoughts: {str(e)}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about thoughts and tasks.

        Returns:
            Dictionary with statistics
        """
        try:
            thoughts_data = self._load_thoughts()
            tasks_data = self._load_tasks()

            total_thoughts = len(thoughts_data["agent_thoughts"])
            total_tasks = len(tasks_data["agent_tasks"])

            # Agent statistics
            agent_counts = {}
            for thought in thoughts_data["agent_thoughts"]:
                agent = thought.get("agent_name", "unknown")
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

            # Task statistics
            task_status_counts = {}
            for task in tasks_data["agent_tasks"]:
                status = task.get("status", "unknown")
                task_status_counts[status] = task_status_counts.get(status, 0) + 1

            # Tag statistics
            tag_counts = {}
            for thought in thoughts_data["agent_thoughts"]:
                for tag in thought.get("tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            return {
                "total_thoughts": total_thoughts,
                "total_tasks": total_tasks,
                "agent_counts": agent_counts,
                "task_status_counts": task_status_counts,
                "top_tags": sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10],
                "unique_agents": len(agent_counts),
                "unique_tags": len(tag_counts),
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}


# Global instance
_thoughts_logger = None
_logger_lock = threading.RLock()


def get_thoughts_logger() -> AgentThoughtsLogger:
    """Get the global thoughts logger instance."""
    global _thoughts_logger
    with _logger_lock:
        if _thoughts_logger is None:
            _thoughts_logger = AgentThoughtsLogger()
        return _thoughts_logger


def log_agent_thought(
    agent_name: str, task_id: str, context: str, decision: str, rationale: str, confidence: float, **kwargs
) -> str:
    """
    Convenience function to log an agent thought.

    Args:
        agent_name: Name of the agent
        task_id: Task identifier
        context: Context of the decision
        decision: The decision made
        rationale: Reasoning behind the decision
        confidence: Confidence level (0.0 to 1.0)
        **kwargs: Additional arguments for log_thought

    Returns:
        str: ID of the logged thought
    """
    return get_thoughts_logger().log_thought(
        agent_name=agent_name,
        task_id=task_id,
        context=context,
        decision=decision,
        rationale=rationale,
        confidence=confidence,
        **kwargs,
    )
