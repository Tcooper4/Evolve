"""
Agent Thoughts Logger for recording decision rationale and enabling recall.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import uuid

logger = logging.getLogger(__name__)

@dataclass
class AgentThought:
    """Represents a single agent thought or decision rationale."""
    thought_id: str
    agent_name: str
    timestamp: str
    context: str
    decision: str
    rationale: str
    confidence: float
    alternatives_considered: List[str]
    factors: Dict[str, Any]
    outcome: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentThoughtsLogger:
    """
    Logger for recording agent decision rationale and enabling recall.
    
    This system allows agents to log their thought processes, decisions,
    and rationale for future analysis and improvement.
    """
    
    def __init__(self, log_file: str = "logs/agent_thoughts.json"):
        """
        Initialize the agent thoughts logger.
        
        Args:
            log_file: Path to the thoughts log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            self._initialize_log_file()
        
        logger.info(f"Initialized AgentThoughtsLogger with log file: {self.log_file}")
    
    def _initialize_log_file(self) -> None:
        """Initialize the log file with basic structure."""
        try:
            initial_data = {
                "agent_thoughts": [],
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "version": "1.0",
                    "description": "Agent decision rationale and thought process logs"
                }
            }
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
                
            logger.info(f"Initialized agent thoughts log file: {self.log_file}")
            
        except Exception as e:
            logger.error(f"Error initializing log file: {str(e)}")
    
    def log_thought(self,
                   agent_name: str,
                   context: str,
                   decision: str,
                   rationale: str,
                   confidence: float,
                   alternatives_considered: Optional[List[str]] = None,
                   factors: Optional[Dict[str, Any]] = None,
                   outcome: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log an agent thought or decision.
        
        Args:
            agent_name: Name of the agent
            context: Context of the decision
            decision: The decision made
            rationale: Reasoning behind the decision
            confidence: Confidence level (0.0 to 1.0)
            alternatives_considered: List of alternatives considered
            factors: Key factors that influenced the decision
            outcome: Outcome of the decision (if known)
            metadata: Additional metadata
            
        Returns:
            str: ID of the logged thought
        """
        try:
            thought_id = str(uuid.uuid4())
            
            thought = AgentThought(
                thought_id=thought_id,
                agent_name=agent_name,
                timestamp=datetime.now().isoformat(),
                context=context,
                decision=decision,
                rationale=rationale,
                confidence=confidence,
                alternatives_considered=alternatives_considered or [],
                factors=factors or {},
                outcome=outcome,
                metadata=metadata or {}
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
            
            logger.info(f"Logged thought for {agent_name}: {decision[:50]}...")
            
            return thought_id
            
        except Exception as e:
            logger.error(f"Error logging thought: {str(e)}")
            return ""
    
    def get_agent_thoughts(self, 
                          agent_name: Optional[str] = None,
                          limit: int = 50,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get agent thoughts with optional filtering.
        
        Args:
            agent_name: Filter by specific agent
            limit: Maximum number of thoughts to return
            start_date: Filter thoughts after this date (ISO format)
            end_date: Filter thoughts before this date (ISO format)
            
        Returns:
            List of agent thoughts
        """
        try:
            thoughts_data = self._load_thoughts()
            thoughts = thoughts_data["agent_thoughts"]
            
            # Filter by agent
            if agent_name:
                thoughts = [t for t in thoughts if t.get("agent_name") == agent_name]
            
            # Filter by date range
            if start_date:
                thoughts = [t for t in thoughts if t.get("timestamp", "") >= start_date]
            
            if end_date:
                thoughts = [t for t in thoughts if t.get("timestamp", "") <= end_date]
            
            # Sort by timestamp (newest first)
            thoughts.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Apply limit
            return thoughts[:limit]
            
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
            outcome: Outcome description
            
        Returns:
            bool: True if updated successfully
        """
        try:
            thoughts_data = self._load_thoughts()
            
            for thought in thoughts_data["agent_thoughts"]:
                if thought.get("thought_id") == thought_id:
                    thought["outcome"] = outcome
                    self._save_thoughts(thoughts_data)
                    logger.info(f"Updated outcome for thought {thought_id}")
                    return True
            
            logger.warning(f"Thought {thought_id} not found for outcome update")
            return False
            
        except Exception as e:
            logger.error(f"Error updating thought outcome: {str(e)}")
            return False
    
    def get_decision_patterns(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze decision patterns from logged thoughts.
        
        Args:
            agent_name: Filter by specific agent
            
        Returns:
            Dictionary containing pattern analysis
        """
        try:
            thoughts = self.get_agent_thoughts(agent_name=agent_name, limit=1000)
            
            if not thoughts:
                return {"message": "No thoughts found for analysis"}
            
            # Analyze patterns
            total_decisions = len(thoughts)
            avg_confidence = sum(t.get("confidence", 0) for t in thoughts) / total_decisions
            
            # Decision frequency
            decision_counts = {}
            for thought in thoughts:
                decision = thought.get("decision", "unknown")
                decision_counts[decision] = decision_counts.get(decision, 0) + 1
            
            # Factor analysis
            factor_frequency = {}
            for thought in thoughts:
                factors = thought.get("factors", {})
                for factor, value in factors.items():
                    if factor not in factor_frequency:
                        factor_frequency[factor] = []
                    factor_frequency[factor].append(value)
            
            # Calculate factor statistics
            factor_stats = {}
            for factor, values in factor_frequency.items():
                if values and all(isinstance(v, (int, float)) for v in values):
                    factor_stats[factor] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values)
                    }
            
            return {
                "total_decisions": total_decisions,
                "average_confidence": avg_confidence,
                "decision_frequency": decision_counts,
                "factor_statistics": factor_stats,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing decision patterns: {str(e)}")
            return {"error": str(e)}
    
    def get_agent_performance_summary(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Performance summary
        """
        try:
            thoughts = self.get_agent_thoughts(agent_name=agent_name, limit=1000)
            
            if not thoughts:
                return {"message": f"No thoughts found for agent {agent_name}"}
            
            # Calculate metrics
            total_decisions = len(thoughts)
            avg_confidence = sum(t.get("confidence", 0) for t in thoughts) / total_decisions
            
            # Recent performance (last 20 decisions)
            recent_thoughts = thoughts[:20]
            recent_avg_confidence = sum(t.get("confidence", 0) for t in recent_thoughts) / len(recent_thoughts)
            
            # Decision variety
            unique_decisions = len(set(t.get("decision", "") for t in thoughts))
            
            # Time-based analysis
            if len(thoughts) >= 2:
                first_thought = thoughts[-1]
                last_thought = thoughts[0]
                
                first_time = datetime.fromisoformat(first_thought.get("timestamp", ""))
                last_time = datetime.fromisoformat(last_thought.get("timestamp", ""))
                
                time_span = (last_time - first_time).total_seconds() / 3600  # hours
                decisions_per_hour = total_decisions / max(time_span, 1)
            else:
                decisions_per_hour = 0
            
            return {
                "agent_name": agent_name,
                "total_decisions": total_decisions,
                "average_confidence": avg_confidence,
                "recent_confidence": recent_avg_confidence,
                "unique_decisions": unique_decisions,
                "decisions_per_hour": decisions_per_hour,
                "first_decision": thoughts[-1].get("timestamp") if thoughts else None,
                "last_decision": thoughts[0].get("timestamp") if thoughts else None,
                "summary_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting agent performance summary: {str(e)}")
            return {"error": str(e)}
    
    def search_thoughts(self, 
                       query: str,
                       agent_name: Optional[str] = None,
                       limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search thoughts by content.
        
        Args:
            query: Search query
            agent_name: Filter by specific agent
            limit: Maximum number of results
            
        Returns:
            List of matching thoughts
        """
        try:
            thoughts = self.get_agent_thoughts(agent_name=agent_name, limit=1000)
            query_lower = query.lower()
            
            matching_thoughts = []
            
            for thought in thoughts:
                # Search in context, decision, and rationale
                context = thought.get("context", "").lower()
                decision = thought.get("decision", "").lower()
                rationale = thought.get("rationale", "").lower()
                
                if (query_lower in context or 
                    query_lower in decision or 
                    query_lower in rationale):
                    matching_thoughts.append(thought)
            
            return matching_thoughts[:limit]
            
        except Exception as e:
            logger.error(f"Error searching thoughts: {str(e)}")
            return []
    
    def _load_thoughts(self) -> Dict[str, Any]:
        """Load thoughts from file."""
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading thoughts: {str(e)}")
            return {"agent_thoughts": [], "metadata": {}}
    
    def _save_thoughts(self, thoughts_data: Dict[str, Any]) -> None:
        """Save thoughts to file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(thoughts_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving thoughts: {str(e)}")
    
    def clear_thoughts(self, agent_name: Optional[str] = None) -> bool:
        """
        Clear thoughts, optionally for a specific agent.
        
        Args:
            agent_name: If provided, clear only thoughts for this agent
            
        Returns:
            bool: True if cleared successfully
        """
        try:
            if agent_name:
                # Clear thoughts for specific agent
                thoughts_data = self._load_thoughts()
                thoughts_data["agent_thoughts"] = [
                    t for t in thoughts_data["agent_thoughts"]
                    if t.get("agent_name") != agent_name
                ]
                self._save_thoughts(thoughts_data)
                logger.info(f"Cleared thoughts for agent {agent_name}")
            else:
                # Clear all thoughts
                self._initialize_log_file()
                logger.info("Cleared all agent thoughts")
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing thoughts: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about logged thoughts."""
        try:
            thoughts_data = self._load_thoughts()
            thoughts = thoughts_data["agent_thoughts"]
            
            if not thoughts:
                return {"message": "No thoughts logged yet"}
            
            # Basic statistics
            total_thoughts = len(thoughts)
            unique_agents = len(set(t.get("agent_name", "") for t in thoughts))
            
            # Time range
            timestamps = [t.get("timestamp", "") for t in thoughts if t.get("timestamp")]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
            else:
                earliest = latest = None
            
            # Average confidence
            avg_confidence = sum(t.get("confidence", 0) for t in thoughts) / total_thoughts
            
            return {
                "total_thoughts": total_thoughts,
                "unique_agents": unique_agents,
                "average_confidence": avg_confidence,
                "earliest_thought": earliest,
                "latest_thought": latest,
                "statistics_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return {"error": str(e)}

# Global instance
agent_thoughts_logger = AgentThoughtsLogger()

def log_agent_thought(agent_name: str,
                     context: str,
                     decision: str,
                     rationale: str,
                     confidence: float,
                     **kwargs) -> str:
    """
    Convenience function to log an agent thought.
    
    Args:
        agent_name: Name of the agent
        context: Context of the decision
        decision: The decision made
        rationale: Reasoning behind the decision
        confidence: Confidence level (0.0 to 1.0)
        **kwargs: Additional arguments for AgentThoughtsLogger.log_thought
        
    Returns:
        str: ID of the logged thought
    """
    return agent_thoughts_logger.log_thought(
        agent_name=agent_name,
        context=context,
        decision=decision,
        rationale=rationale,
        confidence=confidence,
        **kwargs
    ) 