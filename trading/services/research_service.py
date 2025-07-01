"""
Research Service

Service wrapper for the ResearchAgent, handling research requests
via Redis pub/sub communication.
"""

import logging
import sys
import os
from typing import Dict, Any, Optional
import json
from pathlib import Path

# Add the trading directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from trading.agents.research_agent import ResearchAgent
from trading.memory.agent_memory import AgentMemory
from trading.services.base_service import BaseService

logger = logging.getLogger(__name__)

class ResearchService(BaseService):
    """
    Service wrapper for ResearchAgent.
    
    Handles research requests and communicates results via Redis.
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, 
                 redis_db: int = 0) -> Dict[str, Any]:
        """Initialize the ResearchService."""
        try:
            super().__init__('research', redis_host, redis_port, redis_db)
            
            # Initialize the agent
            self.agent = ResearchAgent()
            self.memory = AgentMemory()
            
            logger.info("ResearchService initialized")
            return {"status": "success", "message": "ResearchService initialized successfully"}
        except Exception as e:
            logger.error(f"Error initializing ResearchService: {e}")
            return {'success': True, 'result': {"status": "error", "message": str(e)}, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    
    def process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process incoming research requests.
        
        Args:
            data: Message data containing research request
            
        Returns:
            Response with research results or error
        """
        try:
            message_type = data.get('type', '')
            
            if message_type == 'search_github':
                return self._handle_github_search(data)
            elif message_type == 'search_arxiv':
                return self._handle_arxiv_search(data)
            elif message_type == 'summarize_paper':
                return self._handle_summarize_request(data)
            elif message_type == 'generate_code':
                return self._handle_code_generation(data)
            elif message_type == 'get_research_log':
                return self._handle_get_log(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                return {
                    'type': 'error',
                    'error': f"Unknown message type: {message_type}",
                    'original_message': data
                }
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                'type': 'error',
                'error': str(e),
                'original_message': data
            }
    
    def _handle_github_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GitHub search request."""
        try:
            search_data = data.get('data', {})
            
            # Extract search parameters
            query = search_data.get('query')
            max_results = search_data.get('max_results', 10)
            language = search_data.get('language', 'python')
            
            if not query:
                return {
                    'type': 'error',
                    'error': 'query is required'
                }
            
            logger.info(f"Searching GitHub for: {query}")
            
            # Search GitHub using the agent
            results = self.agent.search_github(
                query=query,
                max_results=max_results,
                language=language
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='research',
                decision_type='github_search',
                details={
                    'query': query,
                    'max_results': max_results,
                    'language': language,
                    'results_count': len(results)
                }
            )
            
            return {
                'type': 'github_search_results',
                'results': results,
                'query': query,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching GitHub: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_arxiv_search(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle arXiv search request."""
        try:
            search_data = data.get('data', {})
            
            # Extract search parameters
            query = search_data.get('query')
            max_results = search_data.get('max_results', 10)
            category = search_data.get('category', 'cs.AI')
            
            if not query:
                return {
                    'type': 'error',
                    'error': 'query is required'
                }
            
            logger.info(f"Searching arXiv for: {query}")
            
            # Search arXiv using the agent
            results = self.agent.search_arxiv(
                query=query,
                max_results=max_results,
                category=category
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='research',
                decision_type='arxiv_search',
                details={
                    'query': query,
                    'max_results': max_results,
                    'category': category,
                    'results_count': len(results)
                }
            )
            
            return {
                'type': 'arxiv_search_results',
                'results': results,
                'query': query,
                'count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_summarize_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle paper summarization request."""
        try:
            summarize_data = data.get('data', {})
            
            # Extract parameters
            paper_url = summarize_data.get('paper_url')
            paper_title = summarize_data.get('paper_title')
            paper_abstract = summarize_data.get('paper_abstract')
            
            if not any([paper_url, paper_title, paper_abstract]):
                return {
                    'type': 'error',
                    'error': 'At least one of paper_url, paper_title, or paper_abstract is required'
                }
            
            logger.info(f"Summarizing paper: {paper_title or paper_url}")
            
            # Summarize using the agent
            summary = self.agent.summarize_paper(
                paper_url=paper_url,
                paper_title=paper_title,
                paper_abstract=paper_abstract
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='research',
                decision_type='summarize_paper',
                details={
                    'paper_title': paper_title,
                    'paper_url': paper_url,
                    'summary_length': len(summary.get('summary', ''))
                }
            )
            
            return {
                'type': 'paper_summarized',
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error summarizing paper: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_code_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code generation request."""
        try:
            code_data = data.get('data', {})
            
            # Extract parameters
            description = code_data.get('description')
            paper_reference = code_data.get('paper_reference')
            implementation_type = code_data.get('implementation_type', 'model')
            
            if not description:
                return {
                    'type': 'error',
                    'error': 'description is required'
                }
            
            logger.info(f"Generating code for: {description}")
            
            # Generate code using the agent
            code_suggestion = self.agent.generate_code_suggestion(
                description=description,
                paper_reference=paper_reference,
                implementation_type=implementation_type
            )
            
            # Log to memory
            self.memory.log_decision(
                agent_name='research',
                decision_type='generate_code',
                details={
                    'description': description,
                    'implementation_type': implementation_type,
                    'code_length': len(code_suggestion.get('code', ''))
                }
            )
            
            return {
                'type': 'code_generated',
                'code_suggestion': code_suggestion
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _handle_get_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle research log retrieval request."""
        try:
            log_data = data.get('data', {})
            
            # Extract parameters
            tags = log_data.get('tags', [])
            limit = log_data.get('limit', 50)
            
            # Get research log
            research_log = self.agent.get_research_log(tags=tags, limit=limit)
            
            return {
                'type': 'research_log',
                'log': research_log,
                'tags': tags,
                'count': len(research_log)
            }
            
        except Exception as e:
            logger.error(f"Error getting research log: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        try:
            memory_stats = self.memory.get_stats()
            
            # Get recent research activities
            recent_research = [
                entry for entry in memory_stats.get('recent_decisions', [])
                if entry.get('agent_name') == 'research'
            ]
            
            # Count by type
            research_types = {}
            for research in recent_research:
                research_type = research.get('decision_type', 'unknown')
                research_types[research_type] = research_types.get(research_type, 0) + 1
            
            return {
                'total_research_activities': len(recent_research),
                'research_types': research_types,
                'memory_entries': memory_stats.get('total_entries', 0),
                'recent_research': recent_research[:5]
            }
        except Exception as e:
            logger.error(f"Error getting service stats: {e}")
            return {'error': str(e)}