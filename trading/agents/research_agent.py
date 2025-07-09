"""
ResearchAgent: Autonomous research agent for discovering new forecasting models and trading strategies.
- Searches GitHub and arXiv
- Summarizes papers and repos using OpenAI API
- Suggests code snippets
- Logs findings to research_log.json with tags
"""

import requests
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult
from .prompt_templates import format_template

@dataclass
class ResearchRequest:
    """Research request."""
    action: str  # 'research', 'search_github', 'search_arxiv', 'summarize', 'code_suggestion'
    topic: Optional[str] = None
    query: Optional[str] = None
    max_results: Optional[int] = None
    text: Optional[str] = None
    description: Optional[str] = None
    prompt: Optional[str] = None

@dataclass
class ResearchResult:
    """Research result."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None

# Optionally import OpenAI API
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

class ResearchAgent(BaseAgent):
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="ResearchAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        
        # Extract config from custom_config or use defaults
        custom_config = config.custom_config or {}
        self.openai_api_key = custom_config.get('openai_api_key') or (openai.api_key if openai else None)
        self.log_path = Path(custom_config.get('log_path', "research_log.json"))
        
        if not self.log_path.exists():
            self.log_path.write_text(json.dumps([]))
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the research logic.
        Args:
            **kwargs: topic, max_results, action, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'research')
            
            if action == 'research':
                topic = kwargs.get('topic')
                max_results = kwargs.get('max_results', 3)
                
                if topic is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: topic"
                    )
                
                findings = self.research(topic, max_results)
                return AgentResult(success=True, data={
                    "findings": findings,
                    "findings_count": len(findings),
                    "topic": topic
                })
                
            elif action == 'search_github':
                query = kwargs.get('query')
                max_results = kwargs.get('max_results', 5)
                
                if query is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: query"
                    )
                
                results = self.search_github(query, max_results)
                return AgentResult(success=True, data={
                    "github_results": results,
                    "results_count": len(results)
                })
                
            elif action == 'search_arxiv':
                query = kwargs.get('query')
                max_results = kwargs.get('max_results', 5)
                
                if query is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: query"
                    )
                
                results = self.search_arxiv(query, max_results)
                return AgentResult(success=True, data={
                    "arxiv_results": results,
                    "results_count": len(results)
                })
                
            elif action == 'summarize':
                text = kwargs.get('text')
                custom_prompt = kwargs.get('prompt')
                
                if text is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: text"
                    )
                
                # Use centralized template if no custom prompt provided
                if custom_prompt is None:
                    prompt = format_template("research_summarize", text=text)
                else:
                    prompt = custom_prompt
                
                summary = self.summarize_with_openai(text, prompt)
                return AgentResult(success=True, data={
                    "summary": summary,
                    "text_length": len(text)
                })
                
            elif action == 'code_suggestion':
                description = kwargs.get('description')
                
                if description is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: description"
                    )
                
                code = self.code_suggestion_with_openai(description)
                return AgentResult(success=True, data={
                    "code_suggestion": code,
                    "description_length": len(description)
                })
                
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
                
        except Exception as e:
            return self.handle_error(e)

    def search_github(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub for repositories related to the query."""
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={max_results}"
        resp = requests.get(url)
        if resp.status_code == 200:
            items = resp.json().get('items', [])
            return [{
                'name': item['name'],
                'full_name': item['full_name'],
                'url': item['html_url'],
                'description': item['description'],
                'stars': item['stargazers_count'],
                'language': item['language'],
                'tag': 'model' if 'model' in item['description'].lower() else 'strategy'
            } for item in items]
        else:
            logger.warning(f"GitHub search failed: {resp.status_code}")
            return []

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv for papers related to the query."""
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        resp = requests.get(url)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.text)
            ns = {'arxiv': 'http://www.w3.org/2005/Atom'}
            entries = []
            for entry in root.findall('arxiv:entry', ns):
                title = entry.find('arxiv:title', ns).text.strip()
                summary = entry.find('arxiv:summary', ns).text.strip()
                link = entry.find('arxiv:id', ns).text.strip()
                entries.append({
                    'title': title,
                    'summary': summary,
                    'url': link,
                    'tag': 'paper'
                })
            return entries
        else:
            logger.warning(f"arXiv search failed: {resp.status_code}")
            return []

    def summarize_with_openai(self, text: str, prompt: str = None) -> str:
        """Use OpenAI API to summarize text."""
        if not openai or not self.openai_api_key:
            return "[OpenAI API not available]"
        
        # Use centralized template if no prompt provided
        if prompt is None:
            prompt = format_template("research_summarize", text=text)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            max_tokens=300
        )
        return response.choices[0].message['content'].strip()

    def code_suggestion_with_openai(self, description: str) -> str:
        """Use OpenAI API to generate code suggestion from a description."""
        if not openai or not self.openai_api_key:
            return "[OpenAI API not available]"
        
        # Use centralized template for code suggestions
        prompt = format_template("research_code_suggestion", description=description)
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": description}
            ],
            max_tokens=300
        )
        return response.choices[0].message['content'].strip()

    def log_finding(self, finding: Dict[str, Any]) -> None:
        """Log a research finding to research_log.json."""
        finding['timestamp'] = datetime.now().isoformat()
        data = json.loads(self.log_path.read_text())
        data.append(finding)
        self.log_path.write_text(json.dumps(data, indent=2))

    def research(self, topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Conduct research on a topic: search, summarize, suggest code, and log findings."""
        findings = []
        # Search GitHub
        github_results = self.search_github(topic, max_results)
        for repo in github_results:
            summary = self.summarize_with_openai(repo['description'] or repo['name'])
            code = self.code_suggestion_with_openai(repo['description'] or repo['name'])
            finding = {
                'type': 'github',
                'tag': repo['tag'],
                'title': repo['name'],
                'url': repo['url'],
                'summary': summary,
                'code_suggestion': code
            }
            self.log_finding(finding)
            findings.append(finding)
        # Search arXiv
        arxiv_results = self.search_arxiv(topic, max_results)
        for paper in arxiv_results:
            summary = self.summarize_with_openai(paper['summary'])
            code = self.code_suggestion_with_openai(paper['summary'])
            finding = {
                'type': 'arxiv',
                'tag': paper['tag'],
                'title': paper['title'],
                'url': paper['url'],
                'summary': summary,
                'code_suggestion': code
            }
            self.log_finding(finding)
            findings.append(finding)
        return findings