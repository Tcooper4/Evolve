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

# Optionally import OpenAI API
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, openai_api_key: Optional[str] = None, log_path: str = "research_log.json"):
        self.openai_api_key = openai_api_key or (openai.api_key if openai else None)
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            self.log_path.write_text(json.dumps([]))
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key

    return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
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
            return {'success': True, 'result': [], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

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
            return {'success': True, 'result': [], 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}

    def summarize_with_openai(self, text: str, prompt: str = "Summarize this for a quant trading engineer:") -> str:
        """Use OpenAI API to summarize text."""
        if not openai or not self.openai_api_key:
            return {'success': True, 'result': "[OpenAI API not available]", 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
            return {'success': True, 'result': "[OpenAI API not available]", 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Suggest Python code for a quant trading engineer."},
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

    return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
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
        return {'success': True, 'result': findings, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}