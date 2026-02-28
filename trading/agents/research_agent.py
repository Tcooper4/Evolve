"""
ResearchAgent: Autonomous research agent for discovering new forecasting models and trading strategies.
- Searches GitHub and arXiv
- Summarizes papers and repos using Claude API (OpenAI fallback)
- Suggests code snippets via Claude
- AGENT_UPGRADE: Claude primary; blocking HTTP/LLM in run_in_executor
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base_agent_interface import AgentConfig, AgentResult, BaseAgent
from .prompt_templates import format_template

# Centralized LLM config (AGENT_UPGRADE)
try:
    from config.llm_config import get_llm_config, CLAUDE_PRIMARY_MODEL
except ImportError:
    get_llm_config = None
    CLAUDE_PRIMARY_MODEL = "claude-sonnet-4-20250514"

# Optionally import OpenAI API
try:
    import openai
except ImportError:
    openai = None


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
                custom_config={},
            )
        super().__init__(config)

        custom_config = config.custom_config or {}
        self.openai_api_key = custom_config.get("openai_api_key") or (
            openai.api_key if openai else None
        )
        self.anthropic_api_key = custom_config.get("anthropic_api_key")
        # AGENT_UPGRADE: Centralized LLM config
        try:
            if get_llm_config:
                llm = get_llm_config()
                if not self.anthropic_api_key:
                    self.anthropic_api_key = getattr(llm, "anthropic_api_key", None)
                if not self.openai_api_key:
                    self.openai_api_key = getattr(llm, "openai_api_key", None)
        except Exception:
            pass
        self.log_path = Path(custom_config.get("log_path", "research_log.json"))

        if not self.log_path.exists():
            self.log_path.write_text(json.dumps([]))
        if openai and self.openai_api_key:
            openai.api_key = self.openai_api_key

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the research logic. Blocking HTTP/LLM run in executor. AGENT_UPGRADE."""
        try:
            action = kwargs.get("action", "research")
            loop = asyncio.get_event_loop()

            if action == "research":
                topic = kwargs.get("topic")
                max_results = kwargs.get("max_results", 3)
                if topic is None:
                    return AgentResult(
                        success=False, error_message="Missing required parameter: topic"
                    )
                findings = await loop.run_in_executor(
                    None, lambda: self.research(topic, max_results)
                )
                return AgentResult(
                    success=True,
                    data={
                        "findings": findings,
                        "findings_count": len(findings),
                        "topic": topic,
                    },
                )

            elif action == "search_github":
                query = kwargs.get("query")
                max_results = kwargs.get("max_results", 5)
                if query is None:
                    return AgentResult(
                        success=False, error_message="Missing required parameter: query"
                    )
                results = await loop.run_in_executor(
                    None, lambda: self.search_github(query, max_results)
                )
                return AgentResult(
                    success=True,
                    data={"github_results": results, "results_count": len(results)},
                )

            elif action == "search_arxiv":
                query = kwargs.get("query")
                max_results = kwargs.get("max_results", 5)
                if query is None:
                    return AgentResult(
                        success=False, error_message="Missing required parameter: query"
                    )
                results = await loop.run_in_executor(
                    None, lambda: self.search_arxiv(query, max_results)
                )
                return AgentResult(
                    success=True,
                    data={"arxiv_results": results, "results_count": len(results)},
                )

            elif action == "summarize":
                text = kwargs.get("text")
                custom_prompt = kwargs.get("prompt")
                if text is None:
                    return AgentResult(
                        success=False, error_message="Missing required parameter: text"
                    )
                prompt = (
                    custom_prompt
                    if custom_prompt is not None
                    else format_template("research_summarize", text=text)
                )
                summary = await loop.run_in_executor(
                    None, lambda: self._summarize(text, prompt)
                )
                return AgentResult(
                    success=True, data={"summary": summary, "text_length": len(text)}
                )

            elif action == "code_suggestion":
                description = kwargs.get("description")
                if description is None:
                    return AgentResult(
                        success=False,
                        error_message="Missing required parameter: description",
                    )
                code = await loop.run_in_executor(
                    None, lambda: self._code_suggestion(description)
                )
                return AgentResult(
                    success=True,
                    data={
                        "code_suggestion": code,
                        "description_length": len(description),
                    },
                )

            else:
                return AgentResult(
                    success=False, error_message=f"Unknown action: {action}"
                )

        except Exception as e:
            return self.handle_error(e)

    def search_github(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub for repositories related to the query."""
        url = f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page={max_results}"
        resp = requests.get(url)
        if resp.status_code == 200:
            items = resp.json().get("items", [])
            return [
                {
                    "name": item["name"],
                    "full_name": item["full_name"],
                    "url": item["html_url"],
                    "description": item["description"],
                    "stars": item["stargazers_count"],
                    "language": item["language"],
                    "tag": (
                        "model"
                        if "model" in item["description"].lower()
                        else "strategy"
                    ),
                }
                for item in items
            ]
        else:
            logger.warning(f"GitHub search failed: {resp.status_code}")
            return []

    def search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv for papers related to the query."""
        # Use vector search instead of keyword for paper matching
        try:
            # First try vector search if available
            if hasattr(self, "_vector_search"):
                return self._vector_search(query, max_results)
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to keyword search: {e}")

        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        resp = requests.get(url)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET

            root = ET.fromstring(resp.text)
            ns = {"arxiv": "http://www.w3.org/2005/Atom"}
            entries = []
            for entry in root.findall("arxiv:entry", ns):
                title = entry.find("arxiv:title", ns).text.strip()
                summary = entry.find("arxiv:summary", ns).text.strip()
                link = entry.find("arxiv:id", ns).text.strip()
                entries.append(
                    {"title": title, "summary": summary, "url": link, "tag": "paper"}
                )
            return entries
        else:
            logger.warning(f"arXiv search failed: {resp.status_code}")
            return []

    def _summarize(self, text: str, prompt: str = None) -> str:
        """Summarize text using Claude (primary) or OpenAI (fallback). AGENT_UPGRADE."""
        if prompt is None:
            prompt = format_template("research_summarize", text=text)
        try:
            if getattr(self, "anthropic_api_key", None):
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                model = CLAUDE_PRIMARY_MODEL
                if get_llm_config:
                    model = getattr(get_llm_config(), "primary_model", model)
                resp = client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.2,
                    system=prompt,
                    messages=[{"role": "user", "content": text[:50000]}],
                )
                return resp.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Claude summarize failed, trying OpenAI: {e}")
        return self.summarize_with_openai(text, prompt)

    def _code_suggestion(self, description: str) -> str:
        """Code suggestion using Claude (primary) or OpenAI (fallback). AGENT_UPGRADE."""
        try:
            if getattr(self, "anthropic_api_key", None):
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                model = CLAUDE_PRIMARY_MODEL
                if get_llm_config:
                    model = getattr(get_llm_config(), "primary_model", model)
                prompt = format_template("research_code_suggestion", description=description)
                resp = client.messages.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.2,
                    system=prompt,
                    messages=[{"role": "user", "content": description}],
                )
                return resp.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Claude code_suggestion failed, trying OpenAI: {e}")
        return self.code_suggestion_with_openai(description)

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
                {"role": "user", "content": text},
            ],
            max_tokens=300,
        )
        return response.choices[0].message["content"].strip()

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
                {"role": "user", "content": description},
            ],
            max_tokens=300,
        )
        return response.choices[0].message["content"].strip()

    def log_finding(self, finding: Dict[str, Any]) -> None:
        """Log a research finding to research_log.json."""
        finding["timestamp"] = datetime.now().isoformat()
        data = json.loads(self.log_path.read_text())
        data.append(finding)
        self.log_path.write_text(json.dumps(data, indent=2))

    def research(self, topic: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Conduct research on a topic: search, summarize, suggest code, and log findings."""
        findings = []
        # Search GitHub
        github_results = self.search_github(topic, max_results)
        for repo in github_results:
            summary = self._summarize(repo["description"] or repo["name"])
            code = self._code_suggestion(repo["description"] or repo["name"])
            finding = {
                "type": "github",
                "tag": repo["tag"],
                "title": repo["name"],
                "url": repo["url"],
                "summary": summary,
                "code_suggestion": code,
            }
            self.log_finding(finding)
            findings.append(finding)
        # Search arXiv
        arxiv_results = self.search_arxiv(topic, max_results)
        for paper in arxiv_results:
            summary = self._summarize(paper["summary"])
            code = self._code_suggestion(paper["summary"])
            finding = {
                "type": "arxiv",
                "tag": paper["tag"],
                "title": paper["title"],
                "url": paper["url"],
                "summary": summary,
                "code_suggestion": code,
            }
            self.log_finding(finding)
            findings.append(finding)
        return findings
