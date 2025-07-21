"""
StrategyResearchAgent - Internet-based strategy discovery and integration

This agent scans multiple sources for new trading strategies and models:
- arXiv papers for academic trading strategies
- SSRN for working papers
- GitHub repositories for open-source strategies
- QuantConnect forums for community strategies

The agent extracts strategy logic, saves discovered strategies, and schedules
periodic testing with the platform's backtester.
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
import schedule
from bs4 import BeautifulSoup

# Local imports
from trading.agents.base_agent_interface import AgentConfig, AgentPriority, BaseAgent
from trading.backtesting import BacktestEngine
from trading.strategies.base_strategy import BaseStrategy
from trading.utils.cache_manager import cache_result
from trading.utils.config_utils import load_config
from utils.safe_json_saver import safe_json_save


@dataclass
class StrategyDiscovery:
    """Represents a discovered trading strategy"""

    source: str
    title: str
    description: str
    authors: List[str]
    url: str
    discovered_date: str
    strategy_type: str  # 'momentum', 'mean_reversion', 'ml', 'options', etc.
    confidence_score: float  # 0-1 based on source quality and content
    code_snippets: List[str]
    parameters: Dict[str, Any]
    requirements: List[str]
    tags: List[str]


class StrategyResearchAgent(BaseAgent):
    """
    Agent for discovering new trading strategies from internet sources
    """

    def __init__(self, config_path: str = "config/app_config.yaml"):
        # Create proper AgentConfig for BaseAgent
        config = AgentConfig(
            name="StrategyResearchAgent",
            enabled=True,
            priority=AgentPriority.NORMAL,
            max_concurrent_runs=1,
            timeout_seconds=300,
            retry_attempts=3,
            description="Agent for discovering new trading strategies from internet sources",
        )
        super().__init__(config)

        # Load configuration
        self.config = load_config(config_path)
        self.research_config = self.config.get("strategy_research", {})

        # Initialize directories
        self.discovered_dir = Path("strategies/discovered")
        self.discovered_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different sources
        (self.discovered_dir / "arxiv").mkdir(exist_ok=True)
        (self.discovered_dir / "ssrn").mkdir(exist_ok=True)
        (self.discovered_dir / "github").mkdir(exist_ok=True)
        (self.discovered_dir / "quantconnect").mkdir(exist_ok=True)

        # Initialize data structures
        self.discovered_strategies: List[StrategyDiscovery] = []
        self.scan_history: Dict[str, datetime] = {}
        self.test_results: Dict[str, Dict] = {}

        # Load existing discoveries
        self._load_existing_discoveries()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Initialize HTTP session
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )

        # GitHub API token (optional)
        self.github_token = os.getenv("GITHUB_TOKEN")
        if self.github_token:
            self.session.headers.update({"Authorization": f"token {self.github_token}"})

    def _load_existing_discoveries(self):
        """Load previously discovered strategies from disk"""
        for source_dir in self.discovered_dir.iterdir():
            if source_dir.is_dir():
                for strategy_file in source_dir.glob("*.json"):
                    try:
                        with open(strategy_file, "r") as f:
                            data = json.load(f)
                            discovery = StrategyDiscovery(**data)
                            self.discovered_strategies.append(discovery)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {strategy_file}: {e}")

    @cache_result(ttl_seconds=3600)  # Cache for 1 hour
    def search_arxiv(
        self, query: str = "trading strategy", max_results: int = 50
    ) -> List[StrategyDiscovery]:
        """
        Search arXiv for trading strategy papers
        """
        discoveries = []

        try:
            # arXiv API endpoint
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f'all:"{query}"',
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            # Parse XML response
            soup = BeautifulSoup(response.content, "xml")
            entries = soup.find_all("entry")

            for entry in entries:
                try:
                    title = entry.find("title").text.strip()
                    summary = entry.find("summary").text.strip()
                    authors = [
                        author.find("name").text for author in entry.find_all("author")
                    ]
                    arxiv_id = entry.find("id").text.split("/")[-1]
                    # Extract publication date for potential future use
                    published_date = entry.find("published").text[:10]

                    # Extract strategy information
                    strategy_info = self._extract_strategy_from_text(summary)

                    if strategy_info["confidence_score"] > 0.3:  # Minimum confidence
                        discovery = StrategyDiscovery(
                            source="arxiv",
                            title=title,
                            description=summary,
                            authors=authors,
                            url=f"https://arxiv.org/abs/{arxiv_id}",
                            discovered_date=datetime.now().isoformat(),
                            strategy_type=strategy_info["strategy_type"],
                            confidence_score=strategy_info["confidence_score"],
                            code_snippets=strategy_info["code_snippets"],
                            parameters=strategy_info["parameters"],
                            requirements=strategy_info["requirements"],
                            tags=strategy_info["tags"],
                        )
                        discoveries.append(discovery)

                except Exception as e:
                    self.logger.warning(f"Failed to parse arXiv entry: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Failed to search arXiv: {e}")

        return discoveries

    @cache_result(ttl_seconds=3600)
    def search_ssrn(
        self, query: str = "trading strategy", max_results: int = 30
    ) -> List[StrategyDiscovery]:
        """
        Search SSRN for trading strategy papers
        """
        discoveries = []

        try:
            # SSRN search URL
            url = "https://papers.ssrn.com/sol3/results.cfm"
            params = {
                "form_name": "journalBrowse",
                "journal_id": "all",
                "search": query,
                "limit": max_results,
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            papers = soup.find_all("div", class_="paper-item")

            for paper in papers:
                try:
                    title_elem = paper.find("h3", class_="title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link = title_elem.find("a")
                    paper_url = urljoin(url, link["href"]) if link else ""

                    # Get paper details
                    paper_details = self._get_ssrn_paper_details(paper_url)

                    if paper_details and paper_details["confidence_score"] > 0.3:
                        discovery = StrategyDiscovery(
                            source="ssrn",
                            title=title,
                            description=paper_details.get("abstract", ""),
                            authors=paper_details.get("authors", []),
                            url=paper_url,
                            discovered_date=datetime.now().isoformat(),
                            strategy_type=paper_details["strategy_type"],
                            confidence_score=paper_details["confidence_score"],
                            code_snippets=paper_details["code_snippets"],
                            parameters=paper_details["parameters"],
                            requirements=paper_details["requirements"],
                            tags=paper_details["tags"],
                        )
                        discoveries.append(discovery)

                except Exception as e:
                    self.logger.warning(f"Failed to parse SSRN paper: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Failed to search SSRN: {e}")

        return discoveries

    def _get_ssrn_paper_details(self, paper_url: str) -> Optional[Dict]:
        """Extract detailed information from SSRN paper page"""
        try:
            response = self.session.get(paper_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract abstract
            abstract_elem = soup.find("div", class_="abstract-text")
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""

            # Extract authors
            authors_elem = soup.find("div", class_="authors")
            authors = []
            if authors_elem:
                author_links = authors_elem.find_all("a")
                authors = [link.get_text(strip=True) for link in author_links]

            # Analyze content for strategy information
            content = abstract
            strategy_info = self._extract_strategy_from_text(content)

            return {"abstract": abstract, "authors": authors, **strategy_info}

        except Exception as e:
            self.logger.warning(f"Failed to get SSRN paper details: {e}")
            return None

    @cache_result(ttl_seconds=3600)
    def search_github(
        self, query: str = "trading strategy", max_results: int = 50
    ) -> List[StrategyDiscovery]:
        """
        Search GitHub for trading strategy repositories
        """
        discoveries = []

        try:
            # GitHub API endpoint
            url = "https://api.github.com/search/repositories"
            params = {
                "q": f"{query} language:python",
                "sort": "updated",
                "order": "desc",
                "per_page": min(max_results, 100),
            }

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            repositories = data.get("items", [])

            for repo in repositories:
                try:
                    repo_name = repo["full_name"]
                    description = repo.get("description", "")
                    repo_url = repo["html_url"]
                    stars = repo.get("stargazers_count", 0)
                    repo.get("language", "")

                    # Only consider repositories with some popularity
                    if stars < 10:
                        continue

                    # Get repository content
                    repo_content = self._analyze_github_repo(repo_name)

                    if repo_content and repo_content["confidence_score"] > 0.4:
                        discovery = StrategyDiscovery(
                            source="github",
                            title=repo_name,
                            description=description,
                            authors=[repo["owner"]["login"]],
                            url=repo_url,
                            discovered_date=datetime.now().isoformat(),
                            strategy_type=repo_content["strategy_type"],
                            confidence_score=repo_content["confidence_score"],
                            code_snippets=repo_content["code_snippets"],
                            parameters=repo_content["parameters"],
                            requirements=repo_content["requirements"],
                            tags=repo_content["tags"],
                        )
                        discoveries.append(discovery)

                except Exception as e:
                    self.logger.warning(
                        f"Failed to analyze GitHub repo {repo_name}: {e}"
                    )
                    continue

        except Exception as e:
            self.logger.error(f"Failed to search GitHub: {e}")

        return discoveries

    def _analyze_github_repo(self, repo_name: str) -> Optional[Dict]:
        """Analyze GitHub repository for trading strategies"""
        try:
            # Get repository contents
            contents_url = f"https://api.github.com/repos/{repo_name}/contents"
            response = self.session.get(contents_url)
            response.raise_for_status()

            contents = response.json()
            strategy_files = []

            # Find Python files that might contain strategies
            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".py"):
                    strategy_files.append(item["name"])
                elif item["type"] == "dir" and item["name"] in [
                    "strategies",
                    "models",
                    "algorithms",
                ]:
                    # Recursively search strategy directories
                    sub_contents = self._get_github_dir_contents(
                        repo_name, item["path"]
                    )
                    strategy_files.extend(sub_contents)

            # Analyze strategy files
            code_snippets = []
            for filename in strategy_files[:10]:  # Limit to first 10 files
                try:
                    file_content = self._get_github_file_content(repo_name, filename)
                    if file_content:
                        code_snippets.append(f"File: {filename}\n{file_content}")
                except Exception:  # noqa: F841 - Exception caught but not used
                    continue

            if not code_snippets:
                return None

            # Analyze combined code content
            combined_content = "\n".join(code_snippets)
            strategy_info = self._extract_strategy_from_text(combined_content)

            # Extract requirements from setup.py or requirements.txt
            requirements = self._extract_requirements_from_repo(repo_name)

            return {
                "code_snippets": code_snippets,
                "requirements": requirements,
                **strategy_info,
            }

        except Exception as e:
            self.logger.warning(f"Failed to analyze GitHub repo {repo_name}: {e}")
            return None

    def _get_github_dir_contents(self, repo_name: str, path: str) -> List[str]:
        """Recursively get contents of GitHub directory"""
        try:
            url = f"https://api.github.com/repos/{repo_name}/contents/{path}"
            response = self.session.get(url)
            response.raise_for_status()

            contents = response.json()
            files = []

            for item in contents:
                if item["type"] == "file" and item["name"].endswith(".py"):
                    files.append(f"{path}/{item['name']}")
                elif item["type"] == "dir":
                    sub_files = self._get_github_dir_contents(
                        repo_name, f"{path}/{item['name']}"
                    )
                    files.extend(sub_files)

            return files

        except Exception as e:
            self.logger.warning(f"Failed to get GitHub dir contents: {e}")
            return []

    def _get_github_file_content(self, repo_name: str, filepath: str) -> Optional[str]:
        """Get content of GitHub file"""
        try:
            url = f"https://api.github.com/repos/{repo_name}/contents/{filepath}"
            response = self.session.get(url)
            response.raise_for_status()

            data = response.json()
            if "content" in data:
                import base64

                content = base64.b64decode(data["content"]).decode("utf-8")
                return content[:5000]  # Limit content size

            return None

        except Exception as e:
            self.logger.warning(f"Failed to get GitHub file content: {e}")
            return None

    def _extract_requirements_from_repo(self, repo_name: str) -> List[str]:
        """Extract requirements from repository"""
        requirements = []

        try:
            # Try to get requirements.txt
            content = self._get_github_file_content(repo_name, "requirements.txt")
            if content:
                requirements = [
                    line.strip()
                    for line in content.split("\n")
                    if line.strip() and not line.startswith("#")
                ]

            # Try to get setup.py
            if not requirements:
                content = self._get_github_file_content(repo_name, "setup.py")
                if content:
                    # Simple regex to extract package names
                    import re

                    packages = re.findall(r'["\']([^"\']+)["\']', content)
                    requirements = [
                        pkg for pkg in packages if "==" in pkg or ">=" in pkg
                    ]

        except Exception as e:
            self.logger.warning(f"Failed to extract requirements: {e}")

        return requirements

    @cache_result(ttl_seconds=3600)
    def search_quantconnect(
        self, query: str = "strategy", max_results: int = 30
    ) -> List[StrategyDiscovery]:
        """
        Search QuantConnect forums for strategies
        """
        discoveries = []

        try:
            # QuantConnect forum search
            url = "https://www.quantconnect.com/forum/search"
            params = {"q": query, "limit": max_results}

            response = self.session.get(url, params=params)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            topics = soup.find_all("div", class_="topic-item")

            for topic in topics:
                try:
                    title_elem = topic.find("h3", class_="topic-title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    link = title_elem.find("a")
                    topic_url = urljoin(url, link["href"]) if link else ""

                    # Get topic details
                    topic_details = self._get_quantconnect_topic_details(topic_url)

                    if topic_details and topic_details["confidence_score"] > 0.3:
                        discovery = StrategyDiscovery(
                            source="quantconnect",
                            title=title,
                            description=topic_details.get("content", ""),
                            authors=topic_details.get("authors", []),
                            url=topic_url,
                            discovered_date=datetime.now().isoformat(),
                            strategy_type=topic_details["strategy_type"],
                            confidence_score=topic_details["confidence_score"],
                            code_snippets=topic_details["code_snippets"],
                            parameters=topic_details["parameters"],
                            requirements=topic_details["requirements"],
                            tags=topic_details["tags"],
                        )
                        discoveries.append(discovery)

                except Exception as e:
                    self.logger.warning(f"Failed to parse QuantConnect topic: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Failed to search QuantConnect: {e}")

        return discoveries

    def _get_quantconnect_topic_details(self, topic_url: str) -> Optional[Dict]:
        """Extract detailed information from QuantConnect topic page"""
        try:
            response = self.session.get(topic_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract content
            content_elem = soup.find("div", class_="topic-content")
            content = content_elem.get_text(strip=True) if content_elem else ""

            # Extract code blocks
            code_blocks = soup.find_all("pre", class_="code-block")
            code_snippets = [block.get_text() for block in code_blocks]

            # Extract authors
            author_elem = soup.find("div", class_="topic-author")
            authors = []
            if author_elem:
                author_name = author_elem.find("span", class_="author-name")
                if author_name:
                    authors = [author_name.get_text(strip=True)]

            # Analyze content for strategy information
            strategy_info = self._extract_strategy_from_text(content)

            return {
                "content": content,
                "authors": authors,
                "code_snippets": code_snippets,
                **strategy_info,
            }

        except Exception as e:
            self.logger.warning(f"Failed to get QuantConnect topic details: {e}")
            return None

    def _extract_strategy_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract strategy information from text content
        """
        text_lower = text.lower()

        # Determine strategy type
        strategy_type = "unknown"
        if any(word in text_lower for word in ["momentum", "trend", "breakout"]):
            strategy_type = "momentum"
        elif any(
            word in text_lower for word in ["mean reversion", "reversion", "oscillator"]
        ):
            strategy_type = "mean_reversion"
        elif any(
            word in text_lower
            for word in ["machine learning", "ml", "neural", "deep learning"]
        ):
            strategy_type = "ml"
        elif any(word in text_lower for word in ["options", "option", "derivatives"]):
            strategy_type = "options"
        elif any(word in text_lower for word in ["arbitrage", "statistical arbitrage"]):
            strategy_type = "arbitrage"

        # Calculate confidence score based on content quality
        confidence_score = 0.3  # Base score

        # Boost confidence for technical indicators
        technical_indicators = [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger",
            "stochastic",
            "atr",
        ]
        indicator_count = sum(
            1 for indicator in technical_indicators if indicator in text_lower
        )
        confidence_score += min(indicator_count * 0.1, 0.3)

        # Boost confidence for code presence
        if "def " in text or "class " in text or "import " in text:
            confidence_score += 0.2

        # Boost confidence for mathematical formulas
        if any(word in text_lower for word in ["formula", "equation", "calculation"]):
            confidence_score += 0.1

        # Cap confidence at 1.0
        confidence_score = min(confidence_score, 1.0)

        # Extract code snippets
        code_snippets = []
        code_patterns = [
            r"```python\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"def\s+\w+\([^)]*\):.*?(?=\n\S|\Z)",
            r"class\s+\w+.*?(?=\n\S|\Z)",
        ]

        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            code_snippets.extend(matches)

        # Extract parameters
        parameters = {}
        param_patterns = {
            "lookback": r"lookback[_\s]*period[_\s]*=?[_\s]*(\d+)",
            "threshold": r"threshold[_\s]*=?[_\s]*([\d.]+)",
            "window": r"window[_\s]*=?[_\s]*(\d+)",
            "period": r"period[_\s]*=?[_\s]*(\d+)",
        }

        for param_name, pattern in param_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    parameters[param_name] = float(match.group(1))
                except ValueError:
                    parameters[param_name] = match.group(1)

        # Extract tags
        tags = []
        tag_keywords = [
            "python",
            "pandas",
            "numpy",
            "scikit-learn",
            "tensorflow",
            "pytorch",
            "backtesting",
            "live trading",
            "paper trading",
        ]

        for keyword in tag_keywords:
            if keyword in text_lower:
                tags.append(keyword)

        return {
            "strategy_type": strategy_type,
            "confidence_score": confidence_score,
            "code_snippets": code_snippets,
            "parameters": parameters,
            "requirements": [],
            "tags": tags,
        }

    def save_discovery(self, discovery: StrategyDiscovery) -> str:
        """
        Save discovered strategy to disk
        """
        try:
            # Create filename
            safe_title = re.sub(r"[^\w\s-]", "", discovery.title)
            safe_title = re.sub(r"[-\s]+", "-", safe_title)
            filename = f"{discovery.source}_{safe_title[:50]}_{int(time.time())}.json"

            # Save to appropriate directory
            filepath = self.discovered_dir / discovery.source / filename

            # Convert to dict and save
            discovery_dict = asdict(discovery)
            safe_json_save(str(filepath), discovery_dict)

            # Add to internal list
            self.discovered_strategies.append(discovery)

            self.logger.info(f"Saved discovery: {filename}")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to save discovery: {e}")
            return ""

    def generate_strategy_code(self, discovery: StrategyDiscovery) -> str:
        """
        Generate executable strategy code from discovery
        """
        try:
            # Template for strategy class
            strategy_template = f'''
"""
Auto-generated strategy from {discovery.source}
Title: {discovery.title}
Authors: {', '.join(discovery.authors)}
URL: {discovery.url}
"""

import pandas as pd
import numpy as np
from trading.strategies.base_strategy import BaseStrategy


class {discovery.title.replace(' ', '').replace('-', '')}Strategy(BaseStrategy):
    """
    {discovery.description[:200]}...
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Set default parameters
        self.lookback_period = kwargs.get('lookback_period', {discovery.parameters.get('lookback', 20)})
        self.threshold = kwargs.get('threshold', {discovery.parameters.get('threshold', 0.5)})
        self.window = kwargs.get('window', {discovery.parameters.get('window', 14)})

    def generate_signals(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate trading signals based on discovered strategy logic
        """
        signals = df.copy()
        signals['signal'] = 0

        # Strategy implementation based on discovered logic
        {self._generate_strategy_logic(discovery)}

        return signals

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy
        """
        df = df.copy()

        # Add basic indicators
        df['sma'] = df['close'].rolling(window=self.window).mean()
        df['ema'] = df['close'].ewm(span=self.window).mean()
        df['rsi'] = self._calculate_rsi(df['close'], self.window)

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI indicator
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
'''

            # Generate strategy logic based on type
            strategy_logic = self._generate_strategy_logic(discovery)
            strategy_template = strategy_template.replace(
                "{self._generate_strategy_logic(discovery)}", strategy_logic
            )

            return strategy_template

        except Exception as e:
            self.logger.error(f"Failed to generate strategy code: {e}")
            return ""

    def _generate_strategy_logic(self, discovery: StrategyDiscovery) -> str:
        """
        Generate strategy logic based on discovery type
        """
        if discovery.strategy_type == "momentum":
            return """
        # Momentum strategy logic
        df = self._calculate_indicators(df)

        # Generate signals based on price momentum
        df['momentum'] = df['close'] / df['close'].shift(self.lookback_period) - 1
        df['signal'] = np.where(df['momentum'] > self.threshold, 1, 0)
        df['signal'] = np.where(df['momentum'] < -self.threshold, -1, df['signal'])

        signals['signal'] = df['signal']
        """

        elif discovery.strategy_type == "mean_reversion":
            return """
        # Mean reversion strategy logic
        df = self._calculate_indicators(df)

        # Generate signals based on mean reversion
        df['deviation'] = (df['close'] - df['sma']) / df['sma']
        df['signal'] = np.where(df['deviation'] > self.threshold, -1, 0)
        df['signal'] = np.where(df['deviation'] < -self.threshold, 1, df['signal'])

        signals['signal'] = df['signal']
        """

        elif discovery.strategy_type == "ml":
            return """
        # Machine learning strategy logic
        df = self._calculate_indicators(df)

        # Simple ML-based signal generation
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['price_change'].rolling(self.window).std()

        # Generate signals based on volatility and price changes
        df['signal'] = np.where(
            (df['price_change'] > self.threshold) & (df['volatility'] < 0.02), 1, 0
        )
        df['signal'] = np.where(
            (df['price_change'] < -self.threshold) & (df['volatility'] < 0.02), -1, df['signal']
        )

        signals['signal'] = df['signal']
        """

        else:
            return """
        # Generic strategy logic
        df = self._calculate_indicators(df)

        # Simple moving average crossover
        df['signal'] = np.where(df['close'] > df['sma'], 1, 0)
        df['signal'] = np.where(df['close'] < df['sma'], -1, df['signal'])

        signals['signal'] = df['signal']
        """

    def test_discovered_strategy(self, discovery: StrategyDiscovery) -> Dict[str, Any]:
        """
        Test discovered strategy using backtester
        """
        try:
            # Generate strategy code
            strategy_code = self.generate_strategy_code(discovery)
            if not strategy_code:
                return {"error": "Failed to generate strategy code"}

            # Save strategy code to temporary file
            strategy_filename = (
                f"discovered_{discovery.title.replace(' ', '_')[:30]}.py"
            )
            strategy_path = self.discovered_dir / discovery.source / strategy_filename

            with open(strategy_path, "w") as f:
                f.write(strategy_code)

            # Import and test strategy
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "discovered_strategy", str(strategy_path)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get strategy class
            strategy_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseStrategy)
                    and attr != BaseStrategy
                ):
                    strategy_class = attr
                    break

            if not strategy_class:
                return {"error": "No valid strategy class found"}

            # Initialize backtester
            backtester = BacktestEngine()

            # Test with sample data
            test_results = backtester.run_backtest(
                strategy_class(),
                symbol="AAPL",
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_capital=10000,
            )

            # Clean up temporary file
            strategy_path.unlink()

            return test_results

        except Exception as e:
            self.logger.error(f"Failed to test discovered strategy: {e}")
            return {"error": str(e)}

    def run_research_scan(self) -> List[StrategyDiscovery]:
        """
        Run a complete research scan across all sources
        """
        self.logger.info("Starting strategy research scan...")

        all_discoveries = []

        # Search all sources
        sources = [
            ("arxiv", self.search_arxiv),
            ("ssrn", self.search_ssrn),
            ("github", self.search_github),
            ("quantconnect", self.search_quantconnect),
        ]

        for source_name, search_func in sources:
            try:
                self.logger.info(f"Searching {source_name}...")
                discoveries = search_func()

                # Filter out already discovered strategies
                new_discoveries = []
                for discovery in discoveries:
                    if not self._is_duplicate(discovery):
                        new_discoveries.append(discovery)
                        all_discoveries.append(discovery)

                self.logger.info(
                    f"Found {len(new_discoveries)} new strategies from {source_name}"
                )

                # Save new discoveries
                for discovery in new_discoveries:
                    self.save_discovery(discovery)

            except Exception as e:
                self.logger.error(f"Failed to search {source_name}: {e}")

        # Update scan history
        self.scan_history["last_scan"] = datetime.now()

        self.logger.info(
            f"Research scan complete. Found {len(all_discoveries)} total new strategies."
        )
        return all_discoveries

    def _is_duplicate(self, discovery: StrategyDiscovery) -> bool:
        """
        Check if discovery is a duplicate of existing ones
        """
        for existing in self.discovered_strategies:
            # Check title similarity
            if self._similarity_score(discovery.title, existing.title) > 0.8:
                return True

            # Check URL
            if discovery.url == existing.url:
                return True

        return False

    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate similarity score between two texts
        """
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def schedule_periodic_scans(self, interval_hours: int = 24):
        """
        Schedule periodic research scans
        """

        def run_scan():
            try:
                discoveries = self.run_research_scan()

                # Test new discoveries
                for discovery in discoveries:
                    if discovery.confidence_score > 0.5:
                        self.logger.info(
                            f"Testing high-confidence strategy: {discovery.title}"
                        )
                        test_results = self.test_discovered_strategy(discovery)
                        self.test_results[discovery.title] = test_results

                        # Save test results
                        results_file = (
                            self.discovered_dir
                            / f"test_results_{discovery.title[:30]}.json"
                        )
                        safe_json_save(str(results_file), test_results)

            except Exception as e:
                self.logger.error(f"Failed to run scheduled scan: {e}")

        # Schedule the scan
        schedule.every(interval_hours).hours.do(run_scan)

        # Start scheduler in background thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute

        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()

        self.logger.info(f"Scheduled periodic scans every {interval_hours} hours")

    def get_discovery_summary(self) -> Dict[str, Any]:
        """
        Get summary of discovered strategies
        """
        summary = {
            "total_discoveries": len(self.discovered_strategies),
            "by_source": {},
            "by_type": {},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "recent_discoveries": [],
            "test_results": len(self.test_results),
        }

        # Group by source
        for discovery in self.discovered_strategies:
            source = discovery.source
            summary["by_source"][source] = summary["by_source"].get(source, 0) + 1

        # Group by type
        for discovery in self.discovered_strategies:
            strategy_type = discovery.strategy_type
            summary["by_type"][strategy_type] = (
                summary["by_type"].get(strategy_type, 0) + 1
            )

        # Confidence distribution
        for discovery in self.discovered_strategies:
            if discovery.confidence_score > 0.7:
                summary["confidence_distribution"]["high"] += 1
            elif discovery.confidence_score > 0.4:
                summary["confidence_distribution"]["medium"] += 1
            else:
                summary["confidence_distribution"]["low"] += 1

        # Recent discoveries (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        recent = [
            discovery
            for discovery in self.discovered_strategies
            if datetime.fromisoformat(discovery.discovered_date) > week_ago
        ]
        summary["recent_discoveries"] = [
            {
                "title": d.title,
                "source": d.source,
                "confidence": d.confidence_score,
                "type": d.strategy_type,
            }
            for d in recent[:10]  # Top 10 recent
        ]

        return summary

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Main agent execution method
        """
        try:
            # Run research scan
            discoveries = self.run_research_scan()

            # Test high-confidence strategies
            tested_strategies = []
            for discovery in discoveries:
                if discovery.confidence_score > 0.5:
                    self.logger.info(f"Testing strategy: {discovery.title}")
                    test_results = self.test_discovered_strategy(discovery)
                    tested_strategies.append(
                        {"discovery": discovery, "test_results": test_results}
                    )

            # Get summary
            summary = self.get_discovery_summary()

            return {
                "status": "success",
                "discoveries_found": len(discoveries),
                "strategies_tested": len(tested_strategies),
                "summary": summary,
                "tested_strategies": tested_strategies,
            }

        except Exception as e:
            self.logger.error(f"Strategy research agent failed: {e}")
            return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Example usage
    agent = StrategyResearchAgent()

    # Run a single scan
    results = agent.run()
    print(json.dumps(results, indent=2))

    # Schedule periodic scans
    agent.schedule_periodic_scans(interval_hours=12)

    # Keep running
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping strategy research agent...")
