"""
ArXiv Research Fetcher Module

This module handles fetching and processing research papers from arXiv
for the auto-evolutionary model generator.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ResearchPaper:
    """Represents a research paper from arXiv."""

    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    published_date: str
    categories: List[str]
    relevance_score: float
    implementation_complexity: str  # "low", "medium", "high"
    potential_impact: str  # "low", "medium", "high"


class ArxivResearchFetcher:
    """Fetches research papers from arXiv API."""

    def __init__(
        self,
        search_terms: Optional[List[str]] = None,
        max_results: int = 100,
        days_back: int = 30,
    ):
        """Initialize research fetcher.

        Args:
            search_terms: Search terms for relevant papers
            max_results: Maximum number of results to fetch
            days_back: Number of days back to search
        """
        self.search_terms = search_terms or [
            "time series forecasting",
            "financial prediction",
            "machine learning trading",
            "neural networks forecasting",
            "quantitative finance",
            "market prediction",
            "deep learning time series",
            "reinforcement learning trading",
        ]
        self.max_results = max_results
        self.days_back = days_back
        self.base_url = "http://export.arxiv.org/api/query"

        # Cache for fetched papers
        self.paper_cache = {}
        self.cache_file = Path("agents/research_cache.json")
        self._load_cache()

        logger.info(
            f"Initialized ArxivResearchFetcher with {len(self.search_terms)} search terms"
        )

    def _load_cache(self):
        """Load cached papers."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    self.paper_cache = json.load(f)
                logger.info(f"Loaded {len(self.paper_cache)} cached papers")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.paper_cache = {}

    def _save_cache(self):
        """Save papers to cache."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w") as f:
                json.dump(self.paper_cache, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _calculate_relevance_score(
        self, title: str, abstract: str, categories: List[str]
    ) -> float:
        """Calculate relevance score for a paper."""
        score = 0.0

        # Keywords that indicate relevance
        relevant_keywords = [
            "time series",
            "forecasting",
            "prediction",
            "financial",
            "trading",
            "market",
            "stock",
            "price",
            "return",
            "volatility",
            "neural network",
            "deep learning",
            "machine learning",
            "reinforcement learning",
            "lstm",
            "transformer",
            "attention",
            "ensemble",
            "optimization",
        ]

        # Check title relevance
        title_lower = title.lower()
        for keyword in relevant_keywords:
            if keyword in title_lower:
                score += 2.0

        # Check abstract relevance
        abstract_lower = abstract.lower()
        for keyword in relevant_keywords:
            if keyword in abstract_lower:
                score += 1.0

        # Check categories relevance
        relevant_categories = ["cs.lg", "cs.ai", "q-fin", "stat.ml", "stat.me"]
        for category in categories:
            if category in relevant_categories:
                score += 3.0

        # Normalize score
        score = min(score / 20.0, 1.0)

        return score

    def _assess_implementation_complexity(self, title: str, abstract: str) -> str:
        """Assess implementation complexity."""
        complexity_indicators = {
            "low": ["simple", "linear", "regression", "basic", "traditional"],
            "medium": ["neural", "network", "ensemble", "gradient", "optimization"],
            "high": [
                "transformer",
                "attention",
                "reinforcement",
                "complex",
                "advanced",
            ],
        }

        text_lower = (title + " " + abstract).lower()

        scores = {}
        for complexity, keywords in complexity_indicators.items():
            scores[complexity] = sum(1 for keyword in keywords if keyword in text_lower)

        # Return complexity with highest score
        return max(scores, key=scores.get)

    def _assess_potential_impact(
        self, title: str, abstract: str, relevance_score: float
    ) -> str:
        """Assess potential impact of the research."""
        impact_indicators = {
            "high": [
                "novel",
                "breakthrough",
                "state-of-the-art",
                "sota",
                "improvement",
            ],
            "medium": ["proposed", "method", "approach", "framework", "model"],
            "low": ["review", "survey", "analysis", "comparison", "study"],
        }

        text_lower = (title + " " + abstract).lower()

        scores = {}
        for impact, keywords in impact_indicators.items():
            scores[impact] = sum(1 for keyword in keywords if keyword in text_lower)

        # Boost score based on relevance
        if relevance_score > 0.7:
            scores["high"] += 2

        return max(scores, key=scores.get)

    async def fetch_papers_async(self, search_term: str) -> List[ResearchPaper]:
        """Fetch papers asynchronously from arXiv."""
        try:
            # Check cache first
            cache_key = f"{search_term}_{self.days_back}"
            if cache_key in self.paper_cache:
                logger.info(f"Using cached results for '{search_term}'")
                return [ResearchPaper(**paper) for paper in self.paper_cache[cache_key]]

            # Build query
            query = (
                f"search_query=all:{search_term}&start=0&max_results={self.max_results}"
            )
            url = f"{self.base_url}?{query}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        xml_content = await response.text()
                        papers = self._parse_arxiv_xml(xml_content, search_term)

                        # Cache results
                        self.paper_cache[cache_key] = [
                            paper.__dict__ for paper in papers
                        ]
                        self._save_cache()

                        logger.info(f"Fetched {len(papers)} papers for '{search_term}'")
                        return papers
                    else:
                        logger.error(
                            f"Failed to fetch papers for '{search_term}': {response.status}"
                        )
                        return []

        except Exception as e:
            logger.error(f"Error fetching papers for '{search_term}': {e}")
            return []

    def _parse_arxiv_xml(
        self, xml_content: str, search_term: str
    ) -> List[ResearchPaper]:
        """Parse arXiv XML response."""
        papers = []

        try:
            # Simple XML parsing with regex (for robustness)
            entry_pattern = r"<entry>(.*?)</entry>"
            entries = re.findall(entry_pattern, xml_content, re.DOTALL)

            for entry in entries:
                try:
                    # Extract title
                    title_match = re.search(r"<title>(.*?)</title>", entry)
                    title = title_match.group(1).strip() if title_match else ""

                    # Extract authors
                    author_pattern = r"<name>(.*?)</name>"
                    authors = re.findall(author_pattern, entry)

                    # Extract abstract
                    abstract_match = re.search(r"<summary>(.*?)</summary>", entry)
                    abstract = abstract_match.group(1).strip() if abstract_match else ""

                    # Extract arXiv ID
                    id_match = re.search(r"<id>(.*?)</id>", entry)
                    arxiv_id = id_match.group(1).split("/")[-1] if id_match else ""

                    # Extract published date
                    published_match = re.search(r"<published>(.*?)</published>", entry)
                    published_date = published_match.group(1) if published_match else ""

                    # Extract categories
                    category_pattern = r'<category term="(.*?)"'
                    categories = re.findall(category_pattern, entry)

                    # Calculate scores
                    relevance_score = self._calculate_relevance_score(
                        title, abstract, categories
                    )
                    implementation_complexity = self._assess_implementation_complexity(
                        title, abstract
                    )
                    potential_impact = self._assess_potential_impact(
                        title, abstract, relevance_score
                    )

                    # Only include relevant papers
                    if relevance_score > 0.3:
                        paper = ResearchPaper(
                            title=title,
                            authors=authors,
                            abstract=abstract,
                            arxiv_id=arxiv_id,
                            published_date=published_date,
                            categories=categories,
                            relevance_score=relevance_score,
                            implementation_complexity=implementation_complexity,
                            potential_impact=potential_impact,
                        )
                        papers.append(paper)

                except Exception as e:
                    logger.warning(f"Error parsing paper entry: {e}")
                    continue

            # Sort by relevance score
            papers.sort(key=lambda x: x.relevance_score, reverse=True)

        except Exception as e:
            logger.error(f"Error parsing arXiv XML: {e}")

        return papers

    async def fetch_recent_papers(self) -> List[ResearchPaper]:
        """Fetch recent papers from all search terms."""
        all_papers = []

        # Fetch papers for each search term
        tasks = [self.fetch_papers_async(term) for term in self.search_terms]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
            else:
                logger.error(f"Error in paper fetching: {result}")

        # Remove duplicates based on arXiv ID
        unique_papers = {}
        for paper in all_papers:
            if paper.arxiv_id not in unique_papers:
                unique_papers[paper.arxiv_id] = paper

        # Sort by relevance and return top papers
        sorted_papers = sorted(
            unique_papers.values(), key=lambda x: x.relevance_score, reverse=True
        )

        logger.info(f"Fetched {len(sorted_papers)} unique relevant papers")
        return sorted_papers[: self.max_results]
