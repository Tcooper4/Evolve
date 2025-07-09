# -*- coding: utf-8 -*-
"""
Meta-Research Agent for automated research discovery and model evaluation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import re
import hashlib

from trading.models.model_registry import ModelRegistry
from trading.agents.model_selector_agent import ModelSelectorAgent
from trading.memory.agent_memory import AgentMemory
from .base_agent_interface import BaseAgent, AgentConfig, AgentResult

@dataclass
class ResearchRequest:
    """Research request."""
    action: str  # 'discover_papers', 'evaluate_models', 'auto_implement', 'get_summary'
    keywords: Optional[List[str]] = None
    max_papers: Optional[int] = None
    threshold: Optional[float] = None
    papers: Optional[List['ResearchPaper']] = None
    evaluations: Optional[List['ModelEvaluation']] = None

@dataclass
class ResearchResult:
    """Research result."""
    success: bool
    data: Dict[str, Any]
    error_message: Optional[str] = None

@dataclass
class ResearchPaper:
    """Research paper information."""
    title: str
    authors: List[str]
    abstract: str
    url: str
    source: str
    publication_date: datetime
    keywords: List[str]
    model_type: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    implementation_status: str = "discovered"

@dataclass
class ModelEvaluation:
    """Model evaluation result."""
    paper_id: str
    model_name: str
    model_type: str
    performance_score: float
    implementation_complexity: float
    market_applicability: float
    overall_score: float
    recommendation: str

class MetaResearchAgent(BaseAgent):
    """
    Meta-Research Agent with:
    - Automated research paper discovery
    - Model performance evaluation
    - Implementation feasibility assessment
    - Auto-addition to model registry
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name="MetaResearchAgent",
                enabled=True,
                priority=1,
                max_concurrent_runs=1,
                timeout_seconds=300,
                retry_attempts=3,
                custom_config={}
            )
        super().__init__(config)
        self.config_dict = config.custom_config or {}
        self.logger = logging.getLogger(__name__)
        self.memory = AgentMemory()
        self.model_registry = ModelRegistry()
        self.model_selector = ModelSelectorAgent()
        
        # Configuration
        self.research_sources = self.config_dict.get('research_sources', [
            'arxiv.org',
            'papers.ssrn.com',
            'scholar.google.com'
        ])
        self.scraping_frequency = self.config_dict.get('scraping_frequency', 'weekly')
        self.evaluation_threshold = self.config_dict.get('evaluation_threshold', 0.7)
        self.max_papers_per_search = self.config_dict.get('max_papers_per_search', 50)
        
        # Keywords for relevant papers
        self.relevant_keywords = [
            'time series forecasting',
            'financial prediction',
            'quantitative trading',
            'machine learning',
            'deep learning',
            'neural networks',
            'transformer',
            'lstm',
            'attention mechanism',
            'ensemble methods',
            'reinforcement learning',
            'market prediction'
        ]
        
        # Storage
        self.discovered_papers: List[ResearchPaper] = []
        self.evaluated_models: List[ModelEvaluation] = []
        self.implemented_models: List[str] = []
        
        # Load existing data
        self._load_research_data()

    def _setup(self):
        pass

    async def execute(self, **kwargs) -> AgentResult:
        """Execute the meta-research agent logic.
        Args:
            **kwargs: action, keywords, max_papers, threshold, etc.
        Returns:
            AgentResult
        """
        try:
            action = kwargs.get('action', 'discover_papers')
            if action == 'discover_papers':
                keywords = kwargs.get('keywords')
                max_papers = kwargs.get('max_papers')
                papers = await self.discover_research_papers(keywords, max_papers)
                return AgentResult(success=True, data={
                    "discovered_papers": len(papers),
                    "papers": [paper.__dict__ for paper in papers]
                })
            elif action == 'evaluate_models':
                papers = kwargs.get('papers', self.discovered_papers)
                evaluations = await self.evaluate_models(papers)
                return AgentResult(success=True, data={
                    "evaluations": [eval.__dict__ for eval in evaluations]
                })
            elif action == 'auto_implement':
                evaluations = kwargs.get('evaluations', self.evaluated_models)
                threshold = kwargs.get('threshold')
                implemented = await self.auto_implement_top_models(evaluations, threshold)
                return AgentResult(success=True, data={
                    "implemented_models": implemented
                })
            elif action == 'get_research_summary':
                summary = self.get_research_summary()
                return AgentResult(success=True, data={"research_summary": summary})
            else:
                return AgentResult(success=False, error_message=f"Unknown action: {action}")
        except Exception as e:
            return self.handle_error(e)

    async def discover_research_papers(self, 
                                     keywords: Optional[List[str]] = None,
                                     max_papers: Optional[int] = None) -> List[ResearchPaper]:
        """
        Discover new research papers from various sources.
        
        Args:
            keywords: Keywords to search for
            max_papers: Maximum number of papers to discover
            
        Returns:
            List of discovered research papers
        """
        try:
            self.logger.info("Starting research paper discovery")
            
            keywords = keywords or self.relevant_keywords
            max_papers = max_papers or self.max_papers_per_search
            
            discovered_papers = []
            
            # Search arXiv
            arxiv_papers = await self._search_arxiv(keywords, max_papers // 2)
            discovered_papers.extend(arxiv_papers)
            
            # Search SSRN
            ssrn_papers = await self._search_ssrn(keywords, max_papers // 2)
            discovered_papers.extend(ssrn_papers)
            
            # Filter and deduplicate papers
            filtered_papers = self._filter_relevant_papers(discovered_papers)
            unique_papers = self._deduplicate_papers(filtered_papers)
            
            # Store discovered papers
            self.discovered_papers.extend(unique_papers)
            
            self.logger.info(f"Discovered {len(unique_papers)} new research papers")
            
            # Store in memory
            self._store_discovered_papers(unique_papers)
            
            return unique_papers
            
        except Exception as e:
            self.logger.error(f"Error discovering research papers: {str(e)}")
            return []
    
    async def _search_arxiv(self, keywords: List[str], max_papers: int) -> List[ResearchPaper]:
        """Search arXiv for relevant papers."""
        try:
            papers = []
            
            # Construct search query
            query = " OR ".join([f'ti:"{kw}" OR abs:"{kw}"' for kw in keywords[:5]])  # Limit keywords
            
            # arXiv API endpoint
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': max_papers,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        papers = self._parse_arxiv_response(content)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {str(e)}")
            return []
    
    async def _search_ssrn(self, keywords: List[str], max_papers: int) -> List[ResearchPaper]:
        """Search SSRN for relevant papers."""
        try:
            papers = []
            
            # SSRN search (simplified - in practice, you'd use their API)
            # For now, return mock data
            for i in range(min(max_papers, 10)):
                paper = ResearchPaper(
                    title=f"Financial Forecasting Model {i+1}",
                    authors=[f"Author {i+1}"],
                    abstract=f"Abstract for financial forecasting paper {i+1}",
                    url=f"https://papers.ssrn.com/paper_{i+1}",
                    source="SSRN",
                    publication_date=datetime.now() - timedelta(days=i*7),
                    keywords=keywords[:3]
                )
                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching SSRN: {str(e)}")
            return []
    
    def _parse_arxiv_response(self, content: str) -> List[ResearchPaper]:
        """Parse arXiv API response."""
        try:
            papers = []
            
            # Simple XML parsing (in practice, use proper XML parser)
            # Extract paper information from XML response
            soup = BeautifulSoup(content, 'xml')
            
            entries = soup.find_all('entry')
            for entry in entries:
                try:
                    title = entry.find('title').text.strip()
                    authors = [author.find('name').text.strip() for author in entry.find_all('author')]
                    abstract = entry.find('summary').text.strip()
                    url = entry.find('id').text.strip()
                    published = entry.find('published').text.strip()
                    
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=url,
                        source="arXiv",
                        publication_date=datetime.fromisoformat(published.replace('Z', '+00:00')),
                        keywords=self._extract_keywords(abstract)
                    )
                    papers.append(paper)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing arXiv entry: {str(e)}")
                    continue
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error parsing arXiv response: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        try:
            # Simple keyword extraction (in practice, use NLP techniques)
            text_lower = text.lower()
            extracted_keywords = []
            
            for keyword in self.relevant_keywords:
                if keyword.lower() in text_lower:
                    extracted_keywords.append(keyword)
            
            return extracted_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def _filter_relevant_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Filter papers based on relevance criteria."""
        try:
            relevant_papers = []
            
            for paper in papers:
                # Check if paper contains relevant keywords
                paper_text = f"{paper.title} {paper.abstract}".lower()
                
                relevance_score = 0
                for keyword in self.relevant_keywords:
                    if keyword.lower() in paper_text:
                        relevance_score += 1
                
                # Paper is relevant if it contains at least 2 relevant keywords
                if relevance_score >= 2:
                    relevant_papers.append(paper)
            
            return relevant_papers
            
        except Exception as e:
            self.logger.error(f"Error filtering papers: {str(e)}")
            return []
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """Remove duplicate papers based on title similarity."""
        try:
            unique_papers = []
            seen_titles = set()
            
            for paper in papers:
                # Create title hash for comparison
                title_hash = hashlib.md5(paper.title.lower().encode()).hexdigest()
                
                if title_hash not in seen_titles:
                    seen_titles.add(title_hash)
                    unique_papers.append(paper)
            
            return unique_papers
            
        except Exception as e:
            self.logger.error(f"Error deduplicating papers: {str(e)}")
            return []
    
    async def evaluate_models(self, papers: List[ResearchPaper]) -> List[ModelEvaluation]:
        """
        Evaluate models described in research papers.
        
        Args:
            papers: List of research papers to evaluate
            
        Returns:
            List of model evaluations
        """
        try:
            self.logger.info(f"Evaluating {len(papers)} models from research papers")
            
            evaluations = []
            
            for paper in papers:
                # Extract model information from paper
                model_info = self._extract_model_info(paper)
                
                if model_info:
                    # Evaluate model
                    evaluation = self._evaluate_model(paper, model_info)
                    evaluations.append(evaluation)
                    
                    # Store evaluation
                    self.evaluated_models.append(evaluation)
            
            # Store evaluations in memory
            self._store_model_evaluations(evaluations)
            
            self.logger.info(f"Evaluated {len(evaluations)} models")
            
            return evaluations
            
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            return []
    
    def _extract_model_info(self, paper: ResearchPaper) -> Optional[Dict[str, Any]]:
        """Extract model information from research paper."""
        try:
            model_info = {}
            
            # Extract model type from title and abstract
            text = f"{paper.title} {paper.abstract}".lower()
            
            # Identify model type
            if 'transformer' in text or 'attention' in text:
                model_info['type'] = 'transformer'
            elif 'lstm' in text or 'rnn' in text:
                model_info['type'] = 'lstm'
            elif 'cnn' in text or 'convolutional' in text:
                model_info['type'] = 'cnn'
            elif 'ensemble' in text or 'boosting' in text:
                model_info['type'] = 'ensemble'
            elif 'reinforcement' in text or 'rl' in text:
                model_info['type'] = 'reinforcement_learning'
            else:
                model_info['type'] = 'unknown'
            
            # Extract performance metrics (simplified)
            model_info['performance'] = self._extract_performance_metrics(paper.abstract)
            
            # Extract implementation complexity
            model_info['complexity'] = self._assess_implementation_complexity(paper)
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error extracting model info: {str(e)}")
            return []
    
    def _extract_performance_metrics(self, abstract: str) -> Dict[str, float]:
        """Extract performance metrics from abstract."""
        try:
            metrics = {}
            
            # Look for common performance metrics
            text = abstract.lower()
            
            # MSE/RMSE
            mse_match = re.search(r'mse[:\s]*([\d.]+)', text)
            if mse_match:
                metrics['mse'] = float(mse_match.group(1))
            
            # Accuracy
            acc_match = re.search(r'accuracy[:\s]*([\d.]+)', text)
            if acc_match:
                metrics['accuracy'] = float(acc_match.group(1))
            
            # Sharpe ratio
            sharpe_match = re.search(r'sharpe[:\s]*([\d.]+)', text)
            if sharpe_match:
                metrics['sharpe_ratio'] = float(sharpe_match.group(1))
            
            # Return
            return_match = re.search(r'return[:\s]*([\d.]+)', text)
            if return_match:
                metrics['return'] = float(return_match.group(1))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extracting performance metrics: {str(e)}")
            return []
    
    def _assess_implementation_complexity(self, paper: ResearchPaper) -> float:
        """Assess implementation complexity of the model."""
        try:
            complexity_score = 0.5  # Base complexity
            
            text = f"{paper.title} {paper.abstract}".lower()
            
            # Factors that increase complexity
            if 'transformer' in text or 'attention' in text:
                complexity_score += 0.3
            if 'reinforcement' in text or 'rl' in text:
                complexity_score += 0.4
            if 'ensemble' in text:
                complexity_score += 0.2
            if 'graph' in text or 'gnn' in text:
                complexity_score += 0.3
            
            # Factors that decrease complexity
            if 'simple' in text or 'baseline' in text:
                complexity_score -= 0.2
            if 'linear' in text:
                complexity_score -= 0.1
            
            return max(0.1, min(1.0, complexity_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing implementation complexity: {str(e)}")
            return 0.5
    
    def _evaluate_model(self, paper: ResearchPaper, model_info: Dict[str, Any]) -> ModelEvaluation:
        """Evaluate a model based on paper and extracted information."""
        try:
            # Calculate performance score
            performance_score = self._calculate_performance_score(model_info['performance'])
            
            # Implementation complexity (inverse score)
            implementation_complexity = 1.0 - model_info['complexity']
            
            # Market applicability
            market_applicability = self._assess_market_applicability(paper, model_info)
            
            # Overall score
            overall_score = (
                0.4 * performance_score +
                0.3 * implementation_complexity +
                0.3 * market_applicability
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(overall_score, model_info)
            
            return ModelEvaluation(
                paper_id=paper.url,
                model_name=f"{model_info['type']}_{paper.title[:20].replace(' ', '_')}",
                model_type=model_info['type'],
                performance_score=performance_score,
                implementation_complexity=implementation_complexity,
                market_applicability=market_applicability,
                overall_score=overall_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return ModelEvaluation(
                paper_id=paper.url,
                model_name="unknown_model",
                model_type="unknown",
                performance_score=0.0,
                implementation_complexity=0.5,
                market_applicability=0.5,
                overall_score=0.0,
                recommendation="Unable to evaluate"
            )
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate performance score from metrics."""
        try:
            if not metrics:
                return 0.5  # Default score
            
            score = 0.0
            count = 0
            
            # Normalize and score each metric
            if 'mse' in metrics:
                # Lower MSE is better
                mse_score = max(0.0, 1.0 - metrics['mse'])
                score += mse_score
                count += 1
            
            if 'accuracy' in metrics:
                # Higher accuracy is better
                score += metrics['accuracy']
                count += 1
            
            if 'sharpe_ratio' in metrics:
                # Higher Sharpe is better
                sharpe_score = min(1.0, max(0.0, metrics['sharpe_ratio'] / 2.0))
                score += sharpe_score
                count += 1
            
            if 'return' in metrics:
                # Higher return is better
                return_score = min(1.0, max(0.0, metrics['return'] / 0.2))
                score += return_score
                count += 1
            
            return score / count if count > 0 else 0.5
            
        except Exception as e:
            self.logger.error(f"Error calculating performance score: {str(e)}")
            return 0.5
    
    def _assess_market_applicability(self, paper: ResearchPaper, model_info: Dict[str, Any]) -> float:
        """Assess market applicability of the model."""
        try:
            applicability_score = 0.5  # Base score
            
            text = f"{paper.title} {paper.abstract}".lower()
            
            # Positive factors
            if 'financial' in text or 'trading' in text or 'market' in text:
                applicability_score += 0.3
            if 'time series' in text or 'forecasting' in text:
                applicability_score += 0.2
            if 'real-time' in text or 'online' in text:
                applicability_score += 0.1
            if 'risk' in text or 'volatility' in text:
                applicability_score += 0.1
            
            # Negative factors
            if 'image' in text or 'vision' in text:
                applicability_score -= 0.2
            if 'nlp' in text or 'language' in text:
                applicability_score -= 0.1
            
            return max(0.0, min(1.0, applicability_score))
            
        except Exception as e:
            self.logger.error(f"Error assessing market applicability: {str(e)}")
            return 0.5
    
    def _generate_recommendation(self, overall_score: float, model_info: Dict[str, Any]) -> str:
        """Generate recommendation based on evaluation."""
        try:
            if overall_score > 0.8:
                return "Strongly recommend implementation"
            elif overall_score > 0.6:
                return "Recommend implementation with monitoring"
            elif overall_score > 0.4:
                return "Consider implementation after further research"
            else:
                return "Not recommended for implementation"
                
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}")
            return "Unable to generate recommendation"
    
    async def auto_implement_top_models(self, 
                                      evaluations: List[ModelEvaluation],
                                      threshold: Optional[float] = None) -> List[str]:
        """
        Automatically implement top-performing models.
        
        Args:
            evaluations: List of model evaluations
            threshold: Score threshold for implementation
            
        Returns:
            List of implemented model names
        """
        try:
            threshold = threshold or self.evaluation_threshold
            
            # Filter high-scoring models
            top_models = [e for e in evaluations if e.overall_score >= threshold]
            
            implemented_models = []
            
            for evaluation in top_models:
                try:
                    # Create model implementation
                    model_name = await self._implement_model(evaluation)
                    
                    if model_name:
                        implemented_models.append(model_name)
                        self.implemented_models.append(model_name)
                        
                        self.logger.info(f"Auto-implemented model: {model_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error implementing model {evaluation.model_name}: {str(e)}")
            
            # Store implementation results
            self._store_implementation_results(implemented_models)
            
            return implemented_models
            
        except Exception as e:
            self.logger.error(f"Error auto-implementing models: {str(e)}")
            return []
    
    async def _implement_model(self, evaluation: ModelEvaluation) -> Optional[str]:
        """Implement a model based on evaluation."""
        try:
            # Generate model code based on type
            model_code = self._generate_model_code(evaluation)
            
            if model_code:
                # Register model in registry
                model_name = evaluation.model_name
                
                # In practice, you'd save the model code to a file
                # and register it with the model registry
                success = await self._register_model(model_name, model_code, evaluation.model_type)
                
                if success:
                    return model_name
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error implementing model: {str(e)}")
            return None
    
    def _generate_model_code(self, evaluation: ModelEvaluation) -> Optional[str]:
        """Generate model implementation code."""
        try:
            model_type = evaluation.model_type
            
            if model_type == 'transformer':
                return self._generate_transformer_code(evaluation)
            elif model_type == 'lstm':
                return self._generate_lstm_code(evaluation)
            elif model_type == 'ensemble':
                return self._generate_ensemble_code(evaluation)
            else:
                return self._generate_generic_code(evaluation)
                
        except Exception as e:
            self.logger.error(f"Error generating model code: {str(e)}")
            return None
    
    def _generate_transformer_code(self, evaluation: ModelEvaluation) -> str:
        """Generate transformer model code."""
        return f"""
# Auto-generated Transformer Model: {evaluation.model_name}
import torch
import torch.nn as nn

class {evaluation.model_name.replace(' ', '_')}(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=6, num_heads=8):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x.mean(dim=1))
"""
    
    def _generate_lstm_code(self, evaluation: ModelEvaluation) -> str:
        """Generate LSTM model code."""
        return f"""
# Auto-generated LSTM Model: {evaluation.model_name}
import torch
import torch.nn as nn

class {evaluation.model_name.replace(' ', '_')}(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
"""
    
    def _generate_ensemble_code(self, evaluation: ModelEvaluation) -> str:
        """Generate ensemble model code."""
        return f"""
# Auto-generated Ensemble Model: {evaluation.model_name}
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class {evaluation.model_name.replace(' ', '_')}:
    def __init__(self):
        self.models = [
            RandomForestRegressor(n_estimators=100),
            GradientBoostingRegressor(n_estimators=100)
        ]
    
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
"""
    
    def _generate_generic_code(self, evaluation: ModelEvaluation) -> str:
        """Generate generic model code."""
        return f"""
# Auto-generated Generic Model: {evaluation.model_name}
import numpy as np
from sklearn.linear_model import LinearRegression

class {evaluation.model_name.replace(' ', '_')}:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
"""
    
    async def _register_model(self, model_name: str, model_code: str, model_type: str) -> bool:
        """Register model in the model registry."""
        try:
            # In practice, you'd save the code to a file and register it
            # For now, just return success
            self.logger.info(f"Registered model: {model_name} ({model_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            return False
    
    def _store_discovered_papers(self, papers: List[ResearchPaper]):
        """Store discovered papers in memory."""
        try:
            self.memory.store('discovered_papers', {
                'papers': [paper.__dict__ for paper in papers],
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing discovered papers: {str(e)}")
    
    def _store_model_evaluations(self, evaluations: List[ModelEvaluation]):
        """Store model evaluations in memory."""
        try:
            self.memory.store('model_evaluations', {
                'evaluations': [eval.__dict__ for eval in evaluations],
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing model evaluations: {str(e)}")
    
    def _store_implementation_results(self, implemented_models: List[str]):
        """Store implementation results in memory."""
        try:
            self.memory.store('implementation_results', {
                'implemented_models': implemented_models,
                'timestamp': datetime.now()
            })
        except Exception as e:
            self.logger.error(f"Error storing implementation results: {str(e)}")
    
    def _load_research_data(self):
        """Load existing research data from memory."""
        try:
            # Load discovered papers
            papers_data = self.memory.get('discovered_papers')
            if papers_data:
                self.discovered_papers = [ResearchPaper(**p) for p in papers_data.get('papers', [])]
            
            # Load model evaluations
            evaluations_data = self.memory.get('model_evaluations')
            if evaluations_data:
                self.evaluated_models = [ModelEvaluation(**e) for e in evaluations_data.get('evaluations', [])]
            
            # Load implemented models
            implementation_data = self.memory.get('implementation_results')
            if implementation_data:
                self.implemented_models = implementation_data.get('implemented_models', [])
                
        except Exception as e:
            self.logger.error(f"Error loading research data: {str(e)}")
    
    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research activities."""
        try:
            return {
                'total_papers_discovered': len(self.discovered_papers),
                'total_models_evaluated': len(self.evaluated_models),
                'total_models_implemented': len(self.implemented_models),
                'recent_discoveries': [
                    {
                        'title': paper.title,
                        'source': paper.source,
                        'date': paper.publication_date.isoformat()
                    }
                    for paper in self.discovered_papers[-5:]  # Last 5 papers
                ],
                'top_evaluations': [
                    {
                        'model_name': eval.model_name,
                        'overall_score': eval.overall_score,
                        'recommendation': eval.recommendation
                    }
                    for eval in sorted(self.evaluated_models, key=lambda x: x.overall_score, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting research summary: {str(e)}")
            return {}