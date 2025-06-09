import logging
from typing import Dict, List, Optional, Union, Any, Set, Callable
from pathlib import Path
import re
from datetime import datetime, timedelta
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import aiohttp
import asyncio
import elasticsearch
import prometheus_client
import grafana_api
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models
import wordcloud
from wordcloud import WordCloud
import jupyter
from jupyter_client import KernelManager
import nbformat
from nbconvert import HTMLExporter
import docutils
from docutils.core import publish_doctree
import sphinx
from sphinx.application import Sphinx
import mistune
from mistune import Markdown
import markdown
from bs4 import BeautifulSoup
import aiofiles
import frontmatter
import pygments
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import rst2html
import rst2pdf
import doc8
import restructuredtext_lint
import markdownlint
import remark
import prettier
import black
import isort
import mypy
import pyright
import pytype
import pyre
import jedi
import rope
import autopep8
import yapf
import flake8
import pydocstyle
import radon
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.visitors import ComplexityVisitor
import mccabe
import xenon
import vulture
import pyflakes
import pycodestyle
from prometheus_client import Counter, Gauge, Histogram, Summary
import textstat
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
import jinja2
import pdfkit
from pydantic import BaseModel, Field, validator
from cachetools import TTLCache, LRUCache
import jwt
import bleach
from ratelimit import limits, sleep_and_retry
import hashlib
import uuid
from typing_extensions import TypedDict

class AnalyticsValidationError(Exception):
    """Raised when analytics validation fails."""
    pass

class AnalyticsAccessError(Exception):
    """Raised when access to analytics is denied."""
    pass

class AnalyticsHealth(BaseModel):
    """Health status of analytics system."""
    status: str
    last_check: datetime
    errors: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

class AnalyticsPlugin(BaseModel):
    """Plugin for extending analytics functionality."""
    name: str
    version: str
    description: str
    metrics: List[str]
    visualizations: List[str]
    handler: Callable

class AnalyticsRetention(BaseModel):
    """Data retention policy."""
    metric_type: str
    retention_days: int
    archive_after_days: int
    delete_after_days: int

@dataclass
class AnalyticsMetric:
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    retention_policy: Optional[AnalyticsRetention] = None

@dataclass
class AnalyticsReport:
    title: str
    metrics: List[AnalyticsMetric]
    visualizations: List[Dict[str, Any]]
    insights: List[str]
    generated_at: datetime
    generated_by: str
    access_control: Dict[str, List[str]] = field(default_factory=dict)
    retention_policy: Optional[AnalyticsRetention] = None

class DocumentationAnalytics:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.analytics_config = config.get('documentation', {}).get('analytics', {})
        self.setup_prometheus()
        self.setup_nlp()
        self.setup_caching()
        self.setup_security()
        self.setup_plugins()
        self.setup_retention()
        self.metrics: List[AnalyticsMetric] = []
        self.reports: List[AnalyticsReport] = []
        self.plugins: Dict[str, AnalyticsPlugin] = {}
        self.lock = asyncio.Lock()
        self.health = AnalyticsHealth(
            status="initializing",
            last_check=datetime.now()
        )

    def setup_logging(self):
        """Configure logging for the analytics system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "analytics.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_prometheus(self):
        """Setup Prometheus metrics."""
        self.page_views = Counter('documentation_page_views_total', 'Total page views')
        self.search_queries = Counter('documentation_search_queries_total', 'Total search queries')
        self.validation_scores = Gauge('documentation_validation_scores', 'Validation scores')
        self.version_changes = Counter('documentation_version_changes_total', 'Total version changes')
        self.user_feedback = Counter('documentation_user_feedback_total', 'Total user feedback')
        self.code_usage = Counter('documentation_code_usage_total', 'Total code usage')
        self.documentation_changes = Counter('documentation_changes_total', 'Total documentation changes')
        
        # New metrics
        self.sentiment_scores = Gauge('documentation_sentiment_scores', 'Sentiment scores')
        self.topic_distribution = Gauge('documentation_topic_distribution', 'Topic distribution')
        self.correlation_scores = Gauge('documentation_correlation_scores', 'Correlation scores')
        self.trend_scores = Gauge('documentation_trend_scores', 'Trend scores')
        self.anomaly_scores = Gauge('documentation_anomaly_scores', 'Anomaly detection scores')
        self.engagement_scores = Gauge('documentation_engagement_scores', 'User engagement scores')
        self.quality_scores = Gauge('documentation_quality_scores', 'Documentation quality scores')
        self.readability_scores = Gauge('documentation_readability_scores', 'Readability scores')
        self.completeness_scores = Gauge('documentation_completeness_scores', 'Completeness scores')
        self.accuracy_scores = Gauge('documentation_accuracy_scores', 'Accuracy scores')
        
        # A/B testing metrics
        self.experiment_views = Counter('documentation_experiment_views_total', 'Total experiment views')
        self.experiment_conversions = Counter('documentation_experiment_conversions_total', 'Total experiment conversions')
        self.experiment_success_rate = Gauge('documentation_experiment_success_rate', 'Experiment success rate')

        # New metrics for system health
        self.system_errors = Counter('documentation_analytics_errors_total', 'Total system errors')
        self.system_warnings = Counter('documentation_analytics_warnings_total', 'Total system warnings')
        self.operation_latency = Histogram('documentation_analytics_operation_latency_seconds', 'Operation latency')
        self.cache_hits = Counter('documentation_analytics_cache_hits_total', 'Total cache hits')
        self.cache_misses = Counter('documentation_analytics_cache_misses_total', 'Total cache misses')
        self.plugin_executions = Counter('documentation_analytics_plugin_executions_total', 'Total plugin executions')
        self.data_retention_operations = Counter('documentation_analytics_retention_operations_total', 'Total retention operations')

    def setup_caching(self):
        """Setup caching for expensive operations."""
        self.metric_cache = TTLCache(
            maxsize=self.analytics_config.get('cache_size', 1000),
            ttl=self.analytics_config.get('cache_ttl', 3600)
        )
        self.visualization_cache = LRUCache(
            maxsize=self.analytics_config.get('visualization_cache_size', 100)
        )
        self.insight_cache = TTLCache(
            maxsize=self.analytics_config.get('insight_cache_size', 100),
            ttl=self.analytics_config.get('insight_cache_ttl', 1800)
        )

    def setup_security(self):
        """Setup security features."""
        self.jwt_secret = self.analytics_config.get('jwt_secret', 'your-secret-key')
        self.access_control = self.analytics_config.get('access_control', {})
        self.rate_limits = self.analytics_config.get('rate_limits', {})
        self.audit_log = []

    def setup_plugins(self):
        """Setup analytics plugins."""
        plugin_config = self.analytics_config.get('plugins', {})
        for name, config in plugin_config.items():
            try:
                plugin = AnalyticsPlugin(
                    name=name,
                    version=config.get('version', '1.0.0'),
                    description=config.get('description', ''),
                    metrics=config.get('metrics', []),
                    visualizations=config.get('visualizations', []),
                    handler=config.get('handler')
                )
                self.plugins[name] = plugin
            except Exception as e:
                self.logger.error(f"Plugin setup error for {name}: {str(e)}")

    def setup_retention(self):
        """Setup data retention policies."""
        retention_config = self.analytics_config.get('retention', {})
        self.retention_policies = {}
        for metric_type, policy in retention_config.items():
            try:
                self.retention_policies[metric_type] = AnalyticsRetention(
                    metric_type=metric_type,
                    retention_days=policy.get('retention_days', 90),
                    archive_after_days=policy.get('archive_after_days', 30),
                    delete_after_days=policy.get('delete_after_days', 365)
                )
            except Exception as e:
                self.logger.error(f"Retention policy setup error for {metric_type}: {str(e)}")

    def validate_input(self, data: Any, schema: Dict) -> bool:
        """Validate input data against schema."""
        try:
            for field, rules in schema.items():
                if field not in data:
                    if rules.get('required', False):
                        raise AnalyticsValidationError(f"Missing required field: {field}")
                    continue
                
                value = data[field]
                if 'type' in rules and not isinstance(value, rules['type']):
                    raise AnalyticsValidationError(f"Invalid type for {field}: expected {rules['type']}")
                
                if 'min' in rules and value < rules['min']:
                    raise AnalyticsValidationError(f"Value too small for {field}: {value} < {rules['min']}")
                
                if 'max' in rules and value > rules['max']:
                    raise AnalyticsValidationError(f"Value too large for {field}: {value} > {rules['max']}")
                
                if 'pattern' in rules and not re.match(rules['pattern'], str(value)):
                    raise AnalyticsValidationError(f"Invalid format for {field}: {value}")
            
            return True
        except Exception as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return False

    def check_access(self, user_id: str, resource: str, action: str) -> bool:
        """Check if user has access to perform action on resource."""
        try:
            if user_id not in self.access_control:
                return False
            
            user_permissions = self.access_control[user_id]
            required_permission = f"{resource}:{action}"
            
            return required_permission in user_permissions or 'admin' in user_permissions
        except Exception as e:
            self.logger.error(f"Access check error: {str(e)}")
            return False

    def sanitize_content(self, content: str) -> str:
        """Sanitize user-generated content."""
        try:
            # Remove potentially harmful HTML
            sanitized = bleach.clean(content)
            
            # Remove control characters
            sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
            
            # Normalize whitespace
            sanitized = ' '.join(sanitized.split())
            
            return sanitized
        except Exception as e:
            self.logger.error(f"Content sanitization error: {str(e)}")
            return content

    def audit_log_operation(self, user_id: str, action: str, resource: str, status: str):
        """Log an audit event."""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'status': status
            }
            self.audit_log.append(event)
            
            # Export to external audit system if configured
            if self.analytics_config.get('audit_export'):
                # Export logic here
                pass
        except Exception as e:
            self.logger.error(f"Audit logging error: {str(e)}")

    @sleep_and_retry
    @limits(calls=100, period=60)
    async def track_page_view(self, page: str, user: str, metadata: Optional[Dict] = None):
        """Track a page view with rate limiting and validation."""
        try:
            # Validate input
            if not self.validate_input({'page': page, 'user': user}, {
                'page': {'type': str, 'required': True},
                'user': {'type': str, 'required': True}
            }):
                raise AnalyticsValidationError("Invalid input for page view tracking")
            
            # Check access
            if not self.check_access(user, 'analytics', 'track'):
                raise AnalyticsAccessError(f"User {user} does not have permission to track page views")
            
            # Sanitize input
            page = self.sanitize_content(page)
            user = self.sanitize_content(user)
            metadata = {k: self.sanitize_content(str(v)) for k, v in (metadata or {}).items()}
            
            async with self.lock:
                self.page_views.inc()
                metric = AnalyticsMetric(
                    name='page_view',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'page': page, 'user': user, **(metadata or {})},
                    user_id=user,
                    session_id=metadata.get('session_id'),
                    retention_policy=self.retention_policies.get('page_view')
                )
                self.metrics.append(metric)
                
                # Audit log
                self.audit_log_operation(user, 'track_page_view', page, 'success')
                
                self.logger.info(f"Tracked page view: {page} by {user}")
        except Exception as e:
            self.logger.error(f"Page view tracking error: {str(e)}")
            self.system_errors.inc()
            self.audit_log_operation(user, 'track_page_view', page, 'error')

    async def check_health(self) -> AnalyticsHealth:
        """Check the health of the analytics system."""
        try:
            health = AnalyticsHealth(
                status="healthy",
                last_check=datetime.now()
            )
            
            # Check Prometheus metrics
            try:
                prometheus_client.generate_latest()
            except Exception as e:
                health.status = "degraded"
                health.errors.append(f"Prometheus error: {str(e)}")
            
            # Check Elasticsearch connection
            try:
                await self.es.ping()
            except Exception as e:
                health.status = "degraded"
                health.errors.append(f"Elasticsearch error: {str(e)}")
            
            # Check Grafana connection
            try:
                self.grafana.health.check()
            except Exception as e:
                health.status = "degraded"
                health.errors.append(f"Grafana error: {str(e)}")
            
            # Check cache health
            if len(self.metric_cache) > self.metric_cache.maxsize * 0.9:
                health.warnings.append("Metric cache near capacity")
            
            # Check disk space
            try:
                disk_usage = Path("automation/logs/documentation").stat().st_size
                if disk_usage > 1e9:  # 1GB
                    health.warnings.append("Log directory size exceeds 1GB")
            except Exception as e:
                health.warnings.append(f"Disk space check error: {str(e)}")
            
            # Update metrics
            health.metrics = {
                'cache_size': len(self.metric_cache),
                'metrics_count': len(self.metrics),
                'reports_count': len(self.reports),
                'plugins_count': len(self.plugins)
            }
            
            self.health = health
            return health
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            return AnalyticsHealth(
                status="error",
                last_check=datetime.now(),
                errors=[str(e)]
            )

    async def apply_retention_policies(self):
        """Apply data retention policies."""
        try:
            now = datetime.now()
            for metric in self.metrics:
                if not metric.retention_policy:
                    continue
                
                age_days = (now - metric.timestamp).days
                
                if age_days > metric.retention_policy.delete_after_days:
                    # Delete metric
                    self.metrics.remove(metric)
                    self.data_retention_operations.inc()
                elif age_days > metric.retention_policy.archive_after_days:
                    # Archive metric
                    # Archive logic here
                    self.data_retention_operations.inc()
            
            self.logger.info("Applied retention policies")
        except Exception as e:
            self.logger.error(f"Retention policy application error: {str(e)}")

    async def register_plugin(self, plugin: AnalyticsPlugin):
        """Register a new analytics plugin."""
        try:
            if plugin.name in self.plugins:
                raise ValueError(f"Plugin {plugin.name} already exists")
            
            self.plugins[plugin.name] = plugin
            self.logger.info(f"Registered plugin: {plugin.name}")
        except Exception as e:
            self.logger.error(f"Plugin registration error: {str(e)}")

    async def execute_plugin(self, plugin_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an analytics plugin."""
        try:
            if plugin_name not in self.plugins:
                raise ValueError(f"Plugin {plugin_name} not found")
            
            plugin = self.plugins[plugin_name]
            result = await plugin.handler(data)
            
            self.plugin_executions.inc()
            return result
        except Exception as e:
            self.logger.error(f"Plugin execution error: {str(e)}")
            return {}

    def setup_nlp(self):
        """Setup NLP tools."""
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('en_core_web_sm')

    def setup_external_tools(self):
        """Setup external analytics tools."""
        # Setup Grafana
        self.grafana = grafana_api.GrafanaAPI(
            auth=self.analytics_config.get('grafana', {}).get('auth'),
            host=self.analytics_config.get('grafana', {}).get('host')
        )
        
        # Setup Elasticsearch
        self.es = elasticsearch.AsyncElasticsearch(
            hosts=self.analytics_config.get('elasticsearch', {}).get('hosts')
        )

    async def track_search_query(self, query: str, user: str, metadata: Optional[Dict] = None):
        """Track a search query."""
        try:
            async with self.lock:
                self.search_queries.inc()
                metric = AnalyticsMetric(
                    name='search_query',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'query': query, 'user': user, **(metadata or {})}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked search query: {query} by {user}")
        except Exception as e:
            self.logger.error(f"Search query tracking error: {str(e)}")

    async def track_validation_score(self, score: float, metadata: Optional[Dict] = None):
        """Track a validation score."""
        try:
            async with self.lock:
                self.validation_scores.set(score)
                metric = AnalyticsMetric(
                    name='validation_score',
                    value=score,
                    timestamp=datetime.now(),
                    metadata=metadata or {}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked validation score: {score}")
        except Exception as e:
            self.logger.error(f"Validation score tracking error: {str(e)}")

    async def track_version_change(self, version: str, user: str, metadata: Optional[Dict] = None):
        """Track a version change."""
        try:
            async with self.lock:
                self.version_changes.inc()
                metric = AnalyticsMetric(
                    name='version_change',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'version': version, 'user': user, **(metadata or {})}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked version change: {version} by {user}")
        except Exception as e:
            self.logger.error(f"Version change tracking error: {str(e)}")

    async def track_user_feedback(self, feedback: str, user: str, metadata: Optional[Dict] = None):
        """Track user feedback."""
        try:
            async with self.lock:
                self.user_feedback.inc()
                metric = AnalyticsMetric(
                    name='user_feedback',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'feedback': feedback, 'user': user, **(metadata or {})}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked user feedback: {feedback} by {user}")
        except Exception as e:
            self.logger.error(f"User feedback tracking error: {str(e)}")

    async def track_code_usage(self, code: str, user: str, metadata: Optional[Dict] = None):
        """Track code usage."""
        try:
            async with self.lock:
                self.code_usage.inc()
                metric = AnalyticsMetric(
                    name='code_usage',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'code': code, 'user': user, **(metadata or {})}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked code usage: {code} by {user}")
        except Exception as e:
            self.logger.error(f"Code usage tracking error: {str(e)}")

    async def track_documentation_change(self, change: str, user: str, metadata: Optional[Dict] = None):
        """Track documentation change."""
        try:
            async with self.lock:
                self.documentation_changes.inc()
                metric = AnalyticsMetric(
                    name='documentation_change',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={'change': change, 'user': user, **(metadata or {})}
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked documentation change: {change} by {user}")
        except Exception as e:
            self.logger.error(f"Documentation change tracking error: {str(e)}")

    async def track_experiment(self, experiment_id: str, variant: str, user: str, action: str, metadata: Optional[Dict] = None):
        """Track an experiment interaction."""
        try:
            async with self.lock:
                self.experiment_views.inc()
                if action == 'conversion':
                    self.experiment_conversions.inc()
                
                metric = AnalyticsMetric(
                    name='experiment',
                    value=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        'experiment_id': experiment_id,
                        'variant': variant,
                        'user': user,
                        'action': action,
                        **(metadata or {})
                    }
                )
                self.metrics.append(metric)
                self.logger.info(f"Tracked experiment: {experiment_id} variant {variant} by {user}")
        except Exception as e:
            self.logger.error(f"Experiment tracking error: {str(e)}")

    async def generate_report(self, title: str) -> AnalyticsReport:
        """Generate an analytics report."""
        try:
            async with self.lock:
                visualizations = await self._generate_visualizations()
                insights = await self._generate_insights()
                report = AnalyticsReport(
                    title=title,
                    metrics=self.metrics,
                    visualizations=visualizations,
                    insights=insights,
                    generated_at=datetime.now(),
                    generated_by=self.config.get('user', 'unknown')
                )
                self.reports.append(report)
                self.logger.info(f"Generated report: {title}")
                return report
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            raise

    async def _generate_visualizations(self) -> List[Dict[str, Any]]:
        """Generate visualizations from metrics."""
        try:
            visualizations = []
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                # Create subplot figure for time series
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        'Page Views Over Time',
                        'Search Queries Over Time',
                        'Validation Scores Over Time',
                        'Version Changes Over Time',
                        'User Feedback Over Time',
                        'Documentation Changes Over Time'
                    )
                )

                # Add traces for each metric
                row = 1
                col = 1
                for name in ['page_view', 'search_query', 'validation_score', 'version_change', 'user_feedback', 'documentation_change']:
                    metric_data = df[df['name'] == name]
                    if not metric_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=metric_data['timestamp'],
                                y=metric_data['value'],
                                name=name,
                                mode='lines+markers'
                            ),
                            row=row, col=col
                        )
                    col += 1
                    if col > 2:
                        col = 1
                        row += 1

                fig.update_layout(height=900, showlegend=True)
                visualizations.append({'type': 'subplot', 'data': fig.to_dict()})

                # Heatmap of usage patterns
                usage_matrix = pd.pivot_table(
                    df,
                    values='value',
                    index=df['timestamp'].dt.date,
                    columns='name',
                    aggfunc='sum'
                ).fillna(0)
                fig = px.imshow(
                    usage_matrix,
                    title='Usage Patterns Heatmap',
                    labels=dict(x='Metric', y='Date', color='Value'),
                    color_continuous_scale='Viridis'
                )
                visualizations.append({'type': 'heatmap', 'data': fig.to_dict()})

                # Network graph of documentation relationships
                if 'dependencies' in df.columns:
                    G = nx.Graph()
                    for _, row in df.iterrows():
                        if 'dependencies' in row['metadata']:
                            for dep in row['metadata']['dependencies']:
                                G.add_edge(row['name'], dep)
                    pos = nx.spring_layout(G)
                    fig = go.Figure()
                    for edge in G.edges():
                        fig.add_trace(go.Scatter(
                            x=[pos[edge[0]][0], pos[edge[1]][0]],
                            y=[pos[edge[0]][1], pos[edge[1]][1]],
                            mode='lines',
                            line=dict(width=1, color='#888'),
                            hoverinfo='none'
                        ))
                    for node in G.nodes():
                        fig.add_trace(go.Scatter(
                            x=[pos[node][0]],
                            y=[pos[node][1]],
                            mode='markers+text',
                            marker=dict(size=10),
                            text=node,
                            hoverinfo='text'
                        ))
                    fig.update_layout(
                        title='Documentation Relationships',
                        showlegend=False,
                        hovermode='closest'
                    )
                    visualizations.append({'type': 'network', 'data': fig.to_dict()})

                # Word cloud of content
                if 'content' in df.columns:
                    text = ' '.join(df['content'].dropna())
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis'
                    ).generate(text)
                    fig = px.imshow(
                        wordcloud,
                        title='Content Word Cloud'
                    )
                    visualizations.append({'type': 'wordcloud', 'data': fig.to_dict()})

                # Box plots for metric distributions
                fig = go.Figure()
                for name in df['name'].unique():
                    metric_data = df[df['name'] == name]['value']
                    fig.add_trace(go.Box(
                        y=metric_data,
                        name=name,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                fig.update_layout(
                    title='Metric Distributions',
                    yaxis_title='Value',
                    showlegend=True
                )
                visualizations.append({'type': 'boxplot', 'data': fig.to_dict()})

                # Correlation matrix heatmap
                corr_matrix = df.pivot_table(
                    values='value',
                    index='timestamp',
                    columns='name'
                ).corr()
                fig = px.imshow(
                    corr_matrix,
                    title='Metric Correlations',
                    color_continuous_scale='RdBu',
                    labels=dict(x='Metric', y='Metric', color='Correlation')
                )
                visualizations.append({'type': 'correlation', 'data': fig.to_dict()})

                # Time series decomposition
                for name in df['name'].unique():
                    metric_data = df[df['name'] == name]
                    if len(metric_data) > 1:
                        ts_data = metric_data.set_index('timestamp')['value']
                        decomposition = seasonal_decompose(ts_data, period=7)
                        
                        fig = make_subplots(
                            rows=3, cols=1,
                            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual')
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=ts_data.index, y=ts_data, name='Original'),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'),
                            row=2, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'),
                            row=3, col=1
                        )
                        
                        fig.update_layout(height=900, title=f'Time Series Decomposition - {name}')
                        visualizations.append({'type': 'decomposition', 'data': fig.to_dict()})

            return visualizations
        except Exception as e:
            self.logger.error(f"Visualization generation error: {str(e)}")
            return []

    async def _generate_insights(self) -> List[str]:
        """Generate insights from metrics."""
        try:
            insights = []
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                # Page views insights
                page_views = df[df['name'] == 'page_view']
                if not page_views.empty:
                    insights.append(f"Total page views: {page_views['value'].sum()}")
                    insights.append(f"Average page views per day: {page_views.groupby(page_views['timestamp'].dt.date)['value'].sum().mean()}")
                # Search queries insights
                search_queries = df[df['name'] == 'search_query']
                if not search_queries.empty:
                    insights.append(f"Total search queries: {search_queries['value'].sum()}")
                    insights.append(f"Average search queries per day: {search_queries.groupby(search_queries['timestamp'].dt.date)['value'].sum().mean()}")
                # Validation scores insights
                validation_scores = df[df['name'] == 'validation_score']
                if not validation_scores.empty:
                    insights.append(f"Average validation score: {validation_scores['value'].mean()}")
                    insights.append(f"Highest validation score: {validation_scores['value'].max()}")
                    insights.append(f"Lowest validation score: {validation_scores['value'].min()}")
                # Version changes insights
                version_changes = df[df['name'] == 'version_change']
                if not version_changes.empty:
                    insights.append(f"Total version changes: {version_changes['value'].sum()}")
                    insights.append(f"Average version changes per day: {version_changes.groupby(version_changes['timestamp'].dt.date)['value'].sum().mean()}")
                # User feedback insights
                user_feedback = df[df['name'] == 'user_feedback']
                if not user_feedback.empty:
                    insights.append(f"Total user feedback: {user_feedback['value'].sum()}")
                    insights.append(f"Average user feedback per day: {user_feedback.groupby(user_feedback['timestamp'].dt.date)['value'].sum().mean()}")
                # Code usage insights
                code_usage = df[df['name'] == 'code_usage']
                if not code_usage.empty:
                    insights.append(f"Total code usage: {code_usage['value'].sum()}")
                    insights.append(f"Average code usage per day: {code_usage.groupby(code_usage['timestamp'].dt.date)['value'].sum().mean()}")
                # Documentation changes insights
                documentation_changes = df[df['name'] == 'documentation_change']
                if not documentation_changes.empty:
                    insights.append(f"Total documentation changes: {documentation_changes['value'].sum()}")
                    insights.append(f"Average documentation changes per day: {documentation_changes.groupby(documentation_changes['timestamp'].dt.date)['value'].sum().mean()}")
            return insights
        except Exception as e:
            self.logger.error(f"Insight generation error: {str(e)}")
            return []

    async def export_report(self, report: AnalyticsReport, format: str = 'html', output_path: Optional[str] = None) -> str:
        """Export a report to a file."""
        try:
            if format == 'html':
                output_path = output_path or f"documentation_analytics_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.html"
                with open(output_path, 'w') as f:
                    f.write(f"<h1>{report.title}</h1>")
                    f.write(f"<p>Generated at: {report.generated_at}</p>")
                    f.write("<h2>Metrics</h2>")
                    for metric in report.metrics:
                        f.write(f"<p>{metric.name}: {metric.value} at {metric.timestamp}</p>")
                    f.write("<h2>Visualizations</h2>")
                    for viz in report.visualizations:
                        f.write(f"<div>{viz['data']}</div>")
                    f.write("<h2>Insights</h2>")
                    for insight in report.insights:
                        f.write(f"<p>{insight}</p>")
            elif format == 'pdf':
                output_path = output_path or f"documentation_analytics_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.pdf"
                # PDF generation logic here
            elif format == 'json':
                output_path = output_path or f"documentation_analytics_report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_path, 'w') as f:
                    json.dump({
                        'title': report.title,
                        'metrics': [vars(m) for m in report.metrics],
                        'visualizations': report.visualizations,
                        'insights': report.insights,
                        'generated_at': report.generated_at.isoformat()
                    }, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            return output_path
        except Exception as e:
            self.logger.error(f"Report export error: {str(e)}")
            raise

    async def clear_metrics(self):
        """Clear all metrics."""
        try:
            async with self.lock:
                self.metrics = []
                self.logger.info("Cleared all metrics")
        except Exception as e:
            self.logger.error(f"Metrics clearing error: {str(e)}")

    async def clear_reports(self):
        """Clear all reports."""
        try:
            async with self.lock:
                self.reports = []
                self.logger.info("Cleared all reports")
        except Exception as e:
            self.logger.error(f"Reports clearing error: {str(e)}")

    async def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text."""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            self.sentiment_scores.set(sentiment)
            return sentiment
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            return 0.0

    async def analyze_topics(self, texts: List[str], num_topics: int = 5) -> List[Dict[str, float]]:
        """Analyze topics in texts."""
        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                doc = self.nlp(text)
                tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
                processed_texts.append(tokens)
            
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(processed_texts)
            corpus = [dictionary.doc2bow(text) for text in processed_texts]
            
            # Train LDA model
            lda_model = models.LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary
            )
            
            # Get topic distribution
            topic_dist = []
            for doc in corpus:
                topics = lda_model.get_document_topics(doc)
                topic_dist.append({f"topic_{i}": score for i, score in topics})
            
            # Update metric
            for i, dist in enumerate(topic_dist):
                self.topic_distribution.set(dist.get(f"topic_{i}", 0.0))
            
            return topic_dist
        except Exception as e:
            self.logger.error(f"Topic analysis error: {str(e)}")
            return []

    async def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between metrics."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                # Calculate correlations
                correlations = {}
                metric_names = df['name'].unique()
                for i, name1 in enumerate(metric_names):
                    for name2 in metric_names[i+1:]:
                        corr = df[df['name'] == name1]['value'].corr(df[df['name'] == name2]['value'])
                        correlations[f"{name1}_{name2}"] = corr
                        self.correlation_scores.set(corr)
                return correlations
            return {}
        except Exception as e:
            self.logger.error(f"Correlation analysis error: {str(e)}")
            return {}

    async def analyze_trends(self) -> Dict[str, float]:
        """Analyze trends in metrics."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                # Calculate trends
                trends = {}
                for name in df['name'].unique():
                    metric_data = df[df['name'] == name]
                    if len(metric_data) > 1:
                        # Calculate linear regression
                        x = np.arange(len(metric_data))
                        y = metric_data['value'].values
                        slope = np.polyfit(x, y, 1)[0]
                        trends[name] = slope
                        self.trend_scores.set(slope)
                return trends
            return {}
        except Exception as e:
            self.logger.error(f"Trend analysis error: {str(e)}")
            return {}

    async def export_to_grafana(self, report: AnalyticsReport):
        """Export metrics to Grafana."""
        try:
            # Create dashboard
            dashboard = {
                'dashboard': {
                    'title': report.title,
                    'panels': []
                },
                'overwrite': True
            }
            
            # Add panels for each metric
            for metric in report.metrics:
                panel = {
                    'title': metric.name,
                    'type': 'graph',
                    'datasource': 'Prometheus',
                    'targets': [{
                        'expr': f'documentation_{metric.name}_total'
                    }]
                }
                dashboard['dashboard']['panels'].append(panel)
            
            # Create/update dashboard
            self.grafana.dashboard.update_dashboard(dashboard)
            self.logger.info(f"Exported metrics to Grafana dashboard: {report.title}")
        except Exception as e:
            self.logger.error(f"Grafana export error: {str(e)}")

    async def export_to_elasticsearch(self, report: AnalyticsReport):
        """Export metrics to Elasticsearch."""
        try:
            # Index metrics
            for metric in report.metrics:
                await self.es.index(
                    index='documentation_metrics',
                    document={
                        'name': metric.name,
                        'value': metric.value,
                        'timestamp': metric.timestamp,
                        'metadata': metric.metadata
                    }
                )
            
            # Index report
            await self.es.index(
                index='documentation_reports',
                document={
                    'title': report.title,
                    'generated_at': report.generated_at,
                    'metrics_count': len(report.metrics),
                    'insights': report.insights
                }
            )
            
            self.logger.info(f"Exported metrics to Elasticsearch: {report.title}")
        except Exception as e:
            self.logger.error(f"Elasticsearch export error: {str(e)}")

    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using statistical methods."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                anomalies = []
                for name in df['name'].unique():
                    metric_data = df[df['name'] == name]['value']
                    if len(metric_data) > 1:
                        # Calculate z-scores
                        z_scores = np.abs((metric_data - metric_data.mean()) / metric_data.std())
                        # Identify anomalies (z-score > 3)
                        anomaly_indices = np.where(z_scores > 3)[0]
                        for idx in anomaly_indices:
                            anomalies.append({
                                'metric': name,
                                'value': metric_data.iloc[idx],
                                'timestamp': df[df['name'] == name].iloc[idx]['timestamp'],
                                'z_score': z_scores.iloc[idx]
                            })
                            self.anomaly_scores.set(z_scores.iloc[idx])
                return anomalies
            return []
        except Exception as e:
            self.logger.error(f"Anomaly detection error: {str(e)}")
            return []

    async def predict_metrics(self, days_ahead: int = 7) -> Dict[str, List[float]]:
        """Predict future metric values using time series forecasting."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                predictions = {}
                for name in df['name'].unique():
                    metric_data = df[df['name'] == name]
                    if len(metric_data) > 1:
                        # Prepare time series data
                        ts_data = metric_data.set_index('timestamp')['value']
                        # Fit ARIMA model
                        model = pm.auto_arima(ts_data, seasonal=True, m=7)
                        # Make predictions
                        forecast = model.predict(n_periods=days_ahead)
                        predictions[name] = forecast.tolist()
                return predictions
            return {}
        except Exception as e:
            self.logger.error(f"Metric prediction error: {str(e)}")
            return {}

    async def analyze_engagement(self) -> Dict[str, float]:
        """Analyze user engagement with documentation."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                engagement_metrics = {}
                
                # Calculate time spent
                if 'time_spent' in df.columns:
                    engagement_metrics['avg_time_spent'] = df['time_spent'].mean()
                
                # Calculate bounce rate
                if 'bounce' in df.columns:
                    bounce_rate = (df['bounce'].sum() / len(df)) * 100
                    engagement_metrics['bounce_rate'] = bounce_rate
                
                # Calculate return rate
                if 'return_visits' in df.columns:
                    return_rate = (df['return_visits'].sum() / len(df)) * 100
                    engagement_metrics['return_rate'] = return_rate
                
                # Calculate interaction depth
                if 'interactions' in df.columns:
                    engagement_metrics['avg_interactions'] = df['interactions'].mean()
                
                # Update metric
                for metric, value in engagement_metrics.items():
                    self.engagement_scores.set(value)
                
                return engagement_metrics
            return {}
        except Exception as e:
            self.logger.error(f"Engagement analysis error: {str(e)}")
            return {}

    async def analyze_quality(self, content: str) -> Dict[str, float]:
        """Analyze documentation quality."""
        try:
            quality_metrics = {}
            
            # Calculate readability scores
            readability = textstat.flesch_reading_ease(content)
            quality_metrics['readability'] = readability
            self.readability_scores.set(readability)
            
            # Calculate completeness
            completeness = self._calculate_completeness(content)
            quality_metrics['completeness'] = completeness
            self.completeness_scores.set(completeness)
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(content)
            quality_metrics['accuracy'] = accuracy
            self.accuracy_scores.set(accuracy)
            
            # Overall quality score
            quality_score = (readability + completeness + accuracy) / 3
            quality_metrics['overall_quality'] = quality_score
            self.quality_scores.set(quality_score)
            
            return quality_metrics
        except Exception as e:
            self.logger.error(f"Quality analysis error: {str(e)}")
            return {}

    def _calculate_completeness(self, content: str) -> float:
        """Calculate documentation completeness score."""
        try:
            # Check for required sections
            required_sections = ['overview', 'prerequisites', 'installation', 'usage', 'examples', 'api']
            section_scores = []
            
            for section in required_sections:
                if section.lower() in content.lower():
                    section_scores.append(1.0)
                else:
                    section_scores.append(0.0)
            
            # Check for code examples
            code_blocks = len(re.findall(r'```[\s\S]*?```', content))
            code_score = min(code_blocks / 3, 1.0)  # Cap at 1.0
            
            # Calculate final score
            completeness = (sum(section_scores) / len(required_sections) + code_score) / 2
            return completeness
        except Exception as e:
            self.logger.error(f"Completeness calculation error: {str(e)}")
            return 0.0

    def _calculate_accuracy(self, content: str) -> float:
        """Calculate documentation accuracy score."""
        try:
            # Check for version information
            version_score = 1.0 if re.search(r'version|v\d+\.\d+', content, re.I) else 0.0
            
            # Check for last updated date
            date_score = 1.0 if re.search(r'updated|last modified|date', content, re.I) else 0.0
            
            # Check for references
            ref_score = 1.0 if re.search(r'reference|see also|related', content, re.I) else 0.0
            
            # Calculate final score
            accuracy = (version_score + date_score + ref_score) / 3
            return accuracy
        except Exception as e:
            self.logger.error(f"Accuracy calculation error: {str(e)}")
            return 0.0

    async def analyze_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            # Process texts
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Calculate similarity
            similarity = doc1.similarity(doc2)
            return similarity
        except Exception as e:
            self.logger.error(f"Semantic similarity analysis error: {str(e)}")
            return 0.0

    async def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            # Process text
            doc = self.nlp(text)
            
            # Extract noun phrases
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]
            
            # Extract named entities
            entities = [ent.text for ent in doc.ents]
            
            # Combine and deduplicate
            key_phrases = list(set(noun_phrases + entities))
            
            return key_phrases
        except Exception as e:
            self.logger.error(f"Key phrase extraction error: {str(e)}")
            return []

    async def generate_dashboard(self) -> Dict[str, Any]:
        """Generate an interactive dashboard with all visualizations."""
        try:
            # Get all visualizations
            visualizations = await self._generate_visualizations()
            
            # Create dashboard layout
            dashboard = {
                'title': 'Documentation Analytics Dashboard',
                'layout': {
                    'grid': [
                        {'type': 'header', 'content': 'Documentation Analytics Overview'},
                        {'type': 'metrics', 'content': self._get_summary_metrics()},
                        {'type': 'visualizations', 'content': visualizations},
                        {'type': 'insights', 'content': await self._generate_insights()}
                    ]
                },
                'interactive': True,
                'refresh_interval': 300  # 5 minutes
            }
            
            return dashboard
        except Exception as e:
            self.logger.error(f"Dashboard generation error: {str(e)}")
            return {}

    def _get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for the dashboard."""
        try:
            df = pd.DataFrame([vars(m) for m in self.metrics])
            if not df.empty:
                return {
                    'total_page_views': df[df['name'] == 'page_view']['value'].sum(),
                    'total_search_queries': df[df['name'] == 'search_query']['value'].sum(),
                    'avg_validation_score': df[df['name'] == 'validation_score']['value'].mean(),
                    'total_version_changes': df[df['name'] == 'version_change']['value'].sum(),
                    'total_user_feedback': df[df['name'] == 'user_feedback']['value'].sum(),
                    'total_documentation_changes': df[df['name'] == 'documentation_change']['value'].sum()
                }
            return {}
        except Exception as e:
            self.logger.error(f"Summary metrics error: {str(e)}")
            return {}

    async def generate_exportable_report(self, format: str = 'html') -> str:
        """Generate an exportable report with all analytics data."""
        try:
            # Get all data
            dashboard = await self.generate_dashboard()
            anomalies = await self.detect_anomalies()
            predictions = await self.predict_metrics()
            engagement = await self.analyze_engagement()
            
            # Create report content
            report = {
                'dashboard': dashboard,
                'anomalies': anomalies,
                'predictions': predictions,
                'engagement': engagement,
                'generated_at': datetime.now().isoformat()
            }
            
            # Export based on format
            if format == 'html':
                return self._export_html_report(report)
            elif format == 'pdf':
                return self._export_pdf_report(report)
            elif format == 'json':
                return self._export_json_report(report)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            self.logger.error(f"Report generation error: {str(e)}")
            raise

    def _export_html_report(self, report: Dict[str, Any]) -> str:
        """Export report as HTML."""
        try:
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Documentation Analytics Report</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; }
                    .visualization { margin: 20px 0; }
                    .insight { margin: 10px 0; padding: 10px; background: #f0f0f0; }
                </style>
            </head>
            <body>
                <h1>Documentation Analytics Report</h1>
                <p>Generated at: {{ generated_at }}</p>
                
                <div class="section">
                    <h2>Summary Metrics</h2>
                    {% for name, value in dashboard.layout.grid[1].content.items() %}
                    <div class="metric">
                        <h3>{{ name }}</h3>
                        <p>{{ value }}</p>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    {% for viz in dashboard.layout.grid[2].content %}
                    <div class="visualization">
                        <div id="viz-{{ loop.index }}"></div>
                        <script>
                            Plotly.newPlot('viz-{{ loop.index }}', {{ viz.data | tojson }});
                        </script>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Insights</h2>
                    {% for insight in dashboard.layout.grid[3].content %}
                    <div class="insight">{{ insight }}</div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Anomalies</h2>
                    {% for anomaly in anomalies %}
                    <div class="anomaly">
                        <h3>{{ anomaly.metric }}</h3>
                        <p>Value: {{ anomaly.value }}</p>
                        <p>Timestamp: {{ anomaly.timestamp }}</p>
                        <p>Z-Score: {{ anomaly.z_score }}</p>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Predictions</h2>
                    {% for metric, values in predictions.items() %}
                    <div class="prediction">
                        <h3>{{ metric }}</h3>
                        <p>Next 7 days: {{ values | join(', ') }}</p>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="section">
                    <h2>Engagement Analysis</h2>
                    {% for metric, value in engagement.items() %}
                    <div class="engagement-metric">
                        <h3>{{ metric }}</h3>
                        <p>{{ value }}</p>
                    </div>
                    {% endfor %}
                </div>
            </body>
            </html>
            """
            
            # Render template
            html = jinja2.Template(template).render(**report)
            
            # Save to file
            output_path = f"documentation_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(output_path, 'w') as f:
                f.write(html)
            
            return output_path
        except Exception as e:
            self.logger.error(f"HTML export error: {str(e)}")
            raise

    def _export_pdf_report(self, report: Dict[str, Any]) -> str:
        """Export report as PDF."""
        try:
            # First generate HTML
            html_path = self._export_html_report(report)
            
            # Convert to PDF
            output_path = html_path.replace('.html', '.pdf')
            pdf = pdfkit.from_file(html_path, output_path)
            
            return output_path
        except Exception as e:
            self.logger.error(f"PDF export error: {str(e)}")
            raise

    def _export_json_report(self, report: Dict[str, Any]) -> str:
        """Export report as JSON."""
        try:
            output_path = f"documentation_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return output_path
        except Exception as e:
            self.logger.error(f"JSON export error: {str(e)}")
            raise

    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        try:
            # Get experiment metrics
            experiment_metrics = [m for m in self.metrics if m.name == 'experiment' and m.metadata.get('experiment_id') == experiment_id]
            
            if not experiment_metrics:
                return {}
            
            # Calculate metrics by variant
            variants = {}
            for metric in experiment_metrics:
                variant = metric.metadata['variant']
                if variant not in variants:
                    variants[variant] = {
                        'views': 0,
                        'conversions': 0,
                        'users': set()
                    }
                
                variants[variant]['views'] += 1
                if metric.metadata['action'] == 'conversion':
                    variants[variant]['conversions'] += 1
                variants[variant]['users'].add(metric.metadata['user'])
            
            # Calculate statistics
            results = {}
            for variant, data in variants.items():
                conversion_rate = data['conversions'] / data['views'] if data['views'] > 0 else 0
                unique_users = len(data['users'])
                
                results[variant] = {
                    'views': data['views'],
                    'conversions': data['conversions'],
                    'conversion_rate': conversion_rate,
                    'unique_users': unique_users
                }
            
            return results
        except Exception as e:
            self.logger.error(f"Experiment analysis error: {str(e)}")
            return {}

    def _calculate_p_value(self, conv1: int, views1: int, conv2: int, views2: int) -> float:
        """Calculate p-value for A/B test using chi-square test."""
        try:
            from scipy.stats import chi2_contingency
            
            # Create contingency table
            table = np.array([
                [conv1, views1 - conv1],
                [conv2, views2 - conv2]
            ])
            
            # Calculate chi-square test
            chi2, p_value, dof, expected = chi2_contingency(table)
            return p_value
        except Exception as e:
            self.logger.error(f"P-value calculation error: {str(e)}")
            return 1.0

    async def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate a comprehensive experiment report."""
        try:
            # Get experiment analysis
            analysis = await self.analyze_experiment(experiment_id)
            
            # Create visualizations
            visualizations = []
            
            # Conversion rate comparison
            fig = go.Figure()
            for variant, data in analysis.items():
                if not variant.startswith('significance'):
                    fig.add_trace(go.Bar(
                        name=variant,
                        x=[variant],
                        y=[data['conversion_rate']],
                        text=[f"{data['conversion_rate']:.2%}"],
                        textposition='auto'
                    ))
            fig.update_layout(
                title='Conversion Rate by Variant',
                yaxis_title='Conversion Rate',
                showlegend=True
            )
            visualizations.append({'type': 'bar', 'data': fig.to_dict()})
            
            # User engagement comparison
            fig = go.Figure()
            for variant, data in analysis.items():
                if not variant.startswith('significance'):
                    fig.add_trace(go.Bar(
                        name=variant,
                        x=[variant],
                        y=[data['unique_users']],
                        text=[str(data['unique_users'])],
                        textposition='auto'
                    ))
            fig.update_layout(
                title='Unique Users by Variant',
                yaxis_title='Unique Users',
                showlegend=True
            )
            visualizations.append({'type': 'bar', 'data': fig.to_dict()})
            
            # Create report
            report = {
                'experiment_id': experiment_id,
                'analysis': analysis,
                'visualizations': visualizations,
                'generated_at': datetime.now().isoformat(),
                'recommendations': self._generate_experiment_recommendations(analysis)
            }
            
            return report
        except Exception as e:
            self.logger.error(f"Experiment report generation error: {str(e)}")
            return {}

    def _generate_experiment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on experiment analysis."""
        try:
            recommendations = []
            
            # Find best performing variant
            best_variant = None
            best_rate = 0
            for variant, data in analysis.items():
                if not variant.startswith('significance') and data['conversion_rate'] > best_rate:
                    best_variant = variant
                    best_rate = data['conversion_rate']
            
            if best_variant:
                recommendations.append(f"Variant {best_variant} shows the highest conversion rate at {best_rate:.2%}")
            
            # Check statistical significance
            for key, p_value in analysis.items():
                if key.startswith('significance'):
                    v1, v2 = key.split('_vs_')
                    if p_value < 0.05:
                        recommendations.append(f"Statistically significant difference found between {v1} and {v2} (p={p_value:.4f})")
                    else:
                        recommendations.append(f"No statistically significant difference found between {v1} and {v2} (p={p_value:.4f})")
            
            # Check sample size
            for variant, data in analysis.items():
                if not variant.startswith('significance'):
                    if data['views'] < 1000:
                        recommendations.append(f"Variant {variant} has a small sample size ({data['views']} views). Consider running the experiment longer.")
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Recommendation generation error: {str(e)}")
            return [] 