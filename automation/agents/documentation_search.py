import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime
import asyncio
import aiohttp
from dataclasses import dataclass
import elasticsearch
from elasticsearch import AsyncElasticsearch
import whoosh
from whoosh import index
from whoosh.fields import Schema, TEXT, ID, DATETIME, KEYWORD, STORED
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.analysis import StemmingAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
import gensim
from gensim import corpora, models
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import sentence_transformers
from sentence_transformers import SentenceTransformer
import rank_bm25
from rank_bm25 import BM25Okapi
import meilisearch
from meilisearch import Client as MeiliClient
import redis
import aioredis
import aiofiles
import yaml
import frontmatter
import mistune
from mistune import Markdown
import markdown
from bs4 import BeautifulSoup
import re
import unicodedata
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class SearchResult:
    id: str
    title: str
    content: str
    score: float
    highlights: List[str]
    metadata: Dict[str, Any]
    language: str
    version: str
    created_at: datetime
    updated_at: datetime
    author: str
    tags: List[str]
    categories: List[str]
    path: str

@dataclass
class SearchIndex:
    id: str
    name: str
    description: str
    type: str
    config: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: str
    stats: Dict[str, Any]

class DocumentationSearch:
    def __init__(self, config: Dict):
        self.config = config
        self.setup_logging()
        self.search_config = config.get('documentation', {}).get('search', {})
        self.setup_elasticsearch()
        self.setup_whoosh()
        self.setup_meilisearch()
        self.setup_redis()
        self.setup_nlp()
        self.setup_models()
        self.indices: Dict[str, SearchIndex] = {}
        self.cache = {}
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def setup_logging(self):
        """Configure logging for the search system."""
        log_path = Path("automation/logs/documentation")
        log_path.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path / "search.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_elasticsearch(self):
        """Setup Elasticsearch client."""
        try:
            es_config = self.search_config.get('elasticsearch', {})
            self.es = AsyncElasticsearch(
                hosts=es_config.get('hosts', ['http://localhost:9200']),
                basic_auth=(
                    es_config.get('username', ''),
                    es_config.get('password', '')
                ),
                verify_certs=es_config.get('verify_certs', True),
                ssl_show_warn=es_config.get('ssl_show_warn', True)
            )
            self.logger.info("Initialized Elasticsearch client")
            
        except Exception as e:
            self.logger.error(f"Elasticsearch setup error: {str(e)}")
            raise

    def setup_whoosh(self):
        """Setup Whoosh index."""
        try:
            whoosh_config = self.search_config.get('whoosh', {})
            index_path = Path(whoosh_config.get('index_path', 'documentation/search/whoosh'))
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Define schema
            self.whoosh_schema = Schema(
                id=ID(stored=True, unique=True),
                title=TEXT(stored=True),
                content=TEXT(stored=True),
                language=KEYWORD(stored=True),
                version=KEYWORD(stored=True),
                created_at=DATETIME(stored=True),
                updated_at=DATETIME(stored=True),
                author=KEYWORD(stored=True),
                tags=KEYWORD(stored=True, commas=True),
                categories=KEYWORD(stored=True, commas=True),
                path=STORED
            )
            
            # Create index
            self.whoosh_ix = index.create_in(str(index_path), self.whoosh_schema)
            self.logger.info("Initialized Whoosh index")
            
        except Exception as e:
            self.logger.error(f"Whoosh setup error: {str(e)}")
            raise

    def setup_meilisearch(self):
        """Setup MeiliSearch client."""
        try:
            meili_config = self.search_config.get('meilisearch', {})
            self.meili = MeiliClient(
                url=meili_config.get('url', 'http://localhost:7700'),
                api_key=meili_config.get('api_key', '')
            )
            self.logger.info("Initialized MeiliSearch client")
            
        except Exception as e:
            self.logger.error(f"MeiliSearch setup error: {str(e)}")
            raise

    def setup_redis(self):
        """Setup Redis client."""
        try:
            redis_config = self.search_config.get('redis', {})
            self.redis = aioredis.from_url(
                redis_config.get('url', 'redis://localhost:6379'),
                password=redis_config.get('password', ''),
                encoding='utf-8',
                decode_responses=True
            )
            self.logger.info("Initialized Redis client")
            
        except Exception as e:
            self.logger.error(f"Redis setup error: {str(e)}")
            raise

    def setup_nlp(self):
        """Setup NLP tools."""
        try:
            # Download NLTK data
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            
            # Initialize NLTK components
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
            # Load spaCy model
            self.nlp = spacy.load('en_core_web_sm')
            self.matcher = PhraseMatcher(self.nlp.vocab)
            
            self.logger.info("Initialized NLP tools")
            
        except Exception as e:
            self.logger.error(f"NLP setup error: {str(e)}")
            raise

    def setup_models(self):
        """Setup search models."""
        try:
            # Initialize TF-IDF vectorizer
            self.tfidf = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=10000
            )
            
            # Initialize BM25
            self.bm25 = None
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize FAISS index
            self.faiss_index = None
            
            self.logger.info("Initialized search models")
            
        except Exception as e:
            self.logger.error(f"Model setup error: {str(e)}")
            raise

    async def create_index(
        self,
        name: str,
        description: str,
        type: str = 'elasticsearch',
        config: Optional[Dict] = None
    ) -> SearchIndex:
        """Create a new search index."""
        try:
            async with self.lock:
                # Generate index ID
                index_id = f"index_{hash(f'{name}_{datetime.now()}')}"
                
                # Create index
                search_index = SearchIndex(
                    id=index_id,
                    name=name,
                    description=description,
                    type=type,
                    config=config or {},
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    status='active',
                    stats={}
                )
                
                # Initialize index based on type
                if type == 'elasticsearch':
                    await self._create_elasticsearch_index(index_id, config)
                elif type == 'whoosh':
                    await self._create_whoosh_index(index_id, config)
                elif type == 'meilisearch':
                    await self._create_meilisearch_index(index_id, config)
                
                self.indices[index_id] = search_index
                self.logger.info(f"Created search index: {index_id}")
                return search_index
                
        except Exception as e:
            self.logger.error(f"Index creation error: {str(e)}")
            raise

    async def delete_index(self, index_id: str) -> bool:
        """Delete a search index."""
        try:
            async with self.lock:
                if index_id not in self.indices:
                    raise ValueError(f"Index not found: {index_id}")
                
                search_index = self.indices[index_id]
                
                # Delete index based on type
                if search_index.type == 'elasticsearch':
                    await self._delete_elasticsearch_index(index_id)
                elif search_index.type == 'whoosh':
                    await self._delete_whoosh_index(index_id)
                elif search_index.type == 'meilisearch':
                    await self._delete_meilisearch_index(index_id)
                
                # Remove from memory
                del self.indices[index_id]
                
                self.logger.info(f"Deleted search index: {index_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Index deletion error: {str(e)}")
            return False

    async def index_document(
        self,
        index_id: str,
        document: Dict[str, Any]
    ) -> bool:
        """Index a document."""
        try:
            async with self.lock:
                if index_id not in self.indices:
                    raise ValueError(f"Index not found: {index_id}")
                
                search_index = self.indices[index_id]
                
                # Index document based on type
                if search_index.type == 'elasticsearch':
                    await self._index_elasticsearch_document(index_id, document)
                elif search_index.type == 'whoosh':
                    await self._index_whoosh_document(index_id, document)
                elif search_index.type == 'meilisearch':
                    await self._index_meilisearch_document(index_id, document)
                
                # Update index stats
                search_index.stats['document_count'] = search_index.stats.get('document_count', 0) + 1
                search_index.updated_at = datetime.now()
                
                self.logger.info(f"Indexed document in {index_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Document indexing error: {str(e)}")
            return False

    async def search(
        self,
        index_id: str,
        query: str,
        filters: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 10,
        sort: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents."""
        try:
            async with self.lock:
                if index_id not in self.indices:
                    raise ValueError(f"Index not found: {index_id}")
                
                search_index = self.indices[index_id]
                
                # Check cache
                cache_key = f"search:{index_id}:{hash(f'{query}{filters}{page}{page_size}{sort}')}"
                cached_results = await self.redis.get(cache_key)
                if cached_results:
                    return json.loads(cached_results)
                
                # Search based on type
                if search_index.type == 'elasticsearch':
                    results = await self._search_elasticsearch(index_id, query, filters, page, page_size, sort)
                elif search_index.type == 'whoosh':
                    results = await self._search_whoosh(index_id, query, filters, page, page_size, sort)
                elif search_index.type == 'meilisearch':
                    results = await self._search_meilisearch(index_id, query, filters, page, page_size, sort)
                
                # Cache results
                await self.redis.set(cache_key, json.dumps(results), ex=300)  # Cache for 5 minutes
                
                return results
                
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return []

    async def _create_elasticsearch_index(self, index_id: str, config: Optional[Dict] = None):
        """Create an Elasticsearch index."""
        try:
            # Define index settings
            settings = {
                'settings': {
                    'number_of_shards': config.get('shards', 1),
                    'number_of_replicas': config.get('replicas', 1),
                    'analysis': {
                        'analyzer': {
                            'custom_analyzer': {
                                'type': 'custom',
                                'tokenizer': 'standard',
                                'filter': ['lowercase', 'stop', 'snowball']
                            }
                        }
                    }
                },
                'mappings': {
                    'properties': {
                        'id': {'type': 'keyword'},
                        'title': {'type': 'text', 'analyzer': 'custom_analyzer'},
                        'content': {'type': 'text', 'analyzer': 'custom_analyzer'},
                        'language': {'type': 'keyword'},
                        'version': {'type': 'keyword'},
                        'created_at': {'type': 'date'},
                        'updated_at': {'type': 'date'},
                        'author': {'type': 'keyword'},
                        'tags': {'type': 'keyword'},
                        'categories': {'type': 'keyword'},
                        'path': {'type': 'keyword'}
                    }
                }
            }
            
            # Create index
            await self.es.indices.create(index=index_id, body=settings)
            
        except Exception as e:
            self.logger.error(f"Elasticsearch index creation error: {str(e)}")
            raise

    async def _create_whoosh_index(self, index_id: str, config: Optional[Dict] = None):
        """Create a Whoosh index."""
        try:
            # Create index directory
            index_path = Path(f"documentation/search/whoosh/{index_id}")
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Create index
            whoosh.index.create_in(str(index_path), self.whoosh_schema)
            
        except Exception as e:
            self.logger.error(f"Whoosh index creation error: {str(e)}")
            raise

    async def _create_meilisearch_index(self, index_id: str, config: Optional[Dict] = None):
        """Create a MeiliSearch index."""
        try:
            # Create index
            await self.meili.create_index(index_id, {
                'primaryKey': 'id',
                'searchableAttributes': ['title', 'content'],
                'filterableAttributes': ['language', 'version', 'author', 'tags', 'categories'],
                'sortableAttributes': ['created_at', 'updated_at']
            })
            
        except Exception as e:
            self.logger.error(f"MeiliSearch index creation error: {str(e)}")
            raise

    async def _delete_elasticsearch_index(self, index_id: str):
        """Delete an Elasticsearch index."""
        try:
            await self.es.indices.delete(index=index_id)
            
        except Exception as e:
            self.logger.error(f"Elasticsearch index deletion error: {str(e)}")
            raise

    async def _delete_whoosh_index(self, index_id: str):
        """Delete a Whoosh index."""
        try:
            index_path = Path(f"documentation/search/whoosh/{index_id}")
            if index_path.exists():
                for file in index_path.glob('*'):
                    file.unlink()
                index_path.rmdir()
            
        except Exception as e:
            self.logger.error(f"Whoosh index deletion error: {str(e)}")
            raise

    async def _delete_meilisearch_index(self, index_id: str):
        """Delete a MeiliSearch index."""
        try:
            await self.meili.delete_index(index_id)
            
        except Exception as e:
            self.logger.error(f"MeiliSearch index deletion error: {str(e)}")
            raise

    async def _index_elasticsearch_document(self, index_id: str, document: Dict[str, Any]):
        """Index a document in Elasticsearch."""
        try:
            # Prepare document
            doc = {
                'id': document['id'],
                'title': document['title'],
                'content': document['content'],
                'language': document['language'],
                'version': document['version'],
                'created_at': document['created_at'],
                'updated_at': document['updated_at'],
                'author': document['author'],
                'tags': document['tags'],
                'categories': document['categories'],
                'path': document['path']
            }
            
            # Index document
            await self.es.index(index=index_id, id=document['id'], body=doc)
            
        except Exception as e:
            self.logger.error(f"Elasticsearch document indexing error: {str(e)}")
            raise

    async def _index_whoosh_document(self, index_id: str, document: Dict[str, Any]):
        """Index a document in Whoosh."""
        try:
            # Get index
            index_path = Path(f"documentation/search/whoosh/{index_id}")
            ix = whoosh.index.open_dir(str(index_path))
            
            # Create writer
            writer = ix.writer()
            
            # Add document
            writer.add_document(
                id=document['id'],
                title=document['title'],
                content=document['content'],
                language=document['language'],
                version=document['version'],
                created_at=document['created_at'],
                updated_at=document['updated_at'],
                author=document['author'],
                tags=','.join(document['tags']),
                categories=','.join(document['categories']),
                path=document['path']
            )
            
            # Commit changes
            writer.commit()
            
        except Exception as e:
            self.logger.error(f"Whoosh document indexing error: {str(e)}")
            raise

    async def _index_meilisearch_document(self, index_id: str, document: Dict[str, Any]):
        """Index a document in MeiliSearch."""
        try:
            # Prepare document
            doc = {
                'id': document['id'],
                'title': document['title'],
                'content': document['content'],
                'language': document['language'],
                'version': document['version'],
                'created_at': document['created_at'].isoformat(),
                'updated_at': document['updated_at'].isoformat(),
                'author': document['author'],
                'tags': document['tags'],
                'categories': document['categories'],
                'path': document['path']
            }
            
            # Index document
            await self.meili.index(index_id).add_documents([doc])
            
        except Exception as e:
            self.logger.error(f"MeiliSearch document indexing error: {str(e)}")
            raise

    async def _search_elasticsearch(
        self,
        index_id: str,
        query: str,
        filters: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 10,
        sort: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents in Elasticsearch."""
        try:
            # Prepare query
            search_query = {
                'query': {
                    'bool': {
                        'must': [
                            {
                                'multi_match': {
                                    'query': query,
                                    'fields': ['title^2', 'content'],
                                    'type': 'best_fields',
                                    'operator': 'and'
                                }
                            }
                        ]
                    }
                },
                'from': (page - 1) * page_size,
                'size': page_size,
                'highlight': {
                    'fields': {
                        'title': {},
                        'content': {}
                    }
                }
            }
            
            # Add filters
            if filters:
                search_query['query']['bool']['filter'] = []
                for field, value in filters.items():
                    search_query['query']['bool']['filter'].append({
                        'term': {field: value}
                    })
            
            # Add sorting
            if sort:
                search_query['sort'] = [{sort: {'order': 'desc'}}]
            
            # Execute search
            response = await self.es.search(index=index_id, body=search_query)
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                result = SearchResult(
                    id=hit['_id'],
                    title=hit['_source']['title'],
                    content=hit['_source']['content'],
                    score=hit['_score'],
                    highlights=[h for h in hit['highlight'].get('content', [])],
                    metadata=hit['_source'],
                    language=hit['_source']['language'],
                    version=hit['_source']['version'],
                    created_at=datetime.fromisoformat(hit['_source']['created_at']),
                    updated_at=datetime.fromisoformat(hit['_source']['updated_at']),
                    author=hit['_source']['author'],
                    tags=hit['_source']['tags'],
                    categories=hit['_source']['categories'],
                    path=hit['_source']['path']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Elasticsearch search error: {str(e)}")
            return []

    async def _search_whoosh(
        self,
        index_id: str,
        query: str,
        filters: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 10,
        sort: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents in Whoosh."""
        try:
            # Get index
            index_path = Path(f"documentation/search/whoosh/{index_id}")
            ix = whoosh.index.open_dir(str(index_path))
            
            # Create searcher
            with ix.searcher() as searcher:
                # Create query parser
                parser = MultifieldParser(['title', 'content'], ix.schema)
                
                # Parse query
                q = parser.parse(query)
                
                # Add filters
                if filters:
                    for field, value in filters.items():
                        q = q & whoosh.query.Term(field, value)
                
                # Execute search
                results = searcher.search(q, limit=page_size, offset=(page - 1) * page_size)
                
                # Process results
                search_results = []
                for result in results:
                    search_result = SearchResult(
                        id=result['id'],
                        title=result['title'],
                        content=result['content'],
                        score=result.score,
                        highlights=[result.highlights('content')],
                        metadata=result.fields(),
                        language=result['language'],
                        version=result['version'],
                        created_at=result['created_at'],
                        updated_at=result['updated_at'],
                        author=result['author'],
                        tags=result['tags'].split(','),
                        categories=result['categories'].split(','),
                        path=result['path']
                    )
                    search_results.append(search_result)
                
                return search_results
            
        except Exception as e:
            self.logger.error(f"Whoosh search error: {str(e)}")
            return []

    async def _search_meilisearch(
        self,
        index_id: str,
        query: str,
        filters: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 10,
        sort: Optional[str] = None
    ) -> List[SearchResult]:
        """Search documents in MeiliSearch."""
        try:
            # Prepare search options
            search_options = {
                'offset': (page - 1) * page_size,
                'limit': page_size
            }
            
            # Add filters
            if filters:
                search_options['filter'] = []
                for field, value in filters.items():
                    search_options['filter'].append(f"{field} = {value}")
            
            # Add sorting
            if sort:
                search_options['sort'] = [sort]
            
            # Execute search
            response = await self.meili.index(index_id).search(query, search_options)
            
            # Process results
            results = []
            for hit in response['hits']:
                result = SearchResult(
                    id=hit['id'],
                    title=hit['title'],
                    content=hit['content'],
                    score=hit['_rankingScore'],
                    highlights=[hit.get('_formatted', {}).get('content', '')],
                    metadata=hit,
                    language=hit['language'],
                    version=hit['version'],
                    created_at=datetime.fromisoformat(hit['created_at']),
                    updated_at=datetime.fromisoformat(hit['updated_at']),
                    author=hit['author'],
                    tags=hit['tags'],
                    categories=hit['categories'],
                    path=hit['path']
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"MeiliSearch search error: {str(e)}")
            return []

    async def clear_cache(self):
        """Clear the search cache."""
        try:
            await self.redis.flushdb()
            self.logger.info("Cleared search cache")
            
        except Exception as e:
            self.logger.error(f"Cache clearing error: {str(e)}")

    async def optimize_index(self, index_id: str):
        """Optimize a search index."""
        try:
            async with self.lock:
                if index_id not in self.indices:
                    raise ValueError(f"Index not found: {index_id}")
                
                search_index = self.indices[index_id]
                
                # Optimize index based on type
                if search_index.type == 'elasticsearch':
                    await self._optimize_elasticsearch_index(index_id)
                elif search_index.type == 'whoosh':
                    await self._optimize_whoosh_index(index_id)
                elif search_index.type == 'meilisearch':
                    await self._optimize_meilisearch_index(index_id)
                
                self.logger.info(f"Optimized search index: {index_id}")
                
        except Exception as e:
            self.logger.error(f"Index optimization error: {str(e)}")
            raise

    async def _optimize_elasticsearch_index(self, index_id: str):
        """Optimize an Elasticsearch index."""
        try:
            await self.es.indices.forcemerge(index=index_id)
            
        except Exception as e:
            self.logger.error(f"Elasticsearch index optimization error: {str(e)}")
            raise

    async def _optimize_whoosh_index(self, index_id: str):
        """Optimize a Whoosh index."""
        try:
            index_path = Path(f"documentation/search/whoosh/{index_id}")
            ix = whoosh.index.open_dir(str(index_path))
            ix.optimize()
            
        except Exception as e:
            self.logger.error(f"Whoosh index optimization error: {str(e)}")
            raise

    async def _optimize_meilisearch_index(self, index_id: str):
        """Optimize a MeiliSearch index."""
        try:
            await self.meili.index(index_id).update_settings({
                'rankingRules': [
                    'words',
                    'typo',
                    'proximity',
                    'attribute',
                    'sort',
                    'exactness'
                ]
            })
            
        except Exception as e:
            self.logger.error(f"MeiliSearch index optimization error: {str(e)}")
            raise 