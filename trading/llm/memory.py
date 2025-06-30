"""Memory management for LLM agents with long-term storage and recall capabilities."""

from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import asyncio
from dataclasses import dataclass, field
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """A single memory entry with metadata."""
    prompt: str
    response: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    importance: float = 1.0

class MemoryManager:
    """Manages long-term memory for LLM agents."""
    
    def __init__(
        self,
        storage_path: Union[str, Path],
        embedding_model: str = "all-MiniLM-L6-v2",
        max_memories: int = 1000,
        similarity_threshold: float = 0.8
            return {'success': True, 'message': 'Initialization completed', 'timestamp': datetime.now().isoformat()}
    ):
        """Initialize the memory manager.
        
        Args:
            storage_path: Path to store memories
            embedding_model: Model to use for embeddings
            max_memories: Maximum number of memories to store
            similarity_threshold: Threshold for memory recall
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.max_memories = max_memories
        self.similarity_threshold = similarity_threshold
        
        self.memories: List[MemoryEntry] = []
        self.index = None
        self._load_memories()
        
    def _load_memories(self) -> None:
        """Load memories from storage."""
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    memory = MemoryEntry(
                        prompt=entry["prompt"],
                        response=entry["response"],
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        metadata=entry.get("metadata", {}),
                        importance=entry.get("importance", 1.0)
                    )
                    self.memories.append(memory)
        
        # Initialize FAISS index if we have memories
        if self.memories:
            self._rebuild_index()
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index for memory search."""
        if not self.memories:
            return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            
        # Get embeddings for all memories
        texts = [m.prompt for m in self.memories]
        embeddings = self.embedding_model.encode(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Store embeddings in memory entries
        for memory, embedding in zip(self.memories, embeddings):
            memory.embedding = embedding
    
    async def store(
        self,
        prompt: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> None:
        """Store a new memory.
        
        Args:
            prompt: The prompt that generated the response
            response: The response to store
            metadata: Optional metadata about the interaction
            importance: Importance score for the memory
        """
        # Create new memory entry
        memory = MemoryEntry(
            prompt=prompt,
            response=response,
            timestamp=datetime.now(),
            metadata=metadata or {},
            importance=importance
        )
        
        # Get embedding for the prompt
        memory.embedding = self.embedding_model.encode([prompt])[0]
        
        # Add to memories
        self.memories.append(memory)
        
        # Update FAISS index
        if self.index is None:
            self._rebuild_index()
        else:
            self.index.add(memory.embedding.reshape(1, -1).astype('float32'))
        
        # Trim memories if needed
        if len(self.memories) > self.max_memories:
            self._trim_memories()
        
        # Save to disk
        await self._save_memories()
    
    async def recall(
        self,
        prompt: str,
        max_results: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Recall relevant memories for a prompt.
        
        Args:
            prompt: The prompt to find memories for
            max_results: Maximum number of memories to return
            
        Returns:
            Dictionary with recalled memories or None if no matches
        """
        if not self.memories or self.index is None:
            return None
            
        # Get embedding for the prompt
        query_embedding = self.embedding_model.encode([prompt])[0]
        
        # Search for similar memories
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            min(max_results, len(self.memories))
        )
        
        # Filter by similarity threshold
        recalled_memories = []
        for distance, idx in zip(distances[0], indices[0]):
            if distance < (1 - self.similarity_threshold):
                memory = self.memories[idx]
                recalled_memories.append({
                    "prompt": memory.prompt,
                    "response": memory.response,
                    "similarity": 1 - distance,
                    "timestamp": memory.timestamp.isoformat(),
                    "metadata": memory.metadata
                })
        
        if not recalled_memories:
            return None
            
        return {
            "memories": recalled_memories,
            "count": len(recalled_memories),
            "timestamp": datetime.now().isoformat()
        }
    
    def _trim_memories(self) -> None:
        """Trim memories based on importance and recency."""
        # Sort memories by importance and recency
        self.memories.sort(
            key=lambda x: (x.importance, x.timestamp),
            reverse=True
        )
        
        # Keep only the most important/recent memories
        self.memories = self.memories[:self.max_memories]
        
        # Rebuild index
        self._rebuild_index()
    
        return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
    async def _save_memories(self) -> None:
        """Save memories to disk."""
        memory_file = self.storage_path / "memories.json"
        
        # Convert memories to JSON-serializable format
        data = []
        for memory in self.memories:
            entry = {
                "prompt": memory.prompt,
                "response": memory.response,
                "timestamp": memory.timestamp.isoformat(),
                "metadata": memory.metadata,
                "importance": memory.importance
            }
            data.append(entry)
        
        # Save to file
        with open(memory_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        return {'success': True, 'result': {, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}
            "total_memories": len(self.memories),
            "oldest_memory": min(m.timestamp for m in self.memories).isoformat() if self.memories else None,
            "newest_memory": max(m.timestamp for m in self.memories).isoformat() if self.memories else None,
            "average_importance": np.mean([m.importance for m in self.memories]) if self.memories else 0.0
        }
    
    def clear_memories(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.index = None
        memory_file = self.storage_path / "memories.json"
        if memory_file.exists():
            memory_file.unlink() 
                return {'success': True, 'message': 'Operation completed successfully', 'timestamp': datetime.now().isoformat()}