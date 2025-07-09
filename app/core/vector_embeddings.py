"""
Vector Embedding Integration Module

This module provides sophisticated embedding capabilities for the hybrid semantic
reasoning system, including pre-trained embeddings, custom embedding training,
and advanced similarity metrics.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Design by Contract support
from icontract import require, ensure, invariant, ViolationError
from app.core.protocols import EmbeddingProviderProtocol


@dataclass
class EmbeddingMetadata:
    """Metadata for concept embeddings."""
    concept_id: str
    embedding_model: str
    created_at: str
    dimensions: int
    source: str  # 'pretrained', 'custom', 'computed'
    confidence: float = 1.0
    update_count: int = 0


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def get_embedding(self, text: str) -> Optional[NDArray[np.float32]]:
        """Get embedding for text."""
        pass
    
    @abstractmethod
    def get_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute similarity between embeddings."""
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        pass


class RandomEmbeddingProvider(EmbeddingProvider):
    """
    Random embedding provider for testing and development.
    
    Generates consistent random embeddings based on text hashing.
    """
    
    def __init__(self, dimensions: int = 300, seed: int = 42):
        self.dimensions = dimensions
        self.seed = seed
        self.cache: Dict[str, NDArray[np.float32]] = {}
    
    def get_embedding(self, text: str) -> Optional[NDArray[np.float32]]:
        """Get random embedding for text."""
        if text in self.cache:
            return self.cache[text]
        
        # Generate consistent random embedding based on text hash
        text_hash = hash(text) % (2**32)
        np.random.seed(text_hash)
        
        embedding = np.random.normal(0, 1, self.dimensions).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        self.cache[text] = embedding
        return embedding
    
    def get_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute cosine similarity."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def get_dimensions(self) -> int:
        return self.dimensions


class SemanticEmbeddingProvider(EmbeddingProvider):
    """
    Semantic embedding provider that creates embeddings based on concept semantics.
    
    This provider creates embeddings that reflect semantic relationships by
    encoding domain knowledge and conceptual hierarchies.
    """
    
    def __init__(self, dimensions: int = 300):
        self.dimensions = dimensions
        self.cache: Dict[str, NDArray[np.float32]] = {}
        
        # Semantic category mappings
        self.semantic_categories = {
            # Animals
            "animal": [1.0, 0.0, 0.0, 0.0, 0.0],
            "mammal": [1.0, 0.1, 0.0, 0.0, 0.0],
            "bird": [1.0, 0.0, 0.1, 0.0, 0.0],
            "predator": [1.0, 0.0, 0.0, 0.1, 0.0],
            "domestic": [1.0, 0.0, 0.0, 0.0, 0.1],
            
            # People/Roles
            "person": [0.0, 1.0, 0.0, 0.0, 0.0],
            "leader": [0.0, 1.0, 0.1, 0.0, 0.0],
            "royalty": [0.0, 1.0, 0.0, 0.1, 0.0],
            "military": [0.0, 1.0, 0.0, 0.0, 0.1],
            "professional": [0.0, 1.0, 0.1, 0.1, 0.0],
            
            # Objects
            "object": [0.0, 0.0, 1.0, 0.0, 0.0],
            "tool": [0.0, 0.0, 1.0, 0.1, 0.0],
            "weapon": [0.0, 0.0, 1.0, 0.0, 0.1],
            "vehicle": [0.0, 0.0, 1.0, 0.1, 0.1],
            
            # Abstract concepts
            "concept": [0.0, 0.0, 0.0, 1.0, 0.0],
            "emotion": [0.0, 0.0, 0.0, 1.0, 0.1],
            "quality": [0.0, 0.0, 0.0, 1.0, 0.0],
            "relation": [0.0, 0.0, 0.0, 1.0, 0.1],
            
            # Actions
            "action": [0.0, 0.0, 0.0, 0.0, 1.0],
            "motion": [0.0, 0.0, 0.0, 0.1, 1.0],
            "communication": [0.0, 0.1, 0.0, 0.0, 1.0],
            "creation": [0.0, 0.0, 0.1, 0.0, 1.0],
        }
        
        # Concept-to-category mappings
        self.concept_mappings = {
            # Animals
            "lion": ["animal", "mammal", "predator"],
            "tiger": ["animal", "mammal", "predator"],
            "wolf": ["animal", "mammal", "predator"],
            "eagle": ["animal", "bird", "predator"],
            "shark": ["animal", "predator"],
            "dog": ["animal", "mammal", "domestic"],
            "cat": ["animal", "mammal", "domestic"],
            
            # Royalty
            "king": ["person", "leader", "royalty"],
            "queen": ["person", "leader", "royalty"],
            "prince": ["person", "royalty"],
            "princess": ["person", "royalty"],
            "emperor": ["person", "leader", "royalty"],
            
            # Military
            "general": ["person", "leader", "military"],
            "colonel": ["person", "military"],
            "captain": ["person", "military"],
            "sergeant": ["person", "military"],
            "soldier": ["person", "military"],
            
            # Business
            "ceo": ["person", "leader", "professional"],
            "manager": ["person", "professional"],
            "director": ["person", "leader", "professional"],
            "supervisor": ["person", "professional"],
            "employee": ["person", "professional"],
            
            # Sports
            "quarterback": ["person", "professional"],
            "striker": ["person", "professional"],
            "coach": ["person", "leader", "professional"],
            "player": ["person", "professional"],
            
            # Objects
            "car": ["object", "vehicle"],
            "bike": ["object", "vehicle"],
            "plane": ["object", "vehicle"],
            "sword": ["object", "weapon", "tool"],
            "shield": ["object", "tool"],
            
            # Abstract
            "strength": ["concept", "quality"],
            "wisdom": ["concept", "quality"],
            "courage": ["concept", "quality"],
            "fear": ["concept", "emotion"],
            "love": ["concept", "emotion"],
            
            # Actions
            "run": ["action", "motion"],
            "fight": ["action"],
            "speak": ["action", "communication"],
            "create": ["action", "creation"],
        }
    
    def get_embedding(self, text: str) -> Optional[NDArray[np.float32]]:
        """Get semantic embedding for text."""
        if text in self.cache:
            return self.cache[text]
        
        # Get semantic categories for the concept
        categories = self.concept_mappings.get(text.lower(), [])
        
        if not categories:
            # Fallback to basic random embedding
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.normal(0, 0.1, self.dimensions).astype(np.float32)
        else:
            # Build embedding from semantic categories
            embedding = np.zeros(self.dimensions, dtype=np.float32)
            
            # Set base semantic features
            for category in categories:
                if category in self.semantic_categories:
                    semantic_vector = self.semantic_categories[category]
                    for i, value in enumerate(semantic_vector):
                        if i < self.dimensions:
                            embedding[i] += value
            
            # Add some random variation
            np.random.seed(hash(text) % (2**32))
            noise = np.random.normal(0, 0.05, self.dimensions).astype(np.float32)
            embedding += noise
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.cache[text] = embedding
        return embedding
    
    def get_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute cosine similarity."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def get_dimensions(self) -> int:
        return self.dimensions


@invariant(lambda self: hasattr(self, 'providers') and isinstance(self.providers, dict),
           "providers must be a dictionary")
@invariant(lambda self: hasattr(self, 'default_provider') and self.default_provider in self.providers,
           "default_provider must be a valid provider")
class VectorEmbeddingManager(EmbeddingProviderProtocol):
    """
    Advanced vector embedding manager for the hybrid semantic system.
    Implements EmbeddingProviderProtocol for protocol compliance.
    
    Provides sophisticated embedding capabilities including multiple providers,
    embedding persistence, and advanced similarity metrics.
    """
    
    def __init__(self, default_provider: str = "semantic", 
                 cache_dir: Optional[str] = None):
        """
        Initialize embedding manager.
        
        Args:
            default_provider: Default embedding provider to use
            cache_dir: Directory to cache embeddings
        """
        self.providers = {
            "random": RandomEmbeddingProvider(dimensions=300),
            "semantic": SemanticEmbeddingProvider(dimensions=300),
        }
        
        self.default_provider = default_provider
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Embedding storage
        self.embeddings: Dict[str, NDArray[np.float32]] = {}
        self.metadata: Dict[str, EmbeddingMetadata] = {}
        
        # Configuration
        self.auto_save = True
        self.similarity_threshold = 0.7
        
        self.logger = logging.getLogger(__name__)
        
        # Load cached embeddings
        if self.cache_dir:
            self.load_embeddings()
    
    def add_provider(self, name: str, provider: EmbeddingProvider) -> None:
        """Add a new embedding provider."""
        self.providers[name] = provider
        self.logger.info(f"Added embedding provider: {name}")
    
    @require(lambda concept_id: isinstance(concept_id, str) and len(concept_id.strip()) > 0,
             "concept_id must be non-empty string")
    @require(lambda text: isinstance(text, str) and len(text.strip()) > 0,
             "text must be non-empty string")
    @require(lambda provider: provider is None or (isinstance(provider, str) and len(provider.strip()) > 0),
             "provider must be None or non-empty string")
    @ensure(lambda result: result is None or isinstance(result, np.ndarray),
            "result must be None or numpy array")
    def get_embedding(self, concept_id: str, text: str, 
                     provider: Optional[str] = None) -> Optional[NDArray[np.float32]]:
        """
        Get embedding for a concept.
        
        Args:
            concept_id: Unique identifier for the concept
            text: Text representation of the concept
            provider: Embedding provider to use
            
        Returns:
            Embedding vector or None if not available
        """
        # Check cache first
        if concept_id in self.embeddings:
            return self.embeddings[concept_id]
        
        # Use specified provider or default
        provider_name = provider or self.default_provider
        if provider_name not in self.providers:
            self.logger.error(f"Unknown embedding provider: {provider_name}")
            return None
        
        provider_obj = self.providers[provider_name]
        
        # Get embedding
        embedding = provider_obj.get_embedding(text)
        
        if embedding is not None:
            # Store in cache
            self.embeddings[concept_id] = embedding
            
            # Store metadata
            self.metadata[concept_id] = EmbeddingMetadata(
                concept_id=concept_id,
                embedding_model=provider_name,
                created_at=str(np.datetime64('now')),
                dimensions=len(embedding),
                source='computed',
                confidence=1.0
            )
            
            # Auto-save if enabled
            if self.auto_save and self.cache_dir:
                self.save_embeddings()
        
        return embedding
    
    def compute_similarity(self, concept1_id: str, concept2_id: str,
                         metric: str = "cosine") -> float:
        """
        Compute similarity between two concepts.
        
        Args:
            concept1_id: First concept ID
            concept2_id: Second concept ID
            metric: Similarity metric to use
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        emb1 = self.embeddings.get(concept1_id)
        emb2 = self.embeddings.get(concept2_id)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        if metric == "cosine":
            return self._cosine_similarity(emb1, emb2)
        elif metric == "euclidean":
            return self._euclidean_similarity(emb1, emb2)
        elif metric == "manhattan":
            return self._manhattan_similarity(emb1, emb2)
        else:
            self.logger.warning(f"Unknown similarity metric: {metric}")
            return self._cosine_similarity(emb1, emb2)
    
    def _cosine_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute cosine similarity."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def _euclidean_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute Euclidean similarity (inverse of distance)."""
        distance = float(np.linalg.norm(emb1 - emb2))
        return 1.0 / (1.0 + distance)
    
    def _manhattan_similarity(self, emb1: NDArray[np.float32], emb2: NDArray[np.float32]) -> float:
        """Compute Manhattan similarity (inverse of L1 distance)."""
        distance = float(np.sum(np.abs(emb1 - emb2)))
        return 1.0 / (1.0 + distance)
    
    def find_similar_concepts(self, concept_id: str, threshold: float = None,
                            top_k: int = None) -> List[Tuple[str, float]]:
        """
        Find concepts similar to the given concept.
        
        Args:
            concept_id: Target concept ID
            threshold: Minimum similarity threshold
            top_k: Maximum number of results
            
        Returns:
            List of (concept_id, similarity) tuples
        """
        target_embedding = self.embeddings.get(concept_id)
        if target_embedding is None:
            return []
        
        threshold = threshold or self.similarity_threshold
        similarities = []
        
        for other_concept_id, other_embedding in self.embeddings.items():
            if other_concept_id == concept_id:
                continue
            
            similarity = self._cosine_similarity(target_embedding, other_embedding)
            
            if similarity >= threshold:
                similarities.append((other_concept_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k:
            similarities = similarities[:top_k]
        
        return similarities
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        if not self.embeddings:
            return {}
        
        dimensions = [len(emb) for emb in self.embeddings.values()]
        providers = [meta.embedding_model for meta in self.metadata.values()]
        
        return {
            "total_embeddings": len(self.embeddings),
            "avg_dimensions": np.mean(dimensions),
            "providers_used": list(set(providers)),
            "provider_counts": {p: providers.count(p) for p in set(providers)},
            "cache_size_mb": self._estimate_cache_size(),
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB."""
        total_size = 0
        for embedding in self.embeddings.values():
            total_size += embedding.nbytes
        return total_size / (1024 * 1024)
    
    def save_embeddings(self) -> None:
        """Save embeddings to disk."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        embeddings_file = self.cache_dir / "embeddings.npy"
        metadata_file = self.cache_dir / "metadata.json"
        
        # Convert to serializable format
        embeddings_data = {
            concept_id: embedding.tolist() 
            for concept_id, embedding in self.embeddings.items()
        }
        
        metadata_data = {
            concept_id: {
                "concept_id": meta.concept_id,
                "embedding_model": meta.embedding_model,
                "created_at": meta.created_at,
                "dimensions": meta.dimensions,
                "source": meta.source,
                "confidence": meta.confidence,
                "update_count": meta.update_count
            }
            for concept_id, meta in self.metadata.items()
        }
        
        # Save files
        with open(embeddings_file, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
        
        self.logger.info(f"Saved {len(self.embeddings)} embeddings to {self.cache_dir}")
    
    def load_embeddings(self) -> None:
        """Load embeddings from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        embeddings_file = self.cache_dir / "embeddings.npy"
        metadata_file = self.cache_dir / "metadata.json"
        
        if not embeddings_file.exists() or not metadata_file.exists():
            return
        
        try:
            # Load embeddings
            with open(embeddings_file, 'r') as f:
                embeddings_data = json.load(f)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata_data = json.load(f)
            
            # Convert back to numpy arrays
            for concept_id, embedding_list in embeddings_data.items():
                self.embeddings[concept_id] = np.array(embedding_list)
            
            # Convert back to metadata objects
            for concept_id, meta_dict in metadata_data.items():
                self.metadata[concept_id] = EmbeddingMetadata(**meta_dict)
            
            self.logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.cache_dir}")
        
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.embeddings.clear()
        self.metadata.clear()
        self.logger.info("Cleared embedding cache")
    
    # Phase 5: Protocol/ABC adoption for embedding manager
    # Ensure VectorEmbeddingManager implements EmbeddingProviderProtocol

    # Add required protocol methods and properties
    def generate_embedding(self, concept: str, context: str = "default") -> NDArray[np.float32]:
        return self.get_embedding(concept, context)  # Alias to existing method

    def compute_similarity(self, emb1, emb2) -> float:
        return float(self._cosine_similarity(emb1, emb2))

    def batch_generate_embeddings(self, concepts: List[str], context: str = "default") -> Dict[str, NDArray[np.float32]]:
        return {c: self.get_embedding(c, context) for c in concepts}

    @property
    def embedding_dimension(self) -> int:
        return 300  # Default dimension; could be dynamic if needed

    @property
    def provider_name(self) -> str:
        return self.default_provider
