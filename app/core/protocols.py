"""
Protocol Interfaces for Type-Safe Soft Logic System
==================================================

This module defines Protocol interfaces that establish contracts for the major
components of our soft logic system. These protocols enable type safety,
runtime validation, and clear architectural boundaries.

DESIGN PHILOSOPHY:
==================

Using Protocol classes from typing provides:
1. **Structural Typing**: Duck typing with static verification
2. **Runtime Checking**: @runtime_checkable enables isinstance() validation
3. **Clear Contracts**: Explicit interface specifications
4. **Flexible Implementation**: Multiple implementations can satisfy same protocol

PROTOCOL HIERARCHY:
===================

- ConceptRegistryProtocol: Core concept management interface
- SemanticReasoningProtocol: Advanced reasoning capabilities  
- EmbeddingProviderProtocol: Vector embedding generation and similarity
- KnowledgeDiscoveryProtocol: Pattern and relationship discovery
"""

from typing import Protocol, TypeVar, Generic, runtime_checkable, List, Dict, Optional, Tuple, Any
import numpy as np
from abc import abstractmethod

# Type variables for generic protocols
T_Concept = TypeVar('T_Concept')
T_ConceptId = TypeVar('T_ConceptId', bound=str, covariant=True)
T_Context = TypeVar('T_Context', bound=str)


@runtime_checkable
class ConceptRegistryProtocol(Protocol, Generic[T_Concept, T_ConceptId]):
    """
    Protocol for concept registry implementations.
    
    Defines the interface for registering, retrieving, and managing concepts
    with type safety guarantees.
    """
    
    def create_concept(
        self, 
        name: str, 
        context: str = "default",
        synset_id: Optional[str] = None,
        disambiguation: Optional[str] = None,
        auto_disambiguate: bool = True
    ) -> T_Concept:
        """Create and register a concept, returning the concept object."""
        ...
    
    def get_concept(
        self, 
        name: str, 
        context: str = "default",
        synset_id: Optional[str] = None
    ) -> Optional[T_Concept]:
        """Retrieve concept by name and context."""
        ...
    
    def find_similar_concepts(
        self, 
        concept: T_Concept, 
        threshold: float = 0.7
    ) -> List[Tuple[T_Concept, float]]:
        """Find concepts similar to the given concept above threshold."""
        ...
    
    def list_concepts(self, context: Optional[T_Context] = None) -> List[T_Concept]:
        """List all concepts, optionally filtered by context."""
        ...
    
    @property
    def concept_count(self) -> int:
        """Return total number of registered concepts."""
        ...


@runtime_checkable
class EmbeddingProviderProtocol(Protocol):
    """
    Protocol for embedding providers.
    
    Defines interface for generating vector embeddings and computing
    semantic similarity between concepts.
    """
    
    def generate_embedding(
        self, 
        concept: str, 
        context: str = "default"
    ) -> np.ndarray:
        """Generate embedding vector for concept in given context."""
        ...
    
    def compute_similarity(
        self, 
        emb1: np.ndarray, 
        emb2: np.ndarray
    ) -> float:
        """Compute similarity score between two embeddings (0.0-1.0)."""
        ...
    
    def batch_generate_embeddings(
        self,
        concepts: List[str],
        context: str = "default"
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for multiple concepts efficiently."""
        ...
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        ...
    
    @property
    def provider_name(self) -> str:
        """Return identifier for this embedding provider."""
        ...


@runtime_checkable
class SemanticReasoningProtocol(Protocol):
    """
    Protocol for semantic reasoning engines.
    
    Defines interface for advanced semantic operations like analogical
    reasoning, semantic field discovery, and cross-domain analysis.
    """
    
    def complete_analogy(
        self, 
        partial_analogy: Dict[str, str], 
        max_completions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete partial analogies.
        
        Args:
            partial_analogy: Dict with known mappings and "?" for completion
            max_completions: Maximum number of completions to return
            
        Returns:
            List of completion results with confidence scores
        """
        ...
    
    def find_analogous_concepts(
        self,
        source_concept: Any,
        frame_context: Optional[str] = None,
        cluster_threshold: float = 0.6,
        frame_threshold: float = 0.6
    ) -> List[Tuple[Any, float, str]]:
        """Find concepts analogous to the given source concept."""
        ...
    
    def discover_semantic_fields(
        self, 
        min_coherence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Discover coherent semantic fields from concept space.
        
        Args:
            min_coherence: Minimum coherence score for discovered fields
            
        Returns:
            List of semantic field descriptions with metadata
        """
        # This is the protocol interface - implementations may override
        return []
    
    def find_cross_domain_analogies(
        self, 
        source_domain: str,
        target_domain: str,
        min_quality: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find structural analogies between different domains."""
        ...


@runtime_checkable
class KnowledgeDiscoveryProtocol(Protocol):
    """
    Protocol for knowledge discovery operations.
    
    Defines interface for pattern extraction, relationship discovery,
    and knowledge base expansion.
    """
    
    def discover_patterns(
        self, 
        domain: str,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover patterns within a specific domain."""
        ...
    
    def extract_relationships(
        self, 
        concepts: List[str],
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships between given concepts."""
        ...
    
    def suggest_new_concepts(
        self,
        existing_concepts: List[str],
        domain: str = "default"
    ) -> List[Dict[str, Any]]:
        """Suggest new concepts that would complement existing ones."""
        ...
    
    def validate_knowledge_consistency(
        self,
        knowledge_base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency of knowledge base and return issues."""
        ...


@runtime_checkable
class FrameRegistryProtocol(Protocol):
    """
    Protocol for semantic frame management.
    
    Defines interface for creating, managing, and querying semantic frames
    in the FrameNet tradition.
    """
    
    def create_frame(
        self,
        name: str,
        definition: str,
        core_elements: List[str],
        peripheral_elements: Optional[List[str]] = None
    ) -> str:
        """Create a new semantic frame and return its ID."""
        ...
    
    def get_frame(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve frame by ID."""
        ...
    
    def find_frames_for_concept(
        self, 
        concept: str
    ) -> List[Dict[str, Any]]:
        """Find all frames that include the given concept."""
        ...
    
    def create_frame_instance(
        self,
        frame_id: str,
        concept_bindings: Dict[str, str]
    ) -> str:
        """Create instance of frame with concept bindings."""
        ...


@runtime_checkable
class ClusterRegistryProtocol(Protocol):
    """
    Protocol for concept clustering operations.
    
    Defines interface for clustering concepts based on embeddings
    and managing cluster-based similarity computations.
    """
    
    def update_clusters(
        self,
        concepts: Optional[List[str]] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update concept clusters and return clustering metadata."""
        ...
    
    def get_cluster_membership(
        self, 
        concept: str
    ) -> Optional[Dict[str, float]]:
        """Get cluster membership probabilities for concept."""
        ...
    
    def find_cluster_neighbors(
        self,
        concept: str,
        max_neighbors: int = 10
    ) -> List[Tuple[str, float]]:
        """Find nearest neighbors within the same cluster."""
        ...
    
    @property
    def is_trained(self) -> bool:
        """Return whether clustering model has been trained."""
        ...
    
    @property
    def cluster_count(self) -> int:
        """Return number of clusters."""
        ...


# Utility type aliases for common protocol combinations
SemanticSystemProtocol = SemanticReasoningProtocol
HybridRegistryProtocol = ConceptRegistryProtocol[Any, str]
