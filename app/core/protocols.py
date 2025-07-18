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
from numpy.typing import NDArray
from abc import abstractmethod

# Type variables for generic protocols
T_Concept = TypeVar('T_Concept')
T_ConceptId = TypeVar('T_ConceptId', bound=str, covariant=True)
T_Context = TypeVar('T_Context', bound=str)
T_Frame = TypeVar('T_Frame')
T_FrameId = TypeVar('T_FrameId', bound=str, covariant=True)
T_ClusterId = TypeVar('T_ClusterId')


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
    ) -> NDArray[np.float32]:
        """Generate embedding vector for concept in given context."""
        ...
    
    def compute_similarity(
        self, 
        emb1: NDArray[np.float32], 
        emb2: NDArray[np.float32]
    ) -> float:
        """Compute similarity score between two embeddings (0.0-1.0)."""
        ...
    
    def batch_generate_embeddings(
        self,
        concepts: List[str],
        context: str = "default"
    ) -> Dict[str, NDArray[np.float32]]:
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
class FrameRegistryProtocol(Protocol, Generic[T_Frame, T_FrameId]):
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
    ) -> T_Frame:
        """Create a new semantic frame and return it."""
        ...

    def get_frame(self, frame_id: T_FrameId) -> Optional[T_Frame]:
        """Retrieve a frame by its ID."""
        ...

    def register_frame(self, frame: T_Frame) -> T_Frame:
        """Register a semantic frame in the registry."""
        ...


@runtime_checkable
class ClusterRegistryProtocol(Protocol, Generic[T_Concept, T_ClusterId]):
    """
    Protocol for managing concept clusters.
    
    Defines interface for clustering concepts based on embeddings
    and managing cluster-based similarity computations.
    """
    
    def update_clusters(
        self,
        concepts: Optional[List[T_Concept]] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update concept clusters and return clustering metadata."""
        ...
    
    def get_cluster_membership(
        self, 
        concept: T_Concept
    ) -> Optional[T_ClusterId]:
        """Get cluster ID for a given concept."""
        ...
    
    def find_cluster_neighbors(
        self,
        concept: T_Concept,
        max_neighbors: int = 10
    ) -> List[Tuple[T_Concept, float]]:
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


@runtime_checkable
class PersistenceProtocol(Protocol):
    """
    Protocol for persistence layer implementations.
    
    Defines the interface for saving, loading, and managing persistent state
    of the soft logic system components.
    """
    
    @abstractmethod
    def save_registry_state(self, registry: Any, context_name: str = "default", 
                           format_type: str = "json") -> Dict[str, Any]:
        """Save complete registry state with versioning."""
        ...
    
    @abstractmethod
    def load_registry_state(self, context_name: str = "default") -> Optional[Dict[str, Any]]:
        """Load complete registry state."""
        ...
    
    @abstractmethod
    def export_knowledge_base(self, format: str = "json", 
                            compressed: bool = False) -> Any:
        """Export complete knowledge base in specified format."""
        ...
    
    @abstractmethod
    def import_knowledge_base(self, source_path: Any, 
                            merge_strategy: str = "overwrite") -> bool:
        """Import knowledge base with conflict resolution."""
        ...


@runtime_checkable
class BatchPersistenceProtocol(Protocol):
    """
    Protocol for batch-aware persistence implementations.
    
    Extends basic persistence with batch operation support and workflow management.
    """
    
    @abstractmethod
    def create_analogy_batch(self, analogies: List[Dict[str, Any]], 
                           workflow_id: Optional[str] = None) -> Any:
        """Create batch of analogies with workflow tracking."""
        ...
    
    @abstractmethod
    def process_analogy_batch(self, workflow_id: str) -> Any:
        """Process pending analogy batch."""
        ...
    
    @abstractmethod
    def delete_analogies_batch(self, criteria: Any, 
                             workflow_id: Optional[str] = None) -> Any:
        """Delete analogies matching criteria."""
        ...
    
    @abstractmethod
    def get_workflow_status(self, workflow_id: str) -> Optional[Any]:
        """Get current workflow status."""
        ...
    
    @abstractmethod
    def stream_analogies(self, domain: Optional[str] = None, 
                        min_quality: Optional[float] = None) -> Any:
        """Stream analogies from storage with optional filtering."""
        ...
    
    @abstractmethod
    def compact_analogies_jsonl(self) -> Dict[str, Any]:
        """Compact analogy storage by removing deleted records."""
        ...


# Utility type aliases for common protocol combinations
SemanticSystemProtocol = SemanticReasoningProtocol
HybridRegistryProtocol = ConceptRegistryProtocol[Any, str]
