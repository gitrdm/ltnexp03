"""
Hybrid Concept Registry: Integration of Concepts, Frames, and Clusters

This module implements a unified registry that combines basic concept management
with semantic frame understanding and clustering-based representations for
advanced analogical reasoning.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import logging

from .concept_registry import ConceptRegistry
from .frame_cluster_registry import FrameRegistry, ClusterRegistry
from .frame_cluster_abstractions import (
    FrameAwareConcept, SemanticFrame, FrameInstance, ConceptCluster,
    AnalogicalMapping, FrameElementType
)
from .protocols import (
    ConceptRegistryProtocol, FrameRegistryProtocol, ClusterRegistryProtocol,
    SemanticReasoningProtocol, KnowledgeDiscoveryProtocol
)


class HybridConceptRegistry(
    ConceptRegistry,
    FrameRegistryProtocol,
    ClusterRegistryProtocol,
    SemanticReasoningProtocol,
    KnowledgeDiscoveryProtocol
):
    """
    Advanced concept registry with frame and cluster integration.
    
    Combines the basic concept management capabilities with semantic frame
    understanding and clustering-based representations to enable sophisticated
    analogical reasoning and concept organization.
    
    ARCHITECTURE:
    =============
    
    HybridConceptRegistry
    ├── ConceptRegistry (base functionality)
    ├── FrameRegistry (semantic frames)
    ├── ClusterRegistry (concept clustering)
    └── Integration layer (hybrid reasoning)
    
    CAPABILITIES:
    =============
    
    1. **Frame-Aware Concept Management**: Concepts understand their roles in frames
    2. **Cluster-Based Similarity**: Similarity based on learned embeddings
    3. **Multi-Level Analogies**: Analogies work at both frame and cluster levels
    4. **Contextual Disambiguation**: Frame and cluster information aids disambiguation
    5. **Compositional Reasoning**: New concepts understood through frame roles and clusters
    """
    
    def __init__(self, download_wordnet: bool = True, embedding_dim: int = 300, 
                 n_clusters: int = 50):
        """
        Initialize hybrid registry with all components.
        
        Args:
            download_wordnet: Whether to initialize WordNet
            embedding_dim: Dimension of concept embeddings
            n_clusters: Number of clusters for concept clustering
        """
        # Initialize base concept registry
        super().__init__(download_wordnet=download_wordnet)
        
        # Initialize specialized registries
        self.frame_registry = FrameRegistry()
        self.cluster_registry = ClusterRegistry(embedding_dim=embedding_dim, n_clusters=n_clusters)
        
        # Hybrid-specific storage
        self.frame_aware_concepts: Dict[str, FrameAwareConcept] = {}
        self.concept_similarities: Dict[str, Dict[str, float]] = {}  # Cached similarities
        
        # Configuration
        self.embedding_dim = embedding_dim
        self.auto_cluster = True  # Automatically update clusters when embeddings change
        
        self.logger = logging.getLogger(__name__)
    
    def create_frame_aware_concept(self, name: str, context: str = "default",
                                 synset_id: Optional[str] = None,
                                 disambiguation: Optional[str] = None,
                                 embedding: Optional[NDArray[np.float32]] = None,
                                 auto_disambiguate: bool = True) -> FrameAwareConcept:
        """
        Create a frame-aware concept with extended capabilities.
        
        This method extends the basic concept creation with frame and cluster
        awareness, providing richer semantic representations.
        """
        # Create base concept first
        base_concept = self.create_concept(name, context, synset_id, disambiguation, auto_disambiguate)
        
        # Create frame-aware version
        frame_concept = FrameAwareConcept(
            name=base_concept.name,
            synset_id=base_concept.synset_id,
            disambiguation=base_concept.disambiguation,
            context=base_concept.context,
            metadata=base_concept.metadata.copy(),
            embedding=embedding
        )
        
        # Store in hybrid registry
        self.frame_aware_concepts[frame_concept.unique_id] = frame_concept
        
        # Add embedding if provided
        if embedding is not None:
            self.add_concept_embedding(frame_concept.unique_id, embedding)
        
        self.logger.info(f"Created frame-aware concept: {frame_concept.unique_id}")
        return frame_concept
    
    def get_frame_aware_concept(self, name: str, context: str = "default",
                              synset_id: Optional[str] = None) -> Optional[FrameAwareConcept]:
        """Retrieve a frame-aware concept."""
        if synset_id:
            unique_id = f"{context}:{name}:{synset_id}"
        else:
            unique_id = f"{context}:{name}"
        
        return self.frame_aware_concepts.get(unique_id)
    
    def register_frame(self, frame: SemanticFrame) -> SemanticFrame:
        """Register a semantic frame."""
        return self.frame_registry.register_frame(frame)
    
    def create_frame_instance(self, frame_name: str, instance_id: str,
                            concept_bindings: Dict[str, Union[str, FrameAwareConcept]],
                            context: str = "default") -> FrameInstance:
        """
        Create a frame instance with concept bindings.
        
        Args:
            frame_name: Name of the frame to instantiate
            instance_id: Unique identifier for this instance
            concept_bindings: Mapping of frame elements to concepts (by name or object)
            context: Context for the instance
        """
        # Resolve concept names to objects
        resolved_bindings = {}
        for element_name, concept_ref in concept_bindings.items():
            if isinstance(concept_ref, str):
                # Look up concept by name
                concept = self.get_frame_aware_concept(concept_ref, context)
                if concept is None:
                    # Create new concept if not found
                    concept = self.create_frame_aware_concept(concept_ref, context)
            else:
                concept = concept_ref
            
            resolved_bindings[element_name] = concept
        
        return self.frame_registry.create_frame_instance(
            frame_name, instance_id, resolved_bindings, context
        )
    
    def add_concept_embedding(self, concept_id: str, embedding: NDArray[np.float32]) -> None:
        """Add or update a concept's embedding."""
        self.cluster_registry.add_concept_embedding(concept_id, embedding)
        
        # Update the concept object if it exists
        if concept_id in self.frame_aware_concepts:
            self.frame_aware_concepts[concept_id].embedding = embedding
        
        # Auto-update clusters if enabled and we have enough concepts
        if self.auto_cluster and len(self.cluster_registry.concept_embeddings) >= self.cluster_registry.n_clusters:
            self.update_clusters()
    
    def update_clusters(self) -> None:
        """Update concept clusters based on current embeddings."""
        self.cluster_registry.train_clusters()
        
        # Update cluster memberships for all frame-aware concepts
        for concept_id, concept in self.frame_aware_concepts.items():
            memberships = self.cluster_registry.get_concept_cluster_memberships(concept_id)
            concept.cluster_memberships = memberships
            
            # Set primary cluster
            if memberships:
                primary_cluster_id = max(memberships.items(), key=lambda x: x[1])[0]
                concept.primary_cluster = primary_cluster_id
        
        self.logger.info("Updated concept clusters and memberships")
    
    def find_analogous_concepts(self, source_concept: Union[str, FrameAwareConcept],
                              frame_context: Optional[str] = None,
                              cluster_threshold: float = 0.7,
                              frame_threshold: float = 0.6) -> List[Tuple[FrameAwareConcept, float, str]]:
        """
        Find concepts analogous to the source concept.
        
        Uses both frame-based and cluster-based similarity to identify
        analogous concepts. Returns concepts with similarity scores and
        the basis for the analogy (frame/cluster/both).
        
        Args:
            source_concept: Source concept (name or object)
            frame_context: Optional frame to focus the search
            cluster_threshold: Minimum cluster similarity
            frame_threshold: Minimum frame similarity
            
        Returns:
            List of (concept, similarity_score, analogy_basis) tuples
        """
        # Resolve source concept
        if isinstance(source_concept, str):
            source = self.get_frame_aware_concept(source_concept)
            if source is None:
                return []
        else:
            source = source_concept
        
        analogies = []
        
        for target_concept in self.frame_aware_concepts.values():
            if target_concept.unique_id == source.unique_id:
                continue
            
            # Compute frame-based similarity
            frame_similarity = self._compute_frame_similarity(source, target_concept, frame_context)
            
            # Compute cluster-based similarity
            cluster_similarity = self._compute_cluster_similarity(source, target_concept)
            
            # Determine analogy basis and overall score
            analogy_basis = []
            scores = []
            
            if frame_similarity >= frame_threshold:
                analogy_basis.append("frame")
                scores.append(frame_similarity)
            
            if cluster_similarity >= cluster_threshold:
                analogy_basis.append("cluster")
                scores.append(cluster_similarity)
            
            if scores:
                # Combined score (weighted average)
                overall_score = float(np.mean(scores))
                basis_str = "+".join(analogy_basis)
                analogies.append((target_concept, overall_score, basis_str))
        
        # Sort by similarity score
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies
    
    def _compute_frame_similarity(self, concept1: FrameAwareConcept, 
                                concept2: FrameAwareConcept,
                                frame_context: Optional[str] = None) -> float:
        """Compute similarity based on frame role compatibility."""
        frames1 = set(concept1.get_frames())
        frames2 = set(concept2.get_frames())
        
        if frame_context:
            # Focus on specific frame
            if frame_context not in frames1 or frame_context not in frames2:
                return 0.0
            
            role1 = concept1.get_role_in_frame(frame_context)
            role2 = concept2.get_role_in_frame(frame_context)
            
            return 1.0 if role1 == role2 else 0.5  # Same role = 1.0, different role = 0.5
        else:
            # General frame compatibility
            if not frames1 or not frames2:
                return 0.0
            
            # Check for shared frames
            shared_frames = frames1.intersection(frames2)
            if shared_frames:
                # Check if they play similar roles in shared frames
                role_similarities = []
                for frame in shared_frames:
                    role1 = concept1.get_role_in_frame(frame)
                    role2 = concept2.get_role_in_frame(frame)
                    role_similarities.append(1.0 if role1 == role2 else 0.5)
                
                return float(np.mean(role_similarities))
            
            # Check for related frames (through inheritance, etc.)
            related_score = 0.0
            for frame1 in frames1:
                for frame2 in frames2:
                    if self.frame_registry._are_frames_related(frame1, frame2):
                        related_score = max(related_score, 0.7)
            
            return related_score
    
    def _compute_cluster_similarity(self, concept1: FrameAwareConcept, 
                                  concept2: FrameAwareConcept) -> float:
        """Compute similarity based on cluster membership overlap."""
        memberships1 = concept1.cluster_memberships
        memberships2 = concept2.cluster_memberships
        
        if not memberships1 or not memberships2:
            return 0.0
        
        return self.cluster_registry._compute_membership_similarity(memberships1, memberships2)
    
    def create_semantic_frame(self, name: str, definition: str,
                            core_elements: List[str],
                            peripheral_elements: Optional[List[str]] = None,
                            lexical_units: Optional[List[str]] = None) -> SemanticFrame:
        """
        Create a semantic frame with the given specification.
        
        Simplified frame creation for common use cases.
        """
        from .frame_cluster_abstractions import FrameElement
        
        # Create core frame elements
        core_frame_elements = [
            FrameElement(name=elem, description=f"Core element: {elem}", 
                        element_type=FrameElementType.CORE)
            for elem in core_elements
        ]
        
        # Create peripheral frame elements
        peripheral_frame_elements = []
        if peripheral_elements:
            peripheral_frame_elements = [
                FrameElement(name=elem, description=f"Peripheral element: {elem}",
                            element_type=FrameElementType.PERIPHERAL)
                for elem in peripheral_elements
            ]
        
        frame = SemanticFrame(
            name=name,
            definition=definition,
            core_elements=core_frame_elements,
            peripheral_elements=peripheral_frame_elements,
            lexical_units=lexical_units or []
        )
        
        return self.register_frame(frame)
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the hybrid registry."""
        base_stats = self.get_concept_stats()
        
        hybrid_stats = {
            **base_stats,
            "frame_aware_concepts": len(self.frame_aware_concepts),
            "semantic_frames": len(self.frame_registry.frames),
            "frame_instances": len(self.frame_registry.frame_instances),
            "concept_clusters": len(self.cluster_registry.clusters),
            "concepts_with_embeddings": len(self.cluster_registry.concept_embeddings),
            "clustering_trained": self.cluster_registry.is_trained,
        }
        
        # Cluster-specific statistics
        if self.cluster_registry.is_trained:
            cluster_sizes = [len(cluster.members) for cluster in self.cluster_registry.clusters.values()]
            hybrid_stats.update({
                "avg_cluster_size": int(np.mean(cluster_sizes)) if cluster_sizes else 0,
                "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            })
        
        return hybrid_stats
