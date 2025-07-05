"""
Enhanced Frame-Cluster Integration: Advanced Semantic Reasoning Module

This module extends the hybrid registry with advanced semantic reasoning capabilities,
including cross-domain analogy discovery, dynamic frame learning, and sophisticated
embedding-based concept similarity.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from scipy.spatial.distance import cosine

from .hybrid_registry import HybridConceptRegistry
from .frame_cluster_abstractions import (
    FrameAwareConcept, SemanticFrame, FrameInstance, ConceptCluster,
    AnalogicalMapping, FrameElementType, FrameElement
)
from .vector_embeddings import VectorEmbeddingManager, SemanticEmbeddingProvider


@dataclass
class CrossDomainAnalogy:
    """
    Represents a cross-domain analogical mapping.
    
    Example: Military strategy concepts map to business strategy concepts
    through shared abstract structural patterns.
    """
    source_domain: str
    target_domain: str
    
    # Concept mappings across domains
    concept_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Frame mappings (abstract structural patterns)
    frame_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Quality metrics
    structural_coherence: float = 0.0    # How well structures align
    semantic_coherence: float = 0.0      # How well meanings align
    productivity: float = 0.0            # How many new mappings it suggests
    
    # Supporting evidence
    evidence_instances: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def add_concept_mapping(self, source_concept: str, target_concept: str) -> None:
        """Add a concept mapping to the cross-domain analogy."""
        self.concept_mappings[source_concept] = target_concept
    
    def add_frame_mapping(self, source_frame: str, target_frame: str) -> None:
        """Add a frame mapping to the cross-domain analogy."""
        self.frame_mappings[source_frame] = target_frame
    
    def compute_overall_quality(self) -> float:
        """Compute overall analogy quality."""
        return (0.4 * self.structural_coherence + 
                0.3 * self.semantic_coherence + 
                0.3 * self.productivity)


@dataclass
class SemanticField:
    """
    Represents a coherent semantic field discovered through clustering.
    
    Semantic fields are coherent regions of meaning space that contain
    related concepts and frames.
    """
    name: str
    description: str
    
    # Core concepts that define the field
    core_concepts: Set[str] = field(default_factory=set)
    
    # Related concepts with membership strength
    related_concepts: Dict[str, float] = field(default_factory=dict)
    
    # Frames associated with this field
    associated_frames: Set[str] = field(default_factory=set)
    
    # Characteristic patterns
    typical_patterns: List[str] = field(default_factory=list)
    
    # Embedding centroid
    centroid: Optional[np.ndarray] = None
    
    def add_core_concept(self, concept_id: str) -> None:
        """Add a core concept to the semantic field."""
        self.core_concepts.add(concept_id)
    
    def add_related_concept(self, concept_id: str, strength: float) -> None:
        """Add a related concept with membership strength."""
        self.related_concepts[concept_id] = strength
    
    def get_all_concepts(self) -> Set[str]:
        """Get all concepts (core + related) in the field."""
        return self.core_concepts.union(set(self.related_concepts.keys()))


class EnhancedHybridRegistry(HybridConceptRegistry):
    """
    Enhanced hybrid registry with advanced semantic reasoning capabilities.
    
    Extends the base hybrid registry with:
    - Cross-domain analogy discovery
    - Dynamic semantic field identification
    - Advanced embedding-based reasoning
    - Sophisticated concept relationship discovery
    """
    
    def __init__(self, download_wordnet: bool = True, embedding_dim: int = 300, 
                 n_clusters: int = 50, enable_cross_domain: bool = True,
                 embedding_provider: str = "semantic"):
        """Initialize enhanced hybrid registry."""
        super().__init__(download_wordnet, embedding_dim, n_clusters)
        
        # Enhanced storage
        self.semantic_fields: Dict[str, SemanticField] = {}
        self.cross_domain_analogies: List[CrossDomainAnalogy] = []
        self.domain_embeddings: Dict[str, np.ndarray] = {}
        
        # Vector embedding integration
        self.embedding_manager = VectorEmbeddingManager(
            default_provider=embedding_provider,
            cache_dir="embeddings_cache"
        )
        
        # Configuration
        self.enable_cross_domain = enable_cross_domain
        self.min_field_size = 3
        self.analogy_threshold = 0.6
        
        # Caching for performance
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def discover_semantic_fields(self, min_coherence: float = 0.7) -> List[SemanticField]:
        """
        Discover semantic fields from concept clusters and embeddings.
        
        Uses advanced clustering analysis to identify coherent semantic regions.
        """
        if not self.cluster_registry.is_trained:
            self.logger.warning("Clustering not trained - training now")
            self.update_clusters()
        
        discovered_fields = []
        
        # Analyze each cluster for semantic coherence
        for cluster_id, cluster in self.cluster_registry.clusters.items():
            if len(cluster.members) < self.min_field_size:
                continue
            
            # Compute cluster coherence
            coherence = self._compute_semantic_coherence(cluster)
            
            if coherence >= min_coherence:
                # Create semantic field
                field = self._create_semantic_field_from_cluster(cluster, coherence)
                discovered_fields.append(field)
                self.semantic_fields[field.name] = field
        
        self.logger.info(f"Discovered {len(discovered_fields)} semantic fields")
        return discovered_fields
    
    def _compute_semantic_coherence(self, cluster: ConceptCluster) -> float:
        """Compute semantic coherence of a cluster."""
        if not cluster.members:
            return 0.0
        
        # Get embeddings for cluster members
        concept_ids = list(cluster.members.keys())
        embeddings = []
        
        for concept_id in concept_ids:
            if concept_id in self.cluster_registry.concept_embeddings:
                embeddings.append(self.cluster_registry.concept_embeddings[concept_id])
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute average pairwise similarity
        embeddings = np.array(embeddings)
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = 1 - cosine(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return float(np.mean(similarities))
    
    def _create_semantic_field_from_cluster(self, cluster: ConceptCluster, 
                                          coherence: float) -> SemanticField:
        """Create a semantic field from a coherent cluster."""
        # Generate field name based on dominant concepts
        core_concepts = cluster.get_core_members(threshold=0.8)
        field_name = f"field_{cluster.cluster_id}"
        
        if core_concepts:
            # Use most representative concept names
            representative_names = [cid.split(":")[-1] for cid in core_concepts[:3]]
            field_name = f"field_{'_'.join(representative_names)}"
        
        field = SemanticField(
            name=field_name,
            description=f"Semantic field with coherence {coherence:.3f}",
            centroid=cluster.centroid.copy() if cluster.centroid is not None else None
        )
        
        # Add core concepts
        for concept_id in core_concepts:
            field.add_core_concept(concept_id)
        
        # Add related concepts
        for concept_id, strength in cluster.members.items():
            if concept_id not in core_concepts:
                field.add_related_concept(concept_id, strength)
        
        # Find associated frames
        for concept_id in field.get_all_concepts():
            if concept_id in self.frame_aware_concepts:
                concept = self.frame_aware_concepts[concept_id]
                field.associated_frames.update(concept.get_frames())
        
        return field
    
    def discover_cross_domain_analogies(self, min_quality: float = 0.6) -> List[CrossDomainAnalogy]:
        """
        Discover cross-domain analogies between semantic fields.
        
        Identifies abstract structural patterns that repeat across domains.
        """
        if not self.semantic_fields:
            self.discover_semantic_fields()
        
        analogies = []
        
        # Compare all pairs of semantic fields
        field_pairs = itertools.combinations(self.semantic_fields.values(), 2)
        
        for field1, field2 in field_pairs:
            analogy = self._compute_cross_domain_analogy(field1, field2)
            
            if analogy.compute_overall_quality() >= min_quality:
                analogies.append(analogy)
        
        # Sort by quality
        analogies.sort(key=lambda x: x.compute_overall_quality(), reverse=True)
        
        # Store best analogies
        self.cross_domain_analogies = analogies
        
        self.logger.info(f"Discovered {len(analogies)} cross-domain analogies")
        return analogies
    
    def _compute_cross_domain_analogy(self, field1: SemanticField, 
                                    field2: SemanticField) -> CrossDomainAnalogy:
        """Compute cross-domain analogy between two semantic fields."""
        analogy = CrossDomainAnalogy(
            source_domain=field1.name,
            target_domain=field2.name
        )
        
        # Find concept mappings based on embedding similarity
        concept_mappings = self._find_concept_mappings(field1, field2)
        for source, target in concept_mappings:
            analogy.add_concept_mapping(source, target)
        
        # Find frame mappings based on structural similarity
        frame_mappings = self._find_frame_mappings(field1, field2)
        for source, target in frame_mappings:
            analogy.add_frame_mapping(source, target)
        
        # Compute quality metrics
        analogy.structural_coherence = self._compute_structural_coherence(field1, field2)
        analogy.semantic_coherence = self._compute_field_semantic_coherence(field1, field2)
        analogy.productivity = len(concept_mappings) / max(len(field1.get_all_concepts()), 1)
        
        return analogy
    
    def _find_concept_mappings(self, field1: SemanticField, 
                             field2: SemanticField) -> List[Tuple[str, str]]:
        """Find concept mappings between two semantic fields."""
        mappings = []
        
        # Get embeddings for concepts in both fields
        field1_concepts = {}
        field2_concepts = {}
        
        for cid in field1.get_all_concepts():
            emb = self.cluster_registry.concept_embeddings.get(cid)
            if emb is not None:
                field1_concepts[cid] = emb
                
        for cid in field2.get_all_concepts():
            emb = self.cluster_registry.concept_embeddings.get(cid)
            if emb is not None:
                field2_concepts[cid] = emb
        
        # Find best mappings using Hungarian algorithm approximation
        for concept1, emb1 in field1_concepts.items():
            best_match = None
            best_similarity = 0.0
            
            for concept2, emb2 in field2_concepts.items():
                similarity = 1 - cosine(emb1, emb2)
                if similarity > best_similarity and similarity > 0.5:  # Threshold
                    best_similarity = similarity
                    best_match = concept2
            
            if best_match:
                mappings.append((concept1, best_match))
        
        return mappings
    
    def _find_frame_mappings(self, field1: SemanticField, 
                           field2: SemanticField) -> List[Tuple[str, str]]:
        """Find frame mappings between two semantic fields."""
        mappings = []
        
        # Compare frames from both fields
        for frame1 in field1.associated_frames:
            for frame2 in field2.associated_frames:
                if frame1 != frame2:
                    similarity = self._compute_frame_structural_similarity(frame1, frame2)
                    if similarity > 0.6:  # Threshold
                        mappings.append((frame1, frame2))
        
        return mappings
    
    def _compute_frame_structural_similarity(self, frame1_name: str, 
                                           frame2_name: str) -> float:
        """Compute structural similarity between frames."""
        frame1 = self.frame_registry.get_frame(frame1_name)
        frame2 = self.frame_registry.get_frame(frame2_name)
        
        if not frame1 or not frame2:
            return 0.0
        
        return self.frame_registry._compute_structural_similarity(frame1, frame2)
    
    def _compute_structural_coherence(self, field1: SemanticField, 
                                    field2: SemanticField) -> float:
        """Compute structural coherence between fields."""
        # Based on frame overlap and structural similarity
        frame_overlap = len(field1.associated_frames.intersection(field2.associated_frames))
        total_frames = len(field1.associated_frames.union(field2.associated_frames))
        
        if total_frames == 0:
            return 0.0
        
        return frame_overlap / total_frames
    
    def _compute_field_semantic_coherence(self, field1: SemanticField, 
                                        field2: SemanticField) -> float:
        """Compute semantic coherence between fields."""
        if field1.centroid is None or field2.centroid is None:
            return 0.0
        
        # Compute centroid similarity
        return 1 - cosine(field1.centroid, field2.centroid)
    
    def find_analogical_completions(self, partial_analogy: Dict[str, str],
                                  max_completions: int = 5) -> List[Dict[str, str]]:
        """
        Find analogical completions for partial analogies.
        
        Given a partial analogy like {"king": "queen", "man": "?"}, 
        find plausible completions.
        """
        completions = []
        
        # Extract source and target concepts
        source_concepts = list(partial_analogy.keys())
        target_concepts = [v for v in partial_analogy.values() if v != "?"]
        
        if not source_concepts or not target_concepts:
            return completions
        
        # Find the missing mapping
        missing_source = None
        for source, target in partial_analogy.items():
            if target == "?":
                missing_source = source
                break
        
        if not missing_source:
            return completions
        
        # Get embeddings for known mappings
        source_embeddings = []
        target_embeddings = []
        
        for source, target in partial_analogy.items():
            if target != "?":
                source_emb = self.cluster_registry.concept_embeddings.get(f"default:{source}")
                target_emb = self.cluster_registry.concept_embeddings.get(f"default:{target}")
                
                if source_emb is not None and target_emb is not None:
                    source_embeddings.append(source_emb)
                    target_embeddings.append(target_emb)
        
        if not source_embeddings:
            return completions
        
        # Compute transformation vector
        transformation = np.mean(np.array(target_embeddings) - np.array(source_embeddings), axis=0)
        
        # Apply transformation to missing source
        missing_source_emb = self.cluster_registry.concept_embeddings.get(f"default:{missing_source}")
        if missing_source_emb is None:
            return completions
        
        target_emb = missing_source_emb + transformation
        
        # Find concepts similar to the transformed embedding
        candidates = []
        for concept_id, emb in self.cluster_registry.concept_embeddings.items():
            if concept_id not in source_concepts and concept_id not in target_concepts:
                similarity = 1 - cosine(target_emb, emb)
                candidates.append((concept_id, similarity))
        
        # Sort by similarity and return top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for concept_id, similarity in candidates[:max_completions]:
            completion = partial_analogy.copy()
            completion[missing_source] = concept_id
            completions.append(completion)
        
        return completions
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the enhanced registry."""
        base_stats = self.get_hybrid_statistics()
        
        enhanced_stats = {
            **base_stats,
            "semantic_fields": len(self.semantic_fields),
            "cross_domain_analogies": len(self.cross_domain_analogies),
            "domain_embeddings": len(self.domain_embeddings),
            "cached_similarities": len(self.similarity_cache),
        }
        
        # Field-specific statistics
        if self.semantic_fields:
            field_sizes = [len(field.get_all_concepts()) for field in self.semantic_fields.values()]
            enhanced_stats.update({
                "avg_field_size": np.mean(field_sizes),
                "max_field_size": max(field_sizes),
                "min_field_size": min(field_sizes),
            })
        
        # Analogy-specific statistics
        if self.cross_domain_analogies:
            analogy_qualities = [analogy.compute_overall_quality() for analogy in self.cross_domain_analogies]
            enhanced_stats.update({
                "avg_analogy_quality": np.mean(analogy_qualities),
                "max_analogy_quality": max(analogy_qualities),
                "min_analogy_quality": min(analogy_qualities),
            })
        
        return enhanced_stats
    
    def create_frame_aware_concept_with_advanced_embedding(self, name: str, context: str = "default",
                                 synset_id: Optional[str] = None,
                                 disambiguation: Optional[str] = None,
                                 embedding: Optional[np.ndarray] = None,
                                 auto_disambiguate: bool = True,
                                 use_semantic_embedding: bool = True) -> FrameAwareConcept:
        """
        Create a frame-aware concept with advanced embedding capabilities.
        """
        # Create base concept first
        concept = self.create_frame_aware_concept(
            name, context, synset_id, disambiguation, 
            embedding if not use_semantic_embedding else None,
            auto_disambiguate
        )
        
        # Generate semantic embedding if requested
        if use_semantic_embedding and embedding is None:
            semantic_embedding = self.embedding_manager.get_embedding(
                concept.unique_id, name
            )
            if semantic_embedding is not None:
                concept.embedding = semantic_embedding
                self.add_concept_embedding(concept.unique_id, semantic_embedding)
        
        return concept
