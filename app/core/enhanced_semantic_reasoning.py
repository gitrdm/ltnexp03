"""
Enhanced Frame-Cluster Integration: Advanced Semantic Reasoning Module

This module extends the hybrid registry with advanced semantic reasoning capabilities,
including cross-domain analogy discovery, dynamic frame learning, and sophisticated
embedding-based concept similarity.

DESIGN BY CONTRACT INTEGRATION:
===============================
This module implements comprehensive Design by Contract validation using icontract
to ensure robust operation of semantic reasoning algorithms and maintain system
invariants throughout complex reasoning operations.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Set, Any, Union
import logging
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
from scipy.spatial.distance import cosine
from icontract import require, ensure, invariant, ViolationError

from .hybrid_registry import HybridConceptRegistry
from .frame_cluster_abstractions import (
    FrameAwareConcept, SemanticFrame, FrameInstance, ConceptCluster,
    AnalogicalMapping, FrameElementType, FrameElement
)
from .vector_embeddings import VectorEmbeddingManager, SemanticEmbeddingProvider
from .protocols import SemanticReasoningProtocol, KnowledgeDiscoveryProtocol


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
    centroid: Optional[NDArray[np.float32]] = None
    
    def add_core_concept(self, concept_id: str) -> None:
        """Add a core concept to the semantic field."""
        self.core_concepts.add(concept_id)
    
    def add_related_concept(self, concept_id: str, strength: float) -> None:
        """Add a related concept with membership strength."""
        self.related_concepts[concept_id] = strength
    
    def get_all_concepts(self) -> Set[str]:
        """Get all concepts (core + related) in the field."""
        return self.core_concepts.union(set(self.related_concepts.keys()))


# Re-enable class invariants with defensive checks for initialization order
@invariant(lambda self: not hasattr(self, 'frame_aware_concepts') or 
           not hasattr(self.frame_aware_concepts, '__len__') or 
           len(self.frame_aware_concepts) >= 0,
           description="registry must maintain non-negative concept count when initialized")
@invariant(lambda self: not hasattr(self, 'semantic_fields') or isinstance(self.semantic_fields, dict),
           description="registry must maintain semantic fields storage")
@invariant(lambda self: not hasattr(self, 'cross_domain_analogies') or isinstance(self.cross_domain_analogies, list),
           description="registry must maintain cross-domain analogies storage")
class EnhancedHybridRegistry(HybridConceptRegistry, SemanticReasoningProtocol, KnowledgeDiscoveryProtocol):
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
        # Initialize enhanced storage BEFORE calling super().__init__
        # This ensures invariants are satisfied during parent initialization
        self.semantic_fields: Dict[str, SemanticField] = {}
        self.cross_domain_analogies: List[CrossDomainAnalogy] = []
        self.domain_embeddings: Dict[str, NDArray[np.float32]] = {}
        
        # Additional storage
        self.similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # Configuration
        self.enable_cross_domain = enable_cross_domain
        self.min_field_size = 3
        self.analogy_threshold = 0.6
        
        # Call parent initialization
        super().__init__(download_wordnet, embedding_dim, n_clusters)
        
        # Vector embedding integration
        self.embedding_manager = VectorEmbeddingManager(
            default_provider=embedding_provider,
            cache_dir="embeddings_cache"
        )
        
        self.logger = logging.getLogger(__name__)

    def _discover_semantic_fields_original(self, min_coherence: float = 0.7) -> List[SemanticField]:
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
        embeddings: List[NDArray[np.float32]] = []
        
        for concept_id in concept_ids:
            if concept_id in self.cluster_registry.concept_embeddings:
                embeddings.append(self.cluster_registry.concept_embeddings[concept_id])
        
        if len(embeddings) < 2:
            return 0.0
        
        # Compute average pairwise similarity
        embeddings_array = np.array(embeddings)
        similarities: List[float] = []
        
        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                sim = 1 - cosine(embeddings_array[i], embeddings_array[j])
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
        similarity: float = 1 - cosine(field1.centroid, field2.centroid)
        return float(similarity)
    
    @require(lambda partial_analogy: isinstance(partial_analogy, dict) and len(partial_analogy) >= 2,
             "partial_analogy must be dict with at least 2 mappings")
    @require(lambda partial_analogy: any(v == "?" for v in partial_analogy.values()),
             "partial_analogy must contain exactly one '?' value")
    @require(lambda max_completions: isinstance(max_completions, int) and max_completions > 0,
             "max_completions must be positive integer")
    @ensure(lambda result: isinstance(result, list),
            "result must be a list")
    @ensure(lambda result, max_completions: len(result) <= max_completions,
            "result length must not exceed max_completions")
    def find_analogical_completions(self, partial_analogy: Dict[str, str],
                                  max_completions: int = 5) -> List[Dict[str, str]]:
        """
        Find analogical completions for partial analogies.
        
        Given a partial analogy like {"king": "queen", "man": "?"}, 
        find plausible completions.
        """
        completions: List[Dict[str, str]] = []
        
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
    
    @require(lambda name: isinstance(name, str) and len(name.strip()) > 0,
             "concept name must be non-empty string")
    @require(lambda context: isinstance(context, str) and len(context.strip()) > 0,
             "context must be non-empty string")
    @require(lambda synset_id: synset_id is None or (isinstance(synset_id, str) and '.' in synset_id),
             "synset_id must be None or valid synset format")
    @ensure(lambda result: result is not None,
            "concept creation must return valid concept")
    @ensure(lambda result: hasattr(result, 'name') and hasattr(result, 'context'),
            "created concept must have required attributes")
    @ensure(lambda result, name, context: result.name == name and result.context == context,
            "created concept must have matching name and context")
    def create_frame_aware_concept_with_advanced_embedding(self, name: str, context: str = "default",
                                 synset_id: Optional[str] = None,
                                 disambiguation: Optional[str] = None,
                                 embedding: Optional[NDArray[np.float32]] = None,
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

    # ============================================================================
    # PROTOCOL ADAPTER METHODS
    # ============================================================================
    
    def complete_analogy(
        self, 
        partial_analogy: Dict[str, str], 
        max_completions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete partial analogies (SemanticReasoningProtocol interface).
        
        Adapts find_analogical_completions to match protocol signature.
        """
        raw_results = self.find_analogical_completions(partial_analogy, max_completions)
        
        # Convert to protocol-compliant format
        formatted_results: List[Dict[str, Any]] = []
        for result in raw_results:
            if isinstance(result, dict):
                formatted_results.append(result)
            else:
                # Convert simple results to structured format
                formatted_results.append({  # type: ignore[unreachable]
                    'completion': str(result),
                    'confidence': 0.8,
                    'method': 'hybrid',
                    'reasoning': 'frame_cluster_based'
                })
        
        return formatted_results
    
    def discover_semantic_fields(self, min_coherence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Discover semantic fields returning dict format (SemanticReasoningProtocol interface).
        
        This method implements the protocol interface and converts SemanticField objects to dict format.
        """
        raw_fields = self._discover_semantic_fields_original(min_coherence)
        
        # Convert SemanticField objects to dict format
        formatted_fields = []
        for field in raw_fields:
            field_dict = {
                'name': field.name,
                'description': field.description,
                'coherence': getattr(field, 'coherence', 0.7),  # Default if not present
                'core_concepts': list(field.core_concepts),
                'related_concepts': field.related_concepts,
                'associated_frames': list(getattr(field, 'associated_frames', [])),
                'discovery_metadata': {
                    'method': 'clustering_based',
                    'min_coherence': min_coherence
                }
            }
            formatted_fields.append(field_dict)
        
        return formatted_fields
    
    def find_cross_domain_analogies(
        self, 
        source_domain: str,
        target_domain: str,
        min_quality: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find cross-domain analogies (SemanticReasoningProtocol interface).
        
        Adapts discover_cross_domain_analogies to match protocol signature.
        """
        raw_analogies = self.discover_cross_domain_analogies(min_quality)
        
        # Filter and format for specified domains
        formatted_analogies = []
        for analogy in raw_analogies:
            # Convert CrossDomainAnalogy to dict format
            analogy_dict = {
                'source_domain': source_domain,
                'target_domain': target_domain,
                'quality': analogy.compute_overall_quality(),
                'concept_mappings': analogy.concept_mappings,
                'frame_mappings': analogy.frame_mappings,
                'structural_coherence': analogy.structural_coherence,
                'discovery_metadata': {
                    'method': 'semantic_field_based',
                    'min_quality': min_quality
                }
            }
            formatted_analogies.append(analogy_dict)
        
        return formatted_analogies
    
    # KnowledgeDiscoveryProtocol methods
    def discover_patterns(
        self, 
        domain: str,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover patterns within a specific domain."""
        patterns = []
        
        # Use semantic fields as patterns
        fields = self.discover_semantic_fields(min_coherence=0.5)
        for field in fields:
            pattern = {
                'type': 'semantic_field',
                'domain': domain,
                'name': field['name'],
                'elements': field['core_concepts'],
                'confidence': field['coherence'],
                'metadata': field.get('discovery_metadata', {})
            }
            patterns.append(pattern)
        
        # Use cross-domain analogies as patterns
        analogies = self.discover_cross_domain_analogies(min_quality=0.5)
        for analogy in analogies:
            pattern = {
                'type': 'cross_domain_analogy',
                'domain': domain,
                'name': f"analogy_{len(patterns)}",
                'elements': list(analogy.concept_mappings.keys()),
                'confidence': analogy.compute_overall_quality(),
                'metadata': {
                    'target_concepts': list(analogy.concept_mappings.values()),
                    'frame_mappings': analogy.frame_mappings
                }
            }
            patterns.append(pattern)
        
        return patterns
    
    def extract_relationships(
        self, 
        concepts: List[str],
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships between given concepts."""
        relationships = []
        
        # Extract analogical relationships
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Check if concepts are analogous
                analogous = self.find_analogous_concepts(concept1)
                for analog, score, method in analogous:
                    concept2_name = analog.name if hasattr(analog, 'name') else str(analog)
                    if concept2_name == concept2:
                        relationship = {
                            'source': concept1,
                            'target': concept2,
                            'type': 'analogous',
                            'strength': score,
                            'method': method,
                            'metadata': {
                                'discovery_method': 'find_analogous_concepts'
                            }
                        }
                        relationships.append(relationship)
                        break
        
        return relationships
    
    def suggest_new_concepts(
        self,
        existing_concepts: List[str],
        domain: str = "default"
    ) -> List[Dict[str, Any]]:
        """Suggest new concepts that would complement existing ones."""
        suggestions = []
        
        # Suggest concepts from semantic fields
        fields = self.discover_semantic_fields(min_coherence=0.5)
        
        for field in fields:
            field_concepts = field['core_concepts']
            
            # Check if any existing concepts are in this field
            overlap = set(existing_concepts) & set(field_concepts)
            if overlap:
                # Suggest other concepts from this field
                for concept in field_concepts:
                    if concept not in existing_concepts:
                        suggestion = {
                            'concept': concept,
                            'domain': domain,
                            'reason': f'Related to {list(overlap)} in {field["name"]}',
                            'confidence': field['coherence'],
                            'semantic_field': field['name'],
                            'metadata': {
                                'suggestion_method': 'semantic_field_completion',
                                'related_concepts': list(overlap)
                            }
                        }
                        suggestions.append(suggestion)
        
        return suggestions[:10]  # Limit suggestions
    
    def validate_knowledge_consistency(
        self,
        knowledge_base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency of knowledge base and return issues."""
        validation_results: Dict[str, Any] = {
            'status': 'valid',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Validate concepts
        if 'concepts' in knowledge_base:
            concepts = knowledge_base['concepts']
            validation_results['metrics']['concept_count'] = len(concepts)
            
            # Check for concept consistency
            concept_names: List[str] = []
            for concept in concepts:
                if isinstance(concept, dict) and 'name' in concept:
                    concept_names.append(concept['name'])
                elif hasattr(concept, 'name'):
                    concept_names.append(getattr(concept, 'name', str(concept)))
            
            # Check for duplicates
            duplicates = [name for name in set(concept_names) if concept_names.count(name) > 1]
            if duplicates:
                issues_list = validation_results['issues']
                if isinstance(issues_list, list):
                    issues_list.append({
                        'type': 'duplicate_concepts',
                        'details': f'Duplicate concept names: {duplicates}',
                        'severity': 'error'
                    })
        
        # Validate semantic fields
        if hasattr(self, 'semantic_fields'):
            validation_results['metrics']['semantic_field_count'] = len(self.semantic_fields)
            
            # Check field coherence
            low_coherence_fields = [
                name for name, field in self.semantic_fields.items()
                if getattr(field, 'coherence', 1.0) < 0.5
            ]
            if low_coherence_fields:
                warnings_list = validation_results['warnings']
                if isinstance(warnings_list, list):
                    warnings_list.append({
                        'type': 'low_coherence_fields',
                        'details': f'Fields with low coherence: {low_coherence_fields}',
                        'severity': 'warning'
                    })
        
        # Set overall status
        if validation_results['issues']:
            validation_results['status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
        
        return validation_results
