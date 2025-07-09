"""
Frame and Cluster Registry: Advanced Concept Management

This module implements registries for managing semantic frames and concept clusters,
extending the basic concept registry with structured semantic representations.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Set, Any, cast
from dataclasses import dataclass
import logging
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import asdict
import uuid

from .concept_registry import ConceptRegistry, SynsetInfo
from .abstractions import Concept
from .protocols import FrameRegistryProtocol, ClusterRegistryProtocol
from .frame_cluster_abstractions import (
    SemanticFrame, FrameElement, FrameInstance, ConceptCluster,
    FrameAwareConcept, FrameRelation, AnalogicalMapping,
    FrameElementType, FrameRelationType
)


class FrameRegistry(FrameRegistryProtocol[SemanticFrame, str]):
    """
    Registry for managing semantic frames and their relationships.
    
    Provides storage, retrieval, and reasoning capabilities for FrameNet-style
    semantic frames, enabling frame-aware analogical reasoning and concept
    organization.
    """
    
    def __init__(self) -> None:
        """Initialize frame registry with storage structures."""
        self.frames: Dict[str, SemanticFrame] = {}
        self.frame_instances: Dict[str, FrameInstance] = {}
        self.frame_relations: List[FrameRelation] = []
        
        # Hierarchical organization
        self.frame_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        self.frame_parents: Dict[str, str] = {}          # child -> parent
        
        # Performance indexes
        self.lexical_unit_index: Dict[str, List[str]] = {}  # lexical_unit -> frame_names
        self.element_index: Dict[str, List[str]] = {}        # element_name -> frame_names
        
        self.logger = logging.getLogger(__name__)
    
    def register_frame(self, frame: SemanticFrame) -> SemanticFrame:
        """Register a semantic frame in the registry."""
        self.frames[frame.name] = frame
        
        # Update lexical unit index
        for lexical_unit in frame.lexical_units:
            if lexical_unit not in self.lexical_unit_index:
                self.lexical_unit_index[lexical_unit] = []
            self.lexical_unit_index[lexical_unit].append(frame.name)
        
        # Update element index
        for element in frame.get_all_elements():
            if element.name not in self.element_index:
                self.element_index[element.name] = []
            self.element_index[element.name].append(frame.name)
        
        # Update hierarchy
        if frame.inherits_from:
            self.frame_parents[frame.name] = frame.inherits_from
            if frame.inherits_from not in self.frame_hierarchy:
                self.frame_hierarchy[frame.inherits_from] = []
            self.frame_hierarchy[frame.inherits_from].append(frame.name)
        
        self.logger.info(f"Registered frame: {frame.name}")
        return frame
    
    def get_frame(self, frame_name: str) -> Optional[SemanticFrame]:
        """Retrieve a frame by name."""
        return self.frames.get(frame_name)
    
    def get_frames_by_lexical_unit(self, lexical_unit: str) -> List[SemanticFrame]:
        """Find frames that can be evoked by a specific word."""
        frame_names = self.lexical_unit_index.get(lexical_unit, [])
        return [self.frames[name] for name in frame_names if name in self.frames]
    
    def get_frames_with_element(self, element_name: str) -> List[SemanticFrame]:
        """Find frames that contain a specific element."""
        frame_names = self.element_index.get(element_name, [])
        return [self.frames[name] for name in frame_names if name in self.frames]
    
    def create_frame(
        self,
        name: str,
        definition: str,
        core_elements: List[str],
        peripheral_elements: Optional[List[str]] = None
    ) -> SemanticFrame:
        """Create a new semantic frame and return it."""
        
        core = [FrameElement(name=el, description="", element_type=FrameElementType.CORE) for el in core_elements]
        peripheral = [FrameElement(name=el, description="", element_type=FrameElementType.PERIPHERAL) for el in (peripheral_elements or [])]
        
        frame = SemanticFrame(
            name=name,
            definition=definition,
            core_elements=core,
            peripheral_elements=peripheral
        )
        
        return self.register_frame(frame)

    def find_frames_for_concept(
        self, 
        concept: str
    ) -> List[Dict[str, Any]]:
        """Find all frames that can be evoked by a concept (lexical unit)."""
        frames = self.get_frames_by_lexical_unit(concept)
        return [asdict(frame) for frame in frames]

    def create_frame_instance(
        self,
        frame_name: str,
        instance_id: str,
        concept_bindings: Dict[str, FrameAwareConcept],
        context: str = "default"
    ) -> FrameInstance:
        """Create a specific instance of a frame with concept bindings."""
        if frame_name not in self.frames:
            raise ValueError(f"Unknown frame: {frame_name}")
        
        # FrameAwareConcept inherits from Concept, so this is safe
        from typing import cast
        concept_bindings_cast = cast(Dict[str, Concept], concept_bindings)
        
        instance = FrameInstance(
            frame_name=frame_name,
            instance_id=instance_id,
            element_bindings=concept_bindings_cast,
            context=context
        )
        
        self.frame_instances[instance_id] = instance
        
        # Update concept frame roles
        for element_name, concept in concept_bindings.items():
            concept.add_frame_role(frame_name, element_name)
            if instance_id not in concept.frame_instances:
                concept.frame_instances.append(instance_id)
        
        return instance
    
    def find_analogous_instances(self, source_instance_id: str,
                               target_frame: Optional[str] = None,
                               threshold: float = 0.7) -> List[AnalogicalMapping]:
        """Find frame instances analogous to the source instance."""
        source_instance = self.frame_instances.get(source_instance_id)
        if not source_instance:
            return []
        
        analogies = []
        
        # Search through all instances or instances of target frame
        candidates = list(self.frame_instances.values())
        if target_frame:
            candidates = [inst for inst in candidates if inst.frame_name == target_frame]
        
        for candidate in candidates:
            if candidate.instance_id == source_instance_id:
                continue
            
            mapping = self._compute_analogical_mapping(source_instance, candidate)
            if mapping.overall_quality >= threshold:
                analogies.append(mapping)
        
        # Sort by quality
        analogies.sort(key=lambda x: x.overall_quality, reverse=True)
        return analogies
    
    def _compute_analogical_mapping(self, source: FrameInstance, 
                                  target: FrameInstance) -> AnalogicalMapping:
        """Compute analogical mapping between two frame instances."""
        mapping = AnalogicalMapping(
            source_instance=source.instance_id,
            target_instance=target.instance_id
        )
        
        source_frame = self.get_frame(source.frame_name)
        target_frame = self.get_frame(target.frame_name)
        
        if not source_frame or not target_frame:
            return mapping
        
        # Compute structural similarity
        mapping.structural_similarity = self._compute_structural_similarity(
            source_frame, target_frame
        )
        
        # Compute semantic similarity of bound concepts
        mapping.semantic_similarity = self._compute_semantic_similarity(
            source.element_bindings, target.element_bindings
        )
        
        # Compute overall quality
        mapping.compute_quality_score()
        
        return mapping
    
    def _compute_structural_similarity(self, frame1: SemanticFrame, 
                                     frame2: SemanticFrame) -> float:
        """Compute structural similarity between frames."""
        if frame1.name == frame2.name:
            return 1.0
        
        # Check inheritance relationships
        if self._are_frames_related(frame1.name, frame2.name):
            return 0.8
        
        # Compare frame elements
        elements1 = {elem.name for elem in frame1.get_all_elements()}
        elements2 = {elem.name for elem in frame2.get_all_elements()}
        
        if not elements1 or not elements2:
            return 0.0
        
        intersection = len(elements1.intersection(elements2))
        union = len(elements1.union(elements2))
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_semantic_similarity(self, bindings1: Dict[str, Any], 
                                   bindings2: Dict[str, Any]) -> float:
        """Compute semantic similarity between concept bindings."""
        if not bindings1 or not bindings2:
            return 0.0
        
        # This is a simplified version - in practice, would use embeddings
        common_elements = set(bindings1.keys()).intersection(set(bindings2.keys()))
        if not common_elements:
            return 0.0
        
        similarities = []
        for element in common_elements:
            concept1 = bindings1[element]
            concept2 = bindings2[element]
            
            # Simple name-based similarity (would use embeddings in practice)
            if concept1.name == concept2.name:
                similarities.append(1.0)
            elif concept1.synset_id and concept2.synset_id:
                # Could compute synset similarity here
                similarities.append(0.5)
            else:
                similarities.append(0.1)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _are_frames_related(self, frame1_name: str, frame2_name: str) -> bool:
        """Check if two frames are related through inheritance."""
        # Check direct inheritance
        if self.frame_parents.get(frame1_name) == frame2_name:
            return True
        if self.frame_parents.get(frame2_name) == frame1_name:
            return True
        
        # Check if they share a common parent
        parent1 = self.frame_parents.get(frame1_name)
        parent2 = self.frame_parents.get(frame2_name)
        if parent1 and parent1 == parent2:
            return True
        
        return False


class ClusterRegistry(ClusterRegistryProtocol[Concept, int]):
    """
    Registry for managing concept clusters and embeddings.
    
    Provides clustering capabilities for concepts based on embeddings,
    enabling cluster-based similarity and analogical reasoning.
    """
    
    def __init__(self, embedding_dim: int = 300, n_clusters: int = 50):
        """Initialize cluster registry."""
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        
        self.clusters: Dict[int, ConceptCluster] = {}
        self.concept_embeddings: Dict[str, NDArray[np.float32]] = {}
        self.cluster_centroids: Optional[NDArray[np.float32]] = None
        
        # Clustering model
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.is_trained_flag = False
        
        self.logger = logging.getLogger(__name__)
    
    def add_concept_embedding(self, concept_id: str, embedding: NDArray[np.float32]) -> None:
        """Add or update a concept's embedding."""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
        
        self.concept_embeddings[concept_id] = embedding
        self.is_trained_flag = False  # Need to retrain clustering
    
    def train_clusters(self) -> None:
        """Train clustering model on current embeddings."""
        if not self.concept_embeddings:
            self.logger.warning("No embeddings available for clustering")
            return
        
        # Prepare data
        concept_ids = list(self.concept_embeddings.keys())
        embeddings = np.stack([self.concept_embeddings[cid] for cid in concept_ids])
        
        # Train clustering model
        self.clustering_model.fit(embeddings)
        self.cluster_centroids = self.clustering_model.cluster_centers_
        
        # Create cluster objects and assign memberships
        self._create_clusters_from_model(concept_ids, embeddings)
        
        self.is_trained_flag = True
        self.logger.info(f"Trained clustering with {len(concept_ids)} concepts into {self.n_clusters} clusters")
    
    def _create_clusters_from_model(self, concept_ids: List[str], embeddings: NDArray[np.float32]) -> None:
        """Create cluster objects from trained model."""
        # Get hard cluster assignments
        hard_assignments = self.clustering_model.labels_
        
        # Compute soft membership scores
        distances = self.clustering_model.transform(embeddings)
        soft_assignments = self._distances_to_soft_memberships(distances)
        
        # Create cluster objects
        self.clusters.clear()
        for cluster_id in range(self.n_clusters):
            cluster = ConceptCluster(cluster_id=cluster_id)
            if self.cluster_centroids is not None:
                cluster.centroid = self.cluster_centroids[cluster_id].copy()
            
            # Add concept memberships
            for i, concept_id in enumerate(concept_ids):
                membership = soft_assignments[i, cluster_id]
                if membership > 0.1:  # Only store significant memberships
                    cluster.add_member(concept_id, membership)
            
            # Compute cluster statistics
            cluster.coherence_score = self._compute_cluster_coherence(cluster_id, embeddings, hard_assignments)
            
            self.clusters[cluster_id] = cluster
    
    def _distances_to_soft_memberships(self, distances: NDArray[np.float32], temperature: float = 1.0) -> NDArray[np.float32]:
        """Convert distances to soft membership probabilities."""
        # Convert distances to similarities
        similarities = np.exp(-distances / temperature)
        
        # Normalize to get soft assignments
        soft_assignments = similarities / np.sum(similarities, axis=1, keepdims=True)
        
        return np.array(soft_assignments, dtype=np.float32)
    
    def _compute_cluster_coherence(self, cluster_id: int, embeddings: NDArray[np.float32], 
                                 assignments: NDArray[np.int32]) -> float:
        """Compute coherence score for a cluster."""
        cluster_members = embeddings[assignments == cluster_id]
        if len(cluster_members) < 2:
            return 0.0
        
        # Compute average pairwise similarity within cluster
        similarities = cosine_similarity(cluster_members)
        # Exclude diagonal (self-similarities)
        mask = np.ones_like(similarities) - np.eye(len(similarities))
        
        return np.sum(similarities * mask) / np.sum(mask) if np.sum(mask) > 0 else 0.0
    
    def get_concept_cluster_memberships(self, concept_id: str) -> Dict[int, float]:
        """Get cluster memberships for a concept."""
        if concept_id not in self.concept_embeddings:
            return {}
        
        if not self.is_trained_flag:
            return {}
        
        embedding = self.concept_embeddings[concept_id].reshape(1, -1)
        distances = self.clustering_model.transform(embedding)[0]
        soft_memberships = self._distances_to_soft_memberships(distances.reshape(1, -1))[0]
        
        return {i: membership for i, membership in enumerate(soft_memberships) if membership > 0.1}
    
    def find_similar_concepts(self, concept_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find concepts similar to the given concept based on cluster membership."""
        concept_memberships = self.get_concept_cluster_memberships(concept_id)
        if not concept_memberships:
            return []
        
        similar_concepts = []
        
        for other_concept_id in self.concept_embeddings:
            if other_concept_id == concept_id:
                continue
            
            other_memberships = self.get_concept_cluster_memberships(other_concept_id)
            
            # Compute membership overlap
            similarity = self._compute_membership_similarity(concept_memberships, other_memberships)
            
            if similarity >= threshold:
                similar_concepts.append((other_concept_id, similarity))
        
        # Sort by similarity
        similar_concepts.sort(key=lambda x: x[1], reverse=True)
        return similar_concepts
    
    def _compute_membership_similarity(self, memberships1: Dict[int, float], 
                                     memberships2: Dict[int, float]) -> float:
        """Compute similarity between two cluster membership distributions."""
        all_clusters = set(memberships1.keys()).union(set(memberships2.keys()))
        
        if not all_clusters:
            return 0.0
        
        # Convert to vectors
        vec1 = np.array([memberships1.get(cluster_id, 0.0) for cluster_id in all_clusters])
        vec2 = np.array([memberships2.get(cluster_id, 0.0) for cluster_id in all_clusters])
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # Protocol Compliance Implementations
    def update_clusters(
        self,
        concepts: Optional[List[str]] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update concept clusters and return clustering metadata."""
        if n_clusters:
            self.n_clusters = n_clusters
            self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42)

        self.train_clusters()
        return {
            "n_clusters": self.n_clusters,
            "n_concepts": len(self.concept_embeddings),
            "is_trained": self.is_trained_flag,
            "coherence_scores": {cid: c.coherence_score for cid, c in self.clusters.items()}
        }

    def get_cluster_membership(
        self, 
        concept: str
    ) -> Optional[Dict[str, float]]:
        """Get cluster membership probabilities for a concept."""
        memberships = self.get_concept_cluster_memberships(concept)
        if not memberships:
            return None
        # Convert cluster IDs (int) to string keys to match protocol
        return {str(cid): score for cid, score in memberships.items()}

    def find_cluster_neighbors(
        self,
        concept: str,
        max_neighbors: int = 10
    ) -> List[Tuple[str, float]]:
        """Find nearest neighbors within the same primary cluster."""
        memberships = self.get_concept_cluster_memberships(concept)
        if not memberships:
            return []

        primary_cluster = max(memberships, key=lambda k: memberships[k])
        
        cluster_members = self.clusters.get(primary_cluster)
        if not cluster_members:
            return []

        neighbors = []
        for member_id, strength in cluster_members.members.items():
            if member_id != concept:
                neighbors.append((member_id, strength))
        
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:max_neighbors]

    @property
    def is_trained(self) -> bool:
        """Return whether clustering model has been trained."""
        return self.is_trained_flag
    
    @property
    def cluster_count(self) -> int:
        """Return number of clusters."""
        return self.n_clusters
