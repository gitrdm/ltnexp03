"""
Extended abstractions for FrameNet and Clustering integration.

This module extends the core abstractions to support semantic frames
and clustering-based concept representations for advanced analogical reasoning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from numpy.typing import NDArray
from enum import Enum
import numpy as np
from .abstractions import Concept, Context


class FrameElementType(Enum):
    """Types of frame elements in FrameNet."""
    CORE = "core"           # Essential semantic roles
    PERIPHERAL = "peripheral"  # Optional additional information
    EXTRATHEMATIC = "extrathematic"  # Context-dependent elements


@dataclass
class FrameElement:
    """Represents a semantic role within a frame."""
    name: str
    description: str
    element_type: FrameElementType
    semantic_constraints: List[str] = field(default_factory=list)
    typical_fillers: List[str] = field(default_factory=list)  # Common concept types


@dataclass
class SemanticFrame:
    """
    FrameNet semantic frame with roles and inheritance.
    
    Represents a conceptual structure that describes a particular type
    of event, relation, or state, along with the participants and props
    involved in it.
    """
    name: str
    definition: str
    core_elements: List[FrameElement] = field(default_factory=list)
    peripheral_elements: List[FrameElement] = field(default_factory=list)
    
    # Frame relationships
    inherits_from: Optional[str] = None
    subframes: List[str] = field(default_factory=list)
    
    # Lexical units that evoke this frame
    lexical_units: List[str] = field(default_factory=list)
    
    # Frame-to-frame relations
    uses_frames: List[str] = field(default_factory=list)
    is_inherited_by: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_elements(self) -> List[FrameElement]:
        """Get all frame elements (core + peripheral)."""
        return self.core_elements + self.peripheral_elements
    
    def get_element_by_name(self, name: str) -> Optional[FrameElement]:
        """Find a frame element by name."""
        for element in self.get_all_elements():
            if element.name.lower() == name.lower():
                return element
        return None


@dataclass
class FrameInstance:
    """
    A specific instantiation of a semantic frame.
    
    Represents a particular event or situation that exemplifies
    the abstract frame structure with concrete concepts filling
    the frame element roles.
    """
    frame_name: str
    instance_id: str
    element_bindings: Dict[str, Concept] = field(default_factory=dict)  # element_name -> concept
    confidence: float = 1.0
    context: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_bound_concept(self, element_name: str) -> Optional[Concept]:
        """Get the concept bound to a specific frame element."""
        return self.element_bindings.get(element_name)
    
    def bind_concept(self, element_name: str, concept: Concept) -> None:
        """Bind a concept to a frame element."""
        self.element_bindings[element_name] = concept


@dataclass
class ConceptCluster:
    """
    Represents a learned cluster of semantically related concepts.
    
    Clusters provide an alternative to hand-crafted taxonomies,
    learning semantic groupings from data through embedding similarities.
    """
    cluster_id: int
    name: Optional[str] = None  # Human-readable cluster name
    description: Optional[str] = None
    
    # Cluster centroid in embedding space
    centroid: Optional[NDArray[np.float32]] = None
    
    # Concept membership (concept_id -> membership_strength)
    members: Dict[str, float] = field(default_factory=dict)
    
    # Cluster statistics
    coherence_score: float = 0.0  # How tightly clustered
    distinctiveness: float = 0.0  # How separated from other clusters
    
    # Relationships to other clusters
    related_clusters: List[Tuple[int, float]] = field(default_factory=list)  # (cluster_id, similarity)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_member(self, concept_id: str, membership_strength: float) -> None:
        """Add a concept to this cluster with given membership strength."""
        self.members[concept_id] = membership_strength
    
    def get_core_members(self, threshold: float = 0.7) -> List[str]:
        """Get concepts with high membership in this cluster."""
        return [concept_id for concept_id, strength in self.members.items() 
                if strength >= threshold]


@dataclass
class FrameAwareConcept(Concept):
    """
    Extended concept with frame and cluster information.
    
    Combines the basic concept representation with additional
    semantic structure from frames and learned cluster memberships.
    """
    # Frame-related metadata
    frame_roles: Dict[str, str] = field(default_factory=dict)  # frame_name -> role_name
    frame_instances: List[str] = field(default_factory=list)  # instance_ids where this concept appears
    
    # Cluster-related metadata  
    cluster_memberships: Dict[int, float] = field(default_factory=dict)  # cluster_id -> membership_strength
    primary_cluster: Optional[int] = None  # Most likely cluster
    
    # Embedding representation
    embedding: Optional[NDArray[np.float32]] = None
    
    def add_frame_role(self, frame_name: str, role_name: str) -> None:
        """Record that this concept can fill a specific role in a frame."""
        self.frame_roles[frame_name] = role_name
    
    def get_frames(self) -> List[str]:
        """Get all frames this concept participates in."""
        return list(self.frame_roles.keys())
    
    def get_role_in_frame(self, frame_name: str) -> Optional[str]:
        """Get the role this concept plays in a specific frame."""
        return self.frame_roles.get(frame_name)
    
    def get_primary_clusters(self, top_k: int = 3) -> List[Tuple[int, float]]:
        """Get the top-k clusters this concept belongs to."""
        sorted_clusters = sorted(self.cluster_memberships.items(), 
                               key=lambda x: x[1], reverse=True)
        return sorted_clusters[:top_k]


class FrameRelationType(Enum):
    """Types of relationships between frames."""
    INHERITANCE = "inheritance"        # Frame A inherits from Frame B
    USING = "using"                   # Frame A uses Frame B
    SUBFRAME = "subframe"             # Frame A is a subframe of Frame B
    PERSPECTIVE = "perspective"        # Different perspectives on same event
    PRECEDES = "precedes"             # Frame A typically precedes Frame B
    INCHOATIVE = "inchoative"         # Frame A is the beginning of Frame B
    CAUSATIVE = "causative"           # Frame A causes Frame B


@dataclass
class FrameRelation:
    """Represents a relationship between two frames."""
    source_frame: str
    target_frame: str
    relation_type: FrameRelationType
    description: str = ""
    confidence: float = 1.0
    
    # Element mappings between frames
    element_mappings: Dict[str, str] = field(default_factory=dict)  # source_element -> target_element


@dataclass
class AnalogicalMapping:
    """
    Represents an analogical mapping between frame instances.
    
    Captures the structural alignment between two situations
    based on frame role correspondences and concept similarities.
    """
    source_instance: str
    target_instance: str
    
    # Role correspondences (source_role -> target_role)
    role_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Concept correspondences (source_concept -> target_concept)
    concept_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Quality metrics
    structural_similarity: float = 0.0  # Frame structure alignment
    semantic_similarity: float = 0.0    # Concept similarity
    overall_quality: float = 0.0        # Combined quality score
    
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_role_mapping(self, source_role: str, target_role: str) -> None:
        """Add a role correspondence to the analogy."""
        self.role_mappings[source_role] = target_role
    
    def add_concept_mapping(self, source_concept: str, target_concept: str) -> None:
        """Add a concept correspondence to the analogy."""
        self.concept_mappings[source_concept] = target_concept
    
    def compute_quality_score(self) -> float:
        """Compute overall analogy quality from components."""
        self.overall_quality = (0.6 * self.structural_similarity + 
                               0.4 * self.semantic_similarity)
        return self.overall_quality
