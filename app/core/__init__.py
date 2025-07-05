"""Core abstractions for the soft logic microservice."""

from .abstractions import (
    Concept,
    Axiom,
    Context,
    FormulaNode,
    AxiomType,
    AxiomClassification,
    OperationType
)

from .parsers import AxiomParser, AxiomParseError
from .concept_registry import ConceptRegistry, SynsetInfo

# Frame and clustering extensions
from .frame_cluster_abstractions import (
    SemanticFrame, FrameElement, FrameInstance, ConceptCluster,
    FrameAwareConcept, FrameRelation, AnalogicalMapping,
    FrameElementType, FrameRelationType
)

from .frame_cluster_registry import FrameRegistry, ClusterRegistry
from .hybrid_registry import HybridConceptRegistry

# Enhanced semantic reasoning
from .enhanced_semantic_reasoning import (
    EnhancedHybridRegistry, CrossDomainAnalogy, SemanticField
)

# Vector embeddings
from .vector_embeddings import (
    VectorEmbeddingManager, EmbeddingProvider, SemanticEmbeddingProvider,
    RandomEmbeddingProvider, EmbeddingMetadata
)

__all__ = [
    "Concept",
    "Axiom", 
    "Context",
    "FormulaNode",
    "AxiomType",
    "AxiomClassification",
    "OperationType",
    "AxiomParser",
    "AxiomParseError",
    "ConceptRegistry",
    "SynsetInfo",
    # Frame and cluster extensions
    "SemanticFrame", "FrameElement", "FrameInstance", "ConceptCluster",
    "FrameAwareConcept", "FrameRelation", "AnalogicalMapping",
    "FrameElementType", "FrameRelationType",
    "FrameRegistry", "ClusterRegistry", "HybridConceptRegistry",
    # Enhanced semantic reasoning
    "EnhancedHybridRegistry", "CrossDomainAnalogy", "SemanticField",
    # Vector embeddings
    "VectorEmbeddingManager", "EmbeddingProvider", "SemanticEmbeddingProvider",
    "RandomEmbeddingProvider", "EmbeddingMetadata"
]
