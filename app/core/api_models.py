"""
TypedDict API Models for Type-Safe Request/Response Handling
===========================================================

This module defines TypedDict specifications for all API request and response
models in our soft logic system. These provide compile-time type checking and
runtime validation for API interfaces.

DESIGN PRINCIPLES:
==================

1. **Type Safety**: All API data structures have explicit type annotations
2. **Validation**: Required vs. optional fields are clearly specified
3. **Documentation**: Each model includes field descriptions and examples
4. **Extensibility**: Models support additional fields where appropriate
5. **Consistency**: Consistent naming and structure across all endpoints

API MODEL CATEGORIES:
=====================

- Concept Management: Create, retrieve, search concepts
- Semantic Reasoning: Analogies, similarity, discovery
- Frame Operations: Create, bind, query semantic frames
- Cluster Operations: Update, query, analyze clusters
- System Status: Health, metrics, configuration
"""

from typing_extensions import TypedDict, NotRequired
from typing import List, Dict, Optional, Any, Union
import numpy as np


# ============================================================================
# CONCEPT MANAGEMENT MODELS
# ============================================================================

class ConceptCreateRequest(TypedDict):
    """Request model for creating a new concept."""
    name: str
    context: str
    synset_id: Optional[str]
    disambiguation: Optional[str]
    metadata: Optional[Dict[str, Any]]
    auto_disambiguate: bool


class ConceptCreateResponse(TypedDict):
    """Response model for concept creation."""
    concept_id: str
    name: str
    synset_id: Optional[str]
    disambiguation: Optional[str]
    context: str
    created_at: str
    metadata: Dict[str, Any]


class ConceptSearchRequest(TypedDict):
    """Request model for concept search operations."""
    query: str
    context: Optional[str]
    similarity_threshold: float
    max_results: int
    include_metadata: bool


class ConceptSearchResponse(TypedDict):
    """Response model for concept search results."""
    concepts: List[Dict[str, Any]]
    total_results: int
    search_metadata: Dict[str, Any]


class ConceptSimilarityRequest(TypedDict):
    """Request model for concept similarity computation."""
    concept1: str
    concept2: str
    context1: Optional[str]
    context2: Optional[str]
    similarity_method: str  # "embedding", "wordnet", "hybrid"


class ConceptSimilarityResponse(TypedDict):
    """Response model for concept similarity results."""
    similarity_score: float
    method_used: str
    confidence: float
    explanation: Optional[str]


# ============================================================================
# SEMANTIC REASONING MODELS
# ============================================================================

class AnalogyRequest(TypedDict):
    """Request model for analogy completion."""
    partial_analogy: Dict[str, str]  # "A:B :: C:?" format
    context: Optional[str]
    max_completions: int
    min_confidence: float


class AnalogyResponse(TypedDict):
    """Response model for analogy completion results."""
    completions: List[Dict[str, Any]]
    reasoning_trace: List[str]
    metadata: Dict[str, Any]


class SemanticFieldDiscoveryRequest(TypedDict):
    """Request model for semantic field discovery."""
    domain: Optional[str]
    min_coherence: float
    max_fields: int
    clustering_method: str


class SemanticFieldDiscoveryResponse(TypedDict):
    """Response model for discovered semantic fields."""
    semantic_fields: List[Dict[str, Any]]
    discovery_metadata: Dict[str, Any]
    coherence_scores: Dict[str, float]


class CrossDomainAnalogiesRequest(TypedDict):
    """Request model for cross-domain analogy discovery."""
    source_domain: str
    target_domain: str
    min_quality: float
    max_analogies: int


class CrossDomainAnalogiesResponse(TypedDict):
    """Response model for cross-domain analogies."""
    analogies: List[Dict[str, Any]]
    quality_scores: Dict[str, float]
    domain_analysis: Dict[str, Any]


# ============================================================================
# FRAME OPERATIONS MODELS
# ============================================================================

class FrameCreateRequest(TypedDict):
    """Request model for creating semantic frames."""
    name: str
    definition: str
    core_elements: List[str]
    peripheral_elements: NotRequired[List[str]]
    lexical_units: NotRequired[List[str]]
    metadata: NotRequired[Dict[str, Any]]


class FrameCreateResponse(TypedDict):
    """Response model for frame creation."""
    frame_id: str
    name: str
    definition: str
    elements: Dict[str, str]
    created_at: str


class FrameInstanceRequest(TypedDict):
    """Request model for creating frame instances."""
    instance_id: str
    concept_bindings: Dict[str, str]
    context: str
    confidence: NotRequired[float]


class FrameInstanceResponse(TypedDict):
    """Response model for frame instance creation."""
    instance_id: str
    frame_id: str
    bindings: Dict[str, Any]
    validation_results: Dict[str, Any]
    created_at: str


class FrameQueryRequest(TypedDict):
    """Request model for frame queries."""
    concept: NotRequired[str]
    frame_pattern: NotRequired[Dict[str, str]]
    context: NotRequired[str]
    max_results: NotRequired[int]


class FrameQueryResponse(TypedDict):
    """Response model for frame query results."""
    frames: List[Dict[str, Any]]
    instances: List[Dict[str, Any]]
    query_metadata: Dict[str, Any]


# ============================================================================
# CLUSTER OPERATIONS MODELS
# ============================================================================

class ClusterUpdateRequest(TypedDict):
    """Request model for cluster updates."""
    concepts: Optional[List[str]]
    n_clusters: Optional[int]
    clustering_algorithm: str
    force_retrain: bool


class ClusterUpdateResponse(TypedDict):
    """Response model for cluster update results."""
    clusters_updated: int
    concepts_clustered: int
    clustering_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]


class ClusterQueryRequest(TypedDict):
    """Request model for cluster queries."""
    concept: str
    context: Optional[str]
    include_neighbors: bool
    max_neighbors: int


class ClusterQueryResponse(TypedDict):
    """Response model for cluster query results."""
    cluster_id: int
    membership_probability: float
    neighbors: Optional[List[Dict[str, Any]]]
    cluster_metadata: Dict[str, Any]


class ClusterAnalysisRequest(TypedDict):
    """Request model for cluster analysis."""
    cluster_ids: Optional[List[int]]
    analysis_type: str  # "coherence", "separation", "stability"
    include_visualization: bool


class ClusterAnalysisResponse(TypedDict):
    """Response model for cluster analysis results."""
    analysis_results: Dict[str, Any]
    metrics: Dict[str, float]
    visualization_data: Optional[Dict[str, Any]]


# ============================================================================
# SYSTEM STATUS AND CONFIGURATION MODELS
# ============================================================================

class SystemHealthRequest(TypedDict, total=False):
    """Request model for system health checks."""
    include_detailed_metrics: bool
    check_external_dependencies: bool


class SystemHealthResponse(TypedDict):
    """Response model for system health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: str
    version: str
    components: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Union[int, float, str]]


class ConfigurationRequest(TypedDict, total=False):
    """Request model for configuration updates."""
    embedding_config: Optional[Dict[str, Any]]
    clustering_config: Optional[Dict[str, Any]]
    reasoning_config: Optional[Dict[str, Any]]
    api_config: Optional[Dict[str, Any]]


class ConfigurationResponse(TypedDict):
    """Response model for configuration operations."""
    current_config: Dict[str, Any]
    updated_fields: List[str]
    validation_results: Dict[str, Any]
    restart_required: bool


# ============================================================================
# BATCH OPERATIONS MODELS
# ============================================================================

class BatchConceptRequest(TypedDict):
    """Request model for batch concept operations."""
    operation: str  # "create", "update", "delete"
    concepts: List[Dict[str, Any]]
    context: str
    validation_mode: str  # "strict", "lenient", "skip"


class BatchConceptResponse(TypedDict):
    """Response model for batch concept operations."""
    successful_operations: int
    failed_operations: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]]
    execution_time: float


class BatchEmbeddingRequest(TypedDict):
    """Request model for batch embedding generation."""
    concepts: List[str]
    context: str
    embedding_provider: str
    cache_results: bool


class BatchEmbeddingResponse(TypedDict):
    """Response model for batch embedding results."""
    embeddings: Dict[str, List[float]]
    generation_metadata: Dict[str, Any]
    cache_hits: int
    generation_time: float


# ============================================================================
# BATCH OPERATION MODELS
# ============================================================================

class AnalogiesBatch(TypedDict):
    """Request model for batch analogy creation."""
    analogies: List[Dict[str, Any]]
    workflow_id: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]


class BatchWorkflowResponse(TypedDict):
    """Response model for batch workflow operations."""
    workflow_id: str
    workflow_type: str
    status: str
    created_at: str
    updated_at: str
    items_total: int
    items_processed: int
    error_count: int
    metadata: Dict[str, Any]
    error_log: List[str]


class DeleteCriteriaRequest(TypedDict):
    """Request model for batch deletion criteria."""
    domains: Optional[List[str]]
    frame_types: Optional[List[str]]
    quality_threshold: Optional[float]
    created_before: Optional[str]  # ISO datetime string
    created_after: Optional[str]   # ISO datetime string
    tags: Optional[List[str]]


class CompactionResult(TypedDict):
    """Response model for compaction operations."""
    status: str
    records_removed: int
    active_records: int
    backup_created: Optional[str]


class WorkflowListResponse(TypedDict):
    """Response model for workflow listing."""
    workflows: List[BatchWorkflowResponse]
    total_count: int
    filtered_by_status: Optional[str]


# ============================================================================
# PERSISTENCE API MODELS
# ============================================================================

class ExportRequest(TypedDict):
    """Request model for knowledge base export."""
    context_name: str
    format: str  # "json", "compressed", "sqlite"
    include_embeddings: bool
    include_models: bool


class ExportResponse(TypedDict):
    """Response model for export operations."""
    export_id: str
    file_path: str
    file_size: int
    format: str
    created_at: str
    components_exported: List[str]


class ImportRequest(TypedDict):
    """Request model for knowledge base import."""
    merge_strategy: str  # "overwrite", "merge", "skip_conflicts"
    target_context: Optional[str]
    import_embeddings: bool
    import_models: bool


class ImportResponse(TypedDict):
    """Response model for import operations."""
    import_id: str
    status: str
    items_imported: int
    conflicts_found: int
    merge_strategy_used: str
    imported_at: str


# ============================================================================
# ERROR AND VALIDATION MODELS
# ============================================================================

class ValidationError(TypedDict):
    """Standard validation error structure."""
    field: str
    message: str
    code: str
    details: Optional[Dict[str, Any]]


class ApiErrorResponse(TypedDict):
    """Standard API error response."""
    error: str
    message: str
    status_code: int
    timestamp: str
    request_id: str
    validation_errors: Optional[List[ValidationError]]


class SuccessResponse(TypedDict, total=False):
    """Standard success response wrapper."""
    success: bool
    message: Optional[str]
    data: Optional[Any]
    metadata: Optional[Dict[str, Any]]


# ============================================================================
# UTILITY TYPE ALIASES
# ============================================================================

# Common response patterns
AnyApiResponse = Union[
    ConceptCreateResponse, ConceptSearchResponse, ConceptSimilarityResponse,
    AnalogyResponse, SemanticFieldDiscoveryResponse, CrossDomainAnalogiesResponse,
    FrameCreateResponse, FrameInstanceResponse, FrameQueryResponse,
    ClusterUpdateResponse, ClusterQueryResponse, ClusterAnalysisResponse,
    SystemHealthResponse, ConfigurationResponse,
    BatchConceptResponse, BatchEmbeddingResponse,
    ApiErrorResponse, SuccessResponse
]

# Common request patterns
AnyApiRequest = Union[
    ConceptCreateRequest, ConceptSearchRequest, ConceptSimilarityRequest,
    AnalogyRequest, SemanticFieldDiscoveryRequest, CrossDomainAnalogiesRequest,
    FrameCreateRequest, FrameInstanceRequest, FrameQueryRequest,
    ClusterUpdateRequest, ClusterQueryRequest, ClusterAnalysisRequest,
    SystemHealthRequest, ConfigurationRequest,
    BatchConceptRequest, BatchEmbeddingRequest
]

# Export lists for convenient importing
__all__ = [
    # Concept Management
    'ConceptCreateRequest', 'ConceptCreateResponse',
    'ConceptSearchRequest', 'ConceptSearchResponse',
    'ConceptSimilarityRequest', 'ConceptSimilarityResponse',
    
    # Semantic Reasoning
    'AnalogyRequest', 'AnalogyResponse',
    'SemanticFieldDiscoveryRequest', 'SemanticFieldDiscoveryResponse',
    'CrossDomainAnalogiesRequest', 'CrossDomainAnalogiesResponse',
    
    # Frame Operations
    'FrameCreateRequest', 'FrameCreateResponse',
    'FrameInstanceRequest', 'FrameInstanceResponse',
    'FrameQueryRequest', 'FrameQueryResponse',
    
    # Cluster Operations
    'ClusterUpdateRequest', 'ClusterUpdateResponse',
    'ClusterQueryRequest', 'ClusterQueryResponse',
    'ClusterAnalysisRequest', 'ClusterAnalysisResponse',
    
    # System
    'SystemHealthRequest', 'SystemHealthResponse',
    'ConfigurationRequest', 'ConfigurationResponse',
    
    # Batch Operations
    'BatchConceptRequest', 'BatchConceptResponse',
    'BatchEmbeddingRequest', 'BatchEmbeddingResponse',
    
    # Error Handling
    'ValidationError', 'ApiErrorResponse', 'SuccessResponse',
    
    # Type Aliases
    'AnyApiResponse', 'AnyApiRequest'
]
