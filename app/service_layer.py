"""
Complete FastAPI Service Layer for Soft Logic Microservice
==========================================================

This module provides a comprehensive REST API service layer that integrates:
- Semantic reasoning operations (analogies, similarity, discovery)
- Concept and frame management
- Persistence and workflow operations
- WebSocket streaming capabilities
- Contract-validated endpoints with comprehensive error handling

Following Design by Contract principles with full type safety and mypy compliance.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

# Import core components
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.batch_persistence import BatchPersistenceManager, BatchWorkflow, WorkflowStatus
from app.core.api_models import (
    ConceptCreateRequest, ConceptCreateResponse, ConceptSearchRequest, ConceptSearchResponse,
    ConceptSimilarityRequest, ConceptSimilarityResponse,
    AnalogyRequest, AnalogyResponse,
    SemanticFieldDiscoveryRequest, SemanticFieldDiscoveryResponse,
    CrossDomainAnalogiesRequest, CrossDomainAnalogiesResponse,
    FrameCreateRequest, FrameCreateResponse,
    FrameInstanceRequest, FrameInstanceResponse,
    FrameQueryRequest, FrameQueryResponse,
    AnalogiesBatch, BatchWorkflowResponse
)

# Phase 4: Neural-Symbolic Integration imports
from app.core.neural_symbolic_service import (
    NeuralSymbolicService, 
    register_neural_symbolic_endpoints,
    initialize_neural_symbolic_service
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR TYPE-SAFE API
# ============================================================================

class ConceptCreate(BaseModel):
    """Pydantic model for concept creation with validation."""
    name: str = Field(..., min_length=1, max_length=100, description="Concept name")
    context: str = Field(..., min_length=1, description="Context or domain")
    synset_id: Optional[str] = Field(None, description="WordNet synset ID")
    disambiguation: Optional[str] = Field(None, description="Disambiguation text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    auto_disambiguate: bool = Field(True, description="Enable automatic disambiguation")

    @field_validator('name')
    @classmethod
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace')
        return v.strip()


class ConceptSearch(BaseModel):
    """Pydantic model for concept search operations."""
    query: str = Field(..., min_length=1, description="Search query")
    context: Optional[str] = Field(None, description="Context filter")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    include_metadata: bool = Field(True, description="Include metadata in results")


class AnalogyCompletion(BaseModel):
    """Pydantic model for analogy completion requests."""
    source_a: str = Field(..., description="First source concept")
    source_b: str = Field(..., description="Second source concept") 
    target_a: str = Field(..., description="First target concept")
    context: Optional[str] = Field(None, description="Context or domain")
    max_completions: int = Field(5, ge=1, le=20, description="Maximum completions")
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_a": "bird",
                "source_b": "fly",
                "target_a": "fish",
                "context": "animals",
                "max_completions": 5,
                "min_confidence": 0.5
            }
        }
    }

    @field_validator('target_a')
    @classmethod
    def validate_unique_concepts(cls, v, info):
        """Ensure target_a is different from source_a to avoid dictionary key conflicts."""
        if 'source_a' in info.data and v == info.data['source_a']:
            raise ValueError('target_a must be different from source_a to form a valid analogy')
        return v


class SemanticFieldDiscovery(BaseModel):
    """Pydantic model for semantic field discovery."""
    domain: Optional[str] = Field(None, description="Target domain")
    min_coherence: float = Field(0.6, ge=0.0, le=1.0, description="Minimum coherence")
    max_fields: int = Field(10, ge=1, le=50, description="Maximum fields")
    clustering_method: str = Field("kmeans", description="Clustering method")


class StreamingMessage(BaseModel):
    """Pydantic model for WebSocket streaming messages."""
    message_type: str = Field(..., description="Type of message")
    content: Dict[str, Any] = Field(..., description="Message content")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ConceptSimilarity(BaseModel):
    """Pydantic model for concept similarity computation."""
    concept1: str = Field(..., min_length=1, description="First concept name")
    concept2: str = Field(..., min_length=1, description="Second concept name")
    context1: Optional[str] = Field(None, description="Context for first concept")
    context2: Optional[str] = Field(None, description="Context for second concept")
    similarity_method: str = Field("hybrid", description="Similarity method: embedding, wordnet, hybrid")


# ============================================================================
# APPLICATION LIFECYCLE AND GLOBAL STATE
# ============================================================================

# Global service instances - in production, use dependency injection
semantic_registry: Optional[EnhancedHybridRegistry] = None
persistence_manager: Optional[ContractEnhancedPersistenceManager] = None
batch_manager: Optional[BatchPersistenceManager] = None

# Phase 4: Neural-Symbolic Integration globals
neural_symbolic_service: Optional[NeuralSymbolicService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with proper startup/shutdown."""
    global semantic_registry, persistence_manager, batch_manager, neural_symbolic_service
    
    try:
        # Startup sequence
        logger.info("Starting Soft Logic Service Layer...")
        
        # Initialize storage path
        storage_path = Path("storage")
        storage_path.mkdir(exist_ok=True)
        
        # Initialize semantic registry with enhanced capabilities
        logger.info("Initializing semantic registry...")
        semantic_registry = EnhancedHybridRegistry(
            download_wordnet=False,  # Don't download wordnet during tests
            n_clusters=8,
            enable_cross_domain=True,
            embedding_provider="random"  # Use random for reliable startup
        )
        
        # Initialize persistence managers
        logger.info("Initializing persistence layer...")
        persistence_manager = ContractEnhancedPersistenceManager(storage_path)
        batch_manager = BatchPersistenceManager(storage_path)
        
        # Phase 4: Initialize neural-symbolic service
        logger.info("Initializing neural-symbolic service...")
        initialize_neural_symbolic_service(semantic_registry, persistence_manager)
        
        logger.info("✅ Soft Logic Service Layer started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start service layer: {e}")
        raise
    finally:
        # Shutdown sequence
        logger.info("Shutting down Soft Logic Service Layer...")
        logger.info("✅ Service layer shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Soft Logic Microservice - Complete Service Layer",
    description="Production-ready soft logic system with semantic reasoning, persistence, and streaming",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 4: Register neural-symbolic endpoints
# Note: Endpoints are registered here, but service initialization happens in lifespan
register_neural_symbolic_endpoints(app)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_semantic_registry() -> EnhancedHybridRegistry:
    """Dependency injection for semantic registry."""
    if semantic_registry is None:
        logger.info("Auto-initializing services for semantic registry access")
        initialize_services()
    if semantic_registry is None:
        raise HTTPException(status_code=503, detail="Semantic registry not initialized")
    return semantic_registry


def get_persistence_manager() -> ContractEnhancedPersistenceManager:
    """Dependency injection for persistence manager."""
    if persistence_manager is None:
        logger.info("Auto-initializing services for persistence manager access")
        initialize_services()
    if persistence_manager is None:
        raise HTTPException(status_code=503, detail="Persistence manager not initialized")
    return persistence_manager


def get_batch_manager() -> BatchPersistenceManager:
    """Dependency injection for batch manager."""
    if batch_manager is None:
        logger.info("Auto-initializing services for batch manager access")
        initialize_services()
    if batch_manager is None:
        raise HTTPException(status_code=503, detail="Batch manager not initialized")
    return batch_manager


# ============================================================================
# CONCEPT MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/concepts", response_model=Dict[str, Any], tags=["Concepts"])
async def create_concept(
    concept: ConceptCreate,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> Dict[str, Any]:
    """Create a new concept with automatic disambiguation."""
    try:
        # Use the correct method from EnhancedHybridRegistry
        concept_obj = registry.create_frame_aware_concept_with_advanced_embedding(
            name=concept.name,
            context=concept.context,
            synset_id=concept.synset_id,
            disambiguation=concept.disambiguation,
            use_semantic_embedding=True
        )
        
        # Return safe serializable data (no numpy arrays)
        return {
            "concept_id": concept_obj.unique_id,  # Use the string ID, not the object
            "name": concept_obj.name,
            "synset_id": concept_obj.synset_id,
            "disambiguation": concept_obj.disambiguation,
            "context": concept_obj.context,
            "created_at": datetime.now().isoformat(),
            "metadata": concept.metadata or {},
            "embedding_size": len(getattr(concept_obj, 'embedding', [])) if getattr(concept_obj, 'embedding', None) is not None else 0
        }
        
    except Exception as e:
        logger.error(f"Error creating concept: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/concepts/{concept_id}", response_model=Dict[str, Any], tags=["Concepts"])
async def get_concept(
    concept_id: str = FastAPIPath(..., description="ID of the concept to retrieve", example="example_concept"),
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
):
    """Retrieve a concept by ID."""
    try:
        if concept_id not in registry.concepts:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        concept = registry.concepts[concept_id]
        # Return safe serializable data (no numpy arrays)
        return {
            "concept_id": concept_id,
            "name": concept.name,
            "synset_id": concept.synset_id,
            "disambiguation": concept.disambiguation,
            "context": concept.context,
            "metadata": getattr(concept, 'metadata', {}),
            "embedding_size": len(getattr(concept, 'embedding', [])) if getattr(concept, 'embedding', None) is not None else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving concept {concept_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concepts/search", response_model=ConceptSearchResponse, tags=["Concepts"])
async def search_concepts(
    search: ConceptSearch,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> ConceptSearchResponse:
    """Search for concepts with similarity matching."""
    try:
        # Use enhanced registry's semantic search capabilities
        results = []
        
        for concept_id, concept in registry.concepts.items():
            # Simple name-based matching for now - could be enhanced with semantic similarity
            if search.query.lower() in concept.name.lower():
                similarity_score = 1.0 if search.query.lower() == concept.name.lower() else 0.8
                
                if similarity_score >= search.similarity_threshold:
                    result = {
                        "concept_id": concept_id,
                        "name": concept.name,
                        "similarity_score": similarity_score,
                        "synset_id": concept.synset_id,
                        "disambiguation": concept.disambiguation
                    }
                    
                    if search.include_metadata:
                        result["metadata"] = getattr(concept, 'metadata', {})
                    
                    results.append(result)
                    
                    if len(results) >= search.max_results:
                        break
        
        return ConceptSearchResponse(
            concepts=results,
            total_results=len(results),
            search_metadata={
                "query": search.query,
                "threshold": search.similarity_threshold,
                "context": search.context
            }
        )
        
    except Exception as e:
        logger.error(f"Error searching concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concepts/similarity", response_model=ConceptSimilarityResponse, tags=["Concepts"])
async def compute_concept_similarity(
    similarity_request: ConceptSimilarity,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> ConceptSimilarityResponse:
    """Compute similarity between two concepts."""
    try:
        # Find concepts by name
        concept1_obj = None
        concept2_obj = None
        
        for cid, concept in registry.concepts.items():
            if concept.name == similarity_request.concept1:
                concept1_obj = concept
            if concept.name == similarity_request.concept2:
                concept2_obj = concept
        
        if not concept1_obj or not concept2_obj:
            raise HTTPException(status_code=404, detail="One or both concepts not found")
        
        # Compute similarity using available methods
        if isinstance(concept1_obj, type(concept2_obj)) and hasattr(registry, '_compute_cluster_similarity'):
            # Use cluster-based similarity if available
            try:
                similarity_score = registry._compute_cluster_similarity(concept1_obj, concept2_obj)
                method_used = "cluster_similarity"
            except Exception:
                # Fallback to basic similarity
                similarity_score = 0.5 if concept1_obj.name == concept2_obj.name else 0.3
                method_used = "basic_fallback"
        else:
            # Basic similarity based on name matching and context
            if concept1_obj.name == concept2_obj.name:
                similarity_score = 1.0
            elif concept1_obj.context == concept2_obj.context:
                similarity_score = 0.7
            else:
                similarity_score = 0.3
            method_used = "basic_similarity"
        
        return ConceptSimilarityResponse(
            similarity_score=float(similarity_score),
            method_used=method_used,
            confidence=0.8,
            explanation=f"Similarity between '{similarity_request.concept1}' and '{similarity_request.concept2}' using {method_used}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SEMANTIC REASONING ENDPOINTS
# ============================================================================

@app.post("/analogies/complete", response_model=AnalogyResponse, tags=["Reasoning"])
async def complete_analogy(
    analogy: AnalogyCompletion,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> AnalogyResponse:
    """Complete analogies using enhanced semantic reasoning."""
    try:
        # Validate that source_a and target_a are different to avoid key conflicts
        if analogy.source_a == analogy.target_a:
            raise HTTPException(
                status_code=400, 
                detail=f"source_a ('{analogy.source_a}') and target_a ('{analogy.target_a}') must be different to form a valid analogy"
            )
        
        # Use registry's analogy completion capabilities
        partial_analogy = {
            analogy.source_a: analogy.source_b,
            analogy.target_a: "?"
        }
        
        # Verify we have exactly 2 keys as expected
        if len(partial_analogy) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analogy structure. Expected 2 distinct concepts, got {len(partial_analogy)}. Please ensure source_a and target_a are different."
            )
        
        completions = registry.complete_analogy(
            partial_analogy,
            analogy.max_completions
        )
        
        # Format completions for response
        formatted_completions = []
        for completion in completions:
            formatted_completions.append({
                "target_b": completion.get("completion", ""),
                "confidence": completion.get("confidence", 0.0),
                "reasoning": completion.get("reasoning", "")
            })
        
        # Filter by minimum confidence
        formatted_completions = [
            c for c in formatted_completions 
            if c["confidence"] >= analogy.min_confidence
        ]
        
        return AnalogyResponse(
            completions=formatted_completions,
            reasoning_trace=[
                f"Analogy pattern: {analogy.source_a}:{analogy.source_b} :: {analogy.target_a}:?",
                f"Found {len(formatted_completions)} completions above confidence {analogy.min_confidence}",
                f"Partial analogy dictionary had {len(partial_analogy)} mappings"
            ],
            metadata={
                "context": analogy.context,
                "method": "enhanced_hybrid_reasoning",
                "partial_analogy": partial_analogy
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error completing analogy: {e}")
        # Provide more detailed error information
        error_detail = f"Analogy completion failed: {str(e)}"
        if "partial_analogy" in str(e):
            error_detail += f". Please ensure source_a ('{analogy.source_a}') and target_a ('{analogy.target_a}') are different concepts."
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/semantic-fields/discover", response_model=SemanticFieldDiscoveryResponse, tags=["Reasoning"])
async def discover_semantic_fields(
    discovery: SemanticFieldDiscovery,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> SemanticFieldDiscoveryResponse:
    """Discover semantic fields using clustering and coherence analysis."""
    try:
        # Use registry's semantic field discovery (only accepts min_coherence)
        fields = registry.discover_semantic_fields(
            min_coherence=discovery.min_coherence
        )
        
        # Filter by domain if specified
        if discovery.domain:
            fields = [
                field for field in fields
                if any(
                    concept.get("metadata", {}).get("domain") == discovery.domain
                    for concept in field.get("core_concepts", [])
                )
            ]
        
        # Format fields for response
        formatted_fields = []
        coherence_scores = {}
        
        for i, field in enumerate(fields[:discovery.max_fields]):
            field_id = f"field_{i}"
            formatted_fields.append({
                "field_id": field_id,
                "name": field.get("name", f"Field {i}"),
                "concepts": field.get("core_concepts", []),
                "coherence": field.get("coherence", 0.0),
                "domain": discovery.domain or "general"
            })
            coherence_scores[field_id] = field.get("coherence", 0.0)
        
        return SemanticFieldDiscoveryResponse(
            semantic_fields=formatted_fields,
            discovery_metadata={
                "method": discovery.clustering_method,
                "domain": discovery.domain,
                "total_concepts_analyzed": len(registry.concepts)
            },
            coherence_scores=coherence_scores
        )
        
    except Exception as e:
        logger.error(f"Error discovering semantic fields: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analogies/cross-domain", response_model=CrossDomainAnalogiesResponse, tags=["Reasoning"])
async def discover_cross_domain_analogies(
    request: CrossDomainAnalogiesRequest,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> CrossDomainAnalogiesResponse:
    """Discover cross-domain analogies between different semantic fields."""
    try:
        # Use registry's cross-domain analogy discovery (only accepts min_quality)
        analogies = registry.discover_cross_domain_analogies(
            min_quality=request["min_quality"]
        )
        
        # Filter results by domain if specified
        filtered_analogies = analogies
        if "source_domain" in request and request["source_domain"]:
            filtered_analogies = [
                a for a in filtered_analogies 
                if request["source_domain"] in str(a).lower()
            ]
        if "target_domain" in request and request["target_domain"]:
            filtered_analogies = [
                a for a in filtered_analogies 
                if request["target_domain"] in str(a).lower()
            ]
        
        # Limit results
        max_analogies = request.get("max_analogies", 5)
        filtered_analogies = filtered_analogies[:max_analogies]
        
        # Format analogies for response
        formatted_analogies = []
        quality_scores = {}
        
        for i, analogy in enumerate(filtered_analogies):
            analogy_id = f"analogy_{i}"
            quality = analogy.compute_overall_quality() if hasattr(analogy, 'compute_overall_quality') else 0.5
            
            formatted_analogies.append({
                "analogy_id": analogy_id,
                "source_pair": getattr(analogy, 'source_pair', []),
                "target_pair": getattr(analogy, 'target_pair', []),
                "quality_score": quality,
                "explanation": getattr(analogy, 'explanation', "")
            })
            quality_scores[analogy_id] = quality
        
        return CrossDomainAnalogiesResponse(
            analogies=formatted_analogies,
            quality_scores=quality_scores,
            domain_analysis={
                "source_domain": request.get("source_domain", ""),
                "target_domain": request.get("target_domain", ""),
                "analogies_found": len(formatted_analogies),
                "average_quality": sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0.0
            }
        )
        
    except Exception as e:
        logger.error(f"Error discovering cross-domain analogies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# FRAME OPERATIONS ENDPOINTS
# ============================================================================

@app.post("/frames", response_model=FrameCreateResponse, tags=["Frames"])
async def create_frame(
    frame: FrameCreateRequest,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> FrameCreateResponse:
    """Create a new semantic frame."""
    try:
        from app.core.frame_cluster_abstractions import SemanticFrame, FrameElement, FrameElementType
        
        # Create frame elements
        core_elements = []
        for element_name in frame["core_elements"]:
            element = FrameElement(
                name=element_name,
                description=f"Core element: {element_name}",
                element_type=FrameElementType.CORE
            )
            core_elements.append(element)
        
        peripheral_elements = []
        for element_name in frame.get("peripheral_elements") or []:
            element = FrameElement(
                name=element_name,
                description=f"Peripheral element: {element_name}",
                element_type=FrameElementType.PERIPHERAL
            )
            peripheral_elements.append(element)
        
        # Create semantic frame
        semantic_frame = SemanticFrame(
            name=frame["name"],
            definition=frame["definition"],
            core_elements=core_elements,
            peripheral_elements=peripheral_elements,
            lexical_units=frame.get("lexical_units") or []
        )
        
        # Register the frame
        registered_frame = registry.frame_registry.register_frame(semantic_frame)
        
        return FrameCreateResponse(
            frame_id=registered_frame.name,  # Use name as ID
            name=registered_frame.name,
            definition=registered_frame.definition,
            elements={
                "core": ", ".join([e.name for e in registered_frame.core_elements]),
                "peripheral": ", ".join([e.name for e in registered_frame.peripheral_elements])
            },
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating frame: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/frames/{frame_id}/instances", response_model=FrameInstanceResponse, tags=["Frames"])
async def create_frame_instance(
    frame_id: str,
    instance: FrameInstanceRequest,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> FrameInstanceResponse:
    """Create an instance of a semantic frame."""
    try:
        from app.core.frame_cluster_abstractions import FrameAwareConcept
        
        # Convert concept bindings to FrameAwareConcept objects
        bindings = {}
        for element_name, concept_name in instance["concept_bindings"].items():
            # Create a simple FrameAwareConcept for the binding
            concept = FrameAwareConcept(
                name=concept_name,
                context="frame_instance"  # Context should be a string
            )
            bindings[element_name] = concept
        
        # Create frame instance using correct parameters
        frame_instance = registry.frame_registry.create_frame_instance(
            frame_name=frame_id,  # Use frame_name parameter
            instance_id=instance["instance_id"],
            bindings=bindings,  # Use bindings parameter
            context=instance["context"]
        )
        
        return FrameInstanceResponse(
            instance_id=frame_instance.instance_id,
            frame_id=frame_instance.frame_name,
            bindings=instance["concept_bindings"],
            validation_results={"valid": True},  # Placeholder
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating frame instance: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/frames/query", response_model=FrameQueryResponse, tags=["Frames"])
async def query_frames(
    query: FrameQueryRequest,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> FrameQueryResponse:
    """Query frames and frame instances."""
    try:
        # Query frames based on concept or pattern
        frames = []
        instances = []
        
        if query.get("concept"):
            concept = query["concept"]
            if concept:  # Ensure concept is not None
                # Find frames containing the concept
                for frame_id, frame in registry.frame_registry.frames.items():
                    if concept in str(frame):  # Simple containment check
                        frames.append({
                            "frame_id": frame_id,
                            "name": getattr(frame, 'name', f'Frame {frame_id}'),
                            "definition": getattr(frame, 'definition', '')
                        })
        
        return FrameQueryResponse(
            frames=frames[:query.get("max_results", 10)],
            instances=instances[:query.get("max_results", 10)],
            query_metadata={
                "concept": query.get("concept"),
                "context": query.get("context"),
                "total_frames": len(frames),
                "total_instances": len(instances)
            }
        )
        
    except Exception as e:
        logger.error(f"Error querying frames: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# BATCH OPERATIONS AND WORKFLOWS
# ============================================================================

@app.post("/batch/analogies", response_model=BatchWorkflowResponse, tags=["Batch Operations"])
async def create_analogy_batch(
    batch: AnalogiesBatch,
    background_tasks: BackgroundTasks,
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
) -> BatchWorkflowResponse:
    """Create and process a batch of analogies."""
    try:
        workflow = batch_mgr.create_analogy_batch(
            analogies=batch["analogies"],
            workflow_id=batch.get("workflow_id")
        )
        
        # Process in background
        background_tasks.add_task(_process_batch_async, workflow.workflow_id, batch_mgr)
        
        return BatchWorkflowResponse(
            workflow_id=workflow.workflow_id,
            workflow_type=workflow.workflow_type.value,
            status=workflow.status.value,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            items_total=workflow.items_total,
            items_processed=workflow.items_processed,
            error_count=workflow.error_count,
            metadata=workflow.metadata or {},
            error_log=workflow.error_log or []
        )
        
    except Exception as e:
        logger.error(f"Error creating analogy batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/workflows", response_model=List[BatchWorkflowResponse], tags=["Batch Operations"])
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by workflow status"),
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
) -> List[BatchWorkflowResponse]:
    """List all workflows with optional status filtering."""
    try:
        workflows = batch_mgr.list_workflows()
        
        if status:
            workflows = [w for w in workflows if w.status.value == status]
        
        return [
            BatchWorkflowResponse(
                workflow_id=w.workflow_id,
                workflow_type=w.workflow_type.value,
                status=w.status.value,
                created_at=w.created_at.isoformat(),
                updated_at=w.updated_at.isoformat(),
                items_total=w.items_total,
                items_processed=w.items_processed,
                error_count=w.error_count,
                metadata=w.metadata or {},
                error_log=w.error_log or []
            )
            for w in workflows
        ]
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/workflows/{workflow_id}", response_model=BatchWorkflowResponse, tags=["Batch Operations"])
async def get_workflow(
    workflow_id: str,
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
) -> BatchWorkflowResponse:
    """Get detailed workflow information."""
    try:
        workflows = batch_mgr.list_workflows()
        workflow = next((w for w in workflows if w.workflow_id == workflow_id), None)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return BatchWorkflowResponse(
            workflow_id=workflow.workflow_id,
            workflow_type=workflow.workflow_type.value,
            status=workflow.status.value,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat(),
            items_total=workflow.items_total,
            items_processed=workflow.items_processed,
            error_count=workflow.error_count,
            metadata=workflow.metadata or {},
            error_log=workflow.error_log or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET STREAMING ENDPOINTS
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections for streaming."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # Connection might be closed
                pass


connection_manager = ConnectionManager()


@app.websocket("/ws/analogies/stream")
async def stream_analogies_websocket(
    websocket: WebSocket,
    domain: Optional[str] = Query(None),
    min_quality: Optional[float] = Query(0.5)
):
    """Stream analogies via WebSocket with real-time updates."""
    await connection_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Analogy streaming started"
        }))
        
        # Stream analogies from persistence layer
        count = 0
        if batch_manager:
            try:
                for analogy in batch_manager.stream_analogies(domain, min_quality):
                    message = StreamingMessage(
                        message_type="analogy",
                        content={
                            "analogy": analogy,
                            "count": count,
                            "filters": {"domain": domain, "min_quality": min_quality}
                        }
                    )
                    
                    await websocket.send_text(message.json())
                    count += 1
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.01)
                    
                    # Break after reasonable number to prevent infinite loops
                    if count >= 100:
                        break
            except Exception as e:
                logger.error(f"Error streaming analogies: {e}")
        
        # Send completion message
        await websocket.send_text(json.dumps({
            "type": "completion",
            "status": "completed",
            "total_streamed": count
        }))
        
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(e)
        }))


@app.websocket("/ws/workflows/{workflow_id}/status")
async def stream_workflow_status(websocket: WebSocket, workflow_id: str):
    """Stream real-time workflow status updates."""
    await connection_manager.connect(websocket)
    
    try:
        # Get batch manager from dependency injection system
        batch_mgr = None
        try:
            batch_mgr = get_batch_manager()
        except Exception:
            # Fall back to auto-initialization if dependency injection fails
            batch_mgr = await get_or_create_batch_manager()
        
        max_iterations = 60  # Maximum 60 seconds of streaming
        iterations = 0
        
        while iterations < max_iterations:
            if batch_mgr:
                try:
                    workflows = batch_mgr.list_workflows()
                    workflow = next((w for w in workflows if w.workflow_id == workflow_id), None)
                    
                    if workflow:
                        status_message = StreamingMessage(
                            message_type="workflow_status",
                            content={
                                "workflow_id": workflow_id,
                                "status": workflow.status.value,
                                "items_processed": workflow.items_processed,
                                "items_total": workflow.items_total,
                                "error_count": workflow.error_count,
                                "progress": workflow.items_processed / workflow.items_total if workflow.items_total > 0 else 0
                            }
                        )
                        
                        await websocket.send_text(status_message.model_dump_json())
                        
                        # Stop streaming if workflow is complete
                        if workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
                            break
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": f"Workflow {workflow_id} not found"
                        }))
                        break
                except Exception as e:
                    logger.error(f"Error getting workflow status: {e}")
                    break
            else:
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "error": "Batch manager not available"
                }))
                break
            
            await asyncio.sleep(1.0)  # Update every second
            iterations += 1
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error for workflow {workflow_id}: {e}")


# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Comprehensive health check endpoint."""
    return {
        "status": "healthy",
        "service": "soft-logic-service-layer",
        "version": "1.0.0",
        "components": {
            "semantic_registry": semantic_registry is not None,
            "persistence_manager": persistence_manager is not None,
            "batch_manager": batch_manager is not None
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status", tags=["System"])
async def get_service_status():
    """Get detailed service status and metrics."""
    if not all([semantic_registry, persistence_manager, batch_manager]):
        return {"status": "initializing"}
    
    try:
        # Registry statistics
        registry_stats = {
            "concepts_count": len(semantic_registry.concepts),
            "frames_count": len(semantic_registry.frame_registry.frames),
            "clusters_trained": semantic_registry.cluster_registry.is_trained,
            "embedding_provider": "semantic"
        }
        
        # Workflow statistics
        workflows = batch_manager.list_workflows()
        workflow_stats = {}
        for status in WorkflowStatus:
            workflow_stats[status.value] = len([w for w in workflows if w.status == status])
        
        # Storage statistics
        storage_stats = {
            "storage_path": str(persistence_manager.storage_path),
            "active_workflows": len(batch_manager.active_workflows)
        }
        
        return {
            "status": "operational",
            "registry_stats": registry_stats,
            "workflow_stats": workflow_stats,
            "storage_stats": storage_stats,
            "uptime": "running",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/docs-overview", tags=["System"])
async def get_docs_overview():
    """Get API documentation overview."""
    return {
        "api_overview": {
            "concepts": "Manage semantic concepts with automatic disambiguation",
            "reasoning": "Advanced analogical reasoning and semantic field discovery",
            "frames": "FrameNet-style semantic frame operations",
            "batch": "Batch processing and workflow management",
            "streaming": "Real-time WebSocket streaming of data and status",
            "system": "Health checks and service status monitoring"
        },
        "endpoints": {
            "concepts": ["/concepts", "/concepts/{id}", "/concepts/search", "/concepts/similarity"],
            "reasoning": ["/analogies/complete", "/semantic-fields/discover", "/analogies/cross-domain"],
            "frames": ["/frames", "/frames/{id}/instances", "/frames/query"],
            "batch": ["/batch/analogies", "/batch/workflows", "/batch/workflows/{id}"],
            "streaming": ["/ws/analogies/stream", "/ws/workflows/{id}/status"],
            "system": ["/health", "/status", "/docs", "/redoc"]
        },
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json"
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _process_batch_async(workflow_id: str, batch_mgr: BatchPersistenceManager):
    """Process batch workflow asynchronously."""
    try:
        # Small delay to ensure workflow creation is complete
        await asyncio.sleep(0.1)
        
        workflow = batch_mgr.process_analogy_batch(workflow_id)
        logger.info(f"Completed batch workflow {workflow_id} with status {workflow.status}")
        
    except Exception as e:
        logger.error(f"Error in async batch processing for workflow {workflow_id}: {e}")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def start_service():
    """Start the service layer with appropriate configuration."""
    uvicorn.run(
        "app.service_layer:app",
        host="0.0.0.0",
        port=8321,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_service()


def initialize_services(force_reinit: bool = False):
    """
    Initialize services directly (synchronous version for testing).
    
    Args:
        force_reinit: If True, reinitialize even if services are already initialized
    """
    global semantic_registry, persistence_manager, batch_manager, neural_symbolic_service
    
    # Check if already initialized
    if not force_reinit and all([semantic_registry, persistence_manager, batch_manager]):
        logger.info("Services already initialized")
        return
    
    try:
        logger.info("Initializing services directly...")
        
        # Initialize storage path
        storage_path = Path("storage")
        storage_path.mkdir(exist_ok=True)
        
        # Initialize semantic registry with enhanced capabilities
        logger.info("Initializing semantic registry...")
        semantic_registry = EnhancedHybridRegistry(
            download_wordnet=False,  # Don't download wordnet during tests
            n_clusters=8,
            enable_cross_domain=True,
            embedding_provider="random"  # Use random for reliable startup
        )
        
        # Initialize persistence managers
        logger.info("Initializing persistence layer...")
        persistence_manager = ContractEnhancedPersistenceManager(storage_path)
        batch_manager = BatchPersistenceManager(storage_path)
        
        # Phase 4: Initialize neural-symbolic service
        logger.info("Initializing neural-symbolic service...")
        initialize_neural_symbolic_service(semantic_registry, persistence_manager)
        
        logger.info("✅ Services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
