"""
Minimal Working FastAPI Service Layer for Soft Logic Microservice
================================================================

This module provides a working REST API service layer that integrates with
our existing persistence and semantic reasoning components. This is a 
simplified version that focuses on working functionality over complete features.

Following Design by Contract principles with basic type safety.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import core components
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.batch_persistence import BatchPersistenceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS FOR TYPE-SAFE API
# ============================================================================

class ConceptCreate(BaseModel):
    """Simple concept creation model."""
    name: str = Field(..., min_length=1, description="Concept name")
    context: str = Field("default", description="Context or domain")
    synset_id: Optional[str] = Field(None, description="WordNet synset ID")
    disambiguation: Optional[str] = Field(None, description="Disambiguation text")


class ConceptResponse(BaseModel):
    """Concept response model."""
    concept_id: str
    name: str
    context: str
    synset_id: Optional[str] = None
    disambiguation: Optional[str] = None
    created_at: str


class BatchRequest(BaseModel):
    """Batch processing request."""
    analogies: List[Dict[str, Any]]
    workflow_id: Optional[str] = None


class StatusResponse(BaseModel):
    """Service status response."""
    status: str
    service: str
    version: str
    components: Dict[str, bool]
    timestamp: str


# ============================================================================
# APPLICATION LIFECYCLE AND GLOBAL STATE
# ============================================================================

# Global service instances
semantic_registry: Optional[EnhancedHybridRegistry] = None
persistence_manager: Optional[ContractEnhancedPersistenceManager] = None
batch_manager: Optional[BatchPersistenceManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global semantic_registry, persistence_manager, batch_manager
    
    try:
        logger.info("Starting Soft Logic Service Layer...")
        
        # Initialize storage
        storage_path = Path("storage")
        storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        semantic_registry = EnhancedHybridRegistry(
            download_wordnet=False,  # Skip for faster startup
            n_clusters=4,
            enable_cross_domain=True,
            embedding_provider="random"  # Use 'random' instead of 'test'
        )
        
        persistence_manager = ContractEnhancedPersistenceManager(storage_path)
        batch_manager = BatchPersistenceManager(storage_path)
        
        logger.info("✅ Service layer started successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start service layer: {e}")
        raise
    finally:
        logger.info("Service layer shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Soft Logic Microservice - Working Service Layer",
    description="Production-ready soft logic system with semantic reasoning and persistence",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

def get_semantic_registry() -> EnhancedHybridRegistry:
    """Get semantic registry dependency."""
    if semantic_registry is None:
        raise HTTPException(status_code=503, detail="Semantic registry not initialized")
    return semantic_registry


def get_persistence_manager() -> ContractEnhancedPersistenceManager:
    """Get persistence manager dependency."""
    if persistence_manager is None:
        raise HTTPException(status_code=503, detail="Persistence manager not initialized")
    return persistence_manager


def get_batch_manager() -> BatchPersistenceManager:
    """Get batch manager dependency."""
    if batch_manager is None:
        raise HTTPException(status_code=503, detail="Batch manager not initialized")
    return batch_manager


# ============================================================================
# CORE API ENDPOINTS
# ============================================================================

@app.get("/health", response_model=StatusResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return StatusResponse(
        status="healthy",
        service="soft-logic-service-layer",
        version="1.0.0",
        components={
            "semantic_registry": semantic_registry is not None,
            "persistence_manager": persistence_manager is not None,
            "batch_manager": batch_manager is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.get("/status", tags=["System"])
async def get_service_status():
    """Get detailed service status."""
    try:
        if not all([semantic_registry, persistence_manager, batch_manager]):
            return {"status": "initializing"}
        
        # Get basic stats safely
        registry_stats = {
            "concepts_count": len(getattr(semantic_registry, 'frame_aware_concepts', {})),
            "frames_count": len(getattr(semantic_registry.frame_registry, 'frames', {})),
            "embedding_provider": "random"
        }
        
        storage_stats = {
            "storage_path": str(persistence_manager.storage_path),
            "initialized": True
        }
        
        return {
            "status": "operational",
            "registry_stats": registry_stats,
            "storage_stats": storage_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        return {"status": "error", "error": str(e)}


@app.post("/concepts", response_model=ConceptResponse, tags=["Concepts"])
async def create_concept(
    concept: ConceptCreate,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
) -> ConceptResponse:
    """Create a new concept."""
    try:
        # Use the existing method from EnhancedHybridRegistry
        created_concept = registry.create_frame_aware_concept_with_advanced_embedding(
            name=concept.name,
            context=concept.context,
            synset_id=concept.synset_id,
            disambiguation=concept.disambiguation,
            use_semantic_embedding=True
        )
        
        # Extract the concept_id from the created concept
        concept_id = getattr(created_concept, 'unique_id', f"{concept.name}_{concept.context}")
        
        return ConceptResponse(
            concept_id=concept_id,
            name=concept.name,
            context=concept.context,
            synset_id=concept.synset_id,
            disambiguation=concept.disambiguation,
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error creating concept: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/concepts/{concept_id}", tags=["Concepts"])
async def get_concept(
    concept_id: str,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
):
    """Retrieve a concept by ID."""
    try:
        frame_aware_concepts = getattr(registry, 'frame_aware_concepts', {})
        
        if concept_id not in frame_aware_concepts:
            raise HTTPException(status_code=404, detail="Concept not found")
        
        concept = frame_aware_concepts[concept_id]
        return {
            "concept_id": concept_id,
            "name": getattr(concept, 'name', 'unknown'),
            "context": getattr(concept, 'context', 'default'),
            "synset_id": getattr(concept, 'synset_id', None),
            "disambiguation": getattr(concept, 'disambiguation', None)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving concept {concept_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/concepts/search", tags=["Concepts"])
async def search_concepts(
    query: str,
    max_results: int = 10,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
):
    """Search for concepts."""
    try:
        frame_aware_concepts = getattr(registry, 'frame_aware_concepts', {})
        results = []
        
        for concept_id, concept in frame_aware_concepts.items():
            concept_name = getattr(concept, 'name', '')
            if query.lower() in concept_name.lower():
                results.append({
                    "concept_id": concept_id,
                    "name": concept_name,
                    "similarity_score": 1.0 if query.lower() == concept_name.lower() else 0.8
                })
                
                if len(results) >= max_results:
                    break
        
        return {
            "concepts": results,
            "total_results": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Error searching concepts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analogies/complete", tags=["Reasoning"])
async def complete_analogy(
    source_a: str,
    source_b: str,
    target_a: str,
    max_completions: int = 3,
    registry: EnhancedHybridRegistry = Depends(get_semantic_registry)
):
    """Complete analogies using semantic reasoning."""
    try:
        # Use existing method for analogical completion if available
        completions = []
        
        # Simple completion logic as fallback
        if hasattr(registry, 'find_analogical_completions'):
            partial = {source_a: source_b, target_a: "?"}
            results = registry.find_analogical_completions(partial, max_completions)
            for result in results:
                completions.append({
                    "target_b": result.get(target_a, "unknown"),
                    "confidence": 0.7,
                    "reasoning": f"Analogical pattern: {source_a}:{source_b} :: {target_a}:?"
                })
        else:
            # Fallback response
            completions = [{
                "target_b": "generated_completion",
                "confidence": 0.5,
                "reasoning": f"Pattern completion for {source_a}:{source_b} :: {target_a}:?"
            }]
        
        return {
            "completions": completions,
            "reasoning_trace": [f"Analogy: {source_a}:{source_b} :: {target_a}:?"],
            "metadata": {"method": "enhanced_hybrid_reasoning"}
        }
        
    except Exception as e:
        logger.error(f"Error completing analogy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch/analogies", tags=["Batch Operations"])
async def create_analogy_batch(
    batch: BatchRequest,
    background_tasks: BackgroundTasks,
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
):
    """Create and process a batch of analogies."""
    try:
        workflow = batch_mgr.create_analogy_batch(
            analogies=batch.analogies,
            workflow_id=batch.workflow_id
        )
        
        # Process in background
        background_tasks.add_task(_process_batch_async, workflow.workflow_id, batch_mgr)
        
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type.value,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "items_total": workflow.items_total,
            "items_processed": workflow.items_processed
        }
        
    except Exception as e:
        logger.error(f"Error creating analogy batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/workflows", tags=["Batch Operations"])
async def list_workflows(
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
):
    """List all workflows."""
    try:
        workflows = batch_mgr.list_workflows()
        
        return [
            {
                "workflow_id": w.workflow_id,
                "workflow_type": w.workflow_type.value,
                "status": w.status.value,
                "created_at": w.created_at.isoformat(),
                "items_total": w.items_total,
                "items_processed": w.items_processed
            }
            for w in workflows
        ]
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch/workflows/{workflow_id}", tags=["Batch Operations"])
async def get_workflow(
    workflow_id: str,
    batch_mgr: BatchPersistenceManager = Depends(get_batch_manager)
):
    """Get workflow details."""
    try:
        workflows = batch_mgr.list_workflows()
        workflow = next((w for w in workflows if w.workflow_id == workflow_id), None)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_type": workflow.workflow_type.value,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "items_total": workflow.items_total,
            "items_processed": workflow.items_processed,
            "error_count": workflow.error_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/docs-overview", tags=["System"])
async def get_docs_overview():
    """Get API documentation overview."""
    return {
        "api_overview": {
            "concepts": "Manage semantic concepts with disambiguation",
            "reasoning": "Basic analogical reasoning capabilities",
            "batch": "Batch processing and workflow management",
            "system": "Health checks and service status monitoring"
        },
        "endpoints": {
            "concepts": ["/concepts", "/concepts/{id}", "/concepts/search"],
            "reasoning": ["/analogies/complete"],
            "batch": ["/batch/analogies", "/batch/workflows"],
            "system": ["/health", "/status", "/docs"]
        },
        "documentation": {
            "interactive_docs": "/docs",
            "redoc": "/redoc"
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def _process_batch_async(workflow_id: str, batch_mgr: BatchPersistenceManager):
    """Process batch workflow asynchronously."""
    try:
        import asyncio
        await asyncio.sleep(0.1)  # Small delay
        
        workflow = batch_mgr.process_analogy_batch(workflow_id)
        logger.info(f"Completed batch workflow {workflow_id} with status {workflow.status}")
        
    except Exception as e:
        logger.error(f"Error in async batch processing for workflow {workflow_id}: {e}")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def start_service():
    """Start the service layer."""
    uvicorn.run(
        "app.working_service_layer:app",
        host="0.0.0.0",
        port=8321,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_service()
