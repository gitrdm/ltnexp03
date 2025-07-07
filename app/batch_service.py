"""
FastAPI Service Layer with Batch Persistence Support

This module provides REST API endpoints for batch operations, persistence,
and workflow management in the soft logic microservice.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.batch_persistence import (
    BatchPersistenceManager, BatchWorkflow, DeleteCriteria,
    WorkflowStatus, WorkflowType
)
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.api_models import (
    AnalogiesBatch, BatchWorkflowResponse, DeleteCriteriaRequest,
    CompactionResult, WorkflowListResponse, ExportRequest, ExportResponse,
    ImportRequest, ImportResponse
)


# Global state - in production, use dependency injection
persistence_manager: Optional[BatchPersistenceManager] = None
registry: Optional[EnhancedHybridRegistry] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    global persistence_manager, registry
    
    # Startup
    storage_path = Path("storage")
    persistence_manager = BatchPersistenceManager(storage_path)
    registry = EnhancedHybridRegistry(
        download_wordnet=True,
        n_clusters=8,
        enable_cross_domain=True,
        embedding_provider="semantic"
    )
    
    logging.info("Batch persistence service started")
    yield
    
    # Shutdown
    logging.info("Batch persistence service stopped")


app = FastAPI(
    title="Soft Logic Microservice with Batch Persistence",
    description="Production-ready soft logic system with batch operations and workflow management",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# BATCH ANALOGY OPERATIONS
# ============================================================================

@app.post("/analogies/batch", response_model=BatchWorkflowResponse)
async def create_analogy_batch(batch: AnalogiesBatch, background_tasks: BackgroundTasks) -> BatchWorkflowResponse:
    """Create batch of analogies with workflow tracking."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        workflow = persistence_manager.create_analogy_batch(
            analogies=batch["analogies"],
            workflow_id=batch.get("workflow_id")
        )
        
        # Process batch in background
        background_tasks.add_task(
            process_batch_workflow_async,
            workflow.workflow_id
        )
        
        return _workflow_to_response(workflow)
        
    except Exception as e:
        logging.error(f"Error creating analogy batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analogies/batch/process/{workflow_id}", response_model=BatchWorkflowResponse)
async def process_analogy_batch_sync(workflow_id: str) -> BatchWorkflowResponse:
    """Process analogy batch synchronously."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        workflow = persistence_manager.process_analogy_batch(workflow_id)
        return _workflow_to_response(workflow)
        
    except Exception as e:
        logging.error(f"Error processing analogy batch {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/analogies/batch", response_model=BatchWorkflowResponse)
async def delete_analogies_batch(criteria: DeleteCriteriaRequest) -> BatchWorkflowResponse:
    """Delete analogies matching criteria."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        # Convert request model to internal model
        # Handle date parsing with proper type checking
        created_before = None
        created_before_str = criteria.get("created_before")
        if created_before_str:
            created_before = datetime.fromisoformat(created_before_str)
            
        created_after = None
        created_after_str = criteria.get("created_after")
        if created_after_str:
            created_after = datetime.fromisoformat(created_after_str)
        
        delete_criteria = DeleteCriteria(
            domains=criteria.get("domains"),
            frame_types=criteria.get("frame_types"),
            quality_threshold=criteria.get("quality_threshold"),
            created_before=created_before,
            created_after=created_after,
            tags=criteria.get("tags")
        )
        
        workflow = persistence_manager.delete_analogies_batch(delete_criteria)
        return _workflow_to_response(workflow)
        
    except Exception as e:
        logging.error(f"Error deleting analogies batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# COMPACTION OPERATIONS
# ============================================================================

@app.post("/analogies/compact", response_model=CompactionResult)
async def compact_analogies() -> CompactionResult:
    """Compact analogies storage by removing deleted records."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        result = persistence_manager.compact_analogies_jsonl()
        return CompactionResult(
            status=result["status"],
            records_removed=result["records_removed"],
            active_records=result["active_records"],
            backup_created=result.get("backup_created")
        )
        
    except Exception as e:
        logging.error(f"Error compacting analogies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WORKFLOW MANAGEMENT
# ============================================================================

@app.get("/workflows/{workflow_id}", response_model=BatchWorkflowResponse)
async def get_workflow_status(workflow_id: str) -> BatchWorkflowResponse:
    """Get workflow status."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    workflow = persistence_manager.get_workflow_status(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return _workflow_to_response(workflow)


@app.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(status: Optional[str] = None, limit: int = 50) -> WorkflowListResponse:
    """List workflows, optionally filtered by status."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        workflow_status = WorkflowStatus(status) if status else None
        workflows = persistence_manager.list_workflows(workflow_status)
        workflows = workflows[:limit]  # Apply limit
        
        return WorkflowListResponse(
            workflows=[_workflow_to_response(w) for w in workflows],
            total_count=len(workflows),
            filtered_by_status=status
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    except Exception as e:
        logging.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflows/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str) -> Dict[str, Any]:
    """Cancel a pending workflow."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    success = persistence_manager.cancel_workflow(workflow_id)
    if not success:
        raise HTTPException(status_code=400, detail="Workflow cannot be cancelled")
    
    return {"status": "cancelled", "workflow_id": workflow_id}


# ============================================================================
# QUERY OPERATIONS
# ============================================================================

@app.get("/analogies/stream")
async def stream_analogies(domain: Optional[str] = None, min_quality: Optional[float] = None, limit: int = 100) -> Dict[str, Any]:
    """Stream analogies with optional filtering."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        analogies = []
        count = 0
        
        for analogy in persistence_manager.stream_analogies(domain, min_quality):
            analogies.append(analogy)
            count += 1
            if count >= limit:
                break
        
        return {
            "analogies": analogies,
            "count": count,
            "filters": {
                "domain": domain,
                "min_quality": min_quality
            }
        }
        
    except Exception as e:
        logging.error(f"Error streaming analogies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analogies/high-quality")
async def get_high_quality_analogies(min_quality: float = 0.8) -> Dict[str, Any]:
    """Get high-quality analogies using indexed search."""
    if not persistence_manager:
        raise HTTPException(status_code=500, detail="Persistence manager not initialized")
    
    try:
        analogies = persistence_manager.find_analogies_by_quality(min_quality)
        return {
            "analogies": analogies,
            "count": len(analogies),
            "min_quality": min_quality
        }
        
    except Exception as e:
        logging.error(f"Error finding high-quality analogies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXPORT/IMPORT OPERATIONS
# ============================================================================

@app.post("/contexts/{context_name}/export")
async def export_context(context_name: str, format: str = "json", compressed: bool = False) -> ExportResponse:
    """Export complete context state."""
    if not persistence_manager or not registry:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Use the original persistence manager for export
        from app.core.persistence import PersistenceManager
        
        export_manager = PersistenceManager(persistence_manager.storage_path)
        export_result = export_manager.save_registry_state(
            registry, context_name, format
        )
        
        return ExportResponse(
            export_id=export_result.get("export_id", "unknown"),
            file_path=export_result.get("file_path", ""),
            file_size=export_result.get("file_size", 0),
            format=format,
            created_at=export_result.get("saved_at", ""),
            components_exported=export_result.get("components_saved", [])
        )
        
    except Exception as e:
        logging.error(f"Error exporting context {context_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/contexts/{context_name}/import")
async def import_context(context_name: str, file: UploadFile = File(...), merge_strategy: str = "overwrite") -> ImportResponse:
    """Import context from uploaded file."""
    if not persistence_manager or not registry:
        raise HTTPException(status_code=500, detail="Services not initialized")
    
    try:
        # Validate filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
            
        # Save uploaded file
        import_path = persistence_manager.storage_path / "imports" / file.filename
        import_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(import_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Import would be implemented here
        # For now, return a placeholder response
        return ImportResponse(
            import_id=f"import_{context_name}_{datetime.now().isoformat()}",
            status="completed",
            items_imported=0,
            conflicts_found=0,
            merge_strategy_used=merge_strategy,
            imported_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logging.error(f"Error importing context {context_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH AND STATUS
# ============================================================================

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "soft-logic-batch-persistence",
        "version": "1.0.0",
        "persistence_manager": persistence_manager is not None,
        "registry": registry is not None
    }


@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed service status."""
    if not persistence_manager or not registry:
        return {"status": "initializing"}
    
    try:
        # Get workflow counts
        workflows = persistence_manager.list_workflows()
        workflow_counts = {}
        for status in WorkflowStatus:
            workflow_counts[status.value] = len([w for w in workflows if w.status == status])
        
        # Get storage info
        storage_info = {
            "storage_path": str(persistence_manager.storage_path),
            "active_workflows": len(persistence_manager.active_workflows)
        }
        
        # Get registry info
        registry_info = {
            "concepts_count": len(registry.concepts),
            "frames_count": len(registry.frame_registry.frames),
            "clusters_trained": registry.cluster_registry.is_trained
        }
        
        return {
            "status": "operational",
            "workflow_counts": workflow_counts,
            "storage_info": storage_info,
            "registry_info": registry_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error getting service status: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _workflow_to_response(workflow: BatchWorkflow) -> BatchWorkflowResponse:
    """Convert BatchWorkflow to API response model."""
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


async def process_batch_workflow_async(workflow_id: str) -> None:
    """Process batch workflow asynchronously."""
    if not persistence_manager:
        logging.error("Persistence manager not initialized for async processing")
        return
    
    try:
        # Add small delay to ensure the workflow creation is complete
        await asyncio.sleep(0.1)
        
        workflow = persistence_manager.process_analogy_batch(workflow_id)
        logging.info(f"Completed batch workflow {workflow_id} with status {workflow.status}")
        
    except Exception as e:
        logging.error(f"Error in async batch processing for workflow {workflow_id}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8321)
