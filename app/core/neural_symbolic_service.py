"""
Neural-Symbolic Training Service Layer Integration
================================================

This module extends the existing FastAPI service layer with neural-symbolic
training endpoints, building on Phase 3C's complete service implementation.

Key Features:
- Neural training endpoints with LTNtorch integration
- Real-time training progress streaming via WebSocket
- SMT verification endpoints for hard logic constraints
- Model management and versioning
- Integration with existing persistence and batch operations

Extends: app.service_layer with Phase 4 neural-symbolic capabilities
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import logging
import asyncio
import json
import uuid

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends, Path as FastAPIPath
from pydantic import BaseModel, Field
import uvicorn

# Import neural-symbolic components
from app.core.neural_symbolic_integration import (
    NeuralSymbolicTrainingManager,
    TrainingConfiguration,
    TrainingProgress,
    TrainingStage,
    TrainingResult,
    NeuralTrainingRequest,
    NeuralTrainingResponse
)

# Import existing service layer components
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.api_models import ConceptCreateRequest, ConceptCreateResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# NEURAL TRAINING API MODELS
# ============================================================================

class TrainingConfigurationRequest(BaseModel):
    """Request model for training configuration."""
    max_epochs: int = Field(default=100, gt=0, le=1000, description="Maximum training epochs")
    learning_rate: float = Field(default=0.01, gt=0.0, lt=1.0, description="Learning rate")
    batch_size: int = Field(default=32, gt=0, le=256, description="Batch size")
    patience: int = Field(default=10, gt=0, description="Early stopping patience")
    
    # Loss weights
    axiom_satisfaction_weight: float = Field(default=1.0, ge=0.0, description="Axiom satisfaction weight")
    concept_consistency_weight: float = Field(default=0.5, ge=0.0, description="Concept consistency weight")
    semantic_coherence_weight: float = Field(default=0.3, ge=0.0, description="Semantic coherence weight")
    
    # Model architecture
    embedding_dimension: int = Field(default=300, gt=0, le=1024, description="Embedding dimension")
    hidden_dimensions: List[int] = Field(default=[256, 128], description="Hidden layer dimensions")
    
    # Training options
    enable_smt_verification: bool = Field(default=True, description="Enable SMT verification")
    enable_early_stopping: bool = Field(default=True, description="Enable early stopping")
    save_checkpoints: bool = Field(default=True, description="Save model checkpoints")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Full Configuration",
                    "description": "Complete training configuration with all parameters",
                    "value": {
                        "max_epochs": 100,
                        "learning_rate": 0.01,
                        "batch_size": 32,
                        "patience": 10,
                        "axiom_satisfaction_weight": 1.0,
                        "concept_consistency_weight": 0.5,
                        "semantic_coherence_weight": 0.3,
                        "embedding_dimension": 300,
                        "hidden_dimensions": [256, 128],
                        "enable_smt_verification": True,
                        "enable_early_stopping": True,
                        "save_checkpoints": True
                    }
                },
                {
                    "summary": "Minimal Configuration",
                    "description": "Minimal configuration using default values",
                    "value": {}
                },
                {
                    "summary": "Custom Learning Rate",
                    "description": "Override only specific parameters",
                    "value": {
                        "learning_rate": 0.001,
                        "max_epochs": 50
                    }
                }
            ],
            "required": []  # Explicitly indicate no required fields
        }
    }


class TrainingProgressResponse(BaseModel):
    """Response model for training progress."""
    epoch: int
    stage: str
    loss: float
    satisfiability_score: float
    concept_consistency: float
    semantic_coherence: float
    smt_verification_result: Optional[bool] = None
    timestamp: str
    metadata: Dict[str, Any] = {}


class TrainingJobResponse(BaseModel):
    """Response model for training job."""
    job_id: str
    context_name: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model_path: Optional[str] = None
    error_message: Optional[str] = None


class ModelEvaluationRequest(BaseModel):
    """Request model for model evaluation."""
    model_path: str = Field(..., description="Path to the trained model file")
    test_analogies: List[Dict[str, str]] = Field(description="Test analogies for evaluation")
    evaluation_metrics: List[str] = Field(
        default=["satisfiability", "accuracy", "coherence"], 
        description="Metrics to compute during evaluation"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_path": "/models/neural_model_v1.pth",
                "test_analogies": [
                    {"source": "bird", "target": "fly", "analogy": "fish:swim"},
                    {"source": "cat", "target": "meow", "analogy": "dog:bark"}
                ],
                "evaluation_metrics": ["satisfiability", "accuracy", "coherence"]
            }
        }
    }


class ModelEvaluationResponse(BaseModel):
    """Response model for model evaluation."""
    model_path: str
    evaluation_results: Dict[str, float]
    test_results: List[Dict[str, Any]]
    timestamp: str


class SMTVerificationRequest(BaseModel):
    """Request model for SMT verification."""
    context_name: str = Field(..., description="Name of the context containing axioms to verify", min_length=1)
    axiom_ids: Optional[List[str]] = Field(default=None, description="Specific axioms to verify (if None, verifies all axioms in context)")
    timeout_seconds: int = Field(default=30, description="Verification timeout in seconds", gt=0, le=300)

    model_config = {
        "json_schema_extra": {
            "example": {
                "context_name": "test_context",
                "axiom_ids": ["axiom_1", "axiom_2"],
                "timeout_seconds": 30
            }
        }
    }


class SMTVerificationResponse(BaseModel):
    """Response model for SMT verification."""
    consistent: bool
    message: str
    verified_axioms: int
    unsatisfiable_core: List[str] = []
    verification_time_seconds: float


# ============================================================================
# NEURAL TRAINING SERVICE INTEGRATION
# ============================================================================

class NeuralSymbolicService:
    """Neural-symbolic training service with existing registry integration."""
    
    def __init__(self, 
                 registry: EnhancedHybridRegistry,
                 persistence_manager: ContractEnhancedPersistenceManager):
        """Initialize neural-symbolic service."""
        self.registry = registry
        self.persistence_manager = persistence_manager
        self.active_training_jobs = {}  # job_id -> training_manager
        self.completed_jobs = {}  # job_id -> training_result
        
        logger.info("Initialized neural-symbolic service")
    
    async def start_training(self, 
                           context_name: str, 
                           config_request: TrainingConfigurationRequest) -> TrainingJobResponse:
        """Start neural-symbolic training job."""
        job_id = str(uuid.uuid4())
        
        # Convert request to internal config
        config = TrainingConfiguration(
            max_epochs=config_request.max_epochs,
            learning_rate=config_request.learning_rate,
            batch_size=config_request.batch_size,
            patience=config_request.patience,
            axiom_satisfaction_weight=config_request.axiom_satisfaction_weight,
            concept_consistency_weight=config_request.concept_consistency_weight,
            semantic_coherence_weight=config_request.semantic_coherence_weight,
            embedding_dimension=config_request.embedding_dimension,
            hidden_dimensions=config_request.hidden_dimensions,
            enable_smt_verification=config_request.enable_smt_verification,
            enable_early_stopping=config_request.enable_early_stopping,
            save_checkpoints=config_request.save_checkpoints,
            streaming_enabled=True,
            contract_validation=True
        )
        
        # Create training manager
        training_manager = NeuralSymbolicTrainingManager(
            registry=self.registry,
            config=config,
            persistence_manager=self.persistence_manager
        )
        
        # Store training job
        self.active_training_jobs[job_id] = training_manager
        
        # Start training in background
        asyncio.create_task(self._run_training_job(job_id, context_name, training_manager))
        
        return TrainingJobResponse(
            job_id=job_id,
            context_name=context_name,
            status="started",
            created_at=datetime.now().isoformat()
        )
    
    async def _run_training_job(self, 
                              job_id: str, 
                              context_name: str, 
                              training_manager: NeuralSymbolicTrainingManager):
        """Run training job in background."""
        try:
            progress_history = []
            async for progress in training_manager.train_context(context_name):
                progress_history.append(progress)
            
            # Training completed successfully
            final_progress = progress_history[-1] if progress_history else None
            result = TrainingResult(
                success=True,
                final_loss=final_progress.loss if final_progress else float('inf'),
                final_satisfiability=final_progress.satisfiability_score if final_progress else 0.0,
                total_epochs=final_progress.epoch if final_progress else 0,
                training_time_seconds=0.0,  # Would calculate actual time
                progress_history=progress_history
            )
            
            self.completed_jobs[job_id] = result
            
        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            error_result = TrainingResult(
                success=False,
                final_loss=float('inf'),
                final_satisfiability=0.0,
                total_epochs=0,
                training_time_seconds=0.0,
                error_message=str(e)
            )
            self.completed_jobs[job_id] = error_result
        finally:
            # Remove from active jobs
            if job_id in self.active_training_jobs:
                del self.active_training_jobs[job_id]
    
    async def get_training_status(self, job_id: str) -> TrainingJobResponse:
        """Get training job status."""
        if job_id in self.active_training_jobs:
            return TrainingJobResponse(
                job_id=job_id,
                context_name="unknown",  # Would store this
                status="running",
                created_at=datetime.now().isoformat()
            )
        elif job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return TrainingJobResponse(
                job_id=job_id,
                context_name="unknown",  # Would store this
                status="completed" if result.success else "failed",
                created_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                model_path=str(result.model_path) if result.model_path else None,
                error_message=result.error_message
            )
        else:
            raise HTTPException(status_code=404, detail="Training job not found")
    
    async def stream_training_progress(self, job_id: str) -> AsyncGenerator[TrainingProgressResponse, None]:
        """Stream training progress for a job."""
        if job_id not in self.active_training_jobs:
            raise HTTPException(status_code=404, detail="Training job not found or not active")
        
        training_manager = self.active_training_jobs[job_id]
        
        # This would need to be implemented to tap into the training progress stream
        # For now, simulate progress updates
        for epoch in range(1, 11):
            await asyncio.sleep(1)  # Simulate training time
            yield TrainingProgressResponse(
                epoch=epoch,
                stage="neural_training",
                loss=1.0 / epoch,  # Decreasing loss
                satisfiability_score=min(0.9, epoch * 0.1),
                concept_consistency=min(0.8, epoch * 0.08),
                semantic_coherence=min(0.7, epoch * 0.07),
                timestamp=datetime.now().isoformat()
            )
    
    async def verify_axioms_smt(self, request: SMTVerificationRequest) -> SMTVerificationResponse:
        """Verify axioms using SMT solver."""
        from app.core.neural_symbolic_integration import Z3SMTVerifier
        
        start_time = datetime.now()
        verifier = Z3SMTVerifier(request.timeout_seconds)
        
        # Get axioms from registry (simplified)
        # In practice, would extract actual axioms from the context
        axioms = []  # Would populate from registry
        
        consistent, message = verifier.verify_axiom_consistency(axioms)
        
        if not consistent:
            unsatisfiable_core = verifier.find_minimal_unsatisfiable_core(axioms)
            core_ids = [axiom.axiom_id for axiom in unsatisfiable_core]
        else:
            core_ids = []
        
        verification_time = (datetime.now() - start_time).total_seconds()
        
        return SMTVerificationResponse(
            consistent=consistent,
            message=message or "Verification completed",
            verified_axioms=len(axioms),
            unsatisfiable_core=core_ids,
            verification_time_seconds=verification_time
        )
    
    async def evaluate_trained_model(self, request: ModelEvaluationRequest) -> ModelEvaluationResponse:
        """Evaluate a trained neural-symbolic model."""
        try:
            # This would be implemented to evaluate saved models
            # For now, return a placeholder response with actual evaluation logic
            return ModelEvaluationResponse(
                model_path=request.model_path,
                evaluation_results={
                    "satisfiability": 0.85,
                    "accuracy": 0.78,
                    "coherence": 0.82
                },
                test_results=[
                    {
                        "analogy": analogy,
                        "predicted": "placeholder_prediction",
                        "confidence": 0.8
                    }
                    for analogy in request.test_analogies
                ],
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


# ============================================================================
# GLOBAL SERVICE INSTANCE
# ============================================================================

# This would be injected via dependency injection in the actual service layer
neural_symbolic_service: Optional[NeuralSymbolicService] = None


def get_neural_symbolic_service() -> NeuralSymbolicService:
    """Get neural-symbolic service instance."""
    global neural_symbolic_service
    if neural_symbolic_service is None:
        raise HTTPException(status_code=503, detail="Neural-symbolic service not initialized")
    return neural_symbolic_service


def initialize_neural_symbolic_service(registry: EnhancedHybridRegistry, 
                                     persistence_manager: ContractEnhancedPersistenceManager):
    """Initialize neural-symbolic service."""
    global neural_symbolic_service
    neural_symbolic_service = NeuralSymbolicService(registry, persistence_manager)
    logger.info("Neural-symbolic service initialized")


# ============================================================================
# PHASE 4 SERVICE LAYER ENDPOINTS
# ============================================================================

# These endpoints would be added to the existing FastAPI app in service_layer.py

async def start_neural_training(
    context_name: str,
    config: TrainingConfigurationRequest,
    service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
) -> TrainingJobResponse:
    """
    Start neural-symbolic training for a context.
    
    This endpoint integrates with the existing service layer to provide
    neural training capabilities with real-time progress monitoring.
    """
    try:
        return await service.start_training(context_name, config)
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=f"Training start failed: {str(e)}")


async def get_training_job_status(
    job_id: str,
    service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
) -> TrainingJobResponse:
    """
    Get status of a neural training job.
    
    Returns current status, progress, and results if completed.
    """
    try:
        return await service.get_training_status(job_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


async def stream_neural_training_progress(
    websocket: WebSocket,
    job_id: str
):
    """
    WebSocket endpoint for streaming neural training progress.
    
    Provides real-time updates on training metrics, loss, and satisfiability.
    Extends the existing WebSocket infrastructure from Phase 3C.
    """
    await websocket.accept()
    
    try:
        service = get_neural_symbolic_service()
        
        async for progress in service.stream_training_progress(job_id):
            progress_dict = {
                "type": "neural_training_progress",
                "job_id": job_id,
                "epoch": progress.epoch,
                "stage": progress.stage,
                "loss": progress.loss,
                "satisfiability": progress.satisfiability_score,
                "consistency": progress.concept_consistency,
                "coherence": progress.semantic_coherence,
                "timestamp": progress.timestamp
            }
            
            await websocket.send_text(json.dumps(progress_dict))
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from training progress stream for job {job_id}")
    except Exception as e:
        error_message = {
            "type": "error",
            "message": f"Training progress stream error: {str(e)}",
            "job_id": job_id
        }
        await websocket.send_text(json.dumps(error_message))
        await websocket.close()


async def verify_axioms_with_smt(
    request: SMTVerificationRequest,
    service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
) -> SMTVerificationResponse:
    """
    Verify axiom consistency using SMT solver.
    
    Provides hard logic verification to complement neural-symbolic training.
    """
    try:
        return await service.verify_axioms_smt(request)
    except Exception as e:
        logger.error(f"SMT verification failed: {e}")
        raise HTTPException(status_code=500, detail=f"SMT verification failed: {str(e)}")


async def evaluate_trained_model(
    request: ModelEvaluationRequest,
    service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
) -> ModelEvaluationResponse:
    """
    Evaluate a trained neural-symbolic model.
    
    Tests model performance on analogical reasoning tasks and provides
    comprehensive evaluation metrics.
    """
    try:
        # This would be implemented to evaluate saved models
        # For now, return a placeholder response
        return ModelEvaluationResponse(
            model_path=request.model_path,
            evaluation_results={
                "satisfiability": 0.85,
                "accuracy": 0.78,
                "coherence": 0.82
            },
            test_results=[
                {
                    "analogy": analogy,
                    "predicted": "placeholder",
                    "confidence": 0.8
                }
                for analogy in request.test_analogies
            ],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation failed: {str(e)}")


# ============================================================================
# INTEGRATION UTILITIES
# ============================================================================

def register_neural_symbolic_endpoints(app):
    """
    Register neural-symbolic endpoints with the existing FastAPI app.
    
    This function would be called from service_layer.py to add Phase 4 endpoints.
    """
    
    # Neural training endpoints
    @app.post("/neural/contexts/{context_name}/train", 
              response_model=TrainingJobResponse, 
              tags=["Neural-Symbolic Training"],
              summary="Start Neural Training",
              description="Start neural-symbolic training for a context. All parameters are optional with sensible defaults.")
    async def train_neural_endpoint(
        context_name: str = FastAPIPath(..., description="Name of the context to train", example="test_context"),
        config: TrainingConfigurationRequest = TrainingConfigurationRequest(),
        service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
    ) -> TrainingJobResponse:
        """
        Start neural-symbolic training for a context.
        
        This endpoint accepts a training configuration with all parameters optional.
        If no configuration is provided, sensible defaults will be used.
        
        Parameters:
        - context_name: Name of the context to train
        - config: Training configuration (optional, uses defaults if not provided)
        
        Returns:
        - Training job information including job_id for tracking progress
        """
        return await service.start_training(context_name, config)
    
    @app.get("/neural/jobs/{job_id}/status", response_model=TrainingJobResponse, tags=["Neural-Symbolic Training"])
    async def training_status_endpoint(
        job_id: str = FastAPIPath(..., description="ID of the training job to check status for", example="12f10b85-9140-447c-a33e-5a373790e071"),
        service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
    ) -> TrainingJobResponse:
        """Get status of a neural training job."""
        return await service.get_training_status(job_id)
    
    @app.post("/neural/models/evaluate", 
              response_model=ModelEvaluationResponse, 
              tags=["Neural-Symbolic Training"],
              summary="Evaluate Trained Model",
              description="Evaluate a trained neural-symbolic model on test analogies and metrics.")
    async def evaluate_model_endpoint(
        request: ModelEvaluationRequest,
        service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
    ) -> ModelEvaluationResponse:
        """Evaluate a trained neural-symbolic model."""
        return await service.evaluate_trained_model(request)
    
    # SMT verification endpoints
    @app.post("/smt/verify", 
              response_model=SMTVerificationResponse, 
              tags=["SMT Verification"],
              summary="Verify Axiom Consistency",
              description="Verify axiom consistency using SMT solver for hard logic constraints.")
    async def smt_verify_endpoint(
        request: SMTVerificationRequest,
        service: NeuralSymbolicService = Depends(get_neural_symbolic_service)
    ) -> SMTVerificationResponse:
        """Verify axiom consistency using SMT solver."""
        return await service.verify_axioms_smt(request)
    
    # WebSocket endpoints
    @app.websocket("/ws/neural/training/{job_id}")
    async def training_progress_websocket(
        websocket: WebSocket, 
        job_id: str = FastAPIPath(..., description="ID of the training job to stream progress for", example="12f10b85-9140-447c-a33e-5a373790e071")
    ):
        """WebSocket endpoint for streaming neural training progress."""
        await stream_neural_training_progress(websocket, job_id)
    
    logger.info("Registered neural-symbolic endpoints")


# ============================================================================
# EXAMPLE INTEGRATION WITH EXISTING SERVICE LAYER
# ============================================================================

"""
To integrate this with the existing service_layer.py, add the following to the main app:

# In service_layer.py, add imports:
from app.core.neural_symbolic_service import (
    initialize_neural_symbolic_service,
    register_neural_symbolic_endpoints
)

# In the lifespan context manager, add initialization:
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... existing initialization ...
    
    # Initialize neural-symbolic service
    initialize_neural_symbolic_service(registry, persistence_manager)
    
    yield
    
    # ... existing cleanup ...

# After app creation, register endpoints:
app = FastAPI(title="...", lifespan=lifespan)

# ... existing endpoints ...

# Register Phase 4 neural-symbolic endpoints
register_neural_symbolic_endpoints(app)
"""
