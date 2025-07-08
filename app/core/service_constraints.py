"""
Service Layer Contract Constraints
=================================

This module provides contract constraints specifically for service layer operations,
extending the core contract framework to cover API-level business rules and
resource management requirements.

These constraints bridge the gap between Pydantic validation (data format) and
business logic validation (domain rules).
"""

from typing import Dict, Any, List, Optional
import re
import uuid
from datetime import datetime


class ServiceConstraints:
    """Contract constraints for service layer operations."""
    
    @staticmethod
    def valid_batch_size(size: int) -> bool:
        """Validate batch operation size limits."""
        return isinstance(size, int) and 1 <= size <= 1000
    
    @staticmethod
    def valid_similarity_threshold(threshold: float) -> bool:
        """Validate similarity threshold range."""
        return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0
    
    @staticmethod
    def valid_confidence_threshold(confidence: float) -> bool:
        """Validate confidence threshold range."""
        return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    
    @staticmethod
    def valid_max_results(max_results: int) -> bool:
        """Validate maximum results parameter."""
        return isinstance(max_results, int) and 1 <= max_results <= 100
    
    @staticmethod
    def valid_workflow_id(workflow_id: str) -> bool:
        """Validate workflow ID format."""
        if not isinstance(workflow_id, str) or len(workflow_id.strip()) == 0:
            return False
        
        # Allow both UUID format and alphanumeric IDs
        try:
            uuid.UUID(workflow_id)
            return True
        except ValueError:
            # Allow alphanumeric workflow IDs (legacy format)
            return bool(re.match(r'^[a-zA-Z0-9_-]+$', workflow_id)) and len(workflow_id) <= 50
    
    @staticmethod
    def valid_job_id(job_id: str) -> bool:
        """Validate training job ID format (must be UUID)."""
        if not isinstance(job_id, str):
            return False
        try:
            uuid.UUID(job_id)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def valid_context_name(context_name: str) -> bool:
        """Validate neural training context name."""
        return (isinstance(context_name, str) and 
                len(context_name.strip()) > 0 and 
                len(context_name.strip()) <= 50 and
                bool(re.match(r'^[a-zA-Z0-9_-]+$', context_name)))
    
    @staticmethod
    def valid_analogy_request(analogy: Dict[str, Any]) -> bool:
        """Validate analogy completion request structure."""
        if not isinstance(analogy, dict):
            return False
        
        required_fields = {"source_domain", "target_domain"}
        if not required_fields.issubset(analogy.keys()):
            return False
        
        # Validate source and target domains
        source = analogy.get("source_domain")
        target = analogy.get("target_domain")
        
        if not (isinstance(source, str) and isinstance(target, str)):
            return False
        
        if len(source.strip()) == 0 or len(target.strip()) == 0:
            return False
        
        # Validate optional quality score
        if "quality_score" in analogy:
            score = analogy["quality_score"]
            if not (isinstance(score, (int, float)) and 0.0 <= score <= 1.0):
                return False
        
        return True
    
    @staticmethod
    def valid_frame_instance_bindings(bindings: Dict[str, str]) -> bool:
        """Validate frame instance concept bindings."""
        if not isinstance(bindings, dict) or len(bindings) == 0:
            return False
        
        # All keys and values must be non-empty strings
        for key, value in bindings.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                return False
            if len(key.strip()) == 0 or len(value.strip()) == 0:
                return False
        
        # Reasonable limit on number of bindings
        return len(bindings) <= 20
    
    @staticmethod
    def valid_search_query(query: str) -> bool:
        """Validate search query format."""
        return (isinstance(query, str) and 
                len(query.strip()) > 0 and 
                len(query.strip()) <= 200)
    
    @staticmethod
    def valid_domain_threshold(threshold: float) -> bool:
        """Validate cross-domain similarity threshold range."""
        return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0
    
    @staticmethod
    def valid_frame_instance(instance_data: Dict[str, Any]) -> bool:
        """Validate frame instance structure and content."""
        if not isinstance(instance_data, dict) or len(instance_data) == 0:
            return False
        
        # Must have at least one element binding
        if len(instance_data) == 0:
            return False
        
        # All keys and values must be non-empty strings
        for key, value in instance_data.items():
            if not (isinstance(key, str) and isinstance(value, str)):
                return False
            if len(key.strip()) == 0 or len(value.strip()) == 0:
                return False
        
        # Reasonable limit on number of bindings
        return len(instance_data) <= 20
    
    @staticmethod
    def valid_query_filter(filter_params: Dict[str, Any]) -> bool:
        """Validate frame query filter parameters."""
        if not isinstance(filter_params, dict):
            return False
        
        # Empty filters are valid (return all)
        if len(filter_params) == 0:
            return True
        
        # Validate specific filter types
        valid_filter_keys = {"domain", "core_elements_count", "created_after", "tags"}
        
        for key, value in filter_params.items():
            if key not in valid_filter_keys:
                return False
                
            if key == "domain" and not isinstance(value, str):
                return False
            elif key == "core_elements_count" and not isinstance(value, int):
                return False
            elif key == "created_after" and not isinstance(value, str):
                return False
            elif key == "tags" and not isinstance(value, list):
                return False
        
        return True
    
    @staticmethod
    def valid_axiom_list(axioms: List[Dict[str, Any]]) -> bool:
        """Validate axiom list for SMT verification."""
        if not isinstance(axioms, list) or len(axioms) == 0:
            return False
        
        # Must have between 1 and 50 axioms
        if not (1 <= len(axioms) <= 50):
            return False
        
        # Each axiom must be a dictionary with required fields
        for axiom in axioms:
            if not isinstance(axiom, dict):
                return False
            
            # Must have 'formula' field
            if 'formula' not in axiom or not isinstance(axiom['formula'], str):
                return False
            
            # Formula must be non-empty
            if len(axiom['formula'].strip()) == 0:
                return False
            
            # Optional: validate axiom structure
            if 'name' in axiom and not isinstance(axiom['name'], str):
                return False
        
        return True
    
    @staticmethod
    def valid_pagination_limit(limit: Optional[int]) -> bool:
        """Validate pagination limit parameter."""
        if limit is None:
            return True  # No limit is valid
        return isinstance(limit, int) and 1 <= limit <= 1000
    
    @staticmethod
    def valid_status_filter(status: Optional[str]) -> bool:
        """Validate workflow status filter parameter."""
        if status is None:
            return True  # No filter is valid
        if not isinstance(status, str):
            return False
        valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        return status.lower() in valid_statuses
    
    @staticmethod
    def valid_system_component_name(component: str) -> bool:
        """Validate system component name."""
        if not isinstance(component, str) or len(component.strip()) == 0:
            return False
        valid_components = {"semantic_registry", "persistence_manager", "batch_manager", "neural_service"}
        return component.lower() in valid_components
    
    @staticmethod
    def valid_evaluation_metric(metric: str) -> bool:
        """Validate model evaluation metric name."""
        if not isinstance(metric, str) or len(metric.strip()) == 0:
            return False
        valid_metrics = {"accuracy", "precision", "recall", "f1_score", "bleu", "rouge", "perplexity"}
        return metric.lower() in valid_metrics
    
    @staticmethod
    def valid_model_path(model_path: str) -> bool:
        """Validate model file path."""
        if not isinstance(model_path, str) or len(model_path.strip()) == 0:
            return False
        # Basic path validation - no directory traversal
        return not ('..' in model_path or model_path.startswith('/') or '\\' in model_path)
    
    @staticmethod
    def valid_websocket_domain_filter(domain: Optional[str]) -> bool:
        """Validate WebSocket domain filter parameter."""
        if domain is None:
            return True  # No filter is valid
        if not isinstance(domain, str):
            return False
        # Domain should be a valid identifier
        return len(domain.strip()) > 0 and len(domain.strip()) <= 50 and domain.replace('_', '').replace('-', '').isalnum()
    
    @staticmethod
    def valid_quality_threshold(quality: Optional[float]) -> bool:
        """Validate quality threshold for streaming filters."""
        if quality is None:
            return True  # No threshold is valid
        return isinstance(quality, (int, float)) and 0.0 <= quality <= 1.0
    
    @staticmethod
    def valid_documentation_response(response: Dict[str, Any]) -> bool:
        """Validate documentation endpoint response structure."""
        if not isinstance(response, dict):
            return False
        required_sections = {"api_overview", "endpoints"}
        return all(section in response for section in required_sections)
        

class ResourceConstraints:
    """Contract constraints for resource management."""
    
    @staticmethod
    def sufficient_memory_available() -> bool:
        """Check if sufficient memory is available for operations."""
        # In production, implement actual memory checking
        # For now, assume memory is available
        return True
    
    @staticmethod
    def registry_initialized() -> bool:
        """Check if semantic registry is properly initialized."""
        # This will be implemented in the service layer with actual registry checks
        return True
    
    @staticmethod
    def storage_available() -> bool:
        """Check if storage system is available."""
        # In production, implement actual storage checking
        return True
    
    @staticmethod
    def neural_service_available() -> bool:
        """Check if neural-symbolic service is available."""
        # This will be checked at the service layer
        return True


class WorkflowConstraints:
    """Contract constraints for workflow operations."""
    
    @staticmethod
    def valid_workflow_status(status: str) -> bool:
        """Validate workflow status values."""
        valid_statuses = {"pending", "running", "completed", "failed", "cancelled"}
        return isinstance(status, str) and status.lower() in valid_statuses
    
    @staticmethod
    def valid_batch_workflow(workflow: Dict[str, Any]) -> bool:
        """Validate batch workflow structure."""
        if not isinstance(workflow, dict):
            return False
        
        required_fields = {"workflow_id", "status", "created_at"}
        if not required_fields.issubset(workflow.keys()):
            return False
        
        # Validate workflow ID
        if not ServiceConstraints.valid_workflow_id(workflow["workflow_id"]):
            return False
        
        # Validate status
        if not WorkflowConstraints.valid_workflow_status(workflow["status"]):
            return False
        
        # Validate timestamp format
        created_at = workflow.get("created_at")
        if isinstance(created_at, str):
            try:
                datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                return False
        
        return True


# Re-export core constraints for convenience
from .contracts import ConceptConstraints, EmbeddingConstraints

__all__ = [
    'ServiceConstraints',
    'ResourceConstraints', 
    'WorkflowConstraints',
    'ConceptConstraints',
    'EmbeddingConstraints'
]
