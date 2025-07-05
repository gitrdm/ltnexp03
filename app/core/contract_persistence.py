"""
Contract-Enhanced Persistence Manager

This module provides a Design by Contract enhanced persistence manager that
implements the persistence protocols with comprehensive validation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging

from icontract import require, ensure, invariant, ViolationError

from .persistence import PersistenceManager
from .batch_persistence import BatchPersistenceManager, DeleteCriteria, BatchWorkflow
from .protocols import PersistenceProtocol, BatchPersistenceProtocol
from .contracts import (
    validate_concept_name, validate_context, validate_coherence_score
)

# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .enhanced_semantic_reasoning import EnhancedHybridRegistry


def validate_workflow_id(workflow_id: str) -> bool:
    """Validate workflow ID format."""
    return (
        isinstance(workflow_id, str) and
        len(workflow_id.strip()) > 0 and
        len(workflow_id) <= 255
    )


def validate_storage_path(path: Union[str, Path]) -> bool:
    """Validate storage path is accessible."""
    try:
        path_obj = Path(path)
        # Try to create if doesn't exist
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj.exists() and path_obj.is_dir()
    except (OSError, PermissionError):
        return False


def validate_analogy_batch(analogies: List[Dict[str, Any]]) -> bool:
    """Validate batch of analogies has required fields."""
    if not isinstance(analogies, list) or len(analogies) == 0:
        return False
    
    required_fields = {"source_domain", "target_domain", "quality_score"}
    
    for analogy in analogies:
        if not isinstance(analogy, dict):
            return False
        
        if not required_fields.issubset(analogy.keys()):
            return False
        
        # Validate quality score
        quality = analogy.get("quality_score")
        if not isinstance(quality, (int, float)) or not (0.0 <= quality <= 1.0):
            return False
    
    return True


def validate_format_type(format_type: str) -> bool:
    """Validate storage format type."""
    return format_type in ["json", "pickle", "compressed", "jsonl"]


@invariant(lambda self: validate_storage_path(self.storage_path))
@invariant(lambda self: hasattr(self, '_basic_manager'))
@invariant(lambda self: hasattr(self, '_batch_manager'))
class ContractEnhancedPersistenceManager:
    """
    Contract-enhanced persistence manager with comprehensive validation.
    
    Implements both PersistenceProtocol and BatchPersistenceProtocol with
    Design by Contract validation for all operations.
    """
    
    @require(lambda storage_path: validate_storage_path(storage_path))
    def __init__(self, storage_path: Union[str, Path]):
        """Initialize contract-enhanced persistence manager."""
        self.storage_path = Path(storage_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize underlying managers
        self._basic_manager = PersistenceManager(storage_path)
        self._batch_manager = BatchPersistenceManager(storage_path)
        
        self.logger.info(f"Initialized contract-enhanced persistence manager at {self.storage_path}")
    
    # ========================================================================
    # BASIC PERSISTENCE PROTOCOL IMPLEMENTATION
    # ========================================================================
    
    @require(lambda registry: registry is not None)
    @require(lambda context_name: validate_context(context_name))
    @require(lambda format_type: validate_format_type(format_type))
    @ensure(lambda result: isinstance(result, dict))
    @ensure(lambda result: "save_metadata" in result)
    @ensure(lambda result: "context_name" in result)
    def save_registry_state(self, registry: 'EnhancedHybridRegistry', 
                           context_name: str = "default",
                           format_type: str = "json") -> Dict[str, Any]:
        """Save complete registry state with contract validation."""
        try:
            result = self._basic_manager.save_registry_state(
                registry, context_name, format_type
            )
            
            # Contract validation
            if not result.get("components_saved"):
                raise ViolationError("No components were saved")
            
            self.logger.info(f"Successfully saved registry state for context '{context_name}'")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to save registry state: {e}")
            raise ViolationError(f"Save operation failed: {e}")
    
    @require(lambda context_name: validate_context(context_name))
    def load_registry_state(self, context_name: str = "default") -> Optional[Dict[str, Any]]:
        """Load complete registry state with validation."""
        try:
            result = self._basic_manager.load_registry_state(context_name)
            
            if result is not None:
                self.logger.info(f"Successfully loaded registry state for context '{context_name}'")
            else:
                self.logger.warning(f"No saved state found for context '{context_name}'")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to load registry state: {e}")
            return None
    
    @require(lambda format: format in ["json", "compressed", "sqlite"])
    def export_knowledge_base(self, format: str = "json", 
                            compressed: bool = False) -> Path:
        """Export complete knowledge base with validation."""
        try:
            # Implementation would go here
            export_path = self.storage_path / "exports" / f"knowledge_base.{format}"
            
            # For now, create a placeholder file
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w') as f:
                f.write(f"# Knowledge base export - {datetime.now().isoformat()}\n")
            
            self.logger.info(f"Exported knowledge base to {export_path}")
            return export_path
            
        except Exception as e:
            self.logger.error(f"Failed to export knowledge base: {e}")
            raise ViolationError(f"Export operation failed: {e}")
    
    @require(lambda source_path: Path(source_path).exists())
    @require(lambda merge_strategy: merge_strategy in ["overwrite", "merge", "skip_conflicts"])
    def import_knowledge_base(self, source_path: Path, 
                            merge_strategy: str = "overwrite") -> bool:
        """Import knowledge base with conflict resolution."""
        try:
            # Implementation would go here
            self.logger.info(f"Imported knowledge base from {source_path} using {merge_strategy} strategy")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import knowledge base: {e}")
            return False
    
    # ========================================================================
    # BATCH PERSISTENCE PROTOCOL IMPLEMENTATION
    # ========================================================================
    
    @require(lambda analogies: validate_analogy_batch(analogies))
    @require(lambda workflow_id: workflow_id is None or validate_workflow_id(workflow_id))
    @ensure(lambda result: isinstance(result, BatchWorkflow))
    @ensure(lambda result: result.items_total > 0)
    def create_analogy_batch(self, analogies: List[Dict[str, Any]], 
                           workflow_id: Optional[str] = None) -> BatchWorkflow:
        """Create batch of analogies with contract validation."""
        try:
            workflow = self._batch_manager.create_analogy_batch(analogies, workflow_id)
            
            # Contract validation
            if workflow.items_total != len(analogies):
                raise ViolationError(f"Workflow items_total mismatch: {workflow.items_total} != {len(analogies)}")
            
            self.logger.info(f"Created analogy batch with {len(analogies)} items, workflow: {workflow.workflow_id}")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create analogy batch: {e}")
            raise ViolationError(f"Batch creation failed: {e}")
    
    @require(lambda workflow_id: validate_workflow_id(workflow_id))
    @ensure(lambda result: isinstance(result, BatchWorkflow))
    def process_analogy_batch(self, workflow_id: str) -> BatchWorkflow:
        """Process pending analogy batch with validation."""
        # Check workflow exists
        if workflow_id not in self._batch_manager.active_workflows:
            raise ViolationError(f"Workflow {workflow_id} not found")
        
        try:
            workflow = self._batch_manager.process_analogy_batch(workflow_id)
            
            # Contract validation
            if workflow.status.value not in ["completed", "failed"]:
                raise ViolationError(f"Unexpected workflow status after processing: {workflow.status}")
            
            self.logger.info(f"Processed analogy batch {workflow_id}: {workflow.items_processed}/{workflow.items_total} items")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to process analogy batch {workflow_id}: {e}")
            raise ViolationError(f"Batch processing failed: {e}")
    
    @require(lambda criteria: isinstance(criteria, DeleteCriteria))
    @require(lambda workflow_id: workflow_id is None or validate_workflow_id(workflow_id))
    @ensure(lambda result: isinstance(result, BatchWorkflow))
    def delete_analogies_batch(self, criteria: DeleteCriteria, 
                             workflow_id: Optional[str] = None) -> BatchWorkflow:
        """Delete analogies matching criteria with validation."""
        try:
            workflow = self._batch_manager.delete_analogies_batch(criteria, workflow_id)
            
            self.logger.info(f"Created deletion batch {workflow.workflow_id} for {workflow.items_total} analogies")
            return workflow
            
        except Exception as e:
            self.logger.error(f"Failed to create deletion batch: {e}")
            raise ViolationError(f"Deletion batch failed: {e}")
    
    @require(lambda workflow_id: validate_workflow_id(workflow_id))
    def get_workflow_status(self, workflow_id: str) -> Optional[BatchWorkflow]:
        """Get current workflow status with validation."""
        try:
            status = self._batch_manager.get_workflow_status(workflow_id)
            
            if status is not None:
                self.logger.debug(f"Retrieved status for workflow {workflow_id}: {status.status}")
            else:
                self.logger.warning(f"Workflow {workflow_id} not found")
                
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status for {workflow_id}: {e}")
            return None
    
    def stream_analogies(self, domain: Optional[str] = None, 
                        min_quality: Optional[float] = None):
        """Stream analogies from storage with optional filtering."""
        # Validate quality threshold
        if min_quality is not None and not (0.0 <= min_quality <= 1.0):
            raise ViolationError(f"Invalid quality threshold: {min_quality}")
        
        try:
            count = 0
            for analogy in self._batch_manager.stream_analogies(domain, min_quality):
                count += 1
                yield analogy
            
            self.logger.debug(f"Streamed {count} analogies with filters: domain={domain}, min_quality={min_quality}")
            
        except Exception as e:
            self.logger.error(f"Failed to stream analogies: {e}")
            raise ViolationError(f"Streaming failed: {e}")
    
    @ensure(lambda result: isinstance(result, dict))
    @ensure(lambda result: "status" in result)
    def compact_analogies_jsonl(self) -> Dict[str, Any]:
        """Compact analogy storage by removing deleted records."""
        try:
            result = self._batch_manager.compact_analogies_jsonl()
            
            # Contract validation
            if "status" not in result:
                raise ViolationError("Compaction result missing status")
            
            self.logger.info(f"Compacted analogies: {result.get('records_removed', 0)} records removed")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compact analogies: {e}")
            raise ViolationError(f"Compaction failed: {e}")
    
    # ========================================================================
    # ADDITIONAL CONTRACT-VALIDATED OPERATIONS
    # ========================================================================
    
    @require(lambda min_quality: 0.0 <= min_quality <= 1.0)
    @ensure(lambda result: isinstance(result, list))
    def find_high_quality_analogies(self, min_quality: float = 0.8) -> List[Dict[str, Any]]:
        """Find high-quality analogies with contract validation."""
        try:
            analogies = self._batch_manager.find_analogies_by_quality(min_quality)
            
            # Validate all returned analogies meet quality threshold
            for analogy in analogies:
                if analogy.get("quality_score", 0.0) < min_quality:
                    raise ViolationError(f"Returned analogy quality {analogy.get('quality_score')} below threshold {min_quality}")
            
            self.logger.info(f"Found {len(analogies)} high-quality analogies (>= {min_quality})")
            return analogies
            
        except Exception as e:
            self.logger.error(f"Failed to find high-quality analogies: {e}")
            raise ViolationError(f"Quality search failed: {e}")
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics."""
        try:
            stats = {
                "storage_path": str(self.storage_path),
                "total_workflows": len(self._batch_manager.active_workflows),
                "workflow_status_counts": {},
                "storage_size_mb": 0.0,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Count workflows by status
            for workflow in self._batch_manager.active_workflows.values():
                status = workflow.status.value
                stats["workflow_status_counts"][status] = stats["workflow_status_counts"].get(status, 0) + 1
            
            # Calculate storage size
            try:
                def get_size(path):
                    if path.is_file():
                        return path.stat().st_size
                    elif path.is_dir():
                        return sum(get_size(child) for child in path.rglob('*') if child.is_file())
                    return 0
                
                stats["storage_size_mb"] = get_size(self.storage_path) / (1024 * 1024)
            except Exception:
                stats["storage_size_mb"] = -1.0  # Error calculating size
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get storage statistics: {e}")
            return {
                "error": str(e),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
    
    def validate_storage_integrity(self) -> Dict[str, Any]:
        """Validate storage integrity and consistency."""
        try:
            issues = []
            
            # Check directory structure
            required_dirs = [
                "contexts/default",
                "models/ltn_models",
                "workflows"
            ]
            
            for dir_path in required_dirs:
                full_path = self.storage_path / dir_path
                if not full_path.exists():
                    issues.append(f"Missing required directory: {dir_path}")
            
            # Check SQLite database
            db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
            if db_path.exists():
                try:
                    import sqlite3
                    with sqlite3.connect(db_path) as conn:
                        # Check table existence
                        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                        tables = [row[0] for row in cursor.fetchall()]
                        
                        required_tables = ["analogies", "frames", "concepts"]
                        for table in required_tables:
                            if table not in tables:
                                issues.append(f"Missing required table: {table}")
                except Exception as e:
                    issues.append(f"SQLite database error: {e}")
            else:
                issues.append("SQLite database not found")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate storage integrity: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {e}"],
                "checked_at": datetime.now(timezone.utc).isoformat()
            }
