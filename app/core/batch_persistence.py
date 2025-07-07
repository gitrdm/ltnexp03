"""
Batch-Aware Persistence Manager: Production Workflow Support

This module provides production-ready persistence with batch operation support,
JSONL streaming, SQLite transactions, and workflow management.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from contextlib import contextmanager
import gzip

from icontract import require, ensure, invariant, ViolationError
import numpy as np
try:
    import faiss  # type: ignore[import-untyped]
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


class WorkflowStatus(Enum):
    """Workflow processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowType(Enum):
    """Types of batch workflows."""
    ANALOGY_CREATION = "analogy_creation"
    FRAME_EXPANSION = "frame_expansion"
    CONCEPT_BATCH = "concept_batch"
    CLUSTER_UPDATE = "cluster_update"
    DELETION_BATCH = "deletion_batch"


@dataclass
class BatchWorkflow:
    """Represents a batch operation workflow."""
    workflow_id: str
    workflow_type: WorkflowType
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    items_total: int
    items_processed: int = 0
    error_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    error_log: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if self.error_log is None:
            self.error_log = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "workflow_type": self.workflow_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "items_total": self.items_total,
            "items_processed": self.items_processed,
            "error_count": self.error_count,
            "metadata": self.metadata,
            "error_log": self.error_log
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchWorkflow':
        """Create from dictionary."""
        return cls(
            workflow_id=data["workflow_id"],
            workflow_type=WorkflowType(data["workflow_type"]),
            status=WorkflowStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            items_total=data["items_total"],
            items_processed=data.get("items_processed", 0),
            error_count=data.get("error_count", 0),
            metadata=data.get("metadata", {}),
            error_log=data.get("error_log", [])
        )


@dataclass
class DeleteCriteria:
    """Criteria for batch deletion operations."""
    domains: Optional[List[str]] = None
    frame_types: Optional[List[str]] = None
    quality_threshold: Optional[float] = None
    created_before: Optional[datetime] = None
    created_after: Optional[datetime] = None
    tags: Optional[List[str]] = None


@invariant(lambda self: self.storage_path.exists())
@invariant(lambda self: self.storage_path.is_dir())
class BatchPersistenceManager:
    """
    Production-ready persistence manager with batch workflow support.
    
    Provides:
    - JSONL streaming for incremental operations
    - SQLite transactions for ACID compliance
    - Vector indexes for similarity search
    - Workflow management for batch operations
    - Soft deletes with compaction
    """
    
    @require(lambda storage_path: isinstance(storage_path, (str, Path)))
    def __init__(self, storage_path: Union[str, Path]):
        """Initialize batch persistence manager."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage structure
        self._init_storage_structure()
        
        # Initialize databases
        self._init_sqlite_database()
        
        # Initialize vector indexes
        self._init_vector_indexes()
        
        # Active workflows
        self.active_workflows: Dict[str, BatchWorkflow] = {}
        
        # Load existing workflows
        self._load_workflows()
    
    def _init_storage_structure(self) -> None:
        """Initialize the storage directory structure."""
        directories = [
            "contexts/default",
            "contexts/batch_operations",
            "models/ltn_models",
            "models/clustering_models",
            "models/vector_indexes",
            "workflows/analogy_workflows",
            "workflows/frame_workflows",
            "workflows/concept_workflows",
            "exports/compressed",
            "audit"
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def _init_sqlite_database(self) -> None:
        """Initialize SQLite database with schema."""
        db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
        
        with sqlite3.connect(db_path) as conn:
            # Analogies table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analogies (
                    id TEXT PRIMARY KEY,
                    source_domain TEXT NOT NULL,
                    target_domain TEXT NOT NULL,
                    quality_score REAL NOT NULL,
                    concept_mappings TEXT,  -- JSON
                    frame_mappings TEXT,    -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL,
                    version INTEGER DEFAULT 1,
                    tags TEXT,              -- JSON array
                    metadata TEXT           -- JSON
                )
            """)
            
            # Frames table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    frame_type TEXT NOT NULL,
                    definition TEXT,
                    elements TEXT,          -- JSON
                    lexical_units TEXT,     -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL,
                    version INTEGER DEFAULT 1,
                    metadata TEXT           -- JSON
                )
            """)
            
            # Concepts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    context TEXT NOT NULL,
                    synset_id TEXT,
                    frame_roles TEXT,       -- JSON
                    cluster_memberships TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deleted_at TIMESTAMP NULL,
                    version INTEGER DEFAULT 1,
                    metadata TEXT           -- JSON
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analogies_domains ON analogies(source_domain, target_domain)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analogies_quality ON analogies(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analogies_deleted ON analogies(deleted_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_frames_type ON frames(frame_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_concepts_context ON concepts(context)")
            
            conn.commit()
    
    def _init_vector_indexes(self) -> None:
        """Initialize vector indexes for similarity search."""
        # Will be initialized when first embeddings are added
        self.analogy_index = None
        self.concept_index = None
        self.embedding_dim = 300  # Default, will be updated
    
    def _load_workflows(self) -> None:
        """Load existing workflows from storage."""
        workflows_path = self.storage_path / "workflows"
        
        for workflow_file in workflows_path.glob("**/workflow_*.json"):
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                    workflow = BatchWorkflow.from_dict(workflow_data)
                    self.active_workflows[workflow.workflow_id] = workflow
            except Exception as e:
                self.logger.warning(f"Failed to load workflow {workflow_file}: {e}")
    
    # ========================================================================
    # BATCH ANALOGY OPERATIONS
    # ========================================================================
    
    @require(lambda analogies: len(analogies) > 0)
    @ensure(lambda result: result.workflow_type == WorkflowType.ANALOGY_CREATION)
    def create_analogy_batch(self, analogies: List[Dict[str, Any]], 
                           workflow_id: Optional[str] = None) -> BatchWorkflow:
        """Create batch of analogies with workflow tracking."""
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
        
        # Create workflow
        workflow = BatchWorkflow(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.ANALOGY_CREATION,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            items_total=len(analogies),
            metadata={"batch_size": len(analogies)}
        )
        
        # Save workflow
        self._save_workflow(workflow)
        
        # Save analogies to pending batch
        batch_file = self.storage_path / "contexts" / "batch_operations" / f"pending_analogies_{workflow_id}.jsonl"
        
        with open(batch_file, 'w') as f:
            for analogy in analogies:
                analogy_record = {
                    "type": "analogy",
                    "workflow_id": workflow_id,
                    "id": analogy.get("id", str(uuid.uuid4())),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **analogy
                }
                f.write(json.dumps(analogy_record) + '\n')
        
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    def process_analogy_batch(self, workflow_id: str) -> BatchWorkflow:
        """Process pending analogy batch."""
        workflow = self.active_workflows[workflow_id]
        workflow.status = WorkflowStatus.PROCESSING
        workflow.updated_at = datetime.now(timezone.utc)
        
        batch_file = self.storage_path / "contexts" / "batch_operations" / f"pending_analogies_{workflow_id}.jsonl"
        analogies_jsonl = self.storage_path / "contexts" / "default" / "analogies.jsonl"
        
        try:
            # Process analogies
            with open(batch_file, 'r') as batch_f, open(analogies_jsonl, 'a') as main_f:
                for line in batch_f:
                    try:
                        analogy = json.loads(line)
                        
                        # Write to main JSONL
                        main_f.write(json.dumps(analogy) + '\n')
                        
                        # Write to SQLite
                        self._save_analogy_to_sqlite(analogy)
                        
                        workflow.items_processed += 1
                        
                    except Exception as e:
                        workflow.error_count += 1
                        if workflow.error_log is None:
                            workflow.error_log = []
                        workflow.error_log.append(f"Error processing analogy: {str(e)}")
                        self.logger.error(f"Error processing analogy in workflow {workflow_id}: {e}")
            
            # Update workflow status
            if workflow.error_count == 0:
                workflow.status = WorkflowStatus.COMPLETED
            else:
                workflow.status = WorkflowStatus.FAILED if workflow.error_count > workflow.items_processed / 2 else WorkflowStatus.COMPLETED
            
            workflow.updated_at = datetime.now(timezone.utc)
            
            # Clean up batch file
            batch_file.unlink()
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            if workflow.error_log is None:
                workflow.error_log = []
            workflow.error_log.append(f"Batch processing failed: {str(e)}")
            self.logger.error(f"Batch processing failed for workflow {workflow_id}: {e}")
        
        self._save_workflow(workflow)
        return workflow
    
    def _save_analogy_to_sqlite(self, analogy: Dict[str, Any]) -> None:
        """Save analogy to SQLite database."""
        db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analogies 
                (id, source_domain, target_domain, quality_score, concept_mappings, 
                 frame_mappings, created_at, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analogy["id"],
                analogy.get("source_domain", ""),
                analogy.get("target_domain", ""),
                analogy.get("quality_score", 0.0),
                json.dumps(analogy.get("concept_mappings", {})),
                json.dumps(analogy.get("frame_mappings", {})),
                analogy.get("created_at", datetime.now(timezone.utc).isoformat()),
                json.dumps(analogy.get("tags", [])),
                json.dumps(analogy.get("metadata", {}))
            ))
            conn.commit()
    
    # ========================================================================
    # DELETION OPERATIONS
    # ========================================================================
    
    @require(lambda criteria: criteria is not None)
    def delete_analogies_batch(self, criteria: DeleteCriteria, 
                             workflow_id: Optional[str] = None) -> BatchWorkflow:
        """Delete analogies matching criteria."""
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
        
        # Find matching analogies
        matching_analogies = self._find_analogies_by_criteria(criteria)
        
        # Create workflow
        workflow = BatchWorkflow(
            workflow_id=workflow_id,
            workflow_type=WorkflowType.DELETION_BATCH,
            status=WorkflowStatus.PROCESSING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            items_total=len(matching_analogies),
            metadata={"deletion_criteria": asdict(criteria)}
        )
        
        # Perform soft deletes
        db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
        
        try:
            with sqlite3.connect(db_path) as conn:
                for analogy_id in matching_analogies:
                    try:
                        # Soft delete in SQLite
                        conn.execute("""
                            UPDATE analogies 
                            SET deleted_at = ?, version = version + 1 
                            WHERE id = ? AND deleted_at IS NULL
                        """, (datetime.now(timezone.utc), analogy_id))
                        
                        # Add tombstone to JSONL
                        self._add_tombstone_to_jsonl(analogy_id)
                        
                        workflow.items_processed += 1
                        
                    except Exception as e:
                        workflow.error_count += 1
                        if workflow.error_log is None:
                            workflow.error_log = []
                        workflow.error_log.append(f"Error deleting analogy {analogy_id}: {str(e)}")
                
                conn.commit()
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.updated_at = datetime.now(timezone.utc)
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            if workflow.error_log is None:
                workflow.error_log = []
            workflow.error_log.append(f"Deletion batch failed: {str(e)}")
            self.logger.error(f"Deletion batch failed for workflow {workflow_id}: {e}")
        
        self._save_workflow(workflow)
        self.active_workflows[workflow_id] = workflow
        return workflow
    
    def _find_analogies_by_criteria(self, criteria: DeleteCriteria) -> List[str]:
        """Find analogies matching deletion criteria."""
        db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
        
        conditions = ["deleted_at IS NULL"]
        params: List[Union[str, float]] = []
        
        if criteria.domains:
            domain_conditions = []
            for domain in criteria.domains:
                domain_conditions.append("source_domain = ? OR target_domain = ?")
                params.extend([domain, domain])
            conditions.append(f"({' OR '.join(domain_conditions)})")
        
        if criteria.quality_threshold:
            conditions.append("quality_score < ?")
            params.append(criteria.quality_threshold)
        
        if criteria.created_before:
            conditions.append("created_at < ?")
            params.append(criteria.created_before.isoformat())
        
        if criteria.created_after:
            conditions.append("created_at > ?")
            params.append(criteria.created_after.isoformat())
        
        query = f"SELECT id FROM analogies WHERE {' AND '.join(conditions)}"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(query, params)
            return [row[0] for row in cursor.fetchall()]
    
    def _add_tombstone_to_jsonl(self, analogy_id: str) -> None:
        """Add deletion tombstone to JSONL file."""
        analogies_jsonl = self.storage_path / "contexts" / "default" / "analogies.jsonl"
        
        tombstone = {
            "type": "deletion",
            "target_id": analogy_id,
            "deleted_at": datetime.now(timezone.utc).isoformat()
        }
        
        with open(analogies_jsonl, 'a') as f:
            f.write(json.dumps(tombstone) + '\n')
    
    # ========================================================================
    # COMPACTION OPERATIONS
    # ========================================================================
    
    def compact_analogies_jsonl(self) -> Dict[str, Any]:
        """Compact analogies JSONL by removing deleted records."""
        analogies_jsonl = self.storage_path / "contexts" / "default" / "analogies.jsonl"
        
        if not analogies_jsonl.exists():
            return {"status": "no_file", "records_removed": 0}
        
        active_analogies = {}
        deleted_ids = set()
        total_records = 0
        
        # Read all records
        with open(analogies_jsonl, 'r') as f:
            for line in f:
                total_records += 1
                record = json.loads(line)
                
                if record["type"] == "deletion":
                    deleted_ids.add(record["target_id"])
                elif record["type"] == "analogy":
                    active_analogies[record["id"]] = record
        
        # Remove deleted analogies
        for deleted_id in deleted_ids:
            active_analogies.pop(deleted_id, None)
        
        # Create backup
        backup_path = analogies_jsonl.with_suffix('.jsonl.backup')
        analogies_jsonl.rename(backup_path)
        
        # Rewrite file
        with open(analogies_jsonl, 'w') as f:
            for analogy in active_analogies.values():
                f.write(json.dumps(analogy) + '\n')
        
        records_removed = total_records - len(active_analogies)
        
        return {
            "status": "completed",
            "records_removed": records_removed,
            "active_records": len(active_analogies),
            "backup_created": str(backup_path)
        }
    
    # ========================================================================
    # WORKFLOW MANAGEMENT
    # ========================================================================
    
    def get_workflow_status(self, workflow_id: str) -> Optional[BatchWorkflow]:
        """Get current workflow status."""
        return self.active_workflows.get(workflow_id)
    
    def list_workflows(self, status: Optional[WorkflowStatus] = None) -> List[BatchWorkflow]:
        """List workflows, optionally filtered by status."""
        workflows = list(self.active_workflows.values())
        
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return sorted(workflows, key=lambda w: w.updated_at, reverse=True)
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a pending workflow."""
        workflow = self.active_workflows.get(workflow_id)
        
        if not workflow or workflow.status != WorkflowStatus.PENDING:
            return False
        
        workflow.status = WorkflowStatus.CANCELLED
        workflow.updated_at = datetime.now(timezone.utc)
        
        # Clean up pending batch file
        batch_file = self.storage_path / "contexts" / "batch_operations" / f"pending_analogies_{workflow_id}.jsonl"
        if batch_file.exists():
            batch_file.unlink()
        
        self._save_workflow(workflow)
        return True
    
    def _save_workflow(self, workflow: BatchWorkflow) -> None:
        """Save workflow to storage."""
        workflow_dir = self.storage_path / "workflows" / "analogy_workflows"
        workflow_file = workflow_dir / f"workflow_{workflow.workflow_id}.json"
        
        with open(workflow_file, 'w') as f:
            json.dump(workflow.to_dict(), f, indent=2)
    
    # ========================================================================
    # QUERY OPERATIONS
    # ========================================================================
    
    def stream_analogies(self, domain: Optional[str] = None, 
                        min_quality: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        """Stream analogies from JSONL file."""
        analogies_jsonl = self.storage_path / "contexts" / "default" / "analogies.jsonl"
        
        if not analogies_jsonl.exists():
            return
        
        deleted_ids = set()
        
        # First pass: collect deleted IDs
        with open(analogies_jsonl, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "deletion":
                    deleted_ids.add(record["target_id"])
        
        # Second pass: yield active analogies
        with open(analogies_jsonl, 'r') as f:
            for line in f:
                record = json.loads(line)
                
                if record["type"] != "analogy":
                    continue
                
                if record["id"] in deleted_ids:
                    continue
                
                if domain and record.get("source_domain") != domain and record.get("target_domain") != domain:
                    continue
                
                if min_quality and record.get("quality_score", 0.0) < min_quality:
                    continue
                
                yield record
    
    def find_analogies_by_quality(self, min_quality: float) -> List[Dict[str, Any]]:
        """Find high-quality analogies using SQLite."""
        db_path = self.storage_path / "contexts" / "default" / "concepts.sqlite"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT id, source_domain, target_domain, quality_score, 
                       concept_mappings, frame_mappings, created_at, tags, metadata
                FROM analogies 
                WHERE quality_score >= ? AND deleted_at IS NULL
                ORDER BY quality_score DESC
            """, (min_quality,))
            
            analogies = []
            for row in cursor.fetchall():
                analogy = {
                    "id": row[0],
                    "source_domain": row[1],
                    "target_domain": row[2],
                    "quality_score": row[3],
                    "concept_mappings": json.loads(row[4]) if row[4] else {},
                    "frame_mappings": json.loads(row[5]) if row[5] else {},
                    "created_at": row[6],
                    "tags": json.loads(row[7]) if row[7] else [],
                    "metadata": json.loads(row[8]) if row[8] else {}
                }
                analogies.append(analogy)
            
            return analogies
