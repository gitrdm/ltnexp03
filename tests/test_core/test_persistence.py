"""
Comprehensive Test Suite for Persistence Layer

This module provides complete test coverage for the persistence layer,
including unit tests, integration tests, contract validation, and workflow testing.
"""

import pytest
import tempfile
import shutil
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any
from unittest.mock import Mock, patch

import numpy as np
from icontract import ViolationError

from app.core.persistence import PersistenceManager, StorageFormat
from app.core.batch_persistence import (
    BatchPersistenceManager, BatchWorkflow, DeleteCriteria,
    WorkflowStatus, WorkflowType
)
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.abstractions import Concept
from app.core.frame_cluster_abstractions import SemanticFrame, FrameElement, FrameElementType


class TestStorageFormat:
    """Test storage format validation."""
    
    def test_validate_format_valid(self):
        """Test valid format validation."""
        assert StorageFormat.validate_format("json")
        assert StorageFormat.validate_format("pickle")
        assert StorageFormat.validate_format("compressed")
    
    def test_validate_format_invalid(self):
        """Test invalid format validation."""
        assert not StorageFormat.validate_format("xml")
        assert not StorageFormat.validate_format("yaml")
        assert not StorageFormat.validate_format("")
    
    def test_get_file_extension(self):
        """Test file extension mapping."""
        assert StorageFormat.get_file_extension("json") == ".json"
        assert StorageFormat.get_file_extension("pickle") == ".pkl"
        assert StorageFormat.get_file_extension("compressed") == ".json.gz"
        assert StorageFormat.get_file_extension("unknown") == ".json"


class TestPersistenceManager:
    """Test the core persistence manager."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def persistence_manager(self, temp_storage):
        """Create persistence manager with temporary storage."""
        return PersistenceManager(temp_storage)
    
    @pytest.fixture
    def sample_registry(self):
        """Create sample registry for testing."""
        registry = EnhancedHybridRegistry(
            download_wordnet=False,
            n_clusters=4,
            enable_cross_domain=True,
            embedding_provider="random"
        )
        
        # Add some test concepts (use frame-aware concepts for proper persistence)
        registry.create_frame_aware_concept("king", "medieval")
        registry.create_frame_aware_concept("queen", "medieval")
        registry.create_frame_aware_concept("general", "military")
        
        return registry
    
    def test_initialization(self, persistence_manager, temp_storage):
        """Test persistence manager initialization."""
        assert persistence_manager.storage_path == temp_storage
        assert persistence_manager.storage_path.exists()
        
        # Check directory structure
        expected_dirs = [
            "contexts", "models", "models/clustering_models",
            "models/embedding_models", "exports", "exports/compressed", "audit"
        ]
        
        for dir_name in expected_dirs:
            assert (temp_storage / dir_name).exists()
    
    def test_save_registry_state_json(self, persistence_manager, sample_registry):
        """Test saving registry state in JSON format."""
        result = persistence_manager.save_registry_state(
            sample_registry, "test_context", "json"
        )
        
        # Check that the result has the expected metadata structure
        assert "context_name" in result
        assert result["context_name"] == "test_context"
        assert result["format"] == "json"
        assert "components_saved" in result
        assert "concepts" in result["components_saved"]
        
        # Check files were created
        context_dir = persistence_manager.storage_path / "contexts" / "test_context"
        assert context_dir.exists()
        assert (context_dir / "concepts.json").exists()
    
    def test_save_registry_state_invalid_format(self, persistence_manager, sample_registry):
        """Test saving with invalid format raises error."""
        with pytest.raises(ViolationError):
            persistence_manager.save_registry_state(
                sample_registry, "test_context", "invalid_format"
            )
    
    def test_save_load_roundtrip(self, persistence_manager, sample_registry):
        """Test save/load roundtrip preserves data."""
        # Save registry
        save_result = persistence_manager.save_registry_state(
            sample_registry, "roundtrip_test", "json"
        )
        
        # Load registry (implementation would go here)
        # For now, just verify files exist and are valid JSON
        context_dir = persistence_manager.storage_path / "contexts" / "roundtrip_test"
        
        with open(context_dir / "concepts.json", 'r') as f:
            concepts_data = json.load(f)
            assert "concepts" in concepts_data
            assert len(concepts_data["concepts"]) >= 3  # We added 3 concepts
    
    def test_contract_validation_invalid_context(self, persistence_manager, sample_registry):
        """Test contract validation for invalid context names."""
        # This would test the SoftLogicContracts.valid_context validation
        # when that's properly imported and working
        pass


class TestBatchPersistenceManager:
    """Test the batch-aware persistence manager."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def batch_persistence_manager(self, temp_storage):
        """Create batch persistence manager with temporary storage."""
        return BatchPersistenceManager(temp_storage)
    
    @pytest.fixture
    def sample_analogies(self):
        """Create sample analogies for testing."""
        return [
            {
                "source_domain": "royal",
                "target_domain": "military",
                "quality_score": 0.85,
                "concept_mappings": {"king": "general", "queen": "lieutenant"},
                "frame_mappings": {"hierarchy": "command_structure"}
            },
            {
                "source_domain": "royal",
                "target_domain": "business",
                "quality_score": 0.78,
                "concept_mappings": {"king": "ceo", "queen": "cfo"},
                "frame_mappings": {"hierarchy": "corporate_structure"}
            },
            {
                "source_domain": "military",
                "target_domain": "business",
                "quality_score": 0.65,
                "concept_mappings": {"general": "ceo", "sergeant": "manager"},
                "frame_mappings": {"command_structure": "corporate_structure"}
            }
        ]
    
    def test_initialization(self, batch_persistence_manager, temp_storage):
        """Test batch persistence manager initialization."""
        assert batch_persistence_manager.storage_path == temp_storage
        assert batch_persistence_manager.storage_path.exists()
        
        # Check SQLite database was created
        db_path = temp_storage / "contexts" / "default" / "concepts.sqlite"
        assert db_path.exists()
        
        # Check tables were created
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert "analogies" in tables
            assert "frames" in tables
            assert "concepts" in tables
    
    def test_create_analogy_batch(self, batch_persistence_manager, sample_analogies):
        """Test creating a batch of analogies."""
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        
        assert workflow.workflow_type == WorkflowType.ANALOGY_CREATION
        assert workflow.status == WorkflowStatus.PENDING
        assert workflow.items_total == len(sample_analogies)
        assert workflow.items_processed == 0
        
        # Check workflow is stored
        assert workflow.workflow_id in batch_persistence_manager.active_workflows
        
        # Check batch file was created
        batch_file = batch_persistence_manager.storage_path / "contexts" / "batch_operations" / f"pending_analogies_{workflow.workflow_id}.jsonl"
        assert batch_file.exists()
        
        # Verify batch file contents
        with open(batch_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == len(sample_analogies)
    
    def test_process_analogy_batch(self, batch_persistence_manager, sample_analogies):
        """Test processing a batch of analogies."""
        # Create batch
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        
        # Process batch
        processed_workflow = batch_persistence_manager.process_analogy_batch(workflow.workflow_id)
        
        assert processed_workflow.status == WorkflowStatus.COMPLETED
        assert processed_workflow.items_processed == len(sample_analogies)
        assert processed_workflow.error_count == 0
        
        # Check analogies were written to main JSONL
        analogies_jsonl = batch_persistence_manager.storage_path / "contexts" / "default" / "analogies.jsonl"
        assert analogies_jsonl.exists()
        
        with open(analogies_jsonl, 'r') as f:
            lines = f.readlines()
            assert len(lines) == len(sample_analogies)
        
        # Check analogies were written to SQLite
        db_path = batch_persistence_manager.storage_path / "contexts" / "default" / "concepts.sqlite"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM analogies")
            count = cursor.fetchone()[0]
            assert count == len(sample_analogies)
    
    def test_delete_analogies_batch(self, batch_persistence_manager, sample_analogies):
        """Test batch deletion of analogies."""
        # First create and process analogies
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        batch_persistence_manager.process_analogy_batch(workflow.workflow_id)
        
        # Delete analogies by domain
        criteria = DeleteCriteria(domains=["royal"])
        delete_workflow = batch_persistence_manager.delete_analogies_batch(criteria)
        
        assert delete_workflow.workflow_type == WorkflowType.DELETION_BATCH
        assert delete_workflow.status == WorkflowStatus.COMPLETED
        assert delete_workflow.items_processed == 2  # Two analogies have "royal" domain
        
        # Check soft deletes in SQLite
        db_path = batch_persistence_manager.storage_path / "contexts" / "default" / "concepts.sqlite"
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM analogies WHERE deleted_at IS NOT NULL")
            deleted_count = cursor.fetchone()[0]
            assert deleted_count == 2
    
    def test_stream_analogies(self, batch_persistence_manager, sample_analogies):
        """Test streaming analogies with filtering."""
        # Create and process analogies
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        batch_persistence_manager.process_analogy_batch(workflow.workflow_id)
        
        # Stream all analogies
        all_analogies = list(batch_persistence_manager.stream_analogies())
        assert len(all_analogies) == len(sample_analogies)
        
        # Stream by domain
        royal_analogies = list(batch_persistence_manager.stream_analogies(domain="royal"))
        assert len(royal_analogies) == 2  # Two analogies involve "royal" domain
        
        # Stream by quality
        high_quality = list(batch_persistence_manager.stream_analogies(min_quality=0.8))
        assert len(high_quality) == 1  # Only one analogy has quality >= 0.8
    
    def test_compact_analogies_jsonl(self, batch_persistence_manager, sample_analogies):
        """Test compaction of JSONL file."""
        # Create, process, and delete some analogies
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        batch_persistence_manager.process_analogy_batch(workflow.workflow_id)
        
        criteria = DeleteCriteria(domains=["military"])
        batch_persistence_manager.delete_analogies_batch(criteria)
        
        # Compact
        result = batch_persistence_manager.compact_analogies_jsonl()
        
        assert result["status"] == "completed"
        assert result["records_removed"] > 0
        assert result["active_records"] < len(sample_analogies)
        assert "backup_created" in result
    
    def test_workflow_management(self, batch_persistence_manager, sample_analogies):
        """Test workflow status tracking and management."""
        # Create workflow
        workflow = batch_persistence_manager.create_analogy_batch(sample_analogies)
        
        # Get status
        status = batch_persistence_manager.get_workflow_status(workflow.workflow_id)
        assert status is not None
        assert status.status == WorkflowStatus.PENDING
        
        # List workflows
        workflows = batch_persistence_manager.list_workflows()
        assert len(workflows) >= 1
        assert workflow.workflow_id in [w.workflow_id for w in workflows]
        
        # Cancel workflow
        success = batch_persistence_manager.cancel_workflow(workflow.workflow_id)
        assert success
        
        # Check status after cancellation
        status = batch_persistence_manager.get_workflow_status(workflow.workflow_id)
        assert status.status == WorkflowStatus.CANCELLED


class TestBatchWorkflow:
    """Test BatchWorkflow dataclass functionality."""
    
    def test_workflow_creation(self):
        """Test creating a BatchWorkflow."""
        workflow = BatchWorkflow(
            workflow_id="test_workflow_001",
            workflow_type=WorkflowType.ANALOGY_CREATION,
            status=WorkflowStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            items_total=10
        )
        
        assert workflow.workflow_id == "test_workflow_001"
        assert workflow.items_processed == 0
        assert workflow.error_count == 0
        assert workflow.metadata == {}
        assert workflow.error_log == []
    
    def test_workflow_serialization(self):
        """Test workflow serialization to/from dict."""
        workflow = BatchWorkflow(
            workflow_id="test_workflow_002",
            workflow_type=WorkflowType.FRAME_EXPANSION,
            status=WorkflowStatus.PROCESSING,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            items_total=5,
            items_processed=2,
            error_count=1,
            metadata={"batch_size": 5},
            error_log=["Error processing item 3"]
        )
        
        # Serialize
        workflow_dict = workflow.to_dict()
        assert workflow_dict["workflow_id"] == "test_workflow_002"
        assert workflow_dict["workflow_type"] == "frame_expansion"
        assert workflow_dict["status"] == "processing"
        
        # Deserialize
        restored_workflow = BatchWorkflow.from_dict(workflow_dict)
        assert restored_workflow.workflow_id == workflow.workflow_id
        assert restored_workflow.workflow_type == workflow.workflow_type
        assert restored_workflow.status == workflow.status
        assert restored_workflow.items_total == workflow.items_total


class TestDeleteCriteria:
    """Test DeleteCriteria functionality."""
    
    def test_criteria_creation(self):
        """Test creating DeleteCriteria."""
        criteria = DeleteCriteria(
            domains=["royal", "military"],
            quality_threshold=0.5,
            created_before=datetime.now(timezone.utc)
        )
        
        assert criteria.domains == ["royal", "military"]
        assert criteria.quality_threshold == 0.5
        assert criteria.frame_types is None
        assert criteria.tags is None


class TestIntegrationPersistence:
    """Integration tests for persistence layer."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_integration(self, temp_storage):
        """Test complete workflow from creation to compaction."""
        # Initialize managers
        batch_manager = BatchPersistenceManager(temp_storage)
        
        # Create sample data
        analogies = [
            {
                "source_domain": "medieval",
                "target_domain": "modern",
                "quality_score": 0.9,
                "concept_mappings": {"king": "president", "knight": "soldier"}
            },
            {
                "source_domain": "medieval",
                "target_domain": "business",
                "quality_score": 0.7,
                "concept_mappings": {"king": "ceo", "knight": "employee"}
            }
        ]
        
        # Create and process batch
        workflow = batch_manager.create_analogy_batch(analogies)
        processed_workflow = batch_manager.process_analogy_batch(workflow.workflow_id)
        
        assert processed_workflow.status == WorkflowStatus.COMPLETED
        
        # Query analogies
        high_quality = batch_manager.find_analogies_by_quality(0.8)
        assert len(high_quality) == 1
        
        # Delete low quality
        criteria = DeleteCriteria(quality_threshold=0.8)
        delete_workflow = batch_manager.delete_analogies_batch(criteria)
        
        assert delete_workflow.status == WorkflowStatus.COMPLETED
        
        # Compact storage
        result = batch_manager.compact_analogies_jsonl()
        assert result["status"] == "completed"
        assert result["records_removed"] > 0


# Performance tests
class TestPerformancePersistence:
    """Performance tests for persistence operations."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.performance
    def test_large_batch_performance(self, temp_storage):
        """Test performance with large batches of analogies."""
        batch_manager = BatchPersistenceManager(temp_storage)
        
        # Create large batch (1000 analogies)
        large_batch = []
        for i in range(1000):
            large_batch.append({
                "source_domain": f"domain_{i % 10}",
                "target_domain": f"target_{i % 5}",
                "quality_score": 0.5 + (i % 50) / 100,
                "concept_mappings": {f"concept_{i}": f"mapped_{i}"}
            })
        
        # Time the batch creation and processing
        import time
        
        start_time = time.time()
        workflow = batch_manager.create_analogy_batch(large_batch)
        create_time = time.time() - start_time
        
        start_time = time.time()
        batch_manager.process_analogy_batch(workflow.workflow_id)
        process_time = time.time() - start_time
        
        # Performance assertions (adjust based on requirements)
        assert create_time < 5.0  # Should create batch in under 5 seconds
        assert process_time < 30.0  # Should process 1000 analogies in under 30 seconds
        
        print(f"Create time: {create_time:.2f}s, Process time: {process_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
