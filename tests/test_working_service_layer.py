"""
Working Test Suite for Service Layer
====================================

Tests the working FastAPI service layer with realistic functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from app.working_service_layer import app, get_semantic_registry, get_persistence_manager, get_batch_manager
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.batch_persistence import BatchPersistenceManager


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_registry():
    """Create test semantic registry."""
    return EnhancedHybridRegistry(
        download_wordnet=False,
        n_clusters=4,
        enable_cross_domain=True,
        embedding_provider="random"  # Use 'random' instead of 'test'
    )


@pytest.fixture
def test_persistence_manager(temp_storage):
    """Create test persistence manager."""
    return ContractEnhancedPersistenceManager(temp_storage)


@pytest.fixture
def test_batch_manager(temp_storage):
    """Create test batch manager."""
    return BatchPersistenceManager(temp_storage)


@pytest.fixture
def client_with_mocks(test_registry, test_persistence_manager, test_batch_manager):
    """Create test client with mocked dependencies."""
    
    app.dependency_overrides[get_semantic_registry] = lambda: test_registry
    app.dependency_overrides[get_persistence_manager] = lambda: test_persistence_manager
    app.dependency_overrides[get_batch_manager] = lambda: test_batch_manager
    
    client = TestClient(app)
    yield client
    
    app.dependency_overrides = {}


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestBasicEndpoints:
    """Test basic endpoint functionality."""
    
    def test_health_check(self, client_with_mocks):
        """Test health check endpoint."""
        response = client_with_mocks.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "soft-logic-service-layer"
        assert "components" in data
        assert "timestamp" in data
    
    def test_service_status(self, client_with_mocks):
        """Test service status endpoint."""
        response = client_with_mocks.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "registry_stats" in data
        assert "storage_stats" in data
    
    def test_docs_overview(self, client_with_mocks):
        """Test API documentation overview."""
        response = client_with_mocks.get("/docs-overview")
        
        assert response.status_code == 200
        data = response.json()
        assert "api_overview" in data
        assert "endpoints" in data
        assert "documentation" in data


class TestConceptOperations:
    """Test concept management operations."""
    
    def test_create_concept_success(self, client_with_mocks):
        """Test successful concept creation."""
        concept_data = {
            "name": "knight",
            "context": "medieval",
            "synset_id": "knight.n.01",
            "disambiguation": "armored warrior"
        }
        
        response = client_with_mocks.post("/concepts", json=concept_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == concept_data["name"]
        assert data["context"] == concept_data["context"]
        assert "concept_id" in data
        assert "created_at" in data
    
    def test_create_concept_validation_error(self, client_with_mocks):
        """Test concept creation with validation errors."""
        invalid_concept = {
            "name": "",  # Empty name should fail
            "context": "test"
        }
        
        response = client_with_mocks.post("/concepts", json=invalid_concept)
        assert response.status_code == 422  # Validation error
    
    def test_get_concept_success(self, client_with_mocks):
        """Test concept retrieval."""
        # First create a concept
        concept_data = {
            "name": "sword",
            "context": "weapons",
            "disambiguation": "bladed weapon"
        }
        
        create_response = client_with_mocks.post("/concepts", json=concept_data)
        concept_id = create_response.json()["concept_id"]
        
        # Then retrieve it
        response = client_with_mocks.get(f"/concepts/{concept_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["concept_id"] == concept_id
        assert data["name"] == concept_data["name"]
    
    def test_get_concept_not_found(self, client_with_mocks):
        """Test concept retrieval with non-existent ID."""
        response = client_with_mocks.get("/concepts/nonexistent_id")
        assert response.status_code == 404
    
    def test_search_concepts(self, client_with_mocks):
        """Test concept search functionality."""
        # Create a concept first
        concept_data = {
            "name": "castle",
            "context": "architecture",
            "disambiguation": "fortified structure"
        }
        
        client_with_mocks.post("/concepts", json=concept_data)
        
        # Search for it
        response = client_with_mocks.post("/concepts/search?query=castle&max_results=5")
        
        assert response.status_code == 200
        data = response.json()
        assert "concepts" in data
        assert "total_results" in data
        assert "query" in data


class TestReasoningOperations:
    """Test semantic reasoning operations."""
    
    def test_complete_analogy(self, client_with_mocks):
        """Test analogy completion."""
        # Create some concepts first
        concepts = [
            {"name": "knight", "context": "medieval"},
            {"name": "sword", "context": "weapons"},
            {"name": "wizard", "context": "fantasy"}
        ]
        
        for concept_data in concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Test analogy completion
        response = client_with_mocks.post(
            "/analogies/complete",
            params={
                "source_a": "knight",
                "source_b": "sword", 
                "target_a": "wizard",
                "max_completions": 3
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "completions" in data
        assert "reasoning_trace" in data
        assert "metadata" in data


class TestBatchOperations:
    """Test batch processing operations."""
    
    def test_create_analogy_batch(self, client_with_mocks):
        """Test batch creation."""
        batch_data = {
            "analogies": [
                {
                    "source_pair": ["knight", "sword"],
                    "target_pair": ["wizard", "staff"],
                    "quality_score": 0.8
                }
            ],
            "workflow_id": "test_workflow"
        }
        
        response = client_with_mocks.post("/batch/analogies", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "test_workflow"
        assert "workflow_type" in data
        assert "status" in data
    
    def test_list_workflows(self, client_with_mocks):
        """Test workflow listing."""
        # Create a workflow first
        batch_data = {
            "analogies": [{"test": "data"}],
            "workflow_id": "list_test"
        }
        
        client_with_mocks.post("/batch/analogies", json=batch_data)
        
        # List workflows
        response = client_with_mocks.get("/batch/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_workflow_by_id(self, client_with_mocks):
        """Test workflow retrieval by ID."""
        # Create a workflow first
        batch_data = {
            "analogies": [{"test": "data"}],
            "workflow_id": "get_test_workflow"
        }
        
        create_response = client_with_mocks.post("/batch/analogies", json=batch_data)
        workflow_id = create_response.json()["workflow_id"]
        
        # Get workflow by ID
        response = client_with_mocks.get(f"/batch/workflows/{workflow_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == workflow_id
    
    def test_get_workflow_not_found(self, client_with_mocks):
        """Test retrieval of non-existent workflow."""
        response = client_with_mocks.get("/batch/workflows/nonexistent")
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_service_unavailable(self):
        """Test service unavailable when dependencies not initialized."""
        # Create client without overrides
        client = TestClient(app)
        
        response = client.post("/concepts", json={"name": "test", "context": "test"})
        assert response.status_code == 503
    
    def test_invalid_json(self, client_with_mocks):
        """Test handling of invalid JSON data."""
        response = client_with_mocks.post("/concepts", data="invalid json")
        assert response.status_code == 422


class TestIntegration:
    """Test complete integration scenarios."""
    
    def test_complete_workflow(self, client_with_mocks):
        """Test complete workflow from concept creation to batch processing."""
        # 1. Create concepts
        concepts = [
            {"name": "knight", "context": "medieval"},
            {"name": "sword", "context": "weapons"},
            {"name": "wizard", "context": "fantasy"},
            {"name": "staff", "context": "implements"}
        ]
        
        concept_ids = []
        for concept_data in concepts:
            response = client_with_mocks.post("/concepts", json=concept_data)
            assert response.status_code == 200
            concept_ids.append(response.json()["concept_id"])
        
        # 2. Search for concepts
        search_response = client_with_mocks.post("/concepts/search?query=knight")
        assert search_response.status_code == 200
        assert len(search_response.json()["concepts"]) > 0
        
        # 3. Complete analogy
        analogy_response = client_with_mocks.post(
            "/analogies/complete",
            params={
                "source_a": "knight",
                "source_b": "sword",
                "target_a": "wizard"
            }
        )
        assert analogy_response.status_code == 200
        
        # 4. Create batch
        batch_response = client_with_mocks.post("/batch/analogies", json={
            "analogies": [{"source": "knight", "target": "wizard"}],
            "workflow_id": "integration_test"
        })
        assert batch_response.status_code == 200
        
        # 5. Check batch status
        workflow_id = batch_response.json()["workflow_id"]
        status_response = client_with_mocks.get(f"/batch/workflows/{workflow_id}")
        assert status_response.status_code == 200
        
        print("✅ Complete integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
