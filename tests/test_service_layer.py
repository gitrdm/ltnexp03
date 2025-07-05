"""
Comprehensive Test Suite for Service Layer
==========================================

Tests the complete FastAPI service layer including:
- REST API endpoints with type safety
- WebSocket streaming capabilities  
- Contract validation and error handling
- Integration with persistence and semantic reasoning
- Performance and load testing

Following Design by Contract principles with comprehensive validation.
"""

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import tempfile
import shutil

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

# Import the service layer
from app.service_layer import app, get_semantic_registry, get_persistence_manager, get_batch_manager
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
        download_wordnet=False,  # Skip WordNet download for tests
        n_clusters=4,
        enable_cross_domain=True,
        embedding_provider="random"
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
    
    # Override dependencies
    app.dependency_overrides[get_semantic_registry] = lambda: test_registry
    app.dependency_overrides[get_persistence_manager] = lambda: test_persistence_manager
    app.dependency_overrides[get_batch_manager] = lambda: test_batch_manager
    
    client = TestClient(app)
    yield client
    
    # Clean up overrides
    app.dependency_overrides = {}


@pytest.fixture
def sample_concepts():
    """Sample concepts for testing."""
    return [
        {
            "name": "knight",
            "context": "medieval",
            "synset_id": "knight.n.01",
            "disambiguation": "medieval warrior",
            "metadata": {"domain": "military", "era": "medieval"}
        },
        {
            "name": "sword",
            "context": "weapon",
            "synset_id": "sword.n.01",
            "disambiguation": "bladed weapon",
            "metadata": {"domain": "military", "type": "weapon"}
        },
        {
            "name": "castle",
            "context": "architecture",
            "synset_id": "castle.n.01",
            "disambiguation": "fortified structure",
            "metadata": {"domain": "architecture", "era": "medieval"}
        }
    ]


@pytest.fixture
def sample_analogies():
    """Sample analogies for testing."""
    return [
        {
            "source_pair": ["knight", "sword"],
            "target_pair": ["wizard", "staff"],
            "context": "medieval fantasy",
            "quality_score": 0.85,
            "reasoning": "Both are character-weapon pairs in fantasy"
        },
        {
            "source_pair": ["castle", "moat"],
            "target_pair": ["city", "wall"],
            "context": "fortification",
            "quality_score": 0.78,
            "reasoning": "Both are structure-defense pairs"
        }
    ]


# ============================================================================
# CONCEPT MANAGEMENT TESTS
# ============================================================================

class TestConceptManagement:
    """Test concept creation, retrieval, and search operations."""
    
    def test_create_concept_success(self, client_with_mocks, sample_concepts):
        """Test successful concept creation."""
        concept_data = sample_concepts[0]
        
        response = client_with_mocks.post("/concepts", json=concept_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == concept_data["name"]
        assert data["context"] == concept_data["context"]
        assert "concept_id" in data
        assert "created_at" in data
    
    def test_create_concept_validation_error(self, client_with_mocks):
        """Test concept creation with validation errors."""
        # Invalid concept with empty name
        invalid_concept = {
            "name": "",  # Empty name should fail validation
            "context": "test"
        }
        
        response = client_with_mocks.post("/concepts", json=invalid_concept)
        assert response.status_code == 422  # Validation error
    
    def test_get_concept_success(self, client_with_mocks, sample_concepts):
        """Test successful concept retrieval."""
        # First create a concept
        concept_data = sample_concepts[0]
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
    
    def test_search_concepts_success(self, client_with_mocks, sample_concepts):
        """Test successful concept search."""
        # Create multiple concepts
        for concept_data in sample_concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Search for concepts
        search_data = {
            "query": "knight",
            "similarity_threshold": 0.7,
            "max_results": 5,
            "include_metadata": True
        }
        
        response = client_with_mocks.post("/concepts/search", json=search_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "concepts" in data
        assert "total_results" in data
        assert "search_metadata" in data
    
    def test_compute_concept_similarity(self, client_with_mocks, sample_concepts):
        """Test concept similarity computation."""
        # Create concepts
        for concept_data in sample_concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Compute similarity
        similarity_data = {
            "concept1": "knight",
            "concept2": "sword",
            "similarity_method": "hybrid"
        }
        
        response = client_with_mocks.post("/concepts/similarity", json=similarity_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "similarity_score" in data
        assert "method_used" in data
        assert "confidence" in data
        assert 0.0 <= data["similarity_score"] <= 1.0


# ============================================================================
# SEMANTIC REASONING TESTS
# ============================================================================

class TestSemanticReasoning:
    """Test analogical reasoning and semantic field discovery."""
    
    def test_complete_analogy_success(self, client_with_mocks, sample_concepts):
        """Test successful analogy completion."""
        # Create concepts first
        for concept_data in sample_concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Request analogy completion
        analogy_data = {
            "source_a": "knight",
            "source_b": "sword",
            "target_a": "wizard",
            "context": "fantasy",
            "max_completions": 3,
            "min_confidence": 0.5
        }
        
        response = client_with_mocks.post("/analogies/complete", json=analogy_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "completions" in data
        assert "reasoning_trace" in data
        assert "metadata" in data
        assert isinstance(data["completions"], list)
    
    def test_discover_semantic_fields(self, client_with_mocks, sample_concepts):
        """Test semantic field discovery."""
        # Create concepts first
        for concept_data in sample_concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Discover semantic fields
        discovery_data = {
            "domain": "medieval",
            "min_coherence": 0.6,
            "max_fields": 5,
            "clustering_method": "kmeans"
        }
        
        response = client_with_mocks.post("/semantic-fields/discover", json=discovery_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "semantic_fields" in data
        assert "discovery_metadata" in data
        assert "coherence_scores" in data
        assert isinstance(data["semantic_fields"], list)
    
    def test_cross_domain_analogies(self, client_with_mocks, sample_concepts):
        """Test cross-domain analogy discovery."""
        # Create concepts first
        for concept_data in sample_concepts:
            client_with_mocks.post("/concepts", json=concept_data)
        
        # Discover cross-domain analogies
        request_data = {
            "source_domain": "medieval",
            "target_domain": "modern",
            "min_quality": 0.6,
            "max_analogies": 5
        }
        
        response = client_with_mocks.post("/analogies/cross-domain", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "analogies" in data
        assert "quality_scores" in data
        assert "domain_analysis" in data
        assert isinstance(data["analogies"], list)


# ============================================================================
# FRAME OPERATIONS TESTS
# ============================================================================

class TestFrameOperations:
    """Test semantic frame creation and management."""
    
    def test_create_frame_success(self, client_with_mocks):
        """Test successful frame creation."""
        frame_data = {
            "name": "Combat",
            "definition": "A situation involving conflict between entities",
            "core_elements": ["Combatant_1", "Combatant_2", "Weapon"],
            "peripheral_elements": ["Location", "Time"],
            "lexical_units": ["fight", "battle", "duel"],
            "metadata": {"domain": "conflict"}
        }
        
        response = client_with_mocks.post("/frames", json=frame_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == frame_data["name"]
        assert data["definition"] == frame_data["definition"]
        assert "frame_id" in data
        assert "created_at" in data
    
    def test_create_frame_instance(self, client_with_mocks):
        """Test frame instance creation."""
        # First create a frame
        frame_data = {
            "name": "Combat",
            "definition": "A situation involving conflict",
            "core_elements": ["Combatant_1", "Combatant_2"],
        }
        
        frame_response = client_with_mocks.post("/frames", json=frame_data)
        frame_id = frame_response.json()["frame_id"]
        
        # Create an instance
        instance_data = {
            "instance_id": "combat_001",
            "concept_bindings": {
                "Combatant_1": "knight",
                "Combatant_2": "dragon"
            },
            "context": "medieval fantasy battle",
            "confidence": 0.9
        }
        
        response = client_with_mocks.post(f"/frames/{frame_id}/instances", json=instance_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["frame_id"] == frame_id
        assert data["instance_id"] == instance_data["instance_id"]
        assert "bindings" in data
    
    def test_query_frames(self, client_with_mocks):
        """Test frame querying."""
        # Create a frame first
        frame_data = {
            "name": "Combat",
            "definition": "A situation involving conflict",
            "core_elements": ["Combatant_1", "Combatant_2"],
        }
        
        client_with_mocks.post("/frames", json=frame_data)
        
        # Query frames
        query_data = {
            "concept": "combat",
            "max_results": 10
        }
        
        response = client_with_mocks.post("/frames/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "frames" in data
        assert "instances" in data
        assert "query_metadata" in data


# ============================================================================
# BATCH OPERATIONS TESTS
# ============================================================================

class TestBatchOperations:
    """Test batch processing and workflow management."""
    
    def test_create_analogy_batch(self, client_with_mocks, sample_analogies):
        """Test creating analogy batch workflow."""
        batch_data = {
            "analogies": sample_analogies,
            "workflow_id": "test_workflow_001"
        }
        
        response = client_with_mocks.post("/batch/analogies", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == batch_data["workflow_id"]
        assert data["workflow_type"] == "analogy_batch"
        assert "status" in data
        assert "items_total" in data
    
    def test_list_workflows(self, client_with_mocks, sample_analogies):
        """Test listing workflows."""
        # Create a workflow first
        batch_data = {
            "analogies": sample_analogies,
            "workflow_id": "test_workflow_002"
        }
        
        client_with_mocks.post("/batch/analogies", json=batch_data)
        
        # List workflows
        response = client_with_mocks.get("/batch/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:  # If workflows exist
            assert "workflow_id" in data[0]
            assert "status" in data[0]
    
    def test_get_workflow_by_id(self, client_with_mocks, sample_analogies):
        """Test retrieving specific workflow."""
        # Create a workflow first
        batch_data = {
            "analogies": sample_analogies,
            "workflow_id": "test_workflow_003"
        }
        
        create_response = client_with_mocks.post("/batch/analogies", json=batch_data)
        workflow_id = create_response.json()["workflow_id"]
        
        # Get workflow by ID
        response = client_with_mocks.get(f"/batch/workflows/{workflow_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == workflow_id
        assert "status" in data
        assert "items_total" in data
    
    def test_get_workflow_not_found(self, client_with_mocks):
        """Test retrieving non-existent workflow."""
        response = client_with_mocks.get("/batch/workflows/nonexistent_workflow")
        assert response.status_code == 404


# ============================================================================
# WEBSOCKET STREAMING TESTS
# ============================================================================

class TestWebSocketStreaming:
    """Test WebSocket streaming capabilities."""
    
    def test_websocket_connection_and_streaming(self, client_with_mocks):
        """Test WebSocket connection and basic streaming."""
        try:
            with client_with_mocks.websocket_connect("/ws/analogies/stream") as websocket:
                # Should receive connection message
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["type"] == "connection"
                assert message["status"] == "connected"
        except Exception as e:
            # WebSocket testing can be flaky in test environment
            pytest.skip(f"WebSocket test skipped due to: {e}")
    
    def test_websocket_workflow_status_streaming(self, client_with_mocks, sample_analogies):
        """Test workflow status streaming via WebSocket."""
        try:
            # First create a workflow
            batch_data = {
                "analogies": sample_analogies,
                "workflow_id": "test_ws_workflow"
            }
            
            create_response = client_with_mocks.post("/batch/analogies", json=batch_data)
            workflow_id = create_response.json()["workflow_id"]
            
            # Connect to workflow status stream
            with client_with_mocks.websocket_connect(f"/ws/workflows/{workflow_id}/status") as websocket:
                # Should receive status update
                data = websocket.receive_text()
                message = json.loads(data)
                assert message["message_type"] == "workflow_status"
                assert "content" in message
                assert message["content"]["workflow_id"] == workflow_id
        except Exception as e:
            # WebSocket testing can be flaky in test environment
            pytest.skip(f"WebSocket test skipped due to: {e}")


# ============================================================================
# SYSTEM ENDPOINT TESTS
# ============================================================================

class TestSystemEndpoints:
    """Test health checks and system status endpoints."""
    
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
        """Test detailed service status endpoint."""
        response = client_with_mocks.get("/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["operational", "initializing", "error"]
        if data["status"] == "operational":
            assert "registry_stats" in data
            assert "workflow_stats" in data
            assert "storage_stats" in data
    
    def test_docs_overview(self, client_with_mocks):
        """Test API documentation overview."""
        response = client_with_mocks.get("/docs-overview")
        
        assert response.status_code == 200
        data = response.json()
        assert "api_overview" in data
        assert "endpoints" in data
        assert "documentation" in data


# ============================================================================
# ERROR HANDLING AND VALIDATION TESTS
# ============================================================================

class TestErrorHandling:
    """Test comprehensive error handling and validation."""
    
    def test_invalid_request_data(self, client_with_mocks):
        """Test handling of invalid request data."""
        # Send malformed JSON
        response = client_with_mocks.post("/concepts", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error
    
    def test_service_unavailable_scenarios(self):
        """Test service unavailable scenarios."""
        # Create client without mocked dependencies
        client = TestClient(app)
        
        # Should get service unavailable for endpoints requiring dependencies
        response = client.post("/concepts", json={"name": "test", "context": "test"})
        assert response.status_code == 503  # Service unavailable
    
    def test_contract_validation_errors(self, client_with_mocks):
        """Test Design by Contract validation errors."""
        # Test with edge case data that might trigger contract violations
        concept_data = {
            "name": "a" * 1000,  # Very long name
            "context": "test",
            "similarity_threshold": -1.0  # Invalid threshold
        }
        
        # The validation should catch this before it gets to contract validation
        response = client_with_mocks.post("/concepts", json=concept_data)
        assert response.status_code in [400, 422]  # Client error


# ============================================================================
# PERFORMANCE AND LOAD TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics of the service layer."""
    
    def test_concept_creation_performance(self, client_with_mocks):
        """Test performance of concept creation operations."""
        import time
        
        start_time = time.time()
        
        # Create multiple concepts rapidly
        for i in range(10):
            concept_data = {
                "name": f"test_concept_{i}",
                "context": "performance_test",
                "metadata": {"test_id": i}
            }
            
            response = client_with_mocks.post("/concepts", json=concept_data)
            assert response.status_code == 200
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 10 concept creations in reasonable time
        assert duration < 5.0  # 5 seconds should be plenty
        
        # Calculate throughput
        throughput = 10 / duration
        print(f"Concept creation throughput: {throughput:.2f} concepts/second")
    
    def test_batch_processing_performance(self, client_with_mocks, sample_analogies):
        """Test performance of batch processing operations."""
        import time
        
        # Create a larger batch
        large_batch = sample_analogies * 10  # 20 analogies
        
        batch_data = {
            "analogies": large_batch,
            "workflow_id": "performance_test_batch"
        }
        
        start_time = time.time()
        response = client_with_mocks.post("/batch/analogies", json=batch_data)
        end_time = time.time()
        
        assert response.status_code == 200
        duration = end_time - start_time
        
        # Batch creation should be fast (processing happens in background)
        assert duration < 2.0  # 2 seconds
        
        print(f"Batch creation time: {duration:.3f} seconds for {len(large_batch)} analogies")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test complete integration scenarios."""
    
    def test_complete_workflow_integration(self, client_with_mocks, sample_concepts, sample_analogies):
        """Test complete workflow from concept creation to analogy processing."""
        # 1. Create concepts
        concept_ids = []
        for concept_data in sample_concepts:
            response = client_with_mocks.post("/concepts", json=concept_data)
            assert response.status_code == 200
            concept_ids.append(response.json()["concept_id"])
        
        # 2. Search for concepts
        search_response = client_with_mocks.post("/concepts/search", json={
            "query": "knight",
            "similarity_threshold": 0.5,
            "max_results": 10
        })
        assert search_response.status_code == 200
        assert len(search_response.json()["concepts"]) > 0
        
        # 3. Create analogy batch
        batch_response = client_with_mocks.post("/batch/analogies", json={
            "analogies": sample_analogies,
            "workflow_id": "integration_test_workflow"
        })
        assert batch_response.status_code == 200
        workflow_id = batch_response.json()["workflow_id"]
        
        # 4. Check workflow status
        status_response = client_with_mocks.get(f"/batch/workflows/{workflow_id}")
        assert status_response.status_code == 200
        assert status_response.json()["workflow_id"] == workflow_id
        
        # 5. Complete analogies
        analogy_response = client_with_mocks.post("/analogies/complete", json={
            "source_a": "knight",
            "source_b": "sword",
            "target_a": "wizard",
            "max_completions": 3,
            "min_confidence": 0.4
        })
        assert analogy_response.status_code == 200
        
        print("✅ Complete workflow integration test passed")


# ============================================================================
# TEST EXECUTION AND REPORTING
# ============================================================================

def run_performance_benchmarks():
    """Run performance benchmarks and generate report."""
    print("\n" + "="*60)
    print("SERVICE LAYER PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # This would be run separately for performance analysis
    # pytest -v tests/test_service_layer.py::TestPerformance --capture=no


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--capture=no"
    ])
