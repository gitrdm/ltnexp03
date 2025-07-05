"""
Simple Test for Working Service Layer
=====================================

Basic functionality tests for the working service layer.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Simple test without TestClient issues
def test_import_working_service_layer():
    """Test that we can import the working service layer."""
    try:
        from app.working_service_layer import app, StatusResponse, ConceptCreate
        assert app is not None
        assert StatusResponse is not None
        assert ConceptCreate is not None
        print("✅ Working service layer imports successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import working service layer: {e}")


def test_create_pydantic_models():
    """Test that Pydantic models work correctly."""
    from app.working_service_layer import ConceptCreate, ConceptResponse, BatchRequest, StatusResponse
    
    # Test ConceptCreate
    concept = ConceptCreate(
        name="test_concept",
        context="test_context",
        synset_id="test.n.01",
        disambiguation="test disambiguation"
    )
    assert concept.name == "test_concept"
    assert concept.context == "test_context"
    
    # Test StatusResponse
    status = StatusResponse(
        status="healthy",
        service="test-service",
        version="1.0.0",
        components={"test": True},
        timestamp="2025-01-01T00:00:00"
    )
    assert status.status == "healthy"
    assert status.service == "test-service"
    
    print("✅ Pydantic models work correctly")


def test_create_registries():
    """Test that we can create the core registries."""
    from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
    from app.core.contract_persistence import ContractEnhancedPersistenceManager
    from app.core.batch_persistence import BatchPersistenceManager
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test registry creation
        registry = EnhancedHybridRegistry(
            download_wordnet=False,
            n_clusters=4,
            enable_cross_domain=True,
            embedding_provider="random"
        )
        assert registry is not None
        
        # Test persistence managers
        persistence_manager = ContractEnhancedPersistenceManager(temp_dir)
        assert persistence_manager is not None
        
        batch_manager = BatchPersistenceManager(temp_dir)
        assert batch_manager is not None
        
        print("✅ Core registries created successfully")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_dependency_functions():
    """Test dependency injection functions."""
    from app.working_service_layer import get_semantic_registry, get_persistence_manager, get_batch_manager
    from fastapi import HTTPException
    
    # These should raise HTTPException when called without initialization
    try:
        get_semantic_registry()
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 503
        assert "not initialized" in e.detail
    
    try:
        get_persistence_manager()
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 503
        assert "not initialized" in e.detail
    
    try:
        get_batch_manager()
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 503
        assert "not initialized" in e.detail
    
    print("✅ Dependency injection functions work correctly")


def test_fastapi_app_creation():
    """Test that FastAPI app is created properly."""
    from app.working_service_layer import app
    
    # Check app properties
    assert app.title == "Soft Logic Microservice - Working Service Layer"
    assert app.version == "1.0.0"
    
    # Check routes exist
    route_paths = [route.path for route in app.routes]
    expected_paths = ["/health", "/status", "/concepts", "/docs-overview"]
    
    for path in expected_paths:
        assert path in route_paths, f"Path {path} not found in routes"
    
    print("✅ FastAPI app created with correct configuration")


def test_concept_creation_logic():
    """Test concept creation logic without full HTTP client."""
    from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
    
    # Create registry
    registry = EnhancedHybridRegistry(
        download_wordnet=False,
        n_clusters=4,
        enable_cross_domain=True,
        embedding_provider="random"
    )
    
    # Test concept creation
    concept_id = registry.create_frame_aware_concept_with_advanced_embedding(
        name="test_knight",
        context="medieval_test",
        synset_id="knight.n.01",
        disambiguation="test armored warrior",
        use_semantic_embedding=True
    )
    
    assert concept_id is not None
    assert len(registry.frame_aware_concepts) > 0
    
    # Check that the concept was created (concept_id is the actual concept, not a key)
    assert hasattr(concept_id, 'name')
    assert concept_id.name == "test_knight"
    
    print("✅ Concept creation logic works correctly")


def test_batch_creation_logic():
    """Test batch creation logic without full HTTP client."""
    import tempfile
    from app.core.batch_persistence import BatchPersistenceManager
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        batch_manager = BatchPersistenceManager(temp_dir)
        
        # Test batch creation
        test_analogies = [
            {
                "source_pair": ["knight", "sword"],
                "target_pair": ["wizard", "staff"],
                "quality_score": 0.8
            }
        ]
        
        workflow = batch_manager.create_analogy_batch(
            analogies=test_analogies,
            workflow_id="test_workflow_001"
        )
        
        assert workflow is not None
        assert workflow.workflow_id == "test_workflow_001"
        assert workflow.items_total == 1
        
        print("✅ Batch creation logic works correctly")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestServiceLayerIntegration:
    """Integration tests for service layer components."""
    
    def test_complete_integration(self):
        """Test complete integration of all components."""
        import tempfile
        from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
        from app.core.contract_persistence import ContractEnhancedPersistenceManager
        from app.core.batch_persistence import BatchPersistenceManager
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            # Initialize all components
            registry = EnhancedHybridRegistry(
                download_wordnet=False,
                n_clusters=4,
                enable_cross_domain=True,
                embedding_provider="random"
            )
            
            persistence_manager = ContractEnhancedPersistenceManager(temp_dir)
            batch_manager = BatchPersistenceManager(temp_dir)
            
            # Test concept creation
            concept_id = registry.create_frame_aware_concept_with_advanced_embedding(
                name="integration_test_knight",
                context="test_medieval",
                use_semantic_embedding=True
            )
            
            assert concept_id is not None
            
            # Test batch operations
            test_analogies = [{"test": "analogy"}]
            workflow = batch_manager.create_analogy_batch(
                analogies=test_analogies,
                workflow_id="integration_test"
            )
            
            assert workflow is not None
            
            # Test workflow listing
            workflows = batch_manager.list_workflows()
            assert len(workflows) > 0
            assert any(w.workflow_id == "integration_test" for w in workflows)
            
            print("✅ Complete integration test passed")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests manually
    test_import_working_service_layer()
    test_create_pydantic_models()
    test_create_registries()
    test_dependency_functions()
    test_fastapi_app_creation()
    test_concept_creation_logic()
    test_batch_creation_logic()
    
    integration_test = TestServiceLayerIntegration()
    integration_test.test_complete_integration()
    
    print("\n🎉 All service layer tests passed!")
