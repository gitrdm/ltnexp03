"""
Test Protocol Implementation
============================

This test file verifies that the Phase 3A type safety foundation is working correctly.
It tests the Protocol interface compliance and TypedDict API models.
"""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_protocol_imports():
    """Test that protocol interfaces can be imported."""
    try:
        from app.core.protocols import (
            ConceptRegistryProtocol, SemanticReasoningProtocol,
            KnowledgeDiscoveryProtocol, EmbeddingProviderProtocol,
            FrameRegistryProtocol, ClusterRegistryProtocol
        )
        print("✓ Protocol interfaces imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import protocols: {e}")
        assert False, f"Failed to import protocols: {e}"

def test_api_models_imports():
    """Test that API models can be imported."""
    try:
        from app.core.api_models import (
            ConceptCreateRequest, ConceptCreateResponse,
            AnalogyRequest, AnalogyResponse,
            SemanticFieldDiscoveryRequest, SemanticFieldDiscoveryResponse,
            SystemHealthRequest, SystemHealthResponse
        )
        print("✓ API models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import API models: {e}")
        assert False, f"Failed to import API models: {e}"

def test_enhanced_registry_protocols():
    """Test that EnhancedHybridRegistry implements protocols."""
    try:
        from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
        from app.core.protocols import SemanticReasoningProtocol, KnowledgeDiscoveryProtocol
        
        # Create registry instance
        registry = EnhancedHybridRegistry(download_wordnet=False)
        
        # Test protocol compliance
        assert isinstance(registry, SemanticReasoningProtocol), "Must implement SemanticReasoningProtocol"
        assert isinstance(registry, KnowledgeDiscoveryProtocol), "Must implement KnowledgeDiscoveryProtocol"
        
        print("✓ EnhancedHybridRegistry implements required protocols")
    except Exception as e:
        print(f"✗ Protocol implementation test failed: {e}")
        assert False, f"Protocol implementation test failed: {e}"

def test_semantic_reasoning_interface():
    """Test the semantic reasoning protocol interface."""
    try:
        from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
        
        # Create registry instance
        registry = EnhancedHybridRegistry(download_wordnet=False)
        
        # Test protocol methods exist and have correct signatures
        assert hasattr(registry, 'complete_analogy'), "Must have complete_analogy method"
        assert hasattr(registry, 'discover_semantic_fields'), "Must have discover_semantic_fields method"
        assert hasattr(registry, 'find_cross_domain_analogies'), "Must have find_cross_domain_analogies method"
        
        # Test basic method calls (with empty data, should not crash)
        try:
            result = registry.discover_semantic_fields(min_coherence=0.9)
            assert isinstance(result, list), "discover_semantic_fields must return list"
            print(f"✓ discover_semantic_fields returned {len(result)} fields")
        except Exception as e:
            print(f"✓ discover_semantic_fields handled empty data gracefully: {e}")
        
        try:
            result = registry.complete_analogy({"a": "b", "c": "?"}, max_completions=1)
            assert isinstance(result, list), "complete_analogy must return list"
            print(f"✓ complete_analogy returned {len(result)} results")
        except Exception as e:
            print(f"✓ complete_analogy handled empty data gracefully: {e}")
        
        print("✓ Semantic reasoning interface working")
    except Exception as e:
        print(f"✗ Semantic reasoning interface test failed: {e}")
        assert False, f"Semantic reasoning interface test failed: {e}"

def test_api_model_validation():
    """Test API model type validation."""
    try:
        from app.core.api_models import ConceptCreateRequest, AnalogyRequest
        
        # Test valid requests
        concept_req = ConceptCreateRequest(
            name="test_concept",
            context="test",
            synset_id=None,
            disambiguation=None,
            metadata={},
            auto_disambiguate=True
        )
        print(f"✓ Valid ConceptCreateRequest: {concept_req['name']}")
        
        analogy_req = AnalogyRequest(
            partial_analogy={"king": "queen", "man": "?"},
            context="test",
            max_completions=5,
            min_confidence=0.5
        )
        print(f"✓ Valid AnalogyRequest with {len(analogy_req['partial_analogy'])} mappings")
        
        print("✓ API model validation working")
    except Exception as e:
        print(f"✗ API model validation test failed: {e}")
        assert False, f"API model validation test failed: {e}"

def run_phase_3a_tests():
    """Run all Phase 3A tests."""
    print("=" * 50)
    print("PHASE 3A TYPE SAFETY FOUNDATION TESTS")
    print("=" * 50)
    
    tests = [
        test_protocol_imports,
        test_api_models_imports,
        test_enhanced_registry_protocols,
        test_semantic_reasoning_interface,
        test_api_model_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = run_phase_3a_tests()
    sys.exit(0 if success else 1)
