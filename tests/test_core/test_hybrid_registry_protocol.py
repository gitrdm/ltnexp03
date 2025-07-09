"""
Tests for HybridConceptRegistry Protocol Compliance
"""
import pytest
from app.core.hybrid_registry import HybridConceptRegistry
from app.core.protocols import (
    SemanticReasoningProtocol,
    KnowledgeDiscoveryProtocol,
    FrameRegistryProtocol,
    ClusterRegistryProtocol,
    ConceptRegistryProtocol
)
from app.core.abstractions import Concept
from app.core.frame_cluster_abstractions import SemanticFrame, FrameAwareConcept

@pytest.fixture
def hybrid_registry() -> HybridConceptRegistry:
    """Fixture for a HybridConceptRegistry instance."""
    return HybridConceptRegistry(download_wordnet=False)

def test_hybrid_registry_protocol_compliance(hybrid_registry: HybridConceptRegistry):
    """
    Tests that HybridConceptRegistry satisfies all its declared protocols at a high level.
    """
    assert isinstance(hybrid_registry, ConceptRegistryProtocol)
    assert isinstance(hybrid_registry, FrameRegistryProtocol)
    assert isinstance(hybrid_registry, ClusterRegistryProtocol)
    assert isinstance(hybrid_registry, SemanticReasoningProtocol)
    assert isinstance(hybrid_registry, KnowledgeDiscoveryProtocol)

def test_semantic_reasoning_protocol_methods_exist(hybrid_registry: HybridConceptRegistry):
    """
    Check for the existence of all SemanticReasoningProtocol methods.
    """
    assert hasattr(hybrid_registry, 'find_analogous_concepts')
    assert hasattr(hybrid_registry, 'complete_analogy')
    assert hasattr(hybrid_registry, 'discover_semantic_fields')
    assert hasattr(hybrid_registry, 'find_cross_domain_analogies')

def test_knowledge_discovery_protocol_methods_exist(hybrid_registry: HybridConceptRegistry):
    """
    Check for the existence of all KnowledgeDiscoveryProtocol methods.
    """
    assert hasattr(hybrid_registry, 'discover_patterns')
    assert hasattr(hybrid_registry, 'extract_relationships')
    assert hasattr(hybrid_registry, 'suggest_new_concepts')
    assert hasattr(hybrid_registry, 'validate_knowledge_consistency')

def test_semantic_reasoning_protocol_placeholders(hybrid_registry: HybridConceptRegistry):
    """
    Verify that placeholder methods for SemanticReasoningProtocol are callable.
    """
    with pytest.raises(NotImplementedError):
        hybrid_registry.complete_analogy({})
    
    with pytest.raises(NotImplementedError):
        hybrid_registry.discover_semantic_fields()

    with pytest.raises(NotImplementedError):
        hybrid_registry.find_cross_domain_analogies("domain1", "domain2")

def test_knowledge_discovery_protocol_placeholders(hybrid_registry: HybridConceptRegistry):
    """
    Verify that placeholder methods for KnowledgeDiscoveryProtocol are callable.
    """
    with pytest.raises(NotImplementedError):
        hybrid_registry.discover_patterns("domain")

    with pytest.raises(NotImplementedError):
        hybrid_registry.extract_relationships([])

    with pytest.raises(NotImplementedError):
        hybrid_registry.suggest_new_concepts([])

    with pytest.raises(NotImplementedError):
        hybrid_registry.validate_knowledge_consistency({})
