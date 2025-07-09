"""
Tests for Protocol Compliance
=============================

This module contains tests to verify that concrete classes correctly implement
their specified Protocol interfaces. This ensures that our abstractions are
consistently applied and that type safety is maintained.
"""

import pytest
from app.core.protocols import ConceptRegistryProtocol
from app.core.concept_registry import ConceptRegistry
from app.core.abstractions import Concept

def test_concept_registry_complies_with_protocol():
    """
    Tests that ConceptRegistry is a valid implementation of ConceptRegistryProtocol.
    
    This test leverages the @runtime_checkable decorator on the protocol to
    perform an isinstance() check, which validates structural compliance.
    """
    # Arrange
    registry = ConceptRegistry(download_wordnet=False)
    
    # Act & Assert
    # This check will fail if ConceptRegistry is missing any methods or properties
    # defined in the ConceptRegistryProtocol.
    assert isinstance(registry, ConceptRegistryProtocol), \
        "ConceptRegistry does not comply with ConceptRegistryProtocol"

    # Further check for one of the methods to be sure
    assert hasattr(registry, "create_concept")

