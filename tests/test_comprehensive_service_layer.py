#!/usr/bin/env python3
"""
Complete Service Layer Test Suite
==================================

Tests both the working service layer and the full-featured service layer
to ensure all components work correctly in production.
"""

import pytest
import requests
import json


pytestmark = pytest.mark.usefixtures("live_service")


def test_service_layer_import(service_module):
    """Test if service layer can be imported."""
    assert service_module is not None


def test_basic_endpoints(port, service_name):
    """Test basic endpoints of a service."""
    # Test health endpoint (root)
    response = requests.get(f"http://127.0.0.1:{port}/health")
    assert response.status_code == 200
    health_data = response.json()
    assert health_data.get("status") == "healthy"
    
    # Test status endpoint (mounted on /api)
    response = requests.get(f"http://127.0.0.1:{port}/api/status")
    assert response.status_code == 200


def test_concept_operations(port, service_name):
    """Test concept operations."""
    # Create a concept
    concept_data = {
        "name": f"{service_name}_test_concept",
        "context": "test_domain",
        "auto_disambiguate": True
    }
    
    response = requests.post(
        f"http://127.0.0.1:{port}/api/concepts",
        json=concept_data,
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "concept_id" in data
    
    concept_id = data["concept_id"]
    
    # Retrieve the concept
    response = requests.get(f"http://127.0.0.1:{port}/api/concepts/{concept_id}")
    assert response.status_code == 200


def test_analogy_operations(port, service_name):
    """Test analogy operations."""
    analogy_data = {
        "source_a": "test",
        "source_b": "value",
        "target_a": "example"
    }
    response = requests.post(
        f"http://127.0.0.1:{port}/api/analogies/complete",
        json=analogy_data
    )
    assert response.status_code == 200
    data = response.json()
    assert "completions" in data
