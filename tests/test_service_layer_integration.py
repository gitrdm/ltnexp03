#!/usr/bin/env python3
"""
Service Layer Integration Test (Without TestClient)
===================================================

Tests the service layer functionality by starting a real server and making HTTP requests.
This avoids the TestClient compatibility issues we've been experiencing.
"""

import pytest
import requests
import json

pytestmark = pytest.mark.usefixtures("live_service")

created_concept_id = None

def test_health_endpoint(port):
    """Test the health endpoint."""
    response = requests.get(f"http://127.0.0.1:{port}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_concept_creation(port):
    """Test concept creation."""
    global created_concept_id
    concept_data = {
        "name": "integration_test_knight",
        "context": "test_medieval",
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
    created_concept_id = data["concept_id"]

def test_concept_retrieval(port):
    """Test concept retrieval."""
    global created_concept_id
    assert created_concept_id is not None, "Concept ID not created"
    response = requests.get(f"http://127.0.0.1:{port}/api/concepts/{created_concept_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["concept_id"] == created_concept_id

def test_concept_search(port):
    """Test concept search."""
    search_data = {
        "query": "knight"
    }
    response = requests.post(
        f"http://127.0.0.1:{port}/api/concepts/search",
        json=search_data
    )
    assert response.status_code == 200
    data = response.json()
    assert "concepts" in data

def test_analogy_completion(port):
    """Test analogy completion."""
    analogy_data = {
        "source_a": "knight",
        "source_b": "sword",
        "target_a": "wizard"
    }
    response = requests.post(
        f"http://127.0.0.1:{port}/api/analogies/complete",
        json=analogy_data
    )
    assert response.status_code == 200
    data = response.json()
    assert "completions" in data

def test_batch_workflows(port):
    """Test batch workflow listing."""
    response = requests.get(f"http://127.0.0.1:{port}/api/batch/workflows")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_service_status(port):
    """Test service status endpoint."""
    response = requests.get(f"http://127.0.0.1:{port}/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
