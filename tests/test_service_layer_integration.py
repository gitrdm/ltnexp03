#!/usr/bin/env python3
"""
Service Layer Integration Test (Without TestClient)
===================================================

Tests the service layer functionality by starting a real server and making HTTP requests.
This avoids the TestClient compatibility issues we've been experiencing.
"""

import subprocess
import time
import requests
import json
import signal
import os
import sys
from pathlib import Path


def start_server():
    """Start the working service layer server."""
    print("🚀 Starting service layer server...")
    
    # Start the server in background
    proc = subprocess.Popen([
        sys.executable, "-c",
        "from app.working_service_layer import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port=8321, log_level='warning')"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(10):
        try:
            response = requests.get("http://127.0.0.1:8321/health", timeout=1)
            if response.status_code == 200:
                print("✅ Server started successfully")
                return proc
        except:
            time.sleep(1)
    
    # If we get here, server failed to start
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    print(f"❌ Server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")
    return None


def stop_server(proc):
    """Stop the server process."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✅ Server stopped")


def test_health_endpoint():
    """Test the health endpoint."""
    print("Testing health endpoint...")
    response = requests.get("http://127.0.0.1:8321/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "soft-logic-service-layer"
    assert "components" in data
    print("✅ Health endpoint test passed")


def test_concept_creation():
    """Test concept creation."""
    print("Testing concept creation...")
    
    concept_data = {
        "name": "integration_test_knight",
        "context": "test_medieval",
        "auto_disambiguate": True
    }
    
    response = requests.post(
        "http://127.0.0.1:8321/concepts",
        json=concept_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "concept_id" in data
    assert data["name"] == "integration_test_knight"
    assert data["context"] == "test_medieval"
    
    # Store concept_id for later tests
    global created_concept_id
    created_concept_id = data["concept_id"]
    print(f"✅ Concept creation test passed. Created concept: {created_concept_id}")


def test_concept_retrieval():
    """Test concept retrieval."""
    print("Testing concept retrieval...")
    
    response = requests.get(f"http://127.0.0.1:8321/concepts/{created_concept_id}")
    assert response.status_code == 200
    
    data = response.json()
    assert data["concept_id"] == created_concept_id
    assert data["name"] == "integration_test_knight"
    print("✅ Concept retrieval test passed")


def test_concept_search():
    """Test concept search."""
    print("Testing concept search...")
    
    response = requests.post(
        "http://127.0.0.1:8321/concepts/search?query=knight&max_results=5"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "concepts" in data
    assert len(data["concepts"]) > 0
    print("✅ Concept search test passed")


def test_analogy_completion():
    """Test analogy completion."""
    print("Testing analogy completion...")
    
    response = requests.post(
        "http://127.0.0.1:8321/analogies/complete?source_a=knight&source_b=sword&target_a=wizard&max_completions=3"
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "completions" in data
    assert "reasoning_trace" in data
    print("✅ Analogy completion test passed")


def test_batch_workflows():
    """Test batch workflow listing."""
    print("Testing batch workflows...")
    
    response = requests.get("http://127.0.0.1:8321/batch/workflows")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    print("✅ Batch workflows test passed")


def test_service_status():
    """Test service status endpoint."""
    print("Testing service status...")
    
    response = requests.get("http://127.0.0.1:8321/status")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    print("✅ Service status test passed")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Service Layer Integration Tests")
    print("=" * 60)
    
    proc = None
    try:
        # Start server
        proc = start_server()
        if not proc:
            print("❌ Failed to start server. Exiting.")
            return False
        
        # Run tests
        test_health_endpoint()
        test_service_status()
        test_concept_creation()
        test_concept_retrieval()
        test_concept_search()
        test_analogy_completion()
        test_batch_workflows()
        
        print("=" * 60)
        print("🎉 All integration tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Always stop the server
        stop_server(proc)


if __name__ == "__main__":
    # Change to the correct directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Run tests
    success = main()
    sys.exit(0 if success else 1)
