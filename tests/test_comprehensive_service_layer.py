#!/usr/bin/env python3
"""
Complete Service Layer Test Suite
==================================

Tests both the working service layer and the full-featured service layer
to ensure all components work correctly in production.
"""

import subprocess
import time
import requests
import json
import signal
import os
import sys
from pathlib import Path


def test_service_layer_import(service_module):
    """Test if service layer can be imported."""
    print(f"Testing {service_module} import...")
    try:
        proc = subprocess.run([
            sys.executable, "-c", f"from app.{service_module} import app; print('✅ Import successful')"
        ], capture_output=True, text=True, timeout=30)
        
        if proc.returncode == 0:
            print(f"✅ {service_module} import successful")
            return True
        else:
            print(f"❌ {service_module} import failed: {proc.stderr}")
            return False
    except Exception as e:
        print(f"❌ {service_module} import failed: {e}")
        return False


def start_service_layer_server(service_module, port):
    """Start a service layer server."""
    print(f"🚀 Starting {service_module} server on port {port}...")
    
    # Start the server in background
    proc = subprocess.Popen([
        sys.executable, "-c",
        f"from app.{service_module} import app; import uvicorn; uvicorn.run(app, host='127.0.0.1', port={port}, log_level='error')"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    for i in range(15):
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ {service_module} server started successfully on port {port}")
                return proc
        except:
            time.sleep(2)
    
    # If we get here, server failed to start
    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=10)
        print(f"❌ {service_module} server failed to start.")
        print(f"stdout: {stdout.decode()}")
        print(f"stderr: {stderr.decode()}")
    except:
        print(f"❌ {service_module} server failed to start (timeout)")
    return None


def stop_server(proc, name):
    """Stop the server process."""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"✅ {name} server stopped")


def test_basic_endpoints(port, service_name):
    """Test basic endpoints of a service."""
    print(f"Testing {service_name} basic endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ {service_name} health endpoint failed")
            return False
        
        health_data = response.json()
        if health_data.get("status") != "healthy":
            print(f"❌ {service_name} health status not healthy")
            return False
        
        # Test status endpoint
        response = requests.get(f"http://127.0.0.1:{port}/status", timeout=5)
        if response.status_code != 200:
            print(f"❌ {service_name} status endpoint failed")
            return False
        
        print(f"✅ {service_name} basic endpoints working")
        return True
        
    except Exception as e:
        print(f"❌ {service_name} basic endpoints failed: {e}")
        return False


def test_concept_operations(port, service_name):
    """Test concept operations."""
    print(f"Testing {service_name} concept operations...")
    
    try:
        # Create a concept
        concept_data = {
            "name": f"{service_name}_test_concept",
            "context": "test_domain",
            "auto_disambiguate": True
        }
        
        response = requests.post(
            f"http://127.0.0.1:{port}/concepts",
            json=concept_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ {service_name} concept creation failed: {response.text}")
            return False
        
        data = response.json()
        if "concept_id" not in data:
            print(f"❌ {service_name} concept creation missing concept_id")
            return False
        
        concept_id = data["concept_id"]
        
        # Retrieve the concept
        response = requests.get(f"http://127.0.0.1:{port}/concepts/{concept_id}", timeout=5)
        if response.status_code != 200:
            print(f"❌ {service_name} concept retrieval failed")
            return False
        
        print(f"✅ {service_name} concept operations working")
        return True
        
    except Exception as e:
        print(f"❌ {service_name} concept operations failed: {e}")
        return False


def test_analogy_operations(port, service_name):
    """Test analogy operations."""
    print(f"Testing {service_name} analogy operations...")
    
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/analogies/complete?source_a=test&source_b=value&target_a=example&max_completions=2",
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ {service_name} analogy completion failed: {response.text}")
            return False
        
        data = response.json()
        if "completions" not in data:
            print(f"❌ {service_name} analogy completion missing completions")
            return False
        
        print(f"✅ {service_name} analogy operations working")
        return True
        
    except Exception as e:
        print(f"❌ {service_name} analogy operations failed: {e}")
        return False


def comprehensive_service_test(service_module, port):
    """Run comprehensive tests on a service layer."""
    print(f"\n{'='*60}")
    print(f"Testing {service_module}")
    print(f"{'='*60}")
    
    # Test import
    if not test_service_layer_import(service_module):
        return False
    
    # Start server
    proc = start_service_layer_server(service_module, port)
    if not proc:
        return False
    
    try:
        # Run tests
        success = True
        success &= test_basic_endpoints(port, service_module)
        success &= test_concept_operations(port, service_module)
        success &= test_analogy_operations(port, service_module)
        
        if success:
            print(f"🎉 {service_module} tests passed!")
        else:
            print(f"❌ {service_module} tests failed!")
        
        return success
        
    finally:
        stop_server(proc, service_module)


def main():
    """Run comprehensive service layer tests."""
    print("=" * 80)
    print("COMPREHENSIVE SERVICE LAYER TEST SUITE")
    print("=" * 80)
    
    # Change to correct directory
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Test both service layers
    working_success = comprehensive_service_test("working_service_layer", 8322)
    full_success = comprehensive_service_test("service_layer", 8323)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Working Service Layer: {'✅ PASS' if working_success else '❌ FAIL'}")
    print(f"Full Service Layer:    {'✅ PASS' if full_success else '❌ FAIL'}")
    
    if working_success and full_success:
        print("\n🎉 ALL SERVICE LAYER TESTS PASSED!")
        print("✅ Phase 3C (Service Layer) is complete and functional!")
    elif working_success:
        print("\n⚠️  Working service layer passed, full service layer needs debugging")
        print("✅ Core functionality is working")
    else:
        print("\n❌ Service layer tests failed")
    
    print("=" * 80)
    return working_success and full_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
