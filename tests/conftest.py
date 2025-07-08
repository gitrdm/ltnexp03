import pytest
import sys
import os
import subprocess
import time
import requests
from requests.exceptions import ConnectionError

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def service_name():
    """Service name for testing."""
    return "test_service"

@pytest.fixture(scope="session")
def port():
    """Port for the test service."""
    return 8123

@pytest.fixture(scope="session")
def service_module():
    """Module for the service layer to be tested."""
    from app import service_layer
    return service_layer

@pytest.fixture(scope="session")
def live_service(port):
    """Fixture to start and stop the service for integration tests."""
    command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)]
    process = subprocess.Popen(command)
    
    # Wait for the service to be ready
    retries = 5
    while retries > 0:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health")
            if response.status_code == 200:
                break
        except ConnectionError:
            time.sleep(1)
            retries -= 1
    
    if retries == 0:
        process.terminate()
        pytest.fail("Service did not start in time.")

    yield
    
    process.terminate()
    process.wait()
