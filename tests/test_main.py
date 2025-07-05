import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ltnexp03"


def test_root_endpoint(client):
    """Test the root endpoint - it redirects to API docs"""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307  # RedirectResponse uses 307
    assert "location" in response.headers
    assert "/api/docs" in response.headers["location"]
