"""
Main Application Entry Point for Soft Logic Microservice
========================================================

This module provides the main FastAPI application with integrated:
- Complete service layer with semantic reasoning
- Persistence and batch operations
- Health checks and system monitoring
- Backward compatibility with original endpoints

For the complete API, use the service layer directly.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
import logging

# Import the complete service layer and its initialization functions
from app.service_layer import app as service_app
from app.service_layer import initialize_services

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Main application lifespan that ensures service initialization."""
    try:
        logger.info("🚀 Starting main application...")
        
        # Initialize the services (this triggers the service layer initialization)
        logger.info("Initializing underlying services...")
        initialize_services()
        
        logger.info("✅ Main application startup complete")
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start main application: {e}")
        raise
    finally:
        logger.info("🛑 Main application shutdown complete")


# Create main application with lifespan
app = FastAPI(
    title="LTN Experiment 03 - Main Application",
    description="Soft Logic Microservice with complete service layer integration",
    version="0.1.0",
    lifespan=lifespan
)

# Mount the complete service layer
app.mount("/api", service_app)


@app.get("/")
async def root():
    """Root endpoint - redirect to service layer API docs."""
    return RedirectResponse(url="/api/docs")


@app.get("/health")
async def health_check():
    """Legacy health check endpoint."""
    return {"status": "healthy", "service": "ltnexp03", "version": "0.1.0"}


@app.get("/service-info")
async def service_info():
    """Get information about available services."""
    return {
        "message": "Welcome to LTN Experiment 03",
        "version": "0.1.0",
        "services": {
            "main_api": "Complete service layer mounted at /api",
            "legacy_health": "Basic health check at /health",
            "documentation": {
                "interactive_docs": "/api/docs",
                "redoc": "/api/redoc",
                "api_overview": "/api/docs-overview"
            }
        },
        "endpoints": {
            "concepts": "/api/concepts",
            "reasoning": "/api/analogies",
            "frames": "/api/frames",
            "batch": "/api/batch",
            "streaming": "/api/ws",
            "system": "/api/health"
        }
    }


def start_server():
    """Convenience function to start the server"""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8321,
        reload=True
    )


if __name__ == "__main__":
    start_server()
