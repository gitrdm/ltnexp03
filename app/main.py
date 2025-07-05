from fastapi import FastAPI
import uvicorn

app = FastAPI(
    title="LTN Experiment 03",
    description="A minimal Python microservice",
    version="0.1.0"
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ltnexp03"}


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to LTN Experiment 03", "version": "0.1.0"}


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
