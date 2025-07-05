# LTN Experiment 03

A soft logic microservice built with FastAPI, providing facilities for building "soft" logic vectors with LTNtorch and SMT verification.

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ltnexp03
```

2. Install the project and dependencies with Poetry:
```bash
poetry install
```

This installs the project in editable mode, allowing you to import from `app.core` anywhere.

## Development

### Running the server
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8321 --reload
```

Or use the convenience script:
```bash
poetry run start
```

### Exploring the core abstractions
```bash
# Basic functionality tour
python explore_abstractions.py

# Comprehensive demo with knowledge base creation
python demo_abstractions.py
```

### Running tests
```bash
# Run all tests
poetry run pytest

# Run specific test file
python -m pytest tests/test_core/test_abstractions.py -v
```

### Code formatting
```bash
poetry run black .
poetry run isort .
```

### Linting
```bash
poetry run flake8 .
```

## API

The service runs on port 8321 by default.

- Health check: `GET /health`
- API docs: `GET /docs` (Swagger UI)
- OpenAPI spec: `GET /openapi.json`
