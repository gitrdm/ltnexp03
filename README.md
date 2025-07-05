# LTN Experiment 03

A minimal Python microservice built with FastAPI.

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate ltnexp03
```

2. Install dependencies with Poetry:
```bash
poetry install
```

## Development

### Running the server
```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8321 --reload
```

Or use the convenience script:
```bash
poetry run start
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

### Testing
```bash
poetry run pytest
```

## API

The service runs on port 8321 by default.

- Health check: `GET /health`
- API docs: `GET /docs` (Swagger UI)
- OpenAPI spec: `GET /openapi.json`
