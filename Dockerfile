# Neural-Symbolic AI Platform Dockerfile
# --------------------------------------
# Production-ready Dockerfile for FastAPI + Poetry + Conda

FROM continuumio/miniconda3:latest

# Set workdir
WORKDIR /app

# Copy environment and poetry files
COPY environment.yml ./
COPY pyproject.toml ./
COPY poetry.lock ./

# Create and activate conda environment
RUN conda env create -f environment.yml && \
    echo "conda activate ltnexp03" >> ~/.bashrc

# Install Poetry in the conda environment
RUN /bin/bash -c "source activate ltnexp03 && pip install poetry"

# Copy the rest of the code
COPY . .

# Install project dependencies with Poetry
RUN /bin/bash -c "source activate ltnexp03 && poetry install --no-interaction --no-root"

# Expose FastAPI port
EXPOSE 8321

# Set environment variables for production
ENV API_HOST=0.0.0.0 \
    API_PORT=8321 \
    STORAGE_PATH=/app/storage \
    LOG_LEVEL=INFO \
    ENABLE_CONTRACTS=true

# Default command: start FastAPI server
CMD ["/bin/bash", "-c", "source activate ltnexp03 && poetry run uvicorn app.main:app --host 0.0.0.0 --port 8321 --workers 2"]
