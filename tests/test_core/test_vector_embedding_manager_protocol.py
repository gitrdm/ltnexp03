import pytest
import numpy as np
from app.core.vector_embeddings import VectorEmbeddingManager
from app.core.protocols import EmbeddingProviderProtocol

def test_vector_embedding_manager_protocol_compliance():
    manager = VectorEmbeddingManager(default_provider="random")
    assert isinstance(manager, EmbeddingProviderProtocol)
    # Test protocol methods
    emb = manager.generate_embedding("test_concept")
    assert isinstance(emb, np.ndarray)
    batch = manager.batch_generate_embeddings(["a", "b"])
    assert isinstance(batch, dict)
    assert all(isinstance(v, np.ndarray) for v in batch.values())
    sim = manager.compute_similarity(emb, emb)
    assert isinstance(sim, float)
    assert 0.0 <= sim <= 1.0
    assert isinstance(manager.embedding_dimension, int)
    assert isinstance(manager.provider_name, str)
