"""
Design by Contract Framework for Soft Logic System
==================================================

This module provides a comprehensive Design by Contract framework using icontract
that integrates seamlessly with our Protocol-based architecture and type system.

CONTRACT PHILOSOPHY:
===================

- **Protocol Compliance**: Contracts validate Protocol interface implementations
- **Type Safety**: Works with mypy and our strict typing configuration
- **API Reliability**: Ensures service layer robustness for Phase 3C
- **Neural-Symbolic Safety**: Validates boundaries between reasoning systems

USAGE PATTERNS:
===============

```python
@require(lambda args: len(args.name.strip()) > 0, description="concept_name must be non-empty")
@require(lambda args: args.context in VALID_CONTEXTS, description="context must be valid")
@ensure(lambda result: result is not None, description="result must not be None")
def create_concept(name: str, context: str) -> Concept:
    # Implementation
```

ICONTRACT INTEGRATION:
=====================

This module uses icontract for Design by Contract validation, which provides:
- Excellent mypy integration with your existing type system
- Runtime contract validation that can be disabled in production
- FastAPI-compatible decorators for service layer validation
- Protocol-friendly contract definitions
"""

from typing import Any, Callable, List, Dict, Optional, Union
from functools import wraps
import inspect
from dataclasses import dataclass

# Import icontract decorators
from icontract import require, ensure, invariant, ViolationError
CONTRACTS_AVAILABLE = True

# Re-export for convenience
__all__ = ['require', 'ensure', 'invariant', 'ViolationError', 'CONTRACTS_AVAILABLE', 
           'ConceptConstraints', 'EmbeddingConstraints', 'ReasoningConstraints',
           'validate_concept_name', 'validate_embedding_dimensions', 
           'validate_coherence_score', 'validate_context', 'SoftLogicContracts']


# Domain-specific constraint validators
class ConceptConstraints:
    """Constraint validators for concept operations."""
    
    @staticmethod
    def valid_concept_name(name: str) -> bool:
        """Validate concept name is non-empty and well-formed."""
        return isinstance(name, str) and len(name.strip()) > 0 and len(name) <= 100
    
    @staticmethod
    def valid_context(context: str) -> bool:
        """Validate context name is acceptable."""
        valid_contexts = {"default", "test", "production", "development", "royalty", "medieval", "fantasy"}
        return isinstance(context, str) and (context in valid_contexts or len(context.strip()) > 0)
    
    @staticmethod 
    def valid_synset_id(synset_id: Optional[str]) -> bool:
        """Validate WordNet synset ID format."""
        if synset_id is None:
            return True
        return isinstance(synset_id, str) and '.' in synset_id and len(synset_id) > 5
    
    @staticmethod
    def valid_disambiguation(disambiguation: Optional[str]) -> bool:
        """Validate disambiguation text."""
        if disambiguation is None:
            return True
        return isinstance(disambiguation, str) and len(disambiguation.strip()) > 0


class EmbeddingConstraints:
    """Constraint validators for embedding operations."""
    
    @staticmethod
    def valid_embedding_dimension(dimension: int) -> bool:
        """Validate embedding dimension is reasonable."""
        return isinstance(dimension, int) and 1 <= dimension <= 4096
    
    @staticmethod
    def valid_similarity_score(score: float) -> bool:
        """Validate similarity score is in valid range."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def valid_threshold(threshold: float) -> bool:
        """Validate threshold parameter is reasonable."""
        return isinstance(threshold, (int, float)) and 0.0 <= threshold <= 1.0
    
    @staticmethod
    def valid_embedding_provider(provider: str) -> bool:
        """Validate embedding provider name."""
        valid_providers = {"random", "semantic", "openai", "huggingface"}
        return isinstance(provider, str) and provider in valid_providers


class ReasoningConstraints:
    """Constraint validators for reasoning operations."""
    
    @staticmethod
    def valid_coherence_score(score: float) -> bool:
        """Validate coherence score is in valid range."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def valid_confidence_score(score: float) -> bool:
        """Validate confidence score is in valid range."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def valid_concept_list(concepts: List[str]) -> bool:
        """Validate list of concept names."""
        return (isinstance(concepts, list) and 
                len(concepts) > 0 and 
                all(isinstance(c, str) and len(c.strip()) > 0 for c in concepts))
    
    @staticmethod
    def valid_partial_analogy(analogy: Dict[str, str]) -> bool:
        """Validate partial analogy dictionary."""
        return (isinstance(analogy, dict) and 
                len(analogy) >= 2 and
                "?" in analogy.values() and
                all(isinstance(k, str) and isinstance(v, str) for k, v in analogy.items()))
    
    @staticmethod
    def valid_max_completions(max_comp: int) -> bool:
        """Validate maximum completions parameter."""
        return isinstance(max_comp, int) and 1 <= max_comp <= 50


# Convenience validators that accept full argument objects
def validate_concept_name(args: Any) -> bool:
    """Validate concept name from args object."""
    return hasattr(args, 'name') and ConceptConstraints.valid_concept_name(args.name)

def validate_context(args: Any) -> bool:
    """Validate context from args object."""
    return hasattr(args, 'context') and ConceptConstraints.valid_context(args.context)

def validate_embedding_dimensions(args: Any) -> bool:
    """Validate embedding dimensions from args object."""
    return (hasattr(args, 'embedding') and 
            hasattr(args.embedding, 'shape') and 
            EmbeddingConstraints.valid_embedding_dimension(args.embedding.shape[0]))

def validate_coherence_score(args: Any) -> bool:
    """Validate coherence score from args object."""
    return (hasattr(args, 'coherence') and 
            ReasoningConstraints.valid_coherence_score(args.coherence))


# Contract templates for common patterns
class ContractTemplates:
    """Pre-defined contract templates for common operation patterns."""
    
    @staticmethod
    def concept_creation_contracts() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Standard contracts for concept creation operations."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @require(lambda args: validate_concept_name(args), description="name must be valid")
            @require(lambda args: validate_context(args), description="context must be valid")
            @ensure(lambda result: result is not None, description="result is not None")
            @ensure(lambda result: hasattr(result, 'concept_id'), description="result has concept_id")
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def similarity_operation_contracts() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Standard contracts for similarity operations."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @require(lambda concepts: ReasoningConstraints.valid_concept_list(concepts), description="concepts list is valid")
            @require(lambda concepts: len(concepts) >= 2, description="at least 2 concepts")
            @ensure(lambda result: isinstance(result, (int, float)), description="result is numeric")
            @ensure(lambda result: EmbeddingConstraints.valid_similarity_score(result), description="similarity score is valid")
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)
            return wrapper
        return decorator


# Domain-specific contract decorators
def semantic_field_discovery_contracts(func: Callable[..., Any]) -> Callable[..., Any]:
    """Contracts for semantic field discovery operations."""
    @require(lambda min_coherence: ReasoningConstraints.valid_coherence_score(min_coherence), 
             description="min_coherence must be valid score")
    @ensure(lambda result: isinstance(result, list), description="result must be list")
    @ensure(lambda result: all(hasattr(field, 'coherence') for field in result), 
            description="all fields must have coherence")
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


def analogical_completion_contracts(func: Callable[..., Any]) -> Callable[..., Any]:
    """Contracts for analogical completion operations."""
    @require(lambda partial_analogy, max_completions=5: ReasoningConstraints.valid_partial_analogy(partial_analogy),
             description="partial_analogy must be valid")
    @require(lambda partial_analogy, max_completions=5: ReasoningConstraints.valid_max_completions(max_completions),
             description="max_completions must be valid")
    @ensure(lambda result: isinstance(result, list), description="result must be list")
    @ensure(lambda result, partial_analogy, max_completions=5: len(result) <= max_completions,
            description="result count within limit")
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper


# Class invariant helpers
def registry_consistency_invariant(description: str, condition: Callable[[Any], bool]) -> Any:
    """Create class invariant for registry consistency."""
    return invariant(lambda self: condition(self), description=description)


# Example usage patterns
class ContractedConceptRegistry:
    """Example of contract-enhanced concept registry."""
    
    def __init__(self) -> None:
        self.concepts: Dict[str, Any] = {}
        self.contexts = {"default"}
    
    @require(lambda name: ConceptConstraints.valid_concept_name(name), description="name must be valid")
    @require(lambda name, context: ConceptConstraints.valid_context(context), description="context must be valid")
    @ensure(lambda result: result is not None, description="result not None")
    def create_concept_with_contracts(self, name: str, context: str) -> Any:
        """Create concept with comprehensive contract validation."""
        # Implementation
        concept_id = f"{context}:{name}"
        concept = type('Concept', (), {
            'concept_id': concept_id,
            'name': name,
            'context': context
        })()
        
        self.concepts[concept_id] = concept
        self.contexts.add(context)
        
        return concept


# Registry state validation functions
def validate_registry_state(registry: Any) -> bool:
    """Validate overall registry state consistency."""
    try:
        # Check basic structure
        if not hasattr(registry, 'frame_aware_concepts'):
            return False
        
        # Check concept consistency
        for concept_id, concept in getattr(registry, 'frame_aware_concepts', {}).items():
            if not hasattr(concept, 'name'):
                return False
            if not hasattr(concept, 'context'):
                return False
        
        return True
    except Exception:
        return False


def validate_embedding_consistency(registry: Any) -> bool:
    """Validate embedding system consistency."""
    try:
        if not hasattr(registry, 'cluster_registry'):
            return True  # No cluster registry is fine
        
        cluster_registry = registry.cluster_registry
        if not hasattr(cluster_registry, 'concept_embeddings'):
            return True  # No embeddings is fine
        
        # Check embedding dimensions are consistent
        embeddings = cluster_registry.concept_embeddings
        if not embeddings:
            return True
        
        first_dim = None
        for concept_id, embedding in embeddings.items():
            if hasattr(embedding, 'shape'):
                if first_dim is None:
                    first_dim = embedding.shape[0]
                elif embedding.shape[0] != first_dim:
                    return False  # Inconsistent dimensions
        
        return True
    except Exception:
        return False


def validate_frame_consistency(registry: Any) -> bool:
    """Validate semantic frame consistency."""
    try:
        if not hasattr(registry, 'frame_registry'):
            return True  # No frame registry is fine
        
        frame_registry = registry.frame_registry
        if not hasattr(frame_registry, 'frames'):
            return True  # No frames is fine
        
        # Check frame structure
        for frame_name, frame in getattr(frame_registry, 'frames', {}).items():
            if not hasattr(frame, 'name'):
                return False
            if frame.name != frame_name:
                return False
        
        return True
    except Exception:
        return False


# Domain-specific contract validators
class SoftLogicContracts:
    """Contract validators for soft logic domain operations."""
    
    @staticmethod
    def valid_concept_name(name: str) -> bool:
        """Validate concept name format."""
        return (isinstance(name, str) and 
                len(name.strip()) > 0 and 
                len(name.strip()) <= 100 and
                not name.startswith('_'))
    
    @staticmethod
    def valid_context(context: str) -> bool:
        """Validate context identifier."""
        valid_contexts = {'default', 'wordnet', 'custom', 'neural', 'hybrid', 'royalty', 
                         'animal', 'magic', 'creatures', 'military', 'locations', 'artifacts',
                         'business', 'sports'}
        return isinstance(context, str) and context in valid_contexts
    
    @staticmethod
    def valid_coherence_score(score: float) -> bool:
        """Validate coherence score range."""
        return isinstance(score, (int, float)) and 0.0 <= score <= 1.0
    
    @staticmethod
    def valid_confidence_score(confidence: float) -> bool:
        """Validate confidence score range."""
        return isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
    
    @staticmethod
    def valid_analogy_mapping(mapping: Dict[str, str]) -> bool:
        """Validate analogical mapping structure."""
        return (isinstance(mapping, dict) and 
                len(mapping) >= 2 and
                "?" in mapping.values())
    
    @staticmethod
    def valid_max_results(max_results: int) -> bool:
        """Validate maximum results parameter."""
        return isinstance(max_results, int) and 1 <= max_results <= 100

    @staticmethod
    def valid_embedding_dimension(dimension: int) -> bool:
        """Validate embedding dimension."""
        return isinstance(dimension, int) and 1 <= dimension <= 10000

    @staticmethod
    def valid_training_epochs(epochs: int) -> bool:
        """Validate number of training epochs."""
        return isinstance(epochs, int) and 1 <= epochs <= 10000

    @staticmethod
    def valid_learning_rate(lr: float) -> bool:
        """Validate learning rate."""
        return isinstance(lr, (int, float)) and 0.0 < lr <= 1.0

    @staticmethod
    def valid_batch_size(batch_size: int) -> bool:
        """Validate batch size."""
        return isinstance(batch_size, int) and 1 <= batch_size <= 10000


# Export contract validation functions
__all__.extend([
    'semantic_field_discovery_contracts',
    'analogical_completion_contracts', 
    'registry_consistency_invariant',
    'ContractedConceptRegistry',
    'validate_registry_state',
    'validate_embedding_consistency',
    'validate_frame_consistency',
    'SoftLogicContracts'
])
