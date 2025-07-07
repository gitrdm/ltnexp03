"""
Protocol Implementation Mixins for Type-Safe Interface Compliance
================================================================

This module provides mixin classes that implement Protocol interfaces,
allowing existing classes to gain type-safe interface compliance without
major refactoring.

DESIGN APPROACH:
================

Rather than forcing existing classes to implement all protocol methods
directly, we provide mixin classes that add the required interface methods
while delegating to existing implementations. This provides:

1. **Gradual Migration**: Existing code continues to work
2. **Type Safety**: Protocols ensure interface compliance
3. **Flexibility**: Multiple implementation strategies
4. **Testing**: Clear interface boundaries for testing

USAGE PATTERN:
==============

```python
class MyRegistry(BaseRegistry, SemanticReasoningMixin):
    # Existing implementation continues to work
    # Mixin adds protocol-compliant interface methods
    pass

# Now MyRegistry implements SemanticReasoningProtocol
assert isinstance(MyRegistry(), SemanticReasoningProtocol)
```
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from abc import ABC, abstractmethod

from .protocols import (
    SemanticReasoningProtocol, KnowledgeDiscoveryProtocol,
    EmbeddingProviderProtocol
)


class SemanticReasoningMixin(SemanticReasoningProtocol):
    """
    Mixin that provides SemanticReasoningProtocol implementation.
    
    This mixin adapts existing enhanced semantic reasoning methods
    to match the protocol interface exactly.
    
    EXPECTED METHODS:
    - find_analogical_completions(partial_analogy, max_completions)
    - find_analogous_concepts(source_concept, frame_context, cluster_threshold, frame_threshold)  
    - discover_semantic_fields(min_coherence)
    - discover_cross_domain_analogies(min_quality)
    """
    
    def complete_analogy(
        self, 
        partial_analogy: Dict[str, str], 
        max_completions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete partial analogies using existing find_analogical_completions.
        
        Args:
            partial_analogy: Dict with known mappings and "?" for completion
            max_completions: Maximum number of completions to return
            
        Returns:
            List of completion results with confidence scores
        """
        if hasattr(self, 'find_analogical_completions'):
            # Use existing implementation
            raw_results = self.find_analogical_completions(
                partial_analogy, max_completions
            )
            
            # Convert to protocol-compliant format
            formatted_results = []
            for result in raw_results:
                if isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    # Handle different result formats
                    formatted_results.append({
                        'completion': str(result),
                        'confidence': 0.8,  # Default confidence
                        'method': 'hybrid'
                    })
            
            return formatted_results[:max_completions]
        else:
            # Fallback implementation
            logging.warning("find_analogical_completions not available, using fallback")
            return []
    
    def find_analogous_concepts(
        self,
        source_concept: Any,
        frame_context: Optional[str] = None,
        cluster_threshold: float = 0.6,
        frame_threshold: float = 0.6
    ) -> List[Tuple[Any, float, str]]:
        """
        Find concepts analogous to the given source concept.
        
        Adapts to existing find_analogous_concepts method signature.
        """
        if hasattr(self, 'find_analogous_concepts'):
            # Use existing implementation with compatible signature
            return self.find_analogous_concepts(
                source_concept=source_concept,
                frame_context=frame_context,
                cluster_threshold=cluster_threshold,
                frame_threshold=frame_threshold
            )
        else:
            # Fallback implementation
            logging.warning("find_analogous_concepts not available, using fallback")
            return []
    
    def discover_semantic_fields(
        self, 
        min_coherence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Discover coherent semantic fields from concept space.
        
        Args:
            min_coherence: Minimum coherence score for discovered fields
            
        Returns:
            List of semantic field descriptions with metadata
        """
        if hasattr(self, 'discover_semantic_fields'):
            # Use existing implementation
            raw_fields = self.discover_semantic_fields(min_coherence)
            
            # Convert to protocol-compliant format
            formatted_fields = []
            for field in raw_fields:
                if hasattr(field, '__dict__'):
                    # Convert object to dict
                    field_dict = {
                        'name': getattr(field, 'name', 'Unknown'),
                        'coherence': getattr(field, 'coherence', min_coherence),
                        'core_concepts': getattr(field, 'core_concepts', []),
                        'related_concepts': getattr(field, 'related_concepts', {}),
                        'description': getattr(field, 'description', '')
                    }
                    formatted_fields.append(field_dict)
                else:
                    formatted_fields.append(field)
                    
            return formatted_fields
        else:
            # Fallback implementation
            logging.warning("discover_semantic_fields not available, using fallback")
            return []
    
    def find_cross_domain_analogies(
        self, 
        source_domain: str,
        target_domain: str,
        min_quality: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find structural analogies between different domains."""
        if hasattr(self, 'discover_cross_domain_analogies'):
            # Use existing implementation (adapt interface)
            raw_analogies = self.discover_cross_domain_analogies(min_quality)
            
            # Filter and format for specific domains if possible
            formatted_analogies = []
            for analogy in raw_analogies:
                if hasattr(analogy, '__dict__'):
                    # Convert object to dict
                    analogy_dict = {
                        'source_domain': source_domain,
                        'target_domain': target_domain,
                        'quality': getattr(analogy, 'quality', min_quality),
                        'concept_mappings': getattr(analogy, 'concept_mappings', {}),
                        'frame_mappings': getattr(analogy, 'frame_mappings', {}),
                        'structural_coherence': getattr(analogy, 'structural_coherence', 0.0)
                    }
                    formatted_analogies.append(analogy_dict)
                else:
                    formatted_analogies.append(analogy)
                    
            return formatted_analogies
        else:
            # Fallback implementation
            logging.warning("discover_cross_domain_analogies not available, using fallback")
            return []


class KnowledgeDiscoveryMixin(KnowledgeDiscoveryProtocol):
    """
    Mixin that provides KnowledgeDiscoveryProtocol implementation.
    
    This mixin provides basic knowledge discovery capabilities that
    can be extended by concrete implementations.
    """
    
    def discover_patterns(
        self, 
        domain: str,
        pattern_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Discover patterns within a specific domain."""
        # Basic implementation - can be overridden
        patterns = []
        
        if hasattr(self, 'discover_semantic_fields'):
            # Use semantic fields as patterns
            fields = self.discover_semantic_fields(min_coherence=0.5)
            for field in fields:
                if hasattr(field, '__dict__'):
                    pattern = {
                        'type': 'semantic_field',
                        'domain': domain,
                        'name': getattr(field, 'name', 'Unknown'),
                        'elements': getattr(field, 'core_concepts', []),
                        'confidence': getattr(field, 'coherence', 0.5)
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def extract_relationships(
        self, 
        concepts: List[str],
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships between given concepts."""
        relationships = []
        
        # Basic similarity-based relationships
        if hasattr(self, 'find_analogous_concepts'):
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # Check if concepts are analogous
                    analogous = self.find_analogous_concepts(concept1)
                    for analog, score, method in analogous:
                        if hasattr(analog, 'name') and analog.name == concept2:
                            relationship = {
                                'source': concept1,
                                'target': concept2,
                                'type': 'analogous',
                                'strength': score,
                                'method': method
                            }
                            relationships.append(relationship)
                            break
        
        return relationships
    
    def suggest_new_concepts(
        self,
        existing_concepts: List[str],
        domain: str = "default"
    ) -> List[Dict[str, Any]]:
        """Suggest new concepts that would complement existing ones."""
        suggestions = []
        
        # Basic implementation - suggest concepts from similar semantic fields
        if hasattr(self, 'discover_semantic_fields'):
            fields = self.discover_semantic_fields(min_coherence=0.5)
            
            for field in fields:
                field_concepts = getattr(field, 'core_concepts', [])
                
                # Check if any existing concepts are in this field
                overlap = set(existing_concepts) & set(field_concepts)
                if overlap:
                    # Suggest other concepts from this field
                    for concept in field_concepts:
                        if concept not in existing_concepts:
                            suggestion = {
                                'concept': concept,
                                'domain': domain,
                                'reason': f'Related to {overlap} in {getattr(field, "name", "semantic field")}',
                                'confidence': getattr(field, 'coherence', 0.5),
                                'semantic_field': getattr(field, 'name', 'Unknown')
                            }
                            suggestions.append(suggestion)
        
        return suggestions[:10]  # Limit suggestions
    
    def validate_knowledge_consistency(
        self,
        knowledge_base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate consistency of knowledge base and return issues."""
        validation_results: Dict[str, Any] = {
            'status': 'valid',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Basic validation checks
        if 'concepts' in knowledge_base:
            concepts = knowledge_base['concepts']
            validation_results['metrics']['concept_count'] = len(concepts)
            
            # Check for duplicate names
            concept_names = [c.get('name', '') for c in concepts if isinstance(c, dict)]
            duplicates = [name for name in set(concept_names) if concept_names.count(name) > 1]
            
            if duplicates:
                validation_results['issues'].append({
                    'type': 'duplicate_concepts',
                    'details': f'Duplicate concept names: {duplicates}'
                })
        
        if validation_results['issues']:
            validation_results['status'] = 'invalid'
        elif validation_results['warnings']:
            validation_results['status'] = 'warning'
        
        return validation_results


class EmbeddingProviderMixin(EmbeddingProviderProtocol):
    """
    Mixin that provides EmbeddingProviderProtocol implementation.
    
    This mixin adapts existing embedding functionality to the protocol interface.
    """
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        if hasattr(self, 'embedding_dim'):
            return int(getattr(self, 'embedding_dim', 300))
        return 300  # Default dimension
    
    @property  
    def provider_name(self) -> str:
        """Return identifier for this embedding provider."""
        return getattr(self, 'provider_name', 'default_provider')
    
    def generate_embedding(
        self, 
        concept: str, 
        context: str = "default"
    ) -> Any:  # Using Any instead of np.ndarray to avoid import issues
        """Generate embedding vector for concept in given context."""
        # Delegate to existing embedding generation if available
        if hasattr(self, 'cluster_registry') and hasattr(self.cluster_registry, 'get_concept_embedding'):
            embedding = self.cluster_registry.get_concept_embedding(concept)
            if embedding is not None:
                return embedding
        
        # Fallback: create random embedding
        import numpy as np
        return np.random.random(self.embedding_dimension)
    
    def compute_similarity(
        self, 
        emb1: Any, 
        emb2: Any
    ) -> float:
        """Compute similarity score between two embeddings (0.0-1.0)."""
        # Basic cosine similarity
        try:
            import numpy as np
            emb1_arr = np.array(emb1)
            emb2_arr = np.array(emb2)
            
            # Normalize vectors
            norm1 = np.linalg.norm(emb1_arr)
            norm2 = np.linalg.norm(emb2_arr)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Cosine similarity
            similarity = np.dot(emb1_arr, emb2_arr) / (norm1 * norm2)
            return float(max(0.0, min(1.0, similarity)))  # Clamp to [0, 1]
        except Exception:
            # Fallback
            return 0.5
    
    def batch_generate_embeddings(
        self,
        concepts: List[str],
        context: str = "default"
    ) -> Dict[str, Any]:
        """Generate embeddings for multiple concepts efficiently."""
        embeddings = {}
        for concept in concepts:
            embeddings[concept] = self.generate_embedding(concept, context)
        return embeddings


# Combined mixin for comprehensive protocol compliance
class FullProtocolMixin(SemanticReasoningMixin, KnowledgeDiscoveryMixin, EmbeddingProviderMixin):
    """
    Combined mixin that provides implementation for all major protocols.
    
    Use this when you want a class to implement multiple protocol interfaces
    with sensible default implementations.
    """
    pass


__all__ = [
    'SemanticReasoningMixin',
    'KnowledgeDiscoveryMixin', 
    'EmbeddingProviderMixin',
    'FullProtocolMixin'
]
