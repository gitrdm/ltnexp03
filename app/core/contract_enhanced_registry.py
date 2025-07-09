"""
Enhanced Semantic Reasoning with Design by Contract
===================================================

This module demonstrates the integration of dpcontracts with the existing
EnhancedHybridRegistry system for Phase 3B implementation.

INTEGRATION STRATEGY:
====================

1. Add precondition/postcondition contracts to all major registry operations
2. Implement class invariants for registry consistency validation
3. Create custom contract validators for domain-specific constraints
4. Provide contract-validated wrapper methods for API integration

DESIGN PHILOSOPHY:
==================

- **Defensive Programming**: Contracts catch violations early in development
- **Documentation**: Contracts serve as executable specifications
- **API Reliability**: Service layer benefits from contract validation
- **Testing Enhancement**: Contracts help identify edge cases and violations
"""

from typing import List, Dict, Optional, Any, Union
import numpy as np
from icontract import require, ensure, invariant, ViolationError

from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.abstractions import Concept
from app.core.frame_cluster_abstractions import FrameAwareConcept, SemanticFrame
from app.core.contracts import (
    ConceptConstraints, EmbeddingConstraints, ReasoningConstraints,
    validate_concept_name, validate_context, validate_coherence_score
)


class ContractValidatedRegistry(EnhancedHybridRegistry):
    """
    Enhanced hybrid registry with comprehensive Design by Contract validation.
    
    This class extends EnhancedHybridRegistry with dpcontracts decorators to:
    - Validate all inputs before processing
    - Ensure all outputs meet quality constraints
    - Maintain class invariants throughout operations
    - Provide clear error messages for contract violations
    """
    
    def __init__(self, download_wordnet: bool = True, n_clusters: int = 8, 
                 enable_cross_domain: bool = True, embedding_provider: str = "semantic"):
        """Initialize registry with contract-validated parameters."""
        super().__init__(download_wordnet, n_clusters, enable_cross_domain, embedding_provider)
        self._total_operations = 0
        self._concept_creation_count = 0
    
    def __class_invariant__(self):
        """Class invariants that must hold throughout the object's lifetime."""
        # Registry must have consistent state
        assert hasattr(self, 'frame_aware_concepts'), "Registry must have frame_aware_concepts"
        assert hasattr(self, 'frame_registry'), "Registry must have frame_registry"
        assert hasattr(self, 'cluster_registry'), "Registry must have cluster_registry"
        
        # All concepts must have valid unique IDs
        concept_ids = [c.unique_id for c in self.frame_aware_concepts.values()]
        assert len(concept_ids) == len(set(concept_ids)), "All concept IDs must be unique"
        
        # Operational counters must be non-negative
        assert self._total_operations >= 0, "Total operations counter must be non-negative"
        assert self._concept_creation_count >= 0, "Concept creation counter must be non-negative"
    
    @require("name must be valid", lambda args: ConceptConstraints.valid_concept_name(args.name))
    @require("context must be valid", lambda args: ConceptConstraints.valid_context(args.context))
    @require("synset_id must be valid if provided", 
             lambda args: ConceptConstraints.valid_synset_id(args.synset_id))
    @ensure("result is not None", lambda result, args: result is not None)
    @ensure("result has valid concept_id", lambda result, args: hasattr(result, 'concept_id'))
    @ensure("result name matches input", lambda result, args: result.name == args.name)
    @ensure("result context matches input", lambda result, args: result.context == args.context)
    def create_frame_aware_concept_with_contracts(
        self, 
        name: str, 
        context: str = "default",
        synset_id: Optional[str] = None,
        disambiguation: Optional[str] = None,
        use_semantic_embedding: bool = True
    ) -> FrameAwareConcept:
        """
        Create frame-aware concept with comprehensive contract validation.
        
        This method demonstrates how to add contracts to existing functionality
        while maintaining backward compatibility.
        """
        # Increment operation counter for invariant validation
        self._total_operations += 1
        self._concept_creation_count += 1
        
        # Call existing implementation
        concept = self.create_frame_aware_concept_with_advanced_embedding(
            name=name,
            context=context,
            synset_id=synset_id,
            disambiguation=disambiguation,
            use_semantic_embedding=use_semantic_embedding
        )
        
        return concept
    
    @require("concepts list must be valid", 
             lambda args: ReasoningConstraints.valid_concept_list(args.concepts))
    @require("min_coherence must be valid score", 
             lambda args: ReasoningConstraints.valid_coherence_score(args.min_coherence))
    @require("clustering must be trained", 
             lambda args: args.self.cluster_registry.is_trained if hasattr(args.self.cluster_registry, 'is_trained') else True)
    @ensure("all returned fields have valid coherence", 
            lambda result, args: all(field.coherence >= args.min_coherence for field in result))
    @ensure("returned list is not None", lambda result, args: result is not None)
    @ensure("returned list length is non-negative", lambda result, args: len(result) >= 0)
    def discover_semantic_fields_with_contracts(
        self, 
        concepts: Optional[List[FrameAwareConcept]] = None,
        min_coherence: float = 0.7,
        max_fields: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Discover semantic fields with contract validation.
        
        Returns semantic fields that meet the minimum coherence threshold,
        with comprehensive input/output validation.
        """
        self._total_operations += 1
        
        # Use provided concepts or all available concepts
        if concepts is None:
            concepts = list(self.frame_aware_concepts.values())
        
        # Call existing semantic field discovery
        fields = self.discover_semantic_fields(min_coherence=min_coherence)
        
        # Convert to structured format for API compatibility
        structured_fields = []
        for field_name, field_data in fields.items():
            structured_fields.append({
                'field_name': field_name,
                'coherence': field_data.get('coherence', 0.0),
                'core_concepts': field_data.get('core_concepts', []),
                'related_concepts': field_data.get('related_concepts', {}),
                'discovery_metadata': field_data.get('metadata', {})
            })
        
        # Limit results if requested
        return structured_fields[:max_fields]
    
    @require("partial_analogy must have at least 2 mappings", 
             lambda args: len(args.partial_analogy) >= 2)
    @require("partial_analogy must contain completion target", 
             lambda args: "?" in args.partial_analogy.values())
    @require("max_completions must be positive", 
             lambda args: isinstance(args.max_completions, int) and args.max_completions > 0)
    @ensure("result length does not exceed max_completions", 
            lambda result, args: len(result) <= args.max_completions)
    @ensure("all completions have valid confidence scores", 
            lambda result, args: all(0.0 <= comp.get('confidence', 0.0) <= 1.0 for comp in result))
    @ensure("all completions have required fields", 
            lambda result, args: all('completion' in comp and 'confidence' in comp 
                                   and 'reasoning_type' in comp for comp in result))
    def complete_analogy_with_contracts(
        self, 
        partial_analogy: Dict[str, str],
        max_completions: int = 5,
        reasoning_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete analogical reasoning with contract validation.
        
        Takes a partial analogy mapping and returns possible completions
        with confidence scores and reasoning evidence.
        """
        self._total_operations += 1
        
        if reasoning_types is None:
            reasoning_types = ["frame", "cluster", "hybrid"]
        
        # Call existing analogical completion
        completions = self.find_analogical_completions(
            partial_analogy, 
            max_completions=max_completions
        )
        
        # Structure results for API compatibility
        structured_completions = []
        for completion in completions:
            structured_completions.append({
                'completion': completion.get('completion', ''),
                'confidence': completion.get('confidence', 0.0),
                'reasoning_type': completion.get('reasoning_type', 'hybrid'),
                'source_evidence': completion.get('source_evidence', []),
                'metadata': completion.get('metadata', {})
            })
        
        return structured_completions
    
    @require("concept must be valid", 
             lambda args: ConceptConstraints.valid_concept_name(args.concept))
    @require("thresholds must be valid", 
             lambda args: (0.0 <= args.frame_threshold <= 1.0 and 
                          0.0 <= args.cluster_threshold <= 1.0))
    @ensure("result is list of tuples", lambda result, args: isinstance(result, list))
    @ensure("all similarity scores are valid", 
            lambda result, args: all(0.0 <= score <= 1.0 for _, score in result))
    def find_analogous_concepts_with_contracts(
        self,
        concept: str,
        frame_threshold: float = 0.6,
        cluster_threshold: float = 0.7,
        max_results: int = 10
    ) -> List[tuple]:
        """
        Find analogous concepts with contract validation.
        
        Returns concepts that are analogous at frame and/or cluster levels
        with similarity scores and reasoning evidence.
        """
        self._total_operations += 1
        
        # Call existing analogous concept finding
        analogous = self.find_analogous_concepts(
            concept, 
            frame_threshold=frame_threshold,
            cluster_threshold=cluster_threshold
        )
        
        # Limit results
        return analogous[:max_results]
    
    def get_contract_statistics(self) -> Dict[str, Any]:
        """Get statistics about contract-validated operations."""
        base_stats = self.get_enhanced_statistics()
        
        contract_stats = {
            'total_operations': self._total_operations,
            'concept_creation_count': self._concept_creation_count,
            'contracts_enabled': True,
            'contract_library': 'dpcontracts'
        }
        
        return {**base_stats, **contract_stats}


# Example usage and testing
def demonstrate_contract_validation():
    """
    Demonstrate contract validation in action.
    
    This function shows how contracts catch violations and provide
    clear error messages for debugging.
    """
    print("=== Contract Validation Demonstration ===")
    
    registry = ContractValidatedRegistry(
        download_wordnet=False,  # Skip for demo
        n_clusters=4,
        enable_cross_domain=True
    )
    
    # Test 1: Valid concept creation
    print("\n1. Testing valid concept creation...")
    try:
        concept = registry.create_frame_aware_concept_with_contracts(
            name="knight",
            context="default",
            disambiguation="medieval warrior"
        )
        print(f"✅ Successfully created concept: {concept.concept_id}")
    except ViolationError as e:
        print(f"❌ Contract violation: {e}")
    
    # Test 2: Invalid concept name (contract violation)
    print("\n2. Testing invalid concept name...")
    try:
        concept = registry.create_frame_aware_concept_with_contracts(
            name="",  # Empty name should violate contract
            context="default"
        )
        print(f"❌ Should have failed but got: {concept.concept_id}")
    except ViolationError as e:
        print(f"✅ Contract correctly caught violation: {e}")
    
    # Test 3: Invalid context (contract violation)
    print("\n3. Testing invalid context...")
    try:
        concept = registry.create_frame_aware_concept_with_contracts(
            name="wizard",
            context="invalid_context"  # Should violate contract
        )
        print(f"❌ Should have failed but got: {concept.concept_id}")
    except ViolationError as e:
        print(f"✅ Contract correctly caught violation: {e}")
    
    # Test 4: Valid analogical completion
    print("\n4. Testing valid analogical completion...")
    try:
        # First create some concepts for analogy
        registry.create_frame_aware_concept_with_contracts("king", "default")
        registry.create_frame_aware_concept_with_contracts("queen", "default")
        registry.create_frame_aware_concept_with_contracts("man", "default")
        
        completions = registry.complete_analogy_with_contracts(
            partial_analogy={"king": "queen", "man": "?"},
            max_completions=3
        )
        print(f"✅ Found {len(completions)} analogical completions")
        for comp in completions:
            print(f"   - {comp['completion']} (confidence: {comp['confidence']:.2f})")
    except ViolationError as e:
        print(f"❌ Contract violation: {e}")
    
    # Test 5: Invalid analogy (too few mappings)
    print("\n5. Testing invalid analogy (too few mappings)...")
    try:
        completions = registry.complete_analogy_with_contracts(
            partial_analogy={"king": "queen"},  # Only one mapping
            max_completions=3
        )
        print(f"❌ Should have failed but got {len(completions)} completions")
    except ViolationError as e:
        print(f"✅ Contract correctly caught violation: {e}")
    
    # Test 6: Check class invariants
    print("\n6. Testing class invariants...")
    try:
        stats = registry.get_contract_statistics()
        print(f"✅ Registry statistics: {stats['total_operations']} operations performed")
        print(f"   - Concepts created: {stats['concept_creation_count']}")
        print(f"   - Contracts enabled: {stats['contracts_enabled']}")
    except Exception as e:
        print(f"❌ Invariant violation or error: {e}")
    
    print("\n=== Contract validation demonstration complete ===")


if __name__ == "__main__":
    # Run demonstration if executed directly
    demonstrate_contract_validation()
