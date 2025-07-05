"""
Practical icontract Integration Example for Phase 3B
====================================================

This module demonstrates how to integrate icontract Design by Contract
decorators with your existing EnhancedHybridRegistry for robust validation.

Run this example to see contract validation in action:
python app/core/icontract_demo.py
"""

from typing import List, Dict, Optional, Any
from icontract import require, ensure, invariant, ViolationError
import os
import sys

# Add the app directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
    from app.core.frame_cluster_abstractions import FrameAwareConcept
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("Warning: Core modules not available, using mock implementations")


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
        valid_contexts = {'default', 'wordnet', 'custom', 'neural', 'hybrid', 'royalty'}
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


# Mock implementations for when core modules aren't available
class MockConcept:
    def __init__(self, name: str, context: str = "default"):
        self.name = name
        self.context = context
        self.unique_id = f"{context}:{name}"


class MockRegistry:
    def __init__(self):
        self.frame_aware_concepts = {}
        self._operation_count = 0
    
    def create_frame_aware_concept_with_advanced_embedding(self, **kwargs):
        concept = MockConcept(kwargs['name'], kwargs.get('context', 'default'))
        self.frame_aware_concepts[concept.unique_id] = concept
        return concept
    
    def discover_semantic_fields(self, **kwargs):
        return {
            'test_field': {
                'coherence': kwargs.get('min_coherence', 0.7),
                'core_concepts': ['test1', 'test2'],
                'metadata': {}
            }
        }
    
    def find_analogical_completions(self, **kwargs):
        return [
            {
                'completion': 'woman',
                'confidence': 0.95,
                'reasoning_type': 'hybrid',
                'source_evidence': ['frame_based', 'cluster_based']
            }
        ]


# Contract-enhanced registry
if IMPORTS_AVAILABLE:
    BaseRegistry = EnhancedHybridRegistry
else:
    BaseRegistry = MockRegistry


@invariant(lambda self: len(self.frame_aware_concepts) >= 0, "Registry must maintain non-negative concept count")
@invariant(lambda self: self._operation_count >= 0, "Operation count must be non-negative")
class ContractEnhancedRegistry(BaseRegistry):
    """
    Enhanced registry with comprehensive icontract validation.
    
    Demonstrates how to add Design by Contract validation to existing
    functionality without breaking backward compatibility.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs) if IMPORTS_AVAILABLE else super().__init__()
        self._operation_count = 0
    
    @require(lambda name: SoftLogicContracts.valid_concept_name(name), 
             "Concept name must be valid (non-empty, reasonable length)")
    @require(lambda context: SoftLogicContracts.valid_context(context),
             "Context must be one of the valid context types")
    @ensure(lambda result: result is not None, 
            "Concept creation must return a valid concept object")
    @ensure(lambda result, name: result.name == name,
            "Created concept must have the specified name")
    @ensure(lambda result, context: result.context == context,
            "Created concept must have the specified context")
    def create_concept_with_contracts(
        self, 
        name: str, 
        context: str = "default",
        synset_id: Optional[str] = None,
        disambiguation: Optional[str] = None
    ) -> Any:
        """
        Create frame-aware concept with comprehensive contract validation.
        
        This method wraps the existing concept creation functionality
        with pre/postcondition validation.
        """
        self._operation_count += 1
        
        # Call existing implementation
        concept = self.create_frame_aware_concept_with_advanced_embedding(
            name=name,
            context=context,
            synset_id=synset_id,
            disambiguation=disambiguation,
            use_semantic_embedding=True
        )
        
        return concept
    
    @require(lambda min_coherence: SoftLogicContracts.valid_coherence_score(min_coherence),
             "Coherence score must be between 0.0 and 1.0")
    @require(lambda max_fields: SoftLogicContracts.valid_max_results(max_fields),
             "Maximum fields must be between 1 and 100")
    @ensure(lambda result, min_coherence: all(field['coherence'] >= min_coherence 
                                            for field in result),
            "All returned fields must meet minimum coherence threshold")
    @ensure(lambda result, max_fields: len(result) <= max_fields,
            "Result count must not exceed maximum requested fields")
    def discover_semantic_fields_with_contracts(
        self, 
        min_coherence: float = 0.7,
        max_fields: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Discover semantic fields with contract validation.
        
        Returns structured semantic field data with guaranteed
        quality constraints.
        """
        self._operation_count += 1
        
        # Call existing implementation
        fields = self.discover_semantic_fields(min_coherence=min_coherence)
        
        # Convert to structured format
        structured_fields = []
        for field_name, field_data in fields.items():
            structured_fields.append({
                'field_name': field_name,
                'coherence': field_data.get('coherence', min_coherence),
                'core_concepts': field_data.get('core_concepts', []),
                'related_concepts': field_data.get('related_concepts', {}),
                'discovery_metadata': field_data.get('metadata', {})
            })
        
        # Apply limit
        return structured_fields[:max_fields]
    
    @require(lambda partial_analogy: SoftLogicContracts.valid_analogy_mapping(partial_analogy),
             "Partial analogy must have at least 2 mappings and contain '?' for completion")
    @require(lambda max_completions: SoftLogicContracts.valid_max_results(max_completions),
             "Maximum completions must be between 1 and 100")
    @ensure(lambda result, max_completions: len(result) <= max_completions,
            "Result count must not exceed maximum requested completions")
    @ensure(lambda result: all(SoftLogicContracts.valid_confidence_score(comp['confidence']) 
                              for comp in result),
            "All completions must have valid confidence scores (0.0-1.0)")
    def complete_analogy_with_contracts(
        self, 
        partial_analogy: Dict[str, str],
        max_completions: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete analogical reasoning with contract validation.
        
        Ensures all inputs are valid and all outputs meet quality standards.
        """
        self._operation_count += 1
        
        # Call existing implementation
        completions = self.find_analogical_completions(
            partial_analogy, 
            max_completions=max_completions
        )
        
        # Ensure proper structure
        structured_completions = []
        for comp in completions:
            structured_completions.append({
                'completion': comp.get('completion', ''),
                'confidence': comp.get('confidence', 0.0),
                'reasoning_type': comp.get('reasoning_type', 'hybrid'),
                'source_evidence': comp.get('source_evidence', []),
                'metadata': comp.get('metadata', {})
            })
        
        return structured_completions
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about contract-validated operations."""
        return {
            'total_operations': self._operation_count,
            'total_concepts': len(self.frame_aware_concepts),
            'contracts_enabled': True,
            'contract_library': 'icontract'
        }


def demonstrate_contract_validation():
    """
    Demonstrate contract validation in action.
    
    Shows both successful operations and contract violations
    with clear error messages.
    """
    print("=== icontract Design by Contract Demonstration ===")
    print("Testing contract validation with your soft logic system...\n")
    
    # Initialize contract-enhanced registry
    kwargs = {
        'download_wordnet': False,
        'n_clusters': 4,
        'enable_cross_domain': True,
        'embedding_provider': 'semantic'
    } if IMPORTS_AVAILABLE else {}
    
    registry = ContractEnhancedRegistry(**kwargs)
    
    # Test 1: Valid concept creation
    print("1. Testing valid concept creation...")
    try:
        concept = registry.create_concept_with_contracts(
            name="knight",
            context="default"
        )
        print(f"✅ Successfully created concept: {concept.name} in {concept.context}")
    except ViolationError as e:
        print(f"❌ Unexpected contract violation: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 2: Invalid concept name (empty)
    print("\n2. Testing invalid concept name (empty)...")
    try:
        concept = registry.create_concept_with_contracts(
            name="",  # Should violate contract
            context="default"
        )
        print(f"❌ Should have failed but created: {concept.name}")
    except ViolationError as e:
        print(f"✅ Contract correctly caught invalid name: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 3: Invalid context
    print("\n3. Testing invalid context...")
    try:
        concept = registry.create_concept_with_contracts(
            name="wizard",
            context="invalid_context"  # Should violate contract
        )
        print(f"❌ Should have failed but created: {concept.name}")
    except ViolationError as e:
        print(f"✅ Contract correctly caught invalid context: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 4: Valid semantic field discovery
    print("\n4. Testing valid semantic field discovery...")
    try:
        fields = registry.discover_semantic_fields_with_contracts(
            min_coherence=0.5,
            max_fields=3
        )
        print(f"✅ Discovered {len(fields)} semantic fields")
        for field in fields:
            print(f"   - {field['field_name']}: coherence={field['coherence']:.2f}")
    except ViolationError as e:
        print(f"❌ Contract violation: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 5: Invalid coherence score
    print("\n5. Testing invalid coherence score...")
    try:
        fields = registry.discover_semantic_fields_with_contracts(
            min_coherence=1.5,  # Should violate contract (> 1.0)
            max_fields=3
        )
        print(f"❌ Should have failed but got {len(fields)} fields")
    except ViolationError as e:
        print(f"✅ Contract correctly caught invalid coherence: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 6: Valid analogical completion
    print("\n6. Testing valid analogical completion...")
    try:
        completions = registry.complete_analogy_with_contracts(
            partial_analogy={"king": "queen", "man": "?"},
            max_completions=3
        )
        print(f"✅ Found {len(completions)} analogical completions")
        for comp in completions:
            print(f"   - {comp['completion']} (confidence: {comp['confidence']:.2f})")
    except ViolationError as e:
        print(f"❌ Contract violation: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 7: Invalid analogy (too few mappings)
    print("\n7. Testing invalid analogy (too few mappings)...")
    try:
        completions = registry.complete_analogy_with_contracts(
            partial_analogy={"king": "queen"},  # Only one mapping, no "?"
            max_completions=3
        )
        print(f"❌ Should have failed but got {len(completions)} completions")
    except ViolationError as e:
        print(f"✅ Contract correctly caught invalid analogy: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 8: Check operation statistics
    print("\n8. Checking operation statistics...")
    try:
        stats = registry.get_operation_statistics()
        print(f"✅ Registry statistics:")
        print(f"   - Total operations: {stats['total_operations']}")
        print(f"   - Total concepts: {stats['total_concepts']}")
        print(f"   - Contracts enabled: {stats['contracts_enabled']}")
        print(f"   - Contract library: {stats['contract_library']}")
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
    
    print("\n=== Contract validation demonstration complete ===")
    print("\nKey Benefits Demonstrated:")
    print("✅ Early error detection - invalid inputs caught before processing")
    print("✅ Clear error messages - contracts provide specific violation details")
    print("✅ Maintained functionality - existing code works with added safety")
    print("✅ API reliability - service layer will benefit from these contracts")


if __name__ == "__main__":
    demonstrate_contract_validation()
