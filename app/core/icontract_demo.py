"""
Practical icontract Integration Example for Phase 3B
====================================================

This module demonstrates how to integrate icontract Design by Contract
decorators with your existing EnhancedHybridRegistry for robust validation.

Run this example to see contract validation in action:
python app/core/icontract_demo.py

---

# DATA MODEL STANDARDIZATION PLAN (Phase 4)
# -----------------------------------------
# As part of the abstraction consistency refactor, the following model usage patterns will be enforced:
#   - Pydantic models: for API boundaries and external data validation only
#   - dataclasses: for all core logic and internal data structures
#   - TypedDict: for type hints and static typing only (not for runtime objects)
#
# Steps for this file and related modules:
#   1. Identify all usages of Pydantic, dataclass, and TypedDict models.
#   2. Refactor to use dataclasses for all core logic in this demo and supporting modules.
#   3. Remove or replace any duplicate or redundant model definitions.
#   4. Add conversion utilities if needed (e.g., dataclass <-> Pydantic).
#   5. Update tests and contract checks to use the standardized models.
#   6. Ensure all tests, mypy, and icontract validation pass after each change.
#
# TODO: Begin with ContractEnhancedRegistry and all concept/frame instance models used here.
#
"""

from typing import List, Dict, Optional, Any
from icontract import require, ensure, invariant, ViolationError
from dataclasses import dataclass, field
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
@dataclass
class Concept:
    name: str
    context: str = "default"
    unique_id: str = field(init=False)

    def __post_init__(self):
        self.unique_id = f"{self.context}:{self.name}"

@dataclass
class FrameInstance:
    frame_name: str
    instance_id: str
    concept_bindings: Dict[str, Concept]
    context: str = "default"


class MockRegistry:
    def __init__(self):
        self.frame_aware_concepts = {}
        self._operation_count = 0
        self.frame_instances = {}
    
    def create_frame_aware_concept_with_advanced_embedding(self, **kwargs):
        concept = Concept(kwargs['name'], kwargs.get('context', 'default'))
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
    
    def create_frame_instance(self, frame_name, instance_id, concept_bindings, context="default"):
        instance = FrameInstance(
            frame_name=frame_name,
            instance_id=instance_id,
            concept_bindings=concept_bindings,
            context=context
        )
        self.frame_instances[instance_id] = instance
        return instance


# Contract-enhanced registry
if IMPORTS_AVAILABLE:
    BaseRegistry = EnhancedHybridRegistry
else:
    BaseRegistry = MockRegistry


@invariant(lambda self: not hasattr(self, 'frame_aware_concepts') or len(self.frame_aware_concepts) >= 0, "Registry must maintain non-negative concept count")
@invariant(lambda self: self._operation_count >= 0, "Operation count must be non-negative")
class ContractEnhancedRegistry(BaseRegistry):
    """
    Enhanced registry with comprehensive icontract validation.
    
    Demonstrates how to add Design by Contract validation to existing
    functionality without breaking backward compatibility.
    """
    
    def __init__(self, **kwargs):
        self._operation_count = 0  # Initialize before super() to satisfy invariants
        super().__init__(**kwargs) if IMPORTS_AVAILABLE else super().__init__()
    
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
        
        # Accept both dict and list return types
        if isinstance(fields, dict):
            field_iter = fields.items()
        elif isinstance(fields, list):
            # Already a list of structured fields
            return fields[:max_fields]
        else:
            raise TypeError("discover_semantic_fields must return dict or list")
        
        # Convert to structured format
        structured_fields = []
        for field_name, field_data in field_iter:
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
        'n_clusters': 1,  # Set to 1 to avoid n_samples < n_clusters error
        'enable_cross_domain': True,
        'embedding_provider': 'semantic'
    } if IMPORTS_AVAILABLE else {}
    
    registry = ContractEnhancedRegistry(**kwargs)
    
    # Add more concepts to avoid clustering errors
    registry.create_concept_with_contracts(name="knight", context="default")
    registry.create_concept_with_contracts(name="wizard", context="default")
    registry.create_concept_with_contracts(name="dragon", context="default")
    
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
        # Handle both dict and list return types for fields
        if isinstance(fields, dict):
            field_iter = fields.values()
        else:
            field_iter = fields
        for field in field_iter:
            print(f"   - {field.get('field_name', str(field))}: coherence={field.get('coherence', 0.0):.2f}")
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


def test_create_frame_instance_contracts():
    print("\n=== Frame Instance Contract Validation ===")
    # Setup registry and add a frame
    registry = ContractEnhancedRegistry()
    frame_name = "TestFrame"
    instance_id = "instance_001"
    # Add a frame to the registry (mock registry does not have frame_registry, so use create_frame_instance directly if needed)
    if hasattr(registry, 'frame_registry'):
        registry.frame_registry.create_frame(
            name=frame_name,
            definition="A test frame",
            core_elements=["Role1", "Role2"]
        )
    # Valid concept bindings
    concept1 = registry.create_concept_with_contracts("concept1", "default")
    concept2 = registry.create_concept_with_contracts("concept2", "default")
    bindings = {"Role1": concept1, "Role2": concept2}
    try:
        if hasattr(registry, 'frame_registry'):
            instance = registry.frame_registry.create_frame_instance(
                frame_name=frame_name,
                instance_id=instance_id,
                concept_bindings=bindings,
                context="default"
            )
        else:
            instance = registry.create_frame_instance(
                frame_name=frame_name,
                instance_id=instance_id,
                concept_bindings=bindings,
                context="default"
            )
        print(f"✅ Successfully created frame instance: {instance.instance_id}")
    except ViolationError as e:
        print(f"❌ Contract violation (should not happen): {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Invalid: unknown frame (only test if frame_registry exists)
    if hasattr(registry, 'frame_registry'):
        try:
            registry.frame_registry.create_frame_instance(
                frame_name="UnknownFrame",
                instance_id="bad_instance",
                concept_bindings=bindings,
                context="default"
            )
            print("❌ Should have failed for unknown frame")
        except Exception as e:
            print(f"✅ Correctly failed for unknown frame: {e}")

        # Invalid: missing concept binding (simulate by omitting a required role)
        try:
            registry.frame_registry.create_frame_instance(
                frame_name=frame_name,
                instance_id="bad_instance2",
                concept_bindings={"Role1": concept1},  # Missing Role2
                context="default"
            )
            print("❌ Should have failed for missing binding")
        except Exception as e:
            print(f"✅ Correctly failed for missing binding: {e}")
    else:
        # For mock registry, simulate error cases
        try:
            # Simulate unknown frame by passing a frame_name not tracked (mock does not check, so just print info)
            instance = registry.create_frame_instance(
                frame_name="UnknownFrame",
                instance_id="bad_instance",
                concept_bindings=bindings,
                context="default"
            )
            print(f"(Mock) Created instance for unknown frame: {instance.instance_id}")
        except Exception as e:
            print(f"(Mock) Error for unknown frame: {e}")
        try:
            # Simulate missing binding (mock does not check, so just print info)
            instance = registry.create_frame_instance(
                frame_name=frame_name,
                instance_id="bad_instance2",
                concept_bindings={"Role1": concept1},
                context="default"
            )
            print(f"(Mock) Created instance with missing binding: {instance.instance_id}")
        except Exception as e:
            print(f"(Mock) Error for missing binding: {e}")


if __name__ == "__main__":
    demonstrate_contract_validation()
    test_create_frame_instance_contracts()
