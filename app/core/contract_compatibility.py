"""
Contract Compatibility Layer
============================

This module provides a compatibility layer to preserve existing Design by 
Contract (DbC) contracts during the abstraction refactoring process.

The decorators and mixins defined here allow us to wrap new or modified
classes and methods, ensuring that the original icontract validation rules
are still enforced. This is a critical part of our strategy to refactor
safely without sacrificing correctness.

Key Components:
- ContractPreservationMixin: A mixin class providing decorators.
- preserve_concept_contracts: Decorator for concept-related methods.
- preserve_registry_invariants: Class decorator for registry invariants.
"""

from typing import Protocol, TypeVar, Any
from icontract import require, ensure, invariant

T = TypeVar('T')

class ContractPreservationMixin:
    """Mixin to preserve existing DbC contracts during abstraction changes."""
    
    @staticmethod
    def preserve_concept_contracts(func):
        """Decorator to preserve concept-related contracts."""
        # Copy existing concept validation contracts
        return require(lambda name: isinstance(name, str) and len(name.strip()) > 0)(
            ensure(lambda result: result is not None)(func)
        )
    
    @staticmethod  
    def preserve_registry_invariants(cls):
        """Class decorator to preserve registry invariants."""
        return invariant(lambda self: hasattr(self, 'concepts'))(
            invariant(lambda self: isinstance(self.concepts, dict))(cls)
        )
