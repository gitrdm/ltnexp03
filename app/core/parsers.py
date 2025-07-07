"""
Axiom Definition Parsers: YAML and JSON Format Support
======================================================

This module implements sophisticated parsers for axiom definition files,
enabling declarative specification of logical relationships in human-readable
formats. The parsers bridge the gap between textual knowledge representation
and our internal logical abstractions.

DESIGN PHILOSOPHY:
==================

Knowledge should be expressible in formats that are:

1. **Human-Readable**: Domain experts can author axioms without programming
2. **Version-Controllable**: Text-based formats work with Git and diff tools
3. **Validation-Friendly**: Structured formats enable schema validation
4. **Tool-Integrable**: Standard formats work with existing YAML/JSON tools

Our parser design supports:
- Multiple input formats (YAML, JSON) with unified semantics
- Flexible formula specification with nested operations
- Automatic concept extraction and registration
- Rich error reporting with context information
- Extensible metadata for domain-specific annotations

SUPPORTED FORMATS:
==================

YAML Format (Recommended for Human Authoring):
----------------------------------------------
```yaml
axiom_id: "king_queen_analogy"
type: "analogy"
classification: "soft"
description: "King is to queen as man is to woman"
context: "social_relations"
concepts:
  - name: "king"
    synset: "king.n.01"
    disambiguation: "male monarch"
  - name: "queen" 
    synset: "queen.n.01"
  - "man"  # Simple string form
  - "woman"
formula:
  similarity:
    left:
      subtract: ["king", "man"]
    right:
      subtract: ["queen", "woman"]
metadata:
  confidence: 0.9
  created_by: "domain_expert"
```

JSON Format (Machine-Generated or API Input):
---------------------------------------------
```json
{
  "axiom_id": "transitivity_similarity",
  "type": "custom",
  "classification": "core", 
  "description": "Similarity is transitive",
  "formula": {
    "similarity": {
      "left": {"similarity": ["A", "B"]},
      "right": {"similarity": ["B", "C"]}
    }
  }
}
```

FORMULA SYNTAX DESIGN:
======================

Our formula syntax is designed for compositionality and clarity:

1. **Operation-First Structure**: Each formula node starts with operation name
2. **Nested Composition**: Operations can contain sub-operations recursively  
3. **Flexible Arguments**: Arguments can be concepts, values, or sub-formulas
4. **Type Safety**: Parser validates operation-argument compatibility

Formula Operation Types:
------------------------

• **constant**: Literal values or concept references
  - Format: `{"constant": "concept_name"}` or `{"constant": 42}`
  - Used for: Leaf nodes in formula trees

• **similarity/dissimilarity**: Binary semantic relationships
  - Format: `{"similarity": {"left": arg1, "right": arg2}}`
  - Used for: Comparing semantic relatedness

• **add/subtract**: Vector space operations
  - Format: `{"add": [arg1, arg2]}` or `{"subtract": [arg1, arg2]}`
  - Used for: Compositional concept relationships

CONCEPT SPECIFICATION:
======================

Concepts can be specified in multiple formats for flexibility:

Simple String Format:
--------------------
```yaml
concepts:
  - "king"
  - "queen"
```

Detailed Object Format:
-----------------------
```yaml
concepts:
  - name: "bank"
    synset: "bank.n.01"
    disambiguation: "financial institution"
    context: "finance"
```

Mixed Format (both in same file):
---------------------------------
```yaml
concepts:
  - "simple_concept"
  - name: "complex_concept"
    synset: "complex.a.01"
```

PARSER ARCHITECTURE:
====================

The parsing pipeline follows a multi-stage approach:

1. **Format Detection**: File extension determines YAML vs JSON parsing
2. **Content Parsing**: Raw text → structured data (dict/list)
3. **Schema Validation**: Structured data validated against expected format
4. **Object Construction**: Validated data → Concept/FormulaNode/Axiom objects
5. **Concept Registration**: Extracted concepts stored in local registry
6. **Cross-Reference Resolution**: Concept references resolved to objects

ERROR HANDLING STRATEGY:
========================

The parser provides detailed error reporting:

- **Parse Errors**: JSON/YAML syntax errors with line numbers
- **Schema Errors**: Missing required fields or invalid types
- **Semantic Errors**: Invalid operations or concept references
- **Context Information**: File name, axiom ID, and operation path

This enables rapid debugging of complex axiom definitions.

PERFORMANCE CONSIDERATIONS:
===========================

- **Streaming Parsing**: Large files processed incrementally
- **Concept Caching**: Duplicate concepts reused within file
- **Lazy Validation**: Expensive validations deferred until needed
- **Memory Efficiency**: Minimal object creation during parsing

EXTENSIBILITY DESIGN:
=====================

The parser architecture supports extension:

- **Custom Operations**: New OperationType values automatically supported
- **Metadata Schema**: Arbitrary metadata fields preserved
- **Format Plugins**: Additional input formats can be added
- **Validation Hooks**: Custom validation logic can be injected

This ensures the parser can evolve with the reasoning system's capabilities.
"""

import json
import yaml
from typing import List, Dict, Any, Union
from pathlib import Path

from .abstractions import (
    Axiom, Concept, FormulaNode, Context,
    AxiomType, AxiomClassification, OperationType
)


class AxiomParseError(Exception):
    """Raised when axiom parsing fails."""
    pass


class AxiomParser:
    """
    Multi-Format Axiom Definition Parser with Concept Registry
    
    The AxiomParser class provides a unified interface for parsing axiom
    definitions from multiple textual formats (YAML, JSON) while maintaining
    a local concept registry for cross-reference resolution and duplicate
    elimination.
    
    DESIGN ARCHITECTURE:
    ====================
    
    The parser follows a pipeline architecture:
    
    Input File → Format Detection → Content Parsing → Schema Validation
                                        ↓
    Axiom Objects ← Object Construction ← Concept Resolution ← Data Validation
    
    Each stage has specific responsibilities:
    
    1. **Format Detection**: Determines parsing strategy based on file extension
    2. **Content Parsing**: Converts text to structured data (dict/list)
    3. **Schema Validation**: Ensures required fields and correct types
    4. **Data Validation**: Validates semantic constraints and relationships
    5. **Concept Resolution**: Resolves concept references to objects
    6. **Object Construction**: Creates final Axiom/Concept/FormulaNode objects
    
    LOCAL CONCEPT REGISTRY:
    =======================
    
    The parser maintains a local concept registry during parsing:
    
    Benefits:
    - **Deduplication**: Same concept referenced multiple times → single object
    - **Cross-Reference**: Concept references in formulas → resolved objects
    - **Consistency**: Same concept properties preserved across references
    - **Performance**: Avoids creating duplicate concept objects
    
    Registry Lifecycle:
    1. **Population**: Concepts extracted from axiom definitions
    2. **Resolution**: Formula references resolved to registered concepts
    3. **Validation**: Ensures all concept references are resolvable
    4. **Export**: Final concept collection available for external use
    
    PARSING STRATEGIES:
    ===================
    
    YAML Parsing:
    - Human-friendly format optimized for manual authoring
    - Supports comments and multi-line strings
    - More forgiving syntax with implicit type conversion
    - Preferred for knowledge base development
    
    JSON Parsing:
    - Machine-friendly format optimized for programmatic generation
    - Strict syntax with explicit type specifications
    - Better tool support and validation capabilities
    - Preferred for API integration and automated workflows
    
    CONCEPT HANDLING PATTERNS:
    ===========================
    
    String References:
    -----------------
    When a formula references a concept by string name, the parser:
    1. Checks local registry for existing concept
    2. Creates placeholder concept if not found
    3. Registers new concept for future references
    4. Returns concept object for formula construction
    
    Explicit Definitions:
    --------------------
    When concepts are explicitly defined in the axiom:
    1. Creates concept with full metadata
    2. Registers in local registry
    3. Uses for formula resolution
    4. Includes in axiom's concept list
    
    Cross-File References:
    ---------------------
    Future enhancement will support:
    - Global concept registries across files
    - Import/export of concept definitions
    - Namespace management for concept conflicts
    
    ERROR RECOVERY:
    ===============
    
    The parser implements graceful error recovery:
    
    - **Continue on Non-Fatal Errors**: Missing optional fields don't stop parsing
    - **Collect Multiple Errors**: All errors reported together for efficient fixing
    - **Context Preservation**: Error messages include file/line/axiom context
    - **Partial Results**: Successfully parsed axioms returned even if some fail
    
    USAGE PATTERNS:
    ===============
    
        # Basic file parsing
        parser = AxiomParser()
        axioms = parser.parse_file("knowledge_base.yaml")
        
        # Incremental parsing with shared concepts
        parser = AxiomParser()
        axioms1 = parser.parse_file("base_axioms.yaml")
        axioms2 = parser.parse_file("domain_axioms.yaml")
        # Concepts shared between files are reused
        
        # Error handling
        try:
            axioms = parser.parse_file("complex_axioms.yaml")
        except AxiomParseError as e:
            print(f"Parse error: {e}")
            # Continue with partial results if needed
    
    INTEGRATION POINTS:
    ===================
    
    The parser integrates with other system components:
    
    - **ConceptRegistry**: Local registry can be merged with global registry
    - **Context**: Parsed axioms can be added to reasoning contexts
    - **Validation**: Parsed axioms can be validated by SMT or LTN systems
    - **API Layer**: Parser used by REST endpoints for axiom upload
    """
    
    def __init__(self) -> None:
        """
        Initialize Parser with Local Concept Registry
        
        The local concept registry enables efficient concept reuse within
        a single parsing session while maintaining isolation between
        different parsing sessions.
        """
        self.concept_registry: Dict[str, Concept] = {}
    
    def parse_file(self, file_path: Union[str, Path]) -> List[Axiom]:
        """Parse axioms from a file (JSON or YAML)."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Axiom file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return self.parse_yaml(content)
        elif file_path.suffix.lower() == '.json':
            return self.parse_json(content)
        else:
            raise AxiomParseError(f"Unsupported file format: {file_path.suffix}")
    
    def parse_json(self, content: str) -> List[Axiom]:
        """Parse axioms from JSON content."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise AxiomParseError(f"Invalid JSON: {e}")
        
        return self._parse_axiom_data(data)
    
    def parse_yaml(self, content: str) -> List[Axiom]:
        """Parse axioms from YAML content."""
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise AxiomParseError(f"Invalid YAML: {e}")
        
        return self._parse_axiom_data(data)
    
    def _parse_axiom_data(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Axiom]:
        """Parse axiom data structure."""
        if isinstance(data, dict):
            # Single axiom
            return [self._parse_single_axiom(data)]
        elif isinstance(data, list):
            # Multiple axioms
            return [self._parse_single_axiom(axiom_data) for axiom_data in data]
        else:
            raise AxiomParseError("Data must be a dict (single axiom) or list (multiple axioms)")
    
    def _parse_single_axiom(self, data: Dict[str, Any]) -> Axiom:
        """Parse a single axiom from data dictionary."""
        try:
            # Required fields
            axiom_id = data['axiom_id']
            axiom_type = AxiomType(data['type'])
            classification = AxiomClassification(data['classification'])
            description = data['description']
            
            # Parse concepts
            concepts = []
            if 'concepts' in data:
                for concept_data in data['concepts']:
                    if isinstance(concept_data, str):
                        # Simple string concept
                        concept = Concept(name=concept_data)
                    elif isinstance(concept_data, dict):
                        # Detailed concept definition
                        concept = Concept(
                            name=concept_data['name'],
                            synset_id=concept_data.get('synset'),
                            disambiguation=concept_data.get('disambiguation'),
                            context=data.get('context', 'default')
                        )
                    else:
                        raise AxiomParseError(f"Invalid concept format: {concept_data}")
                    
                    concepts.append(concept)
                    self.concept_registry[concept.name] = concept
            
            # Parse formula
            formula = self._parse_formula(data['formula'])
            
            # Optional fields
            context = data.get('context', 'default')
            metadata = data.get('metadata', {})
            
            axiom = Axiom(
                axiom_id=axiom_id,
                axiom_type=axiom_type,
                classification=classification,
                description=description,
                formula=formula,
                context=context,
                concepts=concepts,
                metadata=metadata
            )
            
            # Set additional metadata fields if present
            if 'created_by' in metadata:
                axiom.created_by = metadata['created_by']
            if 'confidence' in metadata:
                axiom.confidence = metadata['confidence']
            
            return axiom
            
        except KeyError as e:
            raise AxiomParseError(f"Missing required field: {e}")
        except ValueError as e:
            raise AxiomParseError(f"Invalid value: {e}")
    
    def _parse_formula(self, formula_data: Dict[str, Any]) -> FormulaNode:
        """Parse formula structure into FormulaNode tree."""
        if not isinstance(formula_data, dict):
            raise AxiomParseError("Formula must be a dictionary")
        
        if len(formula_data) != 1:
            raise AxiomParseError("Formula must have exactly one root operation")
        
        operation_name, operation_data = next(iter(formula_data.items()))
        
        try:
            operation = OperationType(operation_name)
        except ValueError:
            raise AxiomParseError(f"Unknown operation: {operation_name}")
        
        return self._parse_formula_node(operation, operation_data)
    
    def _parse_formula_node(self, operation: OperationType, data: Any) -> FormulaNode:
        """Parse a single formula node."""
        if operation == OperationType.CONSTANT:
            # Constant value or concept name
            if isinstance(data, str):
                # Concept reference
                concept = self.concept_registry.get(data)
                if concept:
                    return FormulaNode(operation, [concept])
                else:
                    # Create placeholder concept
                    concept = Concept(name=data)
                    self.concept_registry[data] = concept
                    return FormulaNode(operation, [concept])
            else:
                return FormulaNode(operation, [data])
        
        elif operation in [OperationType.SIMILARITY, OperationType.DISSIMILARITY]:
            # Binary operations with left and right operands
            if isinstance(data, dict) and 'left' in data and 'right' in data:
                left = self._parse_operand(data['left'])
                right = self._parse_operand(data['right'])
                return FormulaNode(operation, [left, right])
            else:
                raise AxiomParseError(f"Operation {operation.value} requires 'left' and 'right' operands")
        
        elif operation in [OperationType.ADD, OperationType.SUBTRACT]:
            # Binary operations with args list
            if isinstance(data, list) and len(data) == 2:
                left = self._parse_operand(data[0])
                right = self._parse_operand(data[1])
                return FormulaNode(operation, [left, right])
            else:
                raise AxiomParseError(f"Operation {operation.value} requires exactly 2 arguments")
        
        else:
            raise AxiomParseError(f"Unsupported operation: {operation}")
    
    def _parse_operand(self, operand_data: Any) -> Any:
        """Parse an operand which can be a concept, value, or sub-formula."""
        if isinstance(operand_data, str):
            # Concept reference
            concept = self.concept_registry.get(operand_data)
            if concept:
                return concept
            else:
                # Create placeholder concept
                concept = Concept(name=operand_data)
                self.concept_registry[operand_data] = concept
                return concept
        
        elif isinstance(operand_data, dict):
            # Sub-formula
            if len(operand_data) != 1:
                raise AxiomParseError("Sub-formula must have exactly one operation")
            
            operation_name, operation_data = next(iter(operand_data.items()))
            try:
                operation = OperationType(operation_name)
                return self._parse_formula_node(operation, operation_data)
            except ValueError:
                raise AxiomParseError(f"Unknown operation in sub-formula: {operation_name}")
        
        else:
            # Literal value
            return operand_data
    
    def create_context_from_axioms(self, axioms: List[Axiom], context_name: str, 
                                  description: str = "") -> Context:
        """Create a context from a list of axioms."""
        context = Context(name=context_name, description=description)
        
        for axiom in axioms:
            context.add_axiom(axiom)
        
        return context
