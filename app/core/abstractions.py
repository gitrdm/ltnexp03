"""
Core Abstractions for Soft Logic Reasoning Systems
==================================================

This module defines the fundamental data structures and abstractions that form
the foundation of our soft logic microservice. These abstractions bridge the
gap between symbolic logical reasoning and neural learning systems.

DESIGN PHILOSOPHY:
==================

Our abstractions are designed around several key principles:

1. **Contextual Reasoning**: All concepts and axioms exist within contexts,
   allowing for domain-specific interpretations and avoiding global namespace
   conflicts.

2. **Semantic Grounding**: Concepts can be linked to WordNet synsets for
   precise semantic disambiguation, enabling robust natural language understanding.

3. **Formula Compositionality**: Complex logical relationships are built from
   simple, composable formula nodes, supporting both symbolic reasoning and
   neural optimization.

4. **Dual Classification**: Axioms are classified as either CORE (requiring
   logical consistency via SMT) or SOFT (allowing contradictions for learning),
   enabling hybrid symbolic-neural approaches.

ARCHITECTURE OVERVIEW:
======================

    Concept ←→ FormulaNode ←→ Axiom ←→ Context
       ↓           ↓          ↓        ↓
   [synset_id] [operations] [type] [inheritance]
       ↓           ↓          ↓        ↓
   [WordNet]   [tree-like]  [SMT/LTN] [hierarchy]

KEY ABSTRACTIONS:
=================

• **Concept**: Represents semantic entities with optional WordNet grounding
• **FormulaNode**: Composable formula trees for logical relationships  
• **Axiom**: Logical statements with metadata and type classification
• **Context**: Scoped knowledge domains with axiom and concept organization

This modular design enables clean separation between:
- Symbolic reasoning (SMT solvers, logic programming)
- Neural learning (LTN training, embedding optimization)  
- Knowledge management (contexts, inheritance, versioning)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class AxiomType(Enum):
    """
    Semantic Classification of Logical Axioms
    
    This enumeration defines the semantic types of logical relationships
    that our system can represent and reason about. Each type corresponds
    to a different kind of logical relationship with specific inference
    patterns and learning characteristics.
    
    AXIOM TYPE SEMANTICS:
    =====================
    
    • ANALOGY: Proportional relationships (A:B :: C:D)
      - Used for relational reasoning and pattern completion
      - Example: "king:queen :: man:woman"
    
    • SYNONYM: Semantic equivalence relationships  
      - Indicates concepts with similar meanings
      - Example: "happy" ≈ "joyful"
    
    • ANTONYM: Semantic opposition relationships
      - Indicates concepts with opposing meanings  
      - Example: "hot" ≠ "cold"
    
    • SIMILARITY: Graded semantic closeness
      - Continuous measure of semantic relatedness
      - Example: "dog" ~ "wolf" (high similarity)
    
    • DISSIMILARITY: Graded semantic distance
      - Continuous measure of semantic difference
      - Example: "dog" !~ "mathematics" (high dissimilarity)
    
    • CUSTOM: User-defined relationship types
      - Allows domain-specific logical relationships
      - Example: "causal", "temporal", "spatial" relationships
    
    USAGE IN REASONING:
    ===================
    
    Different axiom types enable different reasoning patterns:
    - ANALOGY: Enables proportional inference and completion
    - SYNONYM/ANTONYM: Enables logical substitution and contradiction detection
    - SIMILARITY/DISSIMILARITY: Enables soft logical inference with confidence
    - CUSTOM: Enables domain-specific reasoning rules
    """
    ANALOGY = "analogy"
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    SIMILARITY = "similarity"
    DISSIMILARITY = "dissimilarity"
    CUSTOM = "custom"


class AxiomClassification(Enum):
    """
    Processing Classification for Logical Axioms
    
    This enumeration determines how axioms are processed in our hybrid
    symbolic-neural reasoning system. The classification affects both
    validation requirements and learning strategies.
    
    CLASSIFICATION SEMANTICS:
    =========================
    
    • CORE: Hard logical constraints requiring consistency
      - Must be logically consistent (verified via SMT solvers)
      - Cannot be violated during learning
      - Used for foundational logical relationships
      - Example: "A thing cannot be both X and not-X"
    
    • SOFT: Probabilistic constraints allowing contradictions
      - Can be violated with associated confidence penalties
      - Learned and optimized via neural methods (LTN)
      - Used for empirical relationships and preferences
      - Example: "Birds typically fly" (exceptions: penguins, ostriches)
    
    HYBRID REASONING STRATEGY:
    ==========================
    
    Our system uses both classifications in a complementary manner:
    
    1. CORE axioms establish logical foundations and consistency constraints
    2. SOFT axioms capture probabilistic patterns and learned associations
    3. The combination enables robust reasoning that is both logically sound
       and capable of handling real-world uncertainty
    
    TECHNICAL IMPLEMENTATION:
    =========================
    
    - CORE: Processed by Z3 SMT solver for consistency checking
    - SOFT: Processed by LTNtorch for neural optimization
    - Integration: Core constraints guide soft learning boundaries
    """
    CORE = "core"      # Must be logically consistent, verified by SMT
    SOFT = "soft"      # Can be contradictory, learned by LTN


class OperationType(Enum):
    """
    Mathematical Operations for Formula Tree Construction
    
    This enumeration defines the primitive operations used to build
    compositional logical formulas. Each operation represents a different
    mathematical relationship that can be combined to create complex
    logical expressions.
    
    OPERATION SEMANTICS:
    ====================
    
    • ADD: Vector addition in semantic space
      - Combines concept embeddings additively
      - Example: "king" + "woman" → "queen"
    
    • SUBTRACT: Vector subtraction in semantic space  
      - Removes one concept's influence from another
      - Example: "king" - "man" → "royalty"
    
    • SIMILARITY: Cosine similarity or distance metric
      - Measures semantic closeness between concepts
      - Example: similarity("dog", "wolf") → 0.8
    
    • DISSIMILARITY: Inverse similarity or distance metric
      - Measures semantic distance between concepts
      - Example: dissimilarity("dog", "mathematics") → 0.9
    
    • CONSTANT: Literal value or concept reference
      - Represents fixed values or concept embeddings
      - Example: constant("king") → embedding_vector
    
    COMPOSITIONAL DESIGN:
    =====================
    
    Operations can be nested to create complex logical expressions:
    
    similarity(
        add("king", "woman"),
        "queen"
    ) → Analogy verification
    
    subtract(
        "monarch",
        "commoner"  
    ) → Hierarchical relationship
    
    This compositional approach enables:
    - Flexible logical expression construction
    - Hierarchical formula parsing and evaluation
    - Efficient neural optimization of complex relationships
    """
    ADD = "add"
    SUBTRACT = "subtract"
    SIMILARITY = "similarity"
    DISSIMILARITY = "dissimilarity"
    CONSTANT = "constant"


@dataclass
class Concept:
    """
    Semantic Concept with WordNet Grounding and Context Awareness
    
    The Concept class represents semantic entities in our reasoning system,
    providing a bridge between natural language terms and formal logical
    representations. Each concept can be precisely disambiguated using
    WordNet synsets and scoped within specific reasoning contexts.
    
    DESIGN RATIONALE:
    =================
    
    Natural language is inherently ambiguous - the word "bank" can refer to
    a financial institution, a river's edge, or an aircraft maneuver. Our
    Concept class addresses this through:
    
    1. **Synset Disambiguation**: Optional WordNet synset IDs provide precise
       semantic grounding (e.g., "bank.n.01" vs "bank.n.09")
    
    2. **Context Scoping**: Concepts exist within named contexts, allowing
       domain-specific interpretations without global conflicts
    
    3. **Unique Identification**: Generated unique IDs ensure concepts can
       be precisely referenced across the entire system
    
    UNIQUE ID STRATEGY:
    ===================
    
    The unique_id property generates globally unique identifiers using the pattern:
    
        "{context}:{name}"                    # Simple concept
        "{context}:{name}:{synset_id}"        # Disambiguated concept
    
    Examples:
        "default:king"                        # Basic concept
        "finance:bank:bank.n.01"             # Financial institution
        "geography:bank:bank.n.09"           # River bank
        "medieval:king:king.n.01"            # Monarch in historical context
    
    This strategy enables:
    - Global uniqueness across all contexts
    - Fast lookup and comparison operations
    - Hierarchical organization by context
    - Precise semantic disambiguation
    
    METADATA EXTENSIBILITY:
    =======================
    
    The metadata field allows arbitrary key-value storage for:
    - Domain-specific properties
    - Learning parameters and weights
    - Provenance and versioning information
    - Performance metrics and statistics
    
    USAGE PATTERNS:
    ===============
    
        # Simple concept creation
        king = Concept("king")
        
        # Disambiguated concept
        bank = Concept(
            name="bank",
            synset_id="bank.n.01", 
            disambiguation="financial institution",
            context="finance"
        )
        
        # Concepts with metadata
        concept = Concept(
            name="temperature",
            context="physics",
            metadata={
                "unit": "celsius", 
                "range": (-273.15, float('inf'))
            }
        )
    
    INTEGRATION POINTS:
    ===================
    
    Concepts integrate with other system components:
    - ConceptRegistry: Centralized storage and retrieval
    - FormulaNode: Used as operands in logical expressions
    - Axiom: Concepts participate in logical relationships
    - Context: Concepts are scoped within reasoning domains
    """
    
    name: str
    synset_id: Optional[str] = None
    disambiguation: Optional[str] = None
    context: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate concept after initialization."""
        if not self.name.strip():
            raise ValueError("Concept name cannot be empty")
        
        # Normalize name to lowercase for consistency
        self.name = self.name.lower().strip()
    
    @property
    def unique_id(self) -> str:
        """Generate unique identifier for this concept."""
        if self.synset_id:
            return f"{self.context}:{self.name}:{self.synset_id}"
        return f"{self.context}:{self.name}"
    
    def __str__(self) -> str:
        if self.synset_id:
            return f"{self.name}({self.synset_id})"
        return self.name
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Concept):
            return False
        return self.unique_id == other.unique_id
    
    def __hash__(self) -> int:
        return hash(self.unique_id)


@dataclass
class FormulaNode:
    """
    Compositional Formula Tree Node for Logical Expressions
    
    The FormulaNode class implements a tree-based representation of logical
    formulas, enabling compositional construction of complex logical
    relationships from simple mathematical operations.
    
    ARCHITECTURAL DESIGN:
    =====================
    
    FormulaNode uses a recursive tree structure where:
    
    1. **Leaf Nodes**: Represent constants (concepts, values)
    2. **Internal Nodes**: Represent operations with child arguments
    3. **Root Node**: Represents the complete logical expression
    
    This design enables:
    - Hierarchical formula construction
    - Recursive evaluation and optimization
    - Efficient parsing from textual representations
    - Compositional neural network architectures
    
    TREE STRUCTURE EXAMPLES:
    ========================
    
    Simple analogy: "king is to queen as man is to woman"
    
        similarity
        ├── subtract
        │   ├── king
        │   └── man
        └── subtract
            ├── queen
            └── woman
    
    Complex relationship: "royal_male = king + man - commoner"
    
        add
        ├── king
        └── subtract
            ├── man
            └── commoner
    
    OPERATION VALIDATION:
    =====================
    
    The __post_init__ method enforces operation-specific argument constraints:
    
    - CONSTANT: Exactly 1 argument (concept or value)
    - BINARY_OPS: Exactly 2 arguments (ADD, SUBTRACT, SIMILARITY, DISSIMILARITY)
    
    This validation ensures:
    - Syntactic correctness of formula trees
    - Consistent operation semantics
    - Early error detection during construction
    
    CONCEPT EXTRACTION:
    ===================
    
    The get_concepts() method recursively extracts all Concept objects
    from the formula tree, enabling:
    
    - Dependency analysis for axiom relationships
    - Concept registry population during parsing
    - Optimization of concept embeddings during learning
    - Validation of concept availability in contexts
    
    NEURAL INTEGRATION:
    ===================
    
    FormulaNode trees can be evaluated in neural networks:
    
    1. **Concept Embeddings**: Leaf concepts → embedding vectors
    2. **Operation Layers**: Tree operations → neural network layers
    3. **Recursive Evaluation**: Tree traversal → forward pass
    4. **Gradient Flow**: Backpropagation through tree structure
    
    This enables end-to-end learning of both:
    - Concept embeddings (what concepts mean)
    - Operation parameters (how operations work)
    
    USAGE PATTERNS:
    ===============
    
        # Simple concept reference
        king_node = FormulaNode(OperationType.CONSTANT, [king_concept])
        
        # Binary operation
        similarity_node = FormulaNode(
            OperationType.SIMILARITY,
            [concept1, concept2]
        )
        
        # Nested formula
        analogy_node = FormulaNode(
            OperationType.SIMILARITY,
            [
                FormulaNode(OperationType.SUBTRACT, [king, man]),
                FormulaNode(OperationType.SUBTRACT, [queen, woman])
            ]
        )
    """
    
    operation: OperationType
    args: List[Any] = field(default_factory=list)  # Can be concepts, other nodes, or values
    
    def __post_init__(self):
        """Validate formula node after initialization."""
        if self.operation == OperationType.CONSTANT and len(self.args) != 1:
            raise ValueError("Constant operation must have exactly one argument")
        elif self.operation in [OperationType.ADD, OperationType.SUBTRACT] and len(self.args) != 2:
            raise ValueError(f"{self.operation.value} operation must have exactly two arguments")
        elif self.operation in [OperationType.SIMILARITY, OperationType.DISSIMILARITY] and len(self.args) != 2:
            raise ValueError(f"{self.operation.value} operation must have exactly two arguments")
    
    def get_concepts(self) -> List[Concept]:
        """Recursively extract all concepts from this formula node."""
        concepts = []
        for arg in self.args:
            if isinstance(arg, Concept):
                concepts.append(arg)
            elif isinstance(arg, FormulaNode):
                concepts.extend(arg.get_concepts())
            elif isinstance(arg, str):
                # String concept names will be resolved later
                pass
        return concepts
    
    def __str__(self) -> str:
        if self.operation == OperationType.CONSTANT:
            return str(self.args[0])
        elif self.operation == OperationType.ADD:
            return f"({self.args[0]} + {self.args[1]})"
        elif self.operation == OperationType.SUBTRACT:
            return f"({self.args[0]} - {self.args[1]})"
        elif self.operation == OperationType.SIMILARITY:
            return f"similar({self.args[0]}, {self.args[1]})"
        elif self.operation == OperationType.DISSIMILARITY:
            return f"dissimilar({self.args[0]}, {self.args[1]})"
        else:
            return f"{self.operation.value}({', '.join(map(str, self.args))})"


@dataclass
class Axiom:
    """
    Logical Axiom with Formula, Metadata, and Processing Classification
    
    The Axiom class represents logical statements that form the knowledge
    base of our reasoning system. Each axiom encapsulates a logical
    relationship between concepts along with metadata for processing,
    validation, and learning.
    
    DESIGN PHILOSOPHY:
    ==================
    
    Axioms are designed to support hybrid symbolic-neural reasoning:
    
    1. **Logical Precision**: Each axiom has a formal logical representation
       via FormulaNode trees, enabling precise symbolic reasoning
    
    2. **Processing Flexibility**: Axioms are classified as CORE or SOFT,
       determining whether they're processed by SMT solvers or neural networks
    
    3. **Rich Metadata**: Extensive metadata supports provenance tracking,
       confidence modeling, and incremental learning
    
    4. **Context Awareness**: Axioms exist within specific reasoning contexts,
       enabling domain-specific knowledge organization
    
    AXIOM LIFECYCLE:
    ================
    
    1. **Creation**: Axioms are created from user input or learned patterns
    2. **Validation**: CORE axioms undergo SMT consistency checking
    3. **Registration**: Axioms are registered in appropriate contexts
    4. **Application**: Axioms participate in reasoning and learning
    5. **Evolution**: Confidence and metadata updated based on performance
    
    DUAL PROCESSING MODEL:
    ======================
    
    Our system processes axioms differently based on classification:
    
    CORE Axioms (classification = CORE):
    - Validated for logical consistency using Z3 SMT solver
    - Cannot be violated during learning or inference
    - Used for foundational logical relationships
    - Example: "If A implies B and B implies C, then A implies C"
    
    SOFT Axioms (classification = SOFT):
    - Learned and optimized using LTNtorch neural networks
    - Can be violated with associated confidence penalties
    - Used for empirical patterns and probabilistic relationships
    - Example: "Birds typically fly" (exceptions allowed)
    
    CONFIDENCE MODELING:
    ====================
    
    The confidence field (0.0 to 1.0) represents:
    - Certainty in the axiom's validity
    - Weight for neural optimization
    - Prior belief strength for Bayesian updating
    - Quality score for ranking and filtering
    
    Confidence evolution:
    - Starts with initial assessment (user-provided or default)
    - Updated based on reasoning success/failure
    - Adjusted during neural network training
    - Used for axiom selection and pruning
    
    METADATA EXTENSIBILITY:
    =======================
    
    The metadata field enables rich axiom annotation:
    
    - Provenance: Source, creation method, author
    - Performance: Success rate, usage frequency, validation results
    - Learning: Gradient norms, parameter updates, convergence metrics
    - Domain: Subject area, expertise level, application context
    
    CONCEPT INTEGRATION:
    ====================
    
    Axioms maintain bidirectional relationships with concepts:
    
    1. **Extraction**: Concepts are automatically extracted from formulas
    2. **Registration**: Concepts are registered in the parent context
    3. **Dependency**: Axiom validity depends on concept availability
    4. **Evolution**: Concept embeddings updated based on axiom performance
    
    USAGE PATTERNS:
    ===============
    
        # Simple analogy axiom
        analogy = Axiom(
            axiom_id="king_queen_analogy",
            axiom_type=AxiomType.ANALOGY,
            classification=AxiomClassification.SOFT,
            description="King is to queen as man is to woman",
            formula=analogy_formula,
            confidence=0.9
        )
        
        # Core logical axiom
        transitivity = Axiom(
            axiom_id="similarity_transitivity",
            axiom_type=AxiomType.CUSTOM,
            classification=AxiomClassification.CORE,
            description="Similarity is transitive",
            formula=transitivity_formula,
            confidence=1.0
        )
    
    INTEGRATION POINTS:
    ===================
    
    Axioms integrate with other system components:
    - Context: Axioms are scoped within reasoning domains
    - ConceptRegistry: Axiom concepts registered automatically
    - SMT Verifier: CORE axioms validated for consistency
    - LTN Trainer: SOFT axioms optimized via neural learning
    - API Layer: Axioms exposed via REST/GraphQL interfaces
    """
    
    axiom_id: str
    axiom_type: AxiomType
    classification: AxiomClassification
    description: str
    formula: FormulaNode
    context: str = "default"
    concepts: List[Concept] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate and process axiom after initialization."""
        if not self.axiom_id.strip():
            self.axiom_id = str(uuid.uuid4())
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Extract concepts from formula if not provided
        if not self.concepts:
            self.concepts = self.formula.get_concepts()
    
    def get_concept_names(self) -> List[str]:
        """Get list of concept names used in this axiom."""
        return [concept.name for concept in self.concepts]
    
    def is_core_axiom(self) -> bool:
        """Check if this is a core axiom requiring SMT verification."""
        return self.classification == AxiomClassification.CORE
    
    def __str__(self) -> str:
        return f"{self.axiom_id}: {self.description} [{self.formula}]"


@dataclass
class Context:
    """
    Hierarchical Knowledge Domain with Axiom and Concept Organization
    
    The Context class represents a scoped reasoning domain that organizes
    related axioms and concepts into coherent knowledge units. Contexts
    enable domain-specific reasoning while supporting inheritance and
    knowledge sharing across related domains.
    
    DESIGN MOTIVATION:
    ==================
    
    Real-world reasoning often involves domain-specific knowledge:
    
    - Financial domain: "bank" means financial institution
    - Geographic domain: "bank" means river's edge  
    - Aviation domain: "bank" means aircraft turning maneuver
    
    Contexts solve this by providing:
    1. **Namespace Isolation**: Same concept names with different meanings
    2. **Domain Organization**: Related knowledge grouped together
    3. **Inheritance Hierarchies**: Shared knowledge across related domains
    4. **Scoped Reasoning**: Domain-specific inference and learning
    
    HIERARCHICAL ORGANIZATION:
    ==========================
    
    Contexts support parent-child relationships for knowledge inheritance:
    
        general_knowledge
        ├── natural_science
        │   ├── physics
        │   └── biology
        └── social_science
            ├── economics
            └── psychology
    
    Child contexts inherit axioms and concepts from parents, enabling:
    - Knowledge reuse across related domains
    - Specialization of general principles
    - Hierarchical reasoning and inference
    - Efficient knowledge organization
    
    KNOWLEDGE ORGANIZATION STRATEGY:
    ================================
    
    Each context maintains:
    
    1. **Axiom Collection**: All logical statements within the domain
       - Organized by type and classification
       - Inherited from parent contexts
       - Locally specialized or overridden
    
    2. **Concept Registry**: All semantic entities within the domain
       - Fast lookup by concept name
       - Automatic registration from axiom formulas
       - Context-specific disambiguation
    
    3. **Metadata Storage**: Domain-specific information
       - Creation and modification history
       - Performance metrics and statistics
       - Domain expertise and validation status
    
    INHERITANCE SEMANTICS:
    ======================
    
    When a context has a parent, it inherits:
    - All parent axioms (unless locally overridden)
    - All parent concepts (unless locally redefined)
    - Parent metadata (merged with local metadata)
    
    Local definitions take precedence over inherited ones, enabling:
    - Domain-specific specialization
    - Exception handling for general rules
    - Gradual refinement of knowledge
    
    AXIOM CLASSIFICATION WITHIN CONTEXTS:
    =====================================
    
    Contexts organize axioms by classification:
    
    - **Core Axioms**: Foundational logical relationships
      - Must be consistent within context
      - Inherited from parents unless overridden
      - Validated by SMT solvers
    
    - **Soft Axioms**: Empirical patterns and preferences
      - Can contradict each other with confidence weights
      - Learned and optimized via neural networks
      - Evolved based on domain-specific performance
    
    USAGE PATTERNS:
    ===============
    
        # Create domain-specific context
        finance_context = Context(
            name="finance",
            description="Financial and economic reasoning",
            parent="general_knowledge"
        )
        
        # Add domain-specific axiom
        bank_axiom = Axiom(
            axiom_id="bank_financial",
            description="Bank refers to financial institution",
            # ... other fields
        )
        finance_context.add_axiom(bank_axiom)
        
        # Context automatically registers concepts
        bank_concept = finance_context.get_concept("bank")
        # Returns the financial institution concept
    
    REASONING INTEGRATION:
    ======================
    
    Contexts integrate with reasoning systems:
    
    1. **SMT Verification**: Core axioms validated within context scope
    2. **LTN Training**: Soft axioms optimized using context-specific data
    3. **Concept Resolution**: Concepts resolved within context hierarchy
    4. **Inference Scoping**: Reasoning limited to relevant contexts
    
    PERFORMANCE CONSIDERATIONS:
    ===========================
    
    - **Fast Lookup**: O(1) concept retrieval within context
    - **Lazy Inheritance**: Parent axioms loaded on-demand
    - **Caching**: Repeated queries cached for performance
    - **Incremental Updates**: Changes propagated efficiently
    """
    
    name: str
    parent: Optional[str] = None
    description: str = ""
    axioms: List[Axiom] = field(default_factory=list)
    concepts: Dict[str, Concept] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate context after initialization."""
        if not self.name.strip():
            raise ValueError("Context name cannot be empty")
        
        # Normalize name
        self.name = self.name.lower().strip()
    
    def add_axiom(self, axiom: Axiom) -> None:
        """Add an axiom to this context."""
        axiom.context = self.name
        self.axioms.append(axiom)
        
        # Register concepts
        for concept in axiom.concepts:
            concept.context = self.name
            self.concepts[concept.name] = concept
    
    def get_core_axioms(self) -> List[Axiom]:
        """Get all core axioms in this context."""
        return [axiom for axiom in self.axioms if axiom.is_core_axiom()]
    
    def get_soft_axioms(self) -> List[Axiom]:
        """Get all soft axioms in this context."""
        return [axiom for axiom in self.axioms if not axiom.is_core_axiom()]
    
    def get_concept(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        return self.concepts.get(name.lower().strip())
    
    def __str__(self) -> str:
        parent_info = f" (inherits from {self.parent})" if self.parent else ""
        return f"Context '{self.name}'{parent_info}: {len(self.axioms)} axioms, {len(self.concepts)} concepts"
