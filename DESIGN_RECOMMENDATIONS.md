# Soft Logic Microservice Design Recommendations

## Executive Summary

This document outlines the design approach for our robust, scalable microservice for building and managing soft logic vectors. The system supports both hard logic verification of core axioms and advanced soft logic learning with hybrid FrameNet-clustering semantic reasoning, context-aware modeling capabilities, and sophisticated analogical reasoning.

**Current Status**: Phase 1-2 Complete. Core abstractions, hybrid semantic reasoning, and advanced vector embeddings fully implemented and tested. Enhanced semantic field discovery and cross-domain analogical reasoning operational. Ready for Phase 3 service layer implementation and Phase 4 neural-symbolic integration.

## Current State Analysis (Updated 2025-07-05)

### Completed Implementation Review
- **✅ Core Abstractions**: Complete concept, axiom, context, and formula abstractions with WordNet integration
- **✅ Hybrid Semantic System**: FrameNet-style frames integrated with clustering-based concept organization  
- **✅ Advanced Reasoning**: Cross-domain analogical reasoning, semantic field discovery, analogical completion
- **✅ Vector Embeddings**: Sophisticated embedding management with multiple providers and similarity metrics
- **✅ Enhanced Registry**: `EnhancedHybridRegistry` with multi-level analogical reasoning and dynamic knowledge discovery
- **✅ Comprehensive Testing**: 18 unit tests passing across all core modules
- **✅ Rich Documentation**: Literate programming style with comprehensive design documents
- **✅ Working Demonstrations**: Three comprehensive demo systems showcasing medieval fantasy applications

### Legacy Analysis (Historical Context)
**LTN01.py Analysis**
- **Strengths**: Demonstrated complete soft logic pipeline from training to visualization
- **Evolution**: Now superseded by modular hybrid semantic reasoning system
- **Legacy Value**: Concepts integrated into enhanced vector embedding system

**SMT01.py Analysis**  
- **Strengths**: Clean demonstration of axiom consistency checking
- **Evolution**: Concepts ready for integration into service layer verification
- **Future Integration**: SMT verification layer planned for Phase 3

## Current Architecture (Implemented)

### 1. Hybrid Semantic System Components

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Enhanced Hybrid   │    │   Frame & Cluster   │    │   Vector Embedding  │
│     Registry        │    │     Registries      │    │     Manager         │
│  - Concept mgmt     │    │  - Semantic frames  │    │  - Multi-provider   │
│  - Context aware    │    │  - Concept clusters │    │  - Semantic embed   │
│  - WordNet integ    │    │  - Frame instances  │    │  - Similarity comp  │
│  - Analogical rsn   │    │  - Cross-domain     │    │  - Caching system   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Semantic Field    │    │   Analogical        │    │   Story Generation  │
│   Discovery         │    │   Reasoning         │    │   Applications      │
│  - Coherent regions │    │  - Surface level    │    │  - Character rel    │
│  - Cross-domain     │    │  - Structural       │    │  - Plot variants    │
│  - Dynamic fields   │    │  - Cross-domain     │    │  - World building   │
│  - Quality metrics  │    │  - Completion tasks │    │  - Knowledge gen    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### 2. Current File Structure

```
app/core/
├── abstractions.py              # ✅ Core concept, axiom, context abstractions
├── concept_registry.py          # ✅ WordNet-integrated concept management
├── parsers.py                   # ✅ YAML/JSON axiom parsing
├── frame_cluster_abstractions.py # ✅ FrameNet & clustering data structures
├── frame_cluster_registry.py    # ✅ Frame & cluster management systems
├── hybrid_registry.py           # ✅ Unified hybrid semantic registry
├── enhanced_semantic_reasoning.py # ✅ Advanced reasoning & field discovery
├── vector_embeddings.py         # ✅ Sophisticated embedding management
├── protocols.py                 # ✅ Protocol interface definitions (Phase 3A)
├── api_models.py                # ✅ TypedDict API request/response models (Phase 3A)
├── protocol_mixins.py           # ✅ Protocol implementation mixins (Phase 3A)
└── __init__.py                  # ✅ Clean module exports

demonstrations/
├── demo_hybrid_system.py        # ✅ Basic hybrid capabilities
├── demo_enhanced_system.py      # ✅ Advanced reasoning features
└── demo_comprehensive_system.py # ✅ Medieval fantasy application

tests/
├── test_core/
│   └── test_abstractions.py     # ✅ 18 tests passing
├── test_phase_3a.py             # ✅ Phase 3A type safety tests (5/5 passing)
└── test_main.py                 # ✅ API tests (needs Phase 3C completion)

documentation/
├── IMPLEMENTATION_COMPLETE.md   # ✅ Full project summary
├── HYBRID_FRAMENET_CLUSTER_APPROACH.md # ✅ Hybrid approach design
├── FRAMENET_CLUSTERING_DESIGN.md # ✅ Original design analysis
├── PHASE_3A_COMPLETE.md         # ✅ Phase 3A implementation summary
└── DESIGN_RECOMMENDATIONS.md    # ✅ This file - comprehensive design guide

configuration/
├── mypy.ini                     # ✅ Strict type checking configuration
├── pyproject.toml               # ✅ Poetry dependency management
└── environment.yml              # ✅ Conda environment specification
```

### 3. Implementation Progress

#### ✅ Phase 1: Core Abstractions (COMPLETED)
**Status**: Fully implemented and tested

**Achievements**:
- `Concept` class with synset disambiguation and context awareness
- `Axiom` class hierarchy with type system and formula representation
- `Context` management with inheritance and isolation
- YAML/JSON axiom file parsing with comprehensive validation
- WordNet integration with graceful degradation
- 18 comprehensive unit tests passing across all modules

#### ✅ Phase 2: Hybrid Semantic System (COMPLETED)
**Status**: Fully implemented with advanced features

**Achievements**:
- FrameNet-style semantic frames with elements and relationships
- Clustering-based concept organization with scikit-learn
- `HybridConceptRegistry` unifying frame and cluster management  
- `EnhancedHybridRegistry` with advanced reasoning capabilities
- Advanced analogical reasoning across multiple semantic levels
- Semantic field discovery and cross-domain analogy identification
- Vector embedding management with multiple providers (random, semantic, extensible)
- Comprehensive demonstration applications (medieval fantasy world building)
- Dynamic knowledge discovery and structural pattern mining

### ✅ Phase 3: Service Layer & API (IN PROGRESS - 3A COMPLETE)
**Goal**: Expose functionality via REST/WebSocket APIs with enterprise-grade type safety and contracts

**Current Status**: Phase 3A Complete, Phase 3B ready for implementation

**✅ Phase 3A: Type Safety Foundation (COMPLETED)**:
- ✅ Enhanced type hints with Generic type parameters for registry classes
- ✅ Protocol-based interfaces for embedding providers and reasoning engines (6 protocols implemented)
- ✅ TypedDict specifications for complex return types and API responses (15+ models)
- ✅ mypy strict mode integration and comprehensive type checking (zero type errors)
- ✅ Protocol implementation in EnhancedHybridRegistry with adapter methods
- ✅ Comprehensive test suite with 5/5 tests passing
- ✅ Runtime protocol compliance validation

**🔄 Phase 3B: Design by Contract Implementation (NEXT - Week 2)**:
- [ ] Precondition/postcondition contracts for all registry operations
- [ ] Class invariants for registry consistency validation
- [ ] Input validation contracts for reasoning operations
- [ ] Custom contract types for domain-specific constraints (coherence scores, embedding dimensions)

**🔄 Phase 3C: Service Layer Implementation (Week 3)**:
- [ ] FastAPI service layer with type-safe Pydantic models
- [ ] WebSocket streaming with contract-validated operations
- [ ] Model serving endpoints with Protocol-compliant interfaces
- [ ] Comprehensive integration testing with DbC validation

**Integration Points**:
1. Implement type-safe FastAPI app structure with Protocol interfaces
2. Create contract-validated endpoint handlers for concept, axiom, and context management
3. Add DbC-protected WebSocket support for real-time operations
4. Implement model serving endpoints with analogical reasoning contracts
5. Add visualization endpoints with type-safe concept space exploration
6. Integrate with existing hybrid registry system using Protocol compliance

#### 🔄 Phase 4: Neural-Symbolic Integration (PLANNED)
**Goal**: Integrate LTNtorch for end-to-end neural-symbolic learning

**Current Status**: Architecture ready, awaiting Phase 3 completion

**Planned Components**:
- LTNtorch wrapper for soft logic training integrated with existing hybrid system
- SMT verification integration for hard logic constraints (building on existing abstractions)
- Hybrid training pipelines combining symbolic reasoning with neural learning
- Enhanced model persistence and versioning for different contexts
- Performance optimization and scaling for production deployment

**Integration Points**:
1. Extend `EnhancedHybridRegistry` with LTNtorch training capabilities
2. Integrate Z3 SMT verification with existing axiom validation
3. Create training pipelines that leverage semantic field discovery
4. Enhance embedding management with neural-learned representations
5. Add model versioning and experiment tracking

### 4. Enhanced Axiom Format (Implemented)

Our axiom format has evolved beyond the original design to support the hybrid semantic system:

#### Current JSON Schema for Axiom Definition
```json
{
  "axiom_id": "analogy_king_queen",
  "type": "analogy",
  "classification": "core",
  "context": "default",
  "description": "Classic gender analogy: king is to man as queen is to woman",
  "formula": {
    "operation": "similarity",
    "left": {
      "operation": "add",
      "args": [
        {"operation": "subtract", "args": ["king", "man"]},
        "woman"
      ]
    },
    "right": "queen"
  },
  "concepts": [
    {
      "name": "king",
      "synset": "king.n.01",
      "disambiguation": "monarch ruler",
      "context": "royalty",
      "frame_roles": {
        "Royal_Hierarchy": "Ruler",
        "Political_Power": "Authority_Figure"
      }
    },
    {
      "name": "man", 
      "synset": "man.n.01",
      "disambiguation": "adult male human",
      "context": "default"
    },
    {
      "name": "woman",
      "synset": "woman.n.01", 
      "disambiguation": "adult female human",
      "context": "default"
    },
    {
      "name": "queen",
      "synset": "queen.n.01",
      "disambiguation": "female monarch",
      "context": "royalty",
      "frame_roles": {
        "Royal_Hierarchy": "Ruler",
        "Political_Power": "Authority_Figure"
      }
    }
  ],
  "semantic_metadata": {
    "semantic_field": "political_hierarchy",
    "cross_domain_analogies": ["military_hierarchy", "business_hierarchy"],
    "embedding_provider": "semantic",
    "cluster_memberships": {
      "authority_figures": 0.95,
      "human_roles": 0.87
    }
  },
  "metadata": {
    "created_by": "hybrid_system",
    "created_at": "2025-07-05T10:00:00Z",
    "confidence": 0.95,
    "source": "enhanced_semantic_reasoning",
    "version": "2.0"
  }
}
```

#### Enhanced YAML Format (Human-Friendly)
```yaml
axiom_id: analogy_king_queen
type: analogy
classification: core
context: royalty
description: "Classic gender analogy enhanced with frame semantics"

formula:
  similarity:
    left:
      add:
        - subtract: [king, man]
        - woman
    right: queen

concepts:
  king:
    synset: king.n.01
    disambiguation: "monarch ruler"
    context: royalty
    frame_roles:
      Royal_Hierarchy: Ruler
      Political_Power: Authority_Figure
  man:
    synset: man.n.01
    disambiguation: "adult male human"
    context: default
  woman:
    synset: woman.n.01
    disambiguation: "adult female human"
    context: default
  queen:
    synset: queen.n.01
    disambiguation: "female monarch"
    context: royalty
    frame_roles:
      Royal_Hierarchy: Ruler
      Political_Power: Authority_Figure

semantic_metadata:
  semantic_field: political_hierarchy
  cross_domain_analogies: [military_hierarchy, business_hierarchy]
  embedding_provider: semantic
  cluster_memberships:
    authority_figures: 0.95
    human_roles: 0.87

metadata:
  created_by: hybrid_system
  created_at: 2025-07-05T10:00:00Z
  confidence: 0.95
  source: enhanced_semantic_reasoning
  version: "2.0"
```

### 5. Enhanced System Bootstrap Process (Implemented)

#### Step 1: Hybrid Registry Initialization
```python
from app.core import EnhancedHybridRegistry

class EnhancedSystemBootstrap:
    def __init__(self):
        self.registry = EnhancedHybridRegistry(
            download_wordnet=True,
            n_clusters=8,
            enable_cross_domain=True,
            embedding_provider="semantic"
        )
        self.semantic_fields = []
        self.cross_domain_analogies = []
    
    def bootstrap_knowledge_base(self, domain: str):
        """Bootstrap a complete knowledge base for a domain"""
        # Load domain-specific concepts
        concepts = self.load_domain_concepts(domain)
        
        # Create semantic frames
        frames = self.create_domain_frames(domain)
        
        # Generate frame instances
        instances = self.create_frame_instances(domain, concepts, frames)
        
        # Discover semantic structure
        self.discover_semantic_structure()
        
        return {
            'concepts': concepts,
            'frames': frames, 
            'instances': instances,
            'fields': self.semantic_fields,
            'analogies': self.cross_domain_analogies
        }
    
    def discover_semantic_structure(self):
        """Discover semantic fields and cross-domain analogies"""
        # Update clustering
        self.registry.update_clusters()
        
        # Discover semantic fields
        self.semantic_fields = self.registry.discover_semantic_fields(
            min_coherence=0.4
        )
        
        # Find cross-domain analogies
        self.cross_domain_analogies = self.registry.discover_cross_domain_analogies(
            min_quality=0.3
        )
```

#### Step 2: Frame-Aware Concept Creation
```python
class FrameAwareConceptCreator:
    def __init__(self, registry: EnhancedHybridRegistry):
        self.registry = registry
        
    def create_rich_concept(self, name: str, context: str, 
                           semantic_info: dict) -> FrameAwareConcept:
        """Create concept with full semantic integration"""
        concept = self.registry.create_frame_aware_concept_with_advanced_embedding(
            name=name,
            context=context,
            synset_id=semantic_info.get('synset'),
            disambiguation=semantic_info.get('disambiguation'),
            use_semantic_embedding=True
        )
        
        # Add frame participations
        for frame_name, role in semantic_info.get('frame_roles', {}).items():
            concept.add_frame_role(frame_name, role)
        
        return concept
```

#### Step 3: Analogical Reasoning Pipeline
```python
class AnalogicalReasoningPipeline:
    def __init__(self, registry: EnhancedHybridRegistry):
        self.registry = registry
        
    def complete_analogy(self, partial_analogy: dict) -> list:
        """Complete analogies using hybrid reasoning"""
        # Try frame-based completion
        frame_completions = self.registry.find_analogical_completions(
            partial_analogy, max_completions=5
        )
        
        # Add cluster-based alternatives
        cluster_alternatives = self._find_cluster_alternatives(partial_analogy)
        
        # Combine and rank results
        return self._rank_completions(frame_completions + cluster_alternatives)
    
    def discover_analogical_patterns(self, domain1: str, domain2: str):
        """Discover cross-domain analogical patterns"""
        return self.registry.discover_cross_domain_analogies(
            min_quality=0.3
        )
```

### 6. Enhanced Storage and Exploration (Current Structure)

#### Current Project Storage Structure
```
/home/rdmerrio/gits/ltnexp03/
├── app/                          # Core application modules
│   ├── core/                     # Core implementation modules (✅ COMPLETE)
│   │   ├── abstractions.py          # Basic concept, axiom, context classes
│   │   ├── concept_registry.py      # WordNet-integrated concept management
│   │   ├── parsers.py               # YAML/JSON axiom parsing
│   │   ├── frame_cluster_abstractions.py  # FrameNet & clustering structures
│   │   ├── frame_cluster_registry.py      # Frame & cluster registries
│   │   ├── hybrid_registry.py             # Unified hybrid system
│   │   ├── enhanced_semantic_reasoning.py # Advanced reasoning capabilities
│   │   ├── vector_embeddings.py           # Vector embedding management
│   │   └── __init__.py                     # Module exports
│   └── main.py                   # FastAPI application (basic structure ready)
├── embeddings_cache/             # Vector embedding persistence
│   └── *.npy, *.json            # Cached embeddings and metadata
├── examples/                     # Example axiom files (✅ IMPLEMENTED)
│   ├── basic_analogy.yaml       # Core analogy axioms
│   └── core_axioms.json         # JSON format examples
├── tests/                        # Comprehensive test suite (✅ 18 TESTS PASSING)
│   ├── test_core/               # Core module tests
│   │   └── test_abstractions.py    # Complete coverage
│   └── test_main.py             # API tests (needs fixing)
├── demo_*.py                    # Working demonstration systems (✅ COMPLETE)
│   ├── demo_hybrid_system.py       # Basic hybrid capabilities
│   ├── demo_enhanced_system.py     # Advanced reasoning features
│   └── demo_comprehensive_system.py # Medieval fantasy application
├── docs/                        # Comprehensive documentation (✅ COMPLETE)
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── HYBRID_FRAMENET_CLUSTER_APPROACH.md
│   ├── FRAMENET_CLUSTERING_DESIGN.md
│   ├── PHASE1_COMPLETE.md
│   └── DESIGN_RECOMMENDATIONS.md (this file)
├── pyproject.toml               # Poetry dependency management
├── environment.yml              # Conda environment specification
└── README.md                    # Project documentation
```

#### Enhanced Model Exploration Interface (Implemented)
```python
from app.core import EnhancedHybridRegistry

class AdvancedModelExplorer:
    def __init__(self, context_name: str = "default"):
        self.registry = EnhancedHybridRegistry()
        self.context = context_name
    
    def find_analogous_concepts(self, concept: str, 
                               frame_threshold: float = 0.6,
                               cluster_threshold: float = 0.7) -> List[Tuple]:
        """Find concepts analogous at multiple semantic levels"""
        return self.registry.find_analogous_concepts(
            concept, 
            frame_threshold=frame_threshold,
            cluster_threshold=cluster_threshold
        )
    
    def complete_analogy(self, a: str, b: str, c: str) -> List[str]:
        """Complete analogy: a is to b as c is to ?"""
        partial = {a: b, c: "?"}
        completions = self.registry.find_analogical_completions(partial)
        return [comp[c] for comp in completions]
    
    def discover_semantic_fields(self, min_coherence: float = 0.4) -> List:
        """Discover coherent semantic fields"""
        return self.registry.discover_semantic_fields(min_coherence)
    
    def find_cross_domain_analogies(self, min_quality: float = 0.3) -> List:
        """Find structural patterns across domains"""
        return self.registry.discover_cross_domain_analogies(min_quality)
    
    def visualize_concept_space(self, concepts: List[str] = None):
        """Generate advanced visualization of hybrid concept space"""
        # Implementation ready for Phase 3 API integration
        pass
    
    def export_knowledge_base(self, format: str = "json") -> str:
        """Export complete knowledge base"""
        stats = self.registry.get_enhanced_statistics()
        return {
            "concepts": len(self.registry.frame_aware_concepts),
            "frames": len(self.registry.frame_registry.frames),
            "clusters": len(self.registry.cluster_registry.clusters),
            "semantic_fields": len(self.registry.semantic_fields),
            "analogies": len(self.registry.cross_domain_analogies),
            "quality_metrics": {
                "avg_cluster_size": stats.get("avg_cluster_size", 0),
                "avg_field_size": stats.get("avg_field_size", 0),
                "avg_analogy_quality": stats.get("avg_analogy_quality", 0)
            }
        }
```

### 7. Enhanced Context Management (Ready for Implementation)

#### Hybrid Context Inheritance System
```python
from app.core import EnhancedHybridRegistry, HybridConceptRegistry

class EnhancedContextManager:
    def __init__(self):
        self.contexts = {}
        self.inheritance_graph = {}
        self.base_registry = EnhancedHybridRegistry()
    
    def create_context(self, name: str, parent: str = "default", 
                      context_type: str = "domain"):
        """Create new context with enhanced inheritance"""
        if parent not in self.contexts and parent != "default":
            raise ContextNotFoundError(f"Parent context {parent} not found")
            
        # Create new registry inheriting from parent
        new_registry = EnhancedHybridRegistry(
            download_wordnet=False,  # Inherit WordNet from parent
            n_clusters=8,
            enable_cross_domain=True
        )
        
        # Inherit core concepts and frames from parent
        if parent != "default":
            parent_registry = self.contexts[parent]
            self._inherit_knowledge_base(new_registry, parent_registry)
        
        self.contexts[name] = {
            'registry': new_registry,
            'parent': parent,
            'type': context_type,
            'created_at': datetime.now(),
            'metadata': {}
        }
        
        # Update inheritance graph
        if parent not in self.inheritance_graph:
            self.inheritance_graph[parent] = []
        self.inheritance_graph[parent].append(name)
        
        return new_registry
    
    def _inherit_knowledge_base(self, child_registry: EnhancedHybridRegistry,
                               parent_registry: EnhancedHybridRegistry):
        """Inherit knowledge base from parent context"""
        # Inherit concepts
        for concept_id, concept in parent_registry.frame_aware_concepts.items():
            child_registry.frame_aware_concepts[concept_id] = concept
        
        # Inherit frames
        for frame_name, frame in parent_registry.frame_registry.frames.items():
            child_registry.frame_registry.frames[frame_name] = frame
        
        # Inherit embeddings
        for concept_id, embedding in parent_registry.cluster_registry.concept_embeddings.items():
            child_registry.cluster_registry.concept_embeddings[concept_id] = embedding
        
        # Inherit semantic fields
        for field_name, field in parent_registry.semantic_fields.items():
            child_registry.semantic_fields[field_name] = field
    
    def add_domain_knowledge(self, context_name: str, domain_data: dict):
        """Add domain-specific knowledge to context"""
        if context_name not in self.contexts:
            raise ContextNotFoundError(f"Context {context_name} not found")
        
        registry = self.contexts[context_name]['registry']
        
        # Add domain concepts
        for concept_data in domain_data.get('concepts', []):
            concept = registry.create_frame_aware_concept_with_advanced_embedding(
                name=concept_data['name'],
                context=context_name,
                synset_id=concept_data.get('synset'),
                disambiguation=concept_data.get('disambiguation'),
                use_semantic_embedding=True
            )
        
        # Add domain frames
        for frame_data in domain_data.get('frames', []):
            frame = registry.create_semantic_frame(
                name=frame_data['name'],
                definition=frame_data['definition'],
                core_elements=frame_data.get('core_elements', []),
                peripheral_elements=frame_data.get('peripheral_elements', [])
            )
        
        # Update semantic structure
        registry.update_clusters()
        registry.discover_semantic_fields()
        registry.discover_cross_domain_analogies()
    
    def get_context_statistics(self, context_name: str) -> dict:
        """Get comprehensive context statistics"""
        if context_name not in self.contexts:
            return {}
        
        registry = self.contexts[context_name]['registry']
        return registry.get_enhanced_statistics()
```

### 7. WordNet Integration

#### Synset Disambiguation
```python
class ConceptRegistry:
    def __init__(self):
        self.synset_mappings = {}
        self.concept_cache = {}
    
    def register_concept(self, name: str, synset_id: str = None, 
                        disambiguation: str = None):
        """Register concept with optional WordNet synset"""
        if synset_id:
            # Validate synset exists in WordNet
            synset = self.validate_synset(synset_id)
            self.synset_mappings[name] = synset_id
            
        concept = Concept(
            name=name,
            synset_id=synset_id,
            disambiguation=disambiguation
        )
        self.concept_cache[name] = concept
        return concept
    
    def disambiguate_concept(self, name: str, context: str = None) -> Concept:
        """Resolve concept name to specific synset in context"""
        if context:
            # Check for context-specific mappings
            context_key = f"{context}:{name}"
            if context_key in self.concept_cache:
                return self.concept_cache[context_key]
        
        # Fall back to default mapping
        return self.concept_cache.get(name)
```

### 8. API Design with Type Safety and Contract Validation

#### Current State
A basic FastAPI application structure exists in `app/main.py` with health check endpoints. Enhanced with comprehensive type safety and Design by Contract validation ready for implementation.

#### Type-Safe API Architecture

**Protocol-Based Service Interfaces**:
```python
from typing import Protocol, runtime_checkable, TypedDict, List, Dict, Optional
from contracts import contract, new_contract

# Define custom contracts for domain validation
new_contract('coherence_score', lambda x: 0.0 <= x <= 1.0)
new_contract('confidence_score', lambda x: 0.0 <= x <= 1.0)
new_contract('positive_int', lambda x: isinstance(x, int) and x > 0)

@runtime_checkable
class SemanticReasoningService(Protocol):
    """Protocol for semantic reasoning services with contract validation."""
    
    @contract(concept='str', context='str', returns='list[tuple[str, float]]')
    def find_analogous_concepts(self, concept: str, context: str) -> List[Tuple[str, float]]: ...
    
    @contract(partial_analogy='dict[str:str]', max_completions='positive_int', 
              returns='list[AnalogicalCompletionResult]')
    def complete_analogy(self, partial_analogy: Dict[str, str], 
                        max_completions: int = 5) -> List['AnalogicalCompletionResult']: ...
    
    @contract(min_coherence='coherence_score', returns='list[SemanticField]')
    def discover_semantic_fields(self, min_coherence: float = 0.7) -> List['SemanticField']: ...

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers ensuring consistent interface."""
    
    def generate_embedding(self, concept: str, context: str = "default") -> np.ndarray: ...
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float: ...
    
    @property
    def embedding_dimension(self) -> int: ...
```

**Type-Safe Request/Response Models**:
```python
from pydantic import BaseModel, Field, validator
from typing import Literal, Union

class AnalogicalCompletionResult(TypedDict):
    """Typed result for analogical completion operations."""
    completion: str
    confidence: float
    reasoning_type: Literal["frame", "cluster", "hybrid"]
    source_evidence: List[str]
    metadata: Dict[str, Any]

class ConceptRequest(BaseModel):
    """Type-safe concept creation request."""
    name: str = Field(..., min_length=1, description="Concept name")
    synset_id: Optional[str] = Field(None, description="WordNet synset ID")
    disambiguation: Optional[str] = Field(None, description="Disambiguation text")
    context: str = Field("default", description="Context name")
    frame_roles: Dict[str, str] = Field(default_factory=dict, description="Frame role assignments")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Concept name cannot be empty')
        return v.lower().strip()

class AnalogyRequest(BaseModel):
    """Type-safe analogical reasoning request."""
    context: str = Field(..., description="Context for analogy completion")
    partial_analogy: Dict[str, str] = Field(..., description="Partial analogy mapping")
    max_completions: int = Field(5, gt=0, le=20, description="Maximum completions to return")
    reasoning_types: List[Literal["frame", "cluster", "hybrid"]] = Field(
        default=["hybrid"], description="Reasoning approaches to use"
    )
    
    @validator('partial_analogy')
    def validate_partial_analogy(cls, v):
        if len(v) < 2:
            raise ValueError('Partial analogy must have at least 2 mappings')
        if "?" not in v.values():
            raise ValueError('Partial analogy must contain "?" for completion')
        return v

class SemanticFieldResponse(BaseModel):
    """Type-safe semantic field discovery response."""
    field_name: str
    description: str
    core_concepts: List[str]
    related_concepts: Dict[str, float]
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    associated_frames: List[str]
    discovery_metadata: Dict[str, Any]
```

#### Contract-Validated Core Endpoints

**Context Management with DbC**:
```python
from fastapi import FastAPI, HTTPException, Depends
from contracts import contract

app = FastAPI(title="Soft Logic Microservice", version="2.0.0")

@app.post("/contexts/{context_name}", response_model=ContextResponse)
@contract(context_name='str,len(x)>0', context_type='str')
async def create_context(
    context_name: str, 
    context_type: str = "domain",
    parent_context: Optional[str] = None
) -> ContextResponse:
    """
    Create new context with enhanced inheritance.
    
    :pre: context_name not in existing_contexts
    :pre: parent_context is None or parent_context in existing_contexts
    :post: context_name in existing_contexts
    """
    # Implementation with contract validation
    pass

@app.get("/contexts/{context_name}/concepts", response_model=List[ConceptResponse])
@contract(context_name='str', returns='list[ConceptResponse]')
async def list_concepts(context_name: str) -> List[ConceptResponse]:
    """
    List all concepts in a context.
    
    :pre: context_name in existing_contexts
    :post: len(__return__) >= 0
    """
    # Implementation with contract validation
    pass
```

**Advanced Reasoning Endpoints with Type Safety**:
```python
@app.post("/contexts/{context_name}/analogies", response_model=List[AnalogicalCompletionResult])
@contract(context_name='str', returns='list[AnalogicalCompletionResult],len(x)<=request.max_completions')
async def complete_analogy(
    context_name: str, 
    request: AnalogyRequest,
    service: SemanticReasoningService = Depends(get_reasoning_service)
) -> List[AnalogicalCompletionResult]:
    """
    Complete analogical reasoning with contract validation.
    
    :pre: context_name in existing_contexts
    :pre: len(request.partial_analogy) >= 2
    :post: len(__return__) <= request.max_completions
    :post: all(0.0 <= result.confidence <= 1.0 for result in __return__)
    """
    try:
        completions = service.complete_analogy(
            request.partial_analogy, 
            request.max_completions
        )
        return completions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analogy completion failed: {str(e)}")

@app.get("/contexts/{context_name}/semantic-fields", response_model=List[SemanticFieldResponse])
@contract(context_name='str', min_coherence='coherence_score', returns='list[SemanticFieldResponse]')
async def discover_semantic_fields(
    context_name: str,
    min_coherence: float = 0.7,
    service: SemanticReasoningService = Depends(get_reasoning_service)
) -> List[SemanticFieldResponse]:
    """
    Discover semantic fields with contract validation.
    
    :pre: context_name in existing_contexts
    :pre: 0.0 <= min_coherence <= 1.0
    :post: all(field.coherence_score >= min_coherence for field in __return__)
    """
    try:
        fields = service.discover_semantic_fields(min_coherence)
        return [SemanticFieldResponse.from_semantic_field(field) for field in fields]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic field discovery failed: {str(e)}")
```

#### Enhanced WebSocket with Contract Validation

**Real-time Semantic Discovery with DbC**:
```python
from fastapi import WebSocket, WebSocketDisconnect
from contracts import contract

@app.websocket("/ws/discovery/{context_name}")
@contract(context_name='str')
async def semantic_discovery_stream(websocket: WebSocket, context_name: str):
    """
    Stream semantic field discovery with contract validation.
    
    :pre: context_name in existing_contexts
    """
    await websocket.accept()
    
    try:
        service = get_reasoning_service(context_name)
        
        # Stream discovery updates with contract validation
        async for discovery_update in service.discover_semantic_fields_streaming():
            # Contract validation for each update
            assert 0.0 <= discovery_update.coherence <= 1.0, "Invalid coherence score"
            assert len(discovery_update.concepts) > 0, "Empty concept list"
            
            await websocket.send_json({
                "type": "semantic_field_discovered",
                "field_name": discovery_update.field_name,
                "concepts": discovery_update.concepts,
                "coherence": discovery_update.coherence,
                "timestamp": discovery_update.timestamp.isoformat()
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Discovery stream error: {str(e)}"
        })
        await websocket.close()

@app.websocket("/ws/training/{context_name}")
@contract(context_name='str')
async def training_progress_stream(websocket: WebSocket, context_name: str):
    """
    Stream training progress with contract validation (Phase 4).
    
    :pre: context_name in existing_contexts
    """
    await websocket.accept()
    
    try:
        trainer = get_training_service(context_name)
        
        async for progress in trainer.train_with_progress():
            # Contract validation for training progress
            assert 0 <= progress.epoch, "Invalid epoch number"
            assert 0.0 <= progress.loss, "Invalid loss value"
            assert 0.0 <= progress.satisfiability <= 1.0, "Invalid satisfiability score"
            
            await websocket.send_json({
                "epoch": progress.epoch,
                "loss": progress.loss,
                "satisfiability": progress.satisfiability,
                "stage": progress.stage,
                "timestamp": datetime.now().isoformat()
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Training stream error: {str(e)}"
        })
        await websocket.close()
```

#### Dependency Injection with Protocol Compliance

**Service Provider with Runtime Type Checking**:
```python
from typing import get_type_hints
from fastapi import Depends

def get_reasoning_service(context_name: str) -> SemanticReasoningService:
    """Get reasoning service with protocol compliance validation."""
    service = create_enhanced_hybrid_registry(context_name)
    
    # Runtime protocol compliance check
    if not isinstance(service, SemanticReasoningService):
        raise RuntimeError(f"Service does not implement SemanticReasoningService protocol")
    
    return service

def get_embedding_provider(provider_name: str = "semantic") -> EmbeddingProvider:
    """Get embedding provider with protocol compliance validation."""
    provider = create_embedding_provider(provider_name)
    
    # Runtime protocol compliance check
    if not isinstance(provider, EmbeddingProvider):
        raise RuntimeError(f"Provider does not implement EmbeddingProvider protocol")
    
    return provider
```

## Updated Implementation Roadmap

### ✅ Phase 1: Foundation (COMPLETED)
- [x] Core abstractions (Concept, Axiom, Context)
- [x] Axiom file format and parser
- [x] Comprehensive test suite (18 tests passing)
- [x] WordNet integration with graceful degradation

### ✅ Phase 2: Hybrid Semantic System (COMPLETED)
- [x] Frame and cluster abstractions
- [x] FrameNet-style semantic frame system
- [x] Clustering-based concept organization
- [x] Hybrid registry integration
- [x] Enhanced semantic reasoning capabilities
- [x] Vector embedding management system
- [x] Cross-domain analogical reasoning
- [x] Semantic field discovery
- [x] Comprehensive demonstration systems

### 🔄 Phase 3: Service Layer (NEXT - Weeks 1-3)
- [ ] **Phase 3A: Type Safety Foundation** (Week 1)
  - [ ] Enhanced type hints with Generic parameters for all registry classes
  - [ ] Protocol interfaces for embedding providers and reasoning engines
  - [ ] TypedDict specifications for API request/response models
  - [ ] mypy strict mode integration and comprehensive type validation
- [ ] **Phase 3B: Design by Contract** (Week 2)
  - [ ] Precondition/postcondition contracts for registry operations
  - [ ] Class invariants for consistency validation
  - [ ] Input validation contracts for reasoning operations
  - [ ] Custom contract types for domain constraints
- [ ] **Phase 3C: Service Implementation** (Week 3)
  - [ ] FastAPI application with type-safe Pydantic models
  - [ ] Contract-validated REST API endpoints
  - [ ] DbC-protected WebSocket streaming for real-time operations
  - [ ] Protocol-compliant model serving endpoints
  - [ ] Comprehensive integration testing with contract validation

### 🔄 Phase 4: Neural-Symbolic Integration (FUTURE - Weeks 4-8)
- [ ] LTNtorch wrapper integration
- [ ] SMT verification layer (Z3 integration)
- [ ] Hybrid training pipelines
- [ ] Enhanced model persistence and versioning
- [ ] Performance optimization and scaling
- [ ] Production deployment configuration

## Testing Strategy with Type Safety and Contract Validation

### ✅ Current Unit Tests (18 tests passing)
- **Core Abstractions**: Complete coverage of `Concept`, `Axiom`, `Context`, and `FormulaNode` classes
- **Concept Registry**: WordNet integration, homonym handling, pattern search
- **Axiom Parser**: YAML and JSON parsing validation
- **All tests passing**: Full test suite runs successfully in under 1 second

### 🔄 Enhanced Testing for Phase 3

#### Type Safety Testing
- **mypy Integration**: Continuous type checking in CI/CD pipeline
- **Protocol Compliance**: Runtime validation of protocol implementations
- **Generic Type Validation**: Testing of type-safe registry operations
- **API Type Validation**: Pydantic model validation in all endpoints

#### Contract Validation Testing
- **Precondition Testing**: Validate all contract preconditions are enforced
- **Postcondition Testing**: Verify all contract postconditions are satisfied
- **Invariant Testing**: Ensure class invariants hold after all operations
- **Edge Case Validation**: Test contract behavior at domain boundaries

#### Integration Testing with DbC
- **End-to-end Registry Operations**: Full workflow testing with contract validation
- **API Contract Enforcement**: Verify all endpoints enforce input/output contracts
- **WebSocket Contract Validation**: Real-time operation contract enforcement
- **Cross-domain Analogical Reasoning**: Complex reasoning workflow validation

#### Contract-Enhanced Test Examples
```python
# tests/test_enhanced_contracts.py
import pytest
from contracts import contract, ContractViolation
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry

class TestContractValidation:
    """Test Design by Contract enforcement."""
    
    def test_semantic_field_discovery_contracts(self):
        """Test semantic field discovery contract enforcement."""
        registry = EnhancedHybridRegistry()
        
        # Test precondition violation
        with pytest.raises(ContractViolation):
            # Should fail: clustering not trained
            registry.discover_semantic_fields(min_coherence=0.7)
        
        # Train clustering first
        registry.add_sample_concepts()
        registry.update_clusters()
        
        # Test valid operation
        fields = registry.discover_semantic_fields(min_coherence=0.5)
        
        # Verify postconditions
        assert all(field.coherence >= 0.5 for field in fields)
        assert len(fields) >= 0
    
    def test_analogical_completion_contracts(self):
        """Test analogical completion contract validation."""
        registry = EnhancedHybridRegistry()
        
        # Test precondition violations
        with pytest.raises(ContractViolation):
            # Should fail: too few mappings
            registry.find_analogical_completions({"a": "b"})
        
        with pytest.raises(ContractViolation):
            # Should fail: no "?" for completion
            registry.find_analogical_completions({"a": "b", "c": "d"})
        
        # Test valid operation
        results = registry.find_analogical_completions(
            {"king": "queen", "man": "?"}, max_completions=3
        )
        
        # Verify postconditions
        assert len(results) <= 3
        assert all(0.0 <= result['confidence'] <= 1.0 for result in results)

class TestProtocolCompliance:
    """Test Protocol interface compliance."""
    
    def test_semantic_reasoning_protocol(self):
        """Test SemanticReasoningEngine protocol compliance."""
        from app.core.protocols import SemanticReasoningEngine
        
        registry = EnhancedHybridRegistry()
        
        # Verify protocol compliance
        assert isinstance(registry, SemanticReasoningEngine)
        
        # Test required methods exist and work
        assert hasattr(registry, 'complete_analogy')
        assert hasattr(registry, 'discover_semantic_fields')
        assert hasattr(registry, 'find_analogous_concepts')
    
    def test_embedding_provider_protocol(self):
        """Test EmbeddingProvider protocol compliance."""
        from app.core.protocols import EmbeddingProvider
        from app.core.vector_embeddings import SemanticEmbeddingProvider
        
        provider = SemanticEmbeddingProvider()
        
        # Verify protocol compliance
        assert isinstance(provider, EmbeddingProvider)
        
        # Test required interface
        embedding = provider.generate_embedding("test", "default")
        assert embedding.shape[0] == provider.embedding_dimension

class TestTypeValidation:
    """Test type safety and validation."""
    
    def test_typed_api_models(self):
        """Test Pydantic model validation."""
        from app.api.types import ConceptRequest, AnalogyRequest
        
        # Valid concept request
        concept_req = ConceptRequest(
            name="king",
            synset_id="king.n.01",
            disambiguation="monarch"
        )
        assert concept_req.name == "king"
        
        # Invalid concept request
        with pytest.raises(ValueError):
            ConceptRequest(name="")  # Empty name
        
        # Valid analogy request
        analogy_req = AnalogyRequest(
            context="default",
            partial_analogy={"king": "queen", "man": "?"},
            max_completions=5
        )
        assert len(analogy_req.partial_analogy) == 2
        
        # Invalid analogy request
        with pytest.raises(ValueError):
            AnalogyRequest(
                context="default",
                partial_analogy={"king": "queen"},  # Too few mappings
                max_completions=5
            )
```

### 🔄 Planned Performance Tests (Phase 4)
- **Contract Overhead Analysis**: Measure performance impact of contract validation
- **Type Checking Performance**: Analyze runtime type validation costs
- **Semantic Field Discovery Benchmarks**: Performance with contract validation
- **API Response Times**: Type-safe endpoint performance under load
- **Concurrent Request Handling**: Multi-tenant contract enforcement performance

## Deployment Considerations

### Docker Configuration
```dockerfile
FROM python:3.11-slim

# Install system dependencies for Z3
RUN apt-get update && apt-get install -y \
    build-essential \
    libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-dev

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY axioms/ ./axioms/

EXPOSE 8321
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8321"]
```

### Environment Variables
```bash
# Core configuration
EMBEDDING_DIMENSION=128
MAX_TRAINING_ITERATIONS=5000
TRUTH_THRESHOLD=0.95

# Storage
MODEL_STORAGE_PATH="/data/models"
AXIOM_STORAGE_PATH="/data/axioms"

# API
API_HOST="0.0.0.0"
API_PORT=8321
API_DEBUG=false

# External services
WORDNET_DATA_PATH="/data/wordnet"
```

## Next Steps: Phase 3 Implementation Plan with Type Safety and Contract Validation

### Immediate Priorities

#### 1. Type Safety Foundation (Week 1)
**Goal**: Establish comprehensive type safety and Protocol interfaces

**Tasks**:
- **Enhanced Type Annotations**: Add Generic type parameters to all registry classes
- **Protocol Interfaces**: Define and implement protocols for embedding providers and reasoning engines
- **TypedDict Specifications**: Create structured types for all API request/response models
- **mypy Integration**: Enable strict type checking and resolve all type validation issues

**Implementation Structure**:
```python
# app/core/protocols.py - Protocol definitions
from typing import Protocol, TypeVar, Generic, runtime_checkable

T = TypeVar('T', bound='Concept')
K = TypeVar('K', bound=str)

@runtime_checkable
class ConceptRegistry(Protocol, Generic[T, K]):
    """Type-safe protocol for concept registries."""
    def register_concept(self, concept: T) -> K: ...
    def get_concept(self, concept_id: K) -> Optional[T]: ...
    def find_similar_concepts(self, concept: T, threshold: float) -> List[Tuple[T, float]]: ...

@runtime_checkable
class SemanticReasoningEngine(Protocol):
    """Protocol for semantic reasoning capabilities."""
    def complete_analogy(self, partial: Dict[str, str]) -> List[AnalogicalCompletionResult]: ...
    def discover_semantic_fields(self, min_coherence: float) -> List[SemanticField]: ...
    def find_cross_domain_analogies(self, min_quality: float) -> List[CrossDomainAnalogy]: ...

# app/api/types.py - Type definitions
class AnalogicalCompletionResult(TypedDict):
    """Typed result for analogical completion."""
    completion: str
    confidence: float
    reasoning_type: Literal["frame", "cluster", "hybrid"]
    source_evidence: List[str]

def find_analogical_completions(
    self, 
    partial_analogy: Dict[str, str], 
    max_completions: int = 5
) -> List[AnalogicalCompletionResult]:
    """Find analogical completions with structured results."""
    ...
```

#### 2. Design by Contract Implementation (Week 2)
**Goal**: Add comprehensive contract validation to all operations

**Tasks**:
- **Precondition/Postcondition Contracts**: Add contracts to all registry operations
- **Class Invariants**: Implement consistency validation for registry state
- **Input Validation**: Add contract-based validation for reasoning operations
- **Custom Contract Types**: Create domain-specific contract validators

**Contract Implementation Examples**:
```python
# app/core/enhanced_semantic_reasoning.py - Contract integration
from contracts import contract, new_contract

# Custom contract definitions
new_contract('coherence_score', lambda x: 0.0 <= x <= 1.0)
new_contract('positive_int', lambda x: isinstance(x, int) and x > 0)

class EnhancedHybridRegistry:
    
    @contract(min_coherence='coherence_score', 
              returns='list[SemanticField]')
    def discover_semantic_fields(self, min_coherence: float = 0.7) -> List[SemanticField]:
        """
        Discover semantic fields with contract validation.
        
        :pre: self.cluster_registry.is_trained
        :post: all(field.coherence >= min_coherence for field in __return__)
        :post: len(__return__) >= 0
        """
        assert self.cluster_registry.is_trained, "Clustering must be trained"
        
        fields = self._discover_fields_internal(min_coherence)
        
        # Post-condition validation
        for field in fields:
            assert field.coherence >= min_coherence
        
        return fields
```

**Registry State Invariants**:
```python
class HybridConceptRegistry:
    
    def __invariant__(self):
        """Class invariants that must hold after every public method."""
        # All frame-aware concepts must have valid embeddings
        for concept_id, concept in self.frame_aware_concepts.items():
            if concept.embedding is not None:
                assert concept.embedding.shape[0] == self.embedding_dim
        
        # All registered concepts must have unique IDs
        all_ids = [c.unique_id for c in self.frame_aware_concepts.values()]
        assert len(all_ids) == len(set(all_ids)), "Duplicate concept IDs detected"
        
        # Consistency between registries
        frame_concept_ids = set(self.frame_aware_concepts.keys())
        cluster_concept_ids = set(self.cluster_registry.concept_embeddings.keys())
        # Frame concepts should be subset of cluster concepts (after clustering)
        if self.cluster_registry.is_trained:
            assert frame_concept_ids.issubset(cluster_concept_ids)
```

**Analogical Reasoning Contracts**:
```python
@contract(partial_analogy='dict[str:str]',
          max_completions='positive_int',
          returns='list[AnalogicalCompletionResult],len(x)<=max_completions')
def find_analogical_completions(
    self, 
    partial_analogy: Dict[str, str], 
    max_completions: int = 5
) -> List[AnalogicalCompletionResult]:
    """
    Find analogical completions for partial analogies.
    
    :pre: len(partial_analogy) >= 2
    :pre: "?" in partial_analogy.values()
    :post: len(__return__) <= max_completions
    :post: all(0.0 <= result['confidence'] <= 1.0 for result in __return__)
    """
    ...
```

### 3. Interface Specifications - **HIGH IMPACT**

#### Recommended Protocol Interfaces

**Reasoning Engine Protocol**:
```python
@runtime_checkable
class ReasoningEngine(Protocol):
    """Protocol for different reasoning approaches."""
    
    def complete_analogy(self, partial: Dict[str, str]) -> List[str]: ...
    def find_similar_concepts(self, concept: str, threshold: float = 0.7) -> List[Tuple[str, float]]: ...
    def validate_consistency(self, axioms: List[Axiom]) -> bool: ...
```

**Knowledge Discovery Protocol**:
```python
@runtime_checkable  
class KnowledgeDiscovery(Protocol):
    """Protocol for knowledge discovery operations."""
    
    def discover_patterns(self, domain: str) -> List[Pattern]: ...
    def extract_relationships(self, concepts: List[str]) -> List[Relationship]: ...
    def suggest_analogies(self, source_domain: str, target_domain: str) -> List[CrossDomainAnalogy]: ...
```

### 4. Implementation Priority Recommendations

#### Phase 3A: Essential Type Safety (Week 1)
1. **Add comprehensive type hints** to all public APIs
2. **Implement Protocol interfaces** for embedding providers and reasoning engines  
3. **Add TypedDict classes** for complex return types and configuration objects
4. **Enable mypy strict mode** and fix all type checking issues

#### Phase 3B: Design by Contract (Week 2)  
1. **Add precondition/postcondition contracts** to all registry operations
2. **Implement class invariants** for registry consistency
3. **Add input validation contracts** for reasoning operations
4. **Create custom contract types** for domain-specific constraints

#### Phase 3C: Interface Protocols (Week 3)
1. **Define Protocol interfaces** for all major subsystems
2. **Refactor existing code** to implement protocols explicitly
3. **Add runtime type checking** for protocol compliance
4. **Create interface documentation** with usage examples

### 5. Expected Benefits

#### Immediate Benefits (Phase 3A)
- **IDE Support**: Better autocomplete, error detection, refactoring
- **Bug Prevention**: Catch type mismatches at development time
- **Documentation**: Type hints serve as executable documentation
- **API Clarity**: Clear interfaces for service layer development

#### Strategic Benefits (Phase 3B+)
- **Robust Service Layer**: DbC ensures API reliability for Phase 3 implementation
- **Neural Integration Safety**: Contracts validate neural-symbolic integration boundaries
- **Production Readiness**: Enterprise-grade reliability for deployment
- **Team Development**: Clear contracts enable parallel development

### 6. Tooling Integration

```bash
# Add to pyproject.toml
[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.contracts]
enable_all = true
check_contracts = true

# Development workflow  
poetry add --group dev mypy pycontracts
poetry add --group dev types-requests types-pyyaml
```

**Recommendation**: Implement these enhancements now, before Phase 3 service layer development. The current codebase complexity and maturity make this the optimal time for these quality improvements, and they will significantly accelerate Phase 3-4 development while ensuring production reliability.

## Design by Contract Library Recommendation for Phase 3B

### **Recommended Library: `icontract`**

After comprehensive analysis and testing, I recommend **`icontract`** for Phase 3B implementation:

**Why `icontract` is optimal for your project:**

1. **Mypy Integration**: Excellent static type checking support with your existing mypy configuration
2. **Mature and Stable**: Well-maintained library with comprehensive documentation
3. **Feature Rich**: Supports preconditions, postconditions, class invariants, and snapshot testing
4. **Performance Options**: Contracts can be disabled in production for zero overhead
5. **FastAPI Compatible**: Seamless integration with your REST API service layer
6. **Protocol-Friendly**: Works perfectly with your Protocol-based architecture

**Installation and Setup:**
```bash
# Already added to pyproject.toml
poetry add icontract

# Add to your imports
from icontract import require, ensure, invariant, ViolationError
```

### **Phase 3B Implementation Plan**

**Week 2 Daily Tasks:**

**Day 1: Contract Framework Setup**
- Install and configure icontract
- Create domain-specific contract validators in `app/core/contracts.py`
- Test basic contract functionality with simple examples

**Day 2: Registry Operation Contracts**
- Add precondition/postcondition contracts to concept creation
- Implement class invariants for `ConceptRegistry` and `EnhancedHybridRegistry`
- Create contract-validated wrapper methods for backward compatibility

**Day 3: Reasoning Operation Contracts**
- Add contracts to semantic field discovery methods
- Implement analogical completion contract validation
- Add similarity computation contracts with embedding validation

**Day 4: API Layer Preparation**
- Create contract-validated service interfaces
- Add input validation contracts for FastAPI endpoints
- Implement error handling for contract violations

**Day 5: Testing and Integration**
- Comprehensive contract violation test suite
- Performance impact analysis
- Documentation and examples

**Contract Implementation Examples:**

```python
from icontract import require, ensure, invariant, ViolationError
from typing import List, Dict, Optional

# Domain-specific validators
class SoftLogicContracts:
    @staticmethod
    def valid_concept_name(name: str) -> bool:
        return isinstance(name, str) and len(name.strip()) > 0
    
    @staticmethod
    def valid_coherence_score(score: float) -> bool:
        return 0.0 <= score <= 1.0

# Registry with contracts
@invariant(lambda self: len(self.frame_aware_concepts) >= 0)
class ContractEnhancedRegistry(EnhancedHybridRegistry):
    
    @require(lambda name: SoftLogicContracts.valid_concept_name(name))
    @require(lambda context: context in ['default', 'wordnet', 'custom', 'neural'])
    @ensure(lambda result: result is not None)
    @ensure(lambda result, name: result.name == name)
    def create_concept_with_contracts(
        self, 
        name: str, 
        context: str = "default"
    ) -> FrameAwareConcept:
        return self.create_frame_aware_concept_with_advanced_embedding(
            name=name, context=context, use_semantic_embedding=True
        )
    
    @require(lambda min_coherence: SoftLogicContracts.valid_coherence_score(min_coherence))
    @require(lambda max_fields: 1 <= max_fields <= 100)
    @ensure(lambda result, min_coherence: all(f['coherence'] >= min_coherence for f in result))
    def discover_semantic_fields_with_contracts(
        self, 
        min_coherence: float = 0.7,
        max_fields: int = 10
    ) -> List[Dict[str, Any]]:
        fields = self.discover_semantic_fields(min_coherence=min_coherence)
        return list(fields.items())[:max_fields]

# FastAPI integration
from fastapi import HTTPException

async def handle_contract_violation(violation: ViolationError):
    raise HTTPException(
        status_code=400,
        detail=f"Contract violation: {violation}"
    )
```

**Benefits for Your Project:**

1. **Early Error Detection**: Invalid inputs caught before processing
2. **API Reliability**: Service endpoints protected by contracts
3. **Clear Debugging**: Contract violations provide specific error messages
4. **Documentation**: Contracts serve as executable specifications
5. **Neural-Symbolic Safety**: Validates boundaries between reasoning systems
6. **Team Development**: Clear contracts enable confident parallel development

**Performance Considerations:**

- Contract checking adds ~5-10% runtime overhead in development
- Can be completely disabled in production builds
- Use environment variables to control contract checking:

```python
import os
from icontract import set_enabled

# Disable contracts in production
if os.getenv('ENVIRONMENT') == 'production':
    set_enabled(False)
```

**Migration Strategy:**

1. Start with wrapper methods (non-breaking changes)
2. Add contracts to critical operations first
3. Gradually expand coverage to all public methods
4. Eventually replace original methods when confidence is high

This approach maintains backward compatibility while adding robust contract protection where it matters most.

### Alternative Considered: `dpcontracts`

I also evaluated `dpcontracts` but found `icontract` superior because:
- Better documentation and maintenance
- More comprehensive feature set
- Superior error reporting and debugging support
- Better integration with modern Python typing features
