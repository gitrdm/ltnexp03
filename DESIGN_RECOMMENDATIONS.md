# Soft Logic Microservice Design Recommendations

## Executive Summary

This document outlines the design approach for our robust, scalable microservice for building and managing soft logic vectors. The system supports both hard logic verification of core axioms and advanced soft logic learning with hybrid FrameNet-clustering semantic reasoning, context-aware modeling capabilities, and sophisticated analogical reasoning.

**Current Status**: Phase 1-2 Complete, Phase 3A-3B Complete. Core abstractions, hybrid semantic reasoning, and advanced vector embeddings fully implemented and tested. Enhanced semantic field discovery and cross-domain analogical reasoning operational. icontract Design by Contract implementation complete with comprehensive validation. Ready for Phase 3C service layer implementation and Phase 4 neural-symbolic integration.

## Current State Analysis (Updated 2025-07-05)

### Completed Implementation Review
- **✅ Core Abstractions**: Complete concept, axiom, context, and formula abstractions with WordNet integration
- **✅ Hybrid Semantic System**: FrameNet-style frames integrated with clustering-based concept organization  
- **✅ Advanced Reasoning**: Cross-domain analogical reasoning, semantic field discovery, analogical completion
- **✅ Vector Embeddings**: Sophisticated embedding management with multiple providers and similarity metrics
- **✅ Enhanced Registry**: `EnhancedHybridRegistry` with multi-level analogical reasoning and dynamic knowledge discovery
- **✅ Design by Contract**: Comprehensive icontract implementation with preconditions, postconditions, and class invariants
- **✅ Contract Validation**: Domain-specific validators and defensive programming throughout core modules
- **✅ Performance Optimization**: Efficient bulk operations with deferred clustering and contract validation
- **✅ Comprehensive Testing**: 18 unit tests passing across all core modules plus contract demonstration tests
- **✅ Rich Documentation**: Literate programming style with comprehensive design documents and implementation summaries
- **✅ Working Demonstrations**: Three comprehensive demo systems showcasing medieval fantasy applications (all optimized and working)

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
├── concept_registry.py          # ✅ WordNet-integrated concept management (with icontract)
├── contracts.py                 # ✅ Domain-specific contract validators (icontract)
├── icontract_demo.py            # ✅ Contract validation demonstration
├── parsers.py                   # ✅ YAML/JSON axiom parsing
├── frame_cluster_abstractions.py # ✅ FrameNet & clustering data structures
├── frame_cluster_registry.py    # ✅ Frame & cluster management systems
├── hybrid_registry.py           # ✅ Unified hybrid semantic registry (with icontract)
├── enhanced_semantic_reasoning.py # ✅ Advanced reasoning & field discovery (with icontract)
├── vector_embeddings.py         # ✅ Sophisticated embedding management (with icontract)
├── protocols.py                 # ✅ Protocol interface definitions (Phase 3A)
├── api_models.py                  # ✅ TypedDict API request/response models (Phase 3A)
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
├── ICONTRACT_IMPLEMENTATION_SUMMARY.md # ✅ Complete icontract integration summary
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

### ✅ Phase 3: Service Layer & API (IN PROGRESS - 3A & 3B COMPLETE)
**Goal**: Expose functionality via REST/WebSocket APIs with enterprise-grade type safety and contracts

**Current Status**: Phase 3A & 3B Complete, Phase 3C ready for implementation

**✅ Phase 3A: Type Safety Foundation (COMPLETED)**:
- ✅ Enhanced type hints with Generic type parameters for registry classes
- ✅ Protocol-based interfaces for embedding providers and reasoning engines (6 protocols implemented)
- ✅ TypedDict specifications for complex return types and API responses (15+ models)
- ✅ mypy strict mode integration and comprehensive type checking (zero type errors)
- ✅ Protocol implementation in EnhancedHybridRegistry with adapter methods
- ✅ Comprehensive test suite with 5/5 tests passing
- ✅ Runtime protocol compliance validation

**✅ Phase 3B: Design by Contract Implementation (COMPLETED)**:
- ✅ Precondition/postcondition contracts for all registry operations using icontract
- ✅ Class invariants for registry consistency validation with defensive checks
- ✅ Input validation contracts for reasoning operations
- ✅ Custom contract types for domain-specific constraints (coherence scores, embedding dimensions)
- ✅ Domain-specific contract validators in `app/core/contracts.py`
- ✅ Contract integration throughout `enhanced_semantic_reasoning.py`, `concept_registry.py`, and `vector_embeddings.py`
- ✅ Performance-optimized contract validation with minimal overhead
- ✅ Comprehensive contract demonstration and testing

**🔄 Phase 3C: Service Layer Implementation (NEXT - Week 3)**:
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
│   ├── core/                     # Core implementation modules (✅ COMPLETE WITH CONTRACTS)
│   │   ├── abstractions.py          # Basic concept, axiom, context classes
│   │   ├── concept_registry.py      # WordNet-integrated concept management (with icontract)
│   │   ├── contracts.py              # Domain-specific contract validators (icontract)
│   │   ├── icontract_demo.py         # Contract validation demonstration
│   │   ├── parsers.py               # YAML/JSON axiom parsing
│   │   ├── frame_cluster_abstractions.py  # FrameNet & clustering structures
│   │   ├── frame_cluster_registry.py      # Frame & cluster registries
│   │   ├── hybrid_registry.py             # Unified hybrid system (with icontract)
│   │   ├── enhanced_semantic_reasoning.py # Advanced reasoning capabilities (with icontract)
│   │   ├── vector_embeddings.py           # Vector embedding management (with icontract)
│   │   ├── protocols.py                   # Protocol interface definitions (Phase 3A)
│   │   ├── api_models.py                  # TypedDict API request/response models (Phase 3A)
│   │   ├── protocol_mixins.py             # Protocol implementation mixins (Phase 3A)
│   │   └── __init__.py                     # Module exports
│   └── main.py                   # FastAPI application (basic structure ready for Phase 3C)
├── embeddings_cache/             # Vector embedding persistence
│   └── *.npy, *.json            # Cached embeddings and metadata
├── examples/                     # Example axiom files (✅ IMPLEMENTED)
│   ├── basic_analogy.yaml       # Core analogy axioms
│   └── core_axioms.json         # JSON format examples
├── tests/                        # Comprehensive test suite (✅ ALL TESTS PASSING)
│   ├── test_core/               # Core module tests
│   │   └── test_abstractions.py    # Complete coverage (18 tests, 0.64s)
│   ├── test_phase_3a.py         # Phase 3A type safety tests (5/5 passing)
│   └── test_main.py             # API tests (ready for Phase 3C)
├── demo_*.py                    # Working demonstration systems (✅ ALL OPTIMIZED)
│   ├── demo_abstractions.py       # Core abstractions showcase
│   ├── demo_hybrid_system.py       # Hybrid registry capabilities
│   ├── demo_enhanced_system.py     # Advanced reasoning features (fixed)
│   └── demo_comprehensive_system.py # Medieval fantasy application (30s optimized)
├── documentation/               # Comprehensive documentation (✅ COMPLETE)
│   ├── IMPLEMENTATION_COMPLETE.md
│   ├── ICONTRACT_IMPLEMENTATION_SUMMARY.md # ✅ Contract implementation summary
│   ├── HYBRID_FRAMENET_CLUSTER_APPROACH.md
│   ├── FRAMENET_CLUSTERING_DESIGN.md
│   ├── PHASE_3A_COMPLETE.md
│   └── DESIGN_RECOMMENDATIONS.md (this file - updated)
├── Makefile                     # ✅ Complete test automation (all targets working)
├── pyproject.toml               # Poetry dependency management (with icontract)
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

#### Current Status
A basic FastAPI application structure exists in `app/main.py` with health check endpoints. Enhanced with comprehensive type safety and Design by Contract validation ready for implementation.

#### Type-Safe API Architecture

**Protocol-Based Service Interfaces**:
```python
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
@contract(context_name='str', returns='list[AnalogicalCompletionResult],len(x)<=max_completions')
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

## 💾 Persistence Layer Strategy

### Current Gap Analysis
The current implementation has a **significant gap** in persistence capabilities:

**✅ Limited Existing Persistence:**
- Vector embeddings cache in `embeddings_cache/` directory
- Basic YAML/JSON axiom file parsing
- No comprehensive state management

**❌ Missing Critical Persistence:**
- **Semantic Frames & Instances**: No storage for FrameNet structures
- **Concept Clusters**: No persistence for trained clustering models  
- **Cross-Domain Analogies**: No storage for discovered patterns
- **Semantic Fields**: No persistence for coherent semantic regions
- **Context Hierarchies**: No inheritance relationship storage
- **Registry State**: No full system serialization/deserialization

### Recommended Persistence Architecture

#### Multi-Format Storage Structure
```
storage/
├── contexts/                    # Context-specific storage
│   ├── default/
│   │   ├── concepts.json       # ✅ Concept definitions with metadata
│   │   ├── frames.json         # ✅ Semantic frame definitions  
│   │   ├── instances.json      # ✅ Frame instance bindings
│   │   ├── clusters.json       # ✅ Cluster assignments and centroids
│   │   ├── fields.json         # ✅ Discovered semantic fields
│   │   ├── analogies.json      # ✅ Cross-domain analogical mappings
│   │   └── embeddings/         # ✅ Vector embeddings cache
│   ├── medieval/               # Domain-specific contexts
│   └── scientific/
├── models/                     # Trained model persistence
│   ├── clustering_models/      # Scikit-learn model files
│   └── embedding_models/       # Pre-trained embedding models
├── exports/                    # Full system exports
│   ├── knowledge_base_v1.0.json
│   └── compressed/
└── audit/                     # Contract validation audit trails
    └── contract_violations.json
```

#### Contract-Validated Persistence Manager
```python
from icontract import require, ensure, invariant
from .contracts import SoftLogicContracts

@invariant(lambda self: self.storage_path.exists())
class PersistenceManager:
    """Comprehensive persistence with contract validation."""
    
    @require(lambda context_name: SoftLogicContracts.valid_context(context_name))
    @ensure(lambda result: result is not None)
    def save_registry_state(self, registry: 'EnhancedHybridRegistry', 
                           context_name: str = "default") -> Dict[str, Any]:
        """Save complete registry state with versioning."""
        
    @require(lambda context_name: SoftLogicContracts.valid_context(context_name))  
    def load_registry_state(self, context_name: str = "default") -> Optional[Dict[str, Any]]:
        """Load complete registry state."""
        
    def export_knowledge_base(self, format: str = "json", 
                            compressed: bool = False) -> Path:
        """Export complete knowledge base."""
```

### Integration with Phase 3C

**FastAPI Persistence Endpoints:**
```python
@app.post("/contexts/{context_name}/export")
@require(lambda context_name: SoftLogicContracts.valid_context(context_name))
async def export_context(context_name: str, format: str = "json"):
    """Export complete context state with contract validation."""
    
@app.post("/contexts/{context_name}/import") 
async def import_context(context_name: str, file: UploadFile):
    """Import context from uploaded file."""
    
@app.get("/contexts/{context_name}/backup")
async def create_backup(context_name: str):
    """Create versioned backup of context."""
```

### Implementation Priority

**Phase 3C++ (Extended)**: Add persistence layer immediately after Phase 3C completion:

1. **Basic Persistence Manager** - Core save/load functionality
2. **Registry Integration** - Auto-save capabilities in registries  
3. **API Endpoints** - Export/import endpoints in FastAPI
4. **Versioning System** - Version control for knowledge bases
5. **Backup & Recovery** - Automated backup strategies

This addresses the critical gap and enables production-ready deployments with full state management.

For complete specifications, see: `PERSISTENCE_LAYER_STRATEGY.md`

---
