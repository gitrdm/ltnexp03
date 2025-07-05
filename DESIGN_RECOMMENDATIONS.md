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
└── __init__.py                  # ✅ Clean module exports

demonstrations/
├── demo_hybrid_system.py        # ✅ Basic hybrid capabilities
├── demo_enhanced_system.py      # ✅ Advanced reasoning features
└── demo_comprehensive_system.py # ✅ Medieval fantasy application

tests/
└── test_core/
    └── test_abstractions.py     # ✅ 18 tests passing

documentation/
├── IMPLEMENTATION_COMPLETE.md   # ✅ Full project summary
├── HYBRID_FRAMENET_CLUSTER_APPROACH.md # ✅ Hybrid approach design
└── FRAMENET_CLUSTERING_DESIGN.md # ✅ Original design analysis
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

#### 🔄 Phase 3: Service Layer & API (IN PROGRESS)
**Goal**: Expose functionality via REST/WebSocket APIs

**Current Status**: Architecture designed, ready for implementation

**Planned Components**:
- FastAPI service layer with comprehensive endpoints
- WebSocket streaming for training progress and real-time updates
- Model serving capabilities for embedding queries
- Visualization endpoints for concept space exploration
- Authentication and authorization for multi-tenant usage
- Integration with existing `EnhancedHybridRegistry` system

**Next Steps**:
1. Implement FastAPI app structure in `app/main.py`
2. Create endpoint handlers for concept, axiom, and context management
3. Add WebSocket support for real-time training progress
4. Implement model serving endpoints for analogical reasoning
5. Add visualization endpoints with concept space exploration
6. Integrate with existing hybrid registry system

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

### 8. API Design

#### Current State
A basic FastAPI application structure exists in `app/main.py` with health check endpoints. Ready for extension with the full API design below.

#### Planned Core Endpoints
```python
# Context Management
POST /contexts/{context_name}           # Create context
GET /contexts                          # List contexts
GET /contexts/{context_name}           # Get context details
DELETE /contexts/{context_name}        # Delete context

# Concept Management
POST /contexts/{context_name}/concepts  # Add concept to context
GET /contexts/{context_name}/concepts   # List concepts in context
GET /contexts/{context_name}/concepts/{concept_name}  # Get concept details
PUT /contexts/{context_name}/concepts/{concept_name}  # Update concept
DELETE /contexts/{context_name}/concepts/{concept_name}  # Delete concept

# Frame and Cluster Operations
GET /contexts/{context_name}/frames     # List semantic frames
POST /contexts/{context_name}/frames    # Create semantic frame
GET /contexts/{context_name}/clusters   # List concept clusters
POST /contexts/{context_name}/cluster   # Trigger clustering update

# Advanced Reasoning Endpoints
POST /contexts/{context_name}/analogies # Complete analogical reasoning
GET /contexts/{context_name}/semantic-fields  # Get discovered semantic fields
POST /contexts/{context_name}/cross-domain-analogies  # Find cross-domain patterns
GET /contexts/{context_name}/similar/{concept}  # Find similar concepts

# Model Operations
POST /contexts/{context_name}/train     # Train context model (future LTN integration)
GET /contexts/{context_name}/model      # Get model info and statistics
POST /contexts/{context_name}/query     # Query model (analogy, similarity)

# Visualization and Export
GET /contexts/{context_name}/visualization     # Generate concept space visualization
GET /contexts/{context_name}/export/{format}   # Export knowledge base (json/yaml)
GET /contexts/{context_name}/statistics        # Get context statistics
```

#### Enhanced WebSocket for Real-time Operations
```python
# Real-time semantic field discovery
@app.websocket("/ws/discovery/{context_name}")
async def semantic_discovery(websocket: WebSocket, context_name: str):
    await websocket.accept()
    
    registry = get_context_registry(context_name)
    
    # Stream discovery progress
    async for discovery_update in registry.discover_semantic_fields_streaming():
        await websocket.send_json({
            "type": "semantic_field_discovered",
            "field_name": discovery_update.field_name,
            "concepts": discovery_update.concepts,
            "coherence": discovery_update.coherence,
            "timestamp": discovery_update.timestamp
        })

# Future training progress (Phase 4)
@app.websocket("/ws/training/{context_name}")
async def training_progress(websocket: WebSocket, context_name: str):
    await websocket.accept()
    
    trainer = SoftLogicTrainer(context_name)
    
    async for progress in trainer.train_with_progress():
        await websocket.send_json({
            "epoch": progress.epoch,
            "loss": progress.loss,
            "satisfiability": progress.satisfiability,
            "stage": progress.stage
        })
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
- [ ] FastAPI application structure (`app/main.py`)
- [ ] REST API endpoints for concept/axiom/context management
- [ ] WebSocket streaming for real-time operations
- [ ] Model serving endpoints for analogical reasoning
- [ ] Visualization endpoints for concept space exploration
- [ ] Integration testing with existing hybrid system

### 🔄 Phase 4: Neural-Symbolic Integration (FUTURE - Weeks 4-8)
- [ ] LTNtorch wrapper integration
- [ ] SMT verification layer (Z3 integration)
- [ ] Hybrid training pipelines
- [ ] Enhanced model persistence and versioning
- [ ] Performance optimization and scaling
- [ ] Production deployment configuration

## Testing Strategy

### ✅ Current Unit Tests (18 tests passing)
- **Core Abstractions**: Complete coverage of `Concept`, `Axiom`, `Context`, and `FormulaNode` classes
- **Concept Registry**: WordNet integration, homonym handling, pattern search
- **Axiom Parser**: YAML and JSON parsing validation
- **All tests passing**: Full test suite runs successfully in under 1 second

### 🔄 Planned Integration Tests (Phase 3)
- End-to-end hybrid registry operations
- Frame and cluster integration behavior
- API endpoint functionality
- WebSocket real-time streaming
- Cross-domain analogical reasoning workflows

### 🔄 Planned Performance Tests (Phase 4)
- Semantic field discovery benchmarks
- Analogical reasoning performance
- Memory usage profiling for large concept spaces
- API response times under load
- Concurrent request handling

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

## Next Steps: Phase 3 Implementation Plan

### Immediate Priorities

#### 1. API Endpoint Implementation (Week 1)
**Goal**: Extend `app/main.py` with full REST API

**Tasks**:
- Create request/response models using Pydantic
- Implement context management endpoints
- Add concept CRUD operations integrated with `EnhancedHybridRegistry`
- Create analogical reasoning endpoints
- Add semantic field discovery endpoints

**Example Implementation Structure**:
```python
# app/api/models.py - Pydantic models
class ConceptRequest(BaseModel):
    name: str
    synset_id: Optional[str] = None
    disambiguation: Optional[str] = None
    frame_roles: Dict[str, str] = {}

class AnalogyRequest(BaseModel):
    context: str
    partial_analogy: Dict[str, str]
    max_completions: int = 5

# app/api/endpoints.py - API route handlers
@app.post("/contexts/{context_name}/analogies")
async def complete_analogy(context_name: str, request: AnalogyRequest):
    registry = get_or_create_registry(context_name)
    completions = registry.find_analogical_completions(
        request.partial_analogy, 
        max_completions=request.max_completions
    )
    return {"completions": completions}
```

#### 2. WebSocket Integration (Week 2)
**Goal**: Real-time semantic discovery and future training support

**Tasks**:
- Implement WebSocket endpoint for semantic field discovery streaming
- Add WebSocket endpoint for clustering updates
- Create WebSocket for future training progress (Phase 4 preparation)
- Add connection management and error handling

#### 3. Integration Testing (Week 3)
**Goal**: Ensure API works seamlessly with existing hybrid system

**Tasks**:
- Fix existing API test issues
- Create comprehensive API integration tests
- Add performance benchmarks for API endpoints
- Test WebSocket functionality
- Validate end-to-end workflows

### Success Criteria for Phase 3

- [ ] Full REST API operational with all planned endpoints
- [ ] WebSocket streaming working for real-time operations
- [ ] Integration tests passing for all API functionality
- [ ] Performance benchmarks established
- [ ] API documentation generated and complete
- [ ] Ready for Phase 4 neural-symbolic integration

## Future Directions: Phase 4 Neural-Symbolic Integration

### LTNtorch Integration Strategy

The existing `EnhancedHybridRegistry` provides an ideal foundation for LTNtorch integration:

1. **Concept Embeddings**: Current vector embedding system can be enhanced with neural-learned representations
2. **Frame-Based Logic**: Semantic frames can be converted to LTN predicates and relations
3. **Analogical Reasoning**: Current analogical completion can be enhanced with neural similarity learning
4. **Semantic Fields**: Discovered semantic fields can guide neural training objectives

### SMT Verification Integration

The existing axiom system is ready for SMT integration:

1. **Hard Constraints**: Current core axioms can be converted to Z3 constraints
2. **Consistency Checking**: Integration with existing axiom validation pipeline
3. **Hybrid Verification**: Combine symbolic verification with neural learning

This design provides a clear path forward while leveraging the robust foundation already implemented.

## Code Quality Enhancement Analysis

### Current Type Safety and Contract Status

The codebase shows excellent documentation and some type hints, but would benefit significantly from enhanced type safety and Design by Contract (DbC) at this mature stage. Here's an analysis of where these improvements would provide the most value:

### 1. Enhanced Type Hints - **HIGH IMPACT**

#### Current State
- Basic type hints present in function signatures
- `typing` module used but not comprehensively
- Some complex return types not fully specified

#### Recommended Enhancements

**Generic Type Parameters for Registry Classes**:
```python
from typing import TypeVar, Generic, Protocol, runtime_checkable

T = TypeVar('T', bound='Concept')
K = TypeVar('K', bound=str)

class ConceptRegistry(Generic[T, K]):
    """Type-safe concept registry with generic concept types."""
    
    def register_concept(self, concept: T) -> K:
        """Register concept and return its unique identifier."""
        ...
    
    def get_concept(self, concept_id: K) -> Optional[T]:
        """Retrieve concept by ID with type safety."""
        ...
```

**Protocol-Based Interfaces for Embedding Providers**:
```python
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers ensuring consistent interface."""
    
    def generate_embedding(self, concept: str, context: str = "default") -> np.ndarray:
        """Generate embedding vector for concept."""
        ...
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute similarity between embeddings."""
        ...
    
    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension."""
        ...
```

**Enhanced Return Type Specifications**:
```python
from typing import Union, Literal, TypedDict

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

### 2. Design by Contract (DbC) - **VERY HIGH IMPACT**

DbC would be extremely valuable for the complex reasoning operations and registry management.

#### Critical Areas for DbC

**Semantic Field Discovery Contracts**:
```python
from contracts import contract, new_contract

# Define custom contracts
new_contract('coherence_score', lambda x: 0.0 <= x <= 1.0)
new_contract('positive_int', lambda x: isinstance(x, int) and x > 0)

class EnhancedHybridRegistry:
    
    @contract(min_coherence='coherence_score', 
              returns='list[SemanticField]')
    def discover_semantic_fields(self, min_coherence: float = 0.7) -> List[SemanticField]:
        """
        Discover semantic fields from concept clusters.
        
        :param min_coherence: Minimum coherence threshold (0.0-1.0)
        :returns: List of discovered semantic fields
        
        :pre: self.cluster_registry.is_trained
        :post: all(field.coherence >= min_coherence for field in __return__)
        :post: len(__return__) >= 0
        """
        assert self.cluster_registry.is_trained, "Clustering must be trained before field discovery"
        
        fields = self._discover_fields_internal(min_coherence)
        
        # Post-condition validation
        for field in fields:
            assert field.coherence >= min_coherence, f"Field {field.name} has coherence {field.coherence} < {min_coherence}"
        
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
4. **Create custom contract types** for domain-specific constraints (coherence scores, embedding dimensions)

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
