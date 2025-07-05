# Soft Logic Microservice Design Recommendations

## Executive Summary

This document outlines a design approach for transforming the experimental LTNtorch and Z3 SMT solver code into a robust, scalable microservice for building and managing soft logic vectors. The system will support both hard logic verification of core axioms and soft logic learning for extended vocabulary, with context-aware modeling capabilities.

## Current State Analysis

### LTN01.py Analysis
- **Strengths**: Demonstrates complete soft logic pipeline from training to visualization
- **Weaknesses**: Monolithic structure, hard-coded concepts, no persistence, limited extensibility
- **Key Components**: Embedding learning, axiom satisfaction, SMT verification, PCA visualization

### SMT01.py Analysis  
- **Strengths**: Clean demonstration of axiom consistency checking
- **Weaknesses**: Limited to simple vector operations, no integration with soft logic
- **Key Components**: Z3 constraint modeling, contradiction detection

## Recommended Architecture

### 1. Core System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Axiom Store   │    │  Logic Engine   │    │   Context       │
│   - Core axioms │    │  - SMT verify   │    │   Manager       │
│   - Soft axioms │    │  - LTN training │    │   - Namespaces  │
│   - Metadata    │    │  - Evaluation   │    │   - Inheritance │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
         │   Model Store   │    │   API Gateway   │    │   Concept       │
         │   - Embeddings  │    │   - REST APIs   │    │   Registry      │
         │   - Checkpoints │    │   - WebSocket   │    │   - WordNet     │
         │   - Metadata    │    │   - Streaming   │    │   - Synsets     │
         └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. Modular Breakdown Strategy

#### Phase 1: Core Abstractions (Weeks 1-2)
**Goal**: Establish foundational interfaces and data structures

**Components**:
- `Concept` class with synset disambiguation
- `Axiom` base class with type hierarchy
- `Context` management system
- Basic file I/O for axiom definitions

**Test Strategy**: Unit tests for each abstraction, JSON schema validation

#### Phase 2: SMT Integration (Weeks 3-4)
**Goal**: Implement hard logic verification system

**Components**:
- `SMTVerifier` class for Z3 integration
- `CoreAxiomValidator` for consistency checking
- `AxiomClassifier` (core vs. soft determination)

**Test Strategy**: SMT solver test cases, axiom contradiction detection

#### Phase 3: LTN Integration (Weeks 5-6)
**Goal**: Implement soft logic learning system

**Components**:
- `EmbeddingLearner` class wrapping LTNtorch
- `TrainingManager` for optimization loops
- `ModelPersistence` for saving/loading embeddings

**Test Strategy**: Training convergence tests, embedding quality metrics

#### Phase 4: API Layer (Weeks 7-8)
**Goal**: Expose functionality via REST/WebSocket APIs

**Components**:
- FastAPI endpoints for CRUD operations
- WebSocket for training progress streaming
- Model serving endpoints

**Test Strategy**: API integration tests, performance benchmarks

### 3. Standardized Axiom Format

#### JSON Schema for Axiom Definition
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
      "disambiguation": "monarch ruler"
    },
    {
      "name": "man", 
      "synset": "man.n.01",
      "disambiguation": "adult male human"
    },
    {
      "name": "woman",
      "synset": "woman.n.01", 
      "disambiguation": "adult female human"
    },
    {
      "name": "queen",
      "synset": "queen.n.01",
      "disambiguation": "female monarch"
    }
  ],
  "metadata": {
    "created_by": "system",
    "created_at": "2025-01-15T10:00:00Z",
    "confidence": 0.95,
    "source": "wordnet_analogy_bootstrap"
  }
}
```

#### YAML Alternative (Human-Friendly)
```yaml
axiom_id: analogy_king_queen
type: analogy
classification: core
context: default
description: "Classic gender analogy: king is to man as queen is to woman"

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
  man:
    synset: man.n.01
    disambiguation: "adult male human"
  woman:
    synset: woman.n.01
    disambiguation: "adult female human"
  queen:
    synset: queen.n.01
    disambiguation: "female monarch"

metadata:
  created_by: system
  created_at: 2025-01-15T10:00:00Z
  confidence: 0.95
  source: wordnet_analogy_bootstrap
```

### 4. Core System Bootstrap Process

#### Step 1: Core Axiom Verification
```python
class CoreAxiomBootstrap:
    def __init__(self):
        self.smt_verifier = SMTVerifier()
        self.core_axioms = []
    
    def load_core_axioms(self, axiom_file: str):
        """Load and validate core axioms from file"""
        axioms = self.parse_axiom_file(axiom_file)
        
        # Classify axioms as core vs soft
        for axiom in axioms:
            if axiom.classification == "core":
                self.verify_core_axiom(axiom)
        
        return axioms
    
    def verify_core_axiom(self, axiom: Axiom):
        """Use SMT to verify axiom consistency"""
        consistency_result = self.smt_verifier.check_consistency(
            axiom, self.core_axioms
        )
        
        if consistency_result.is_consistent:
            self.core_axioms.append(axiom)
        else:
            raise AxiomInconsistencyError(
                f"Axiom {axiom.id} conflicts with: {consistency_result.conflicts}"
            )
```

#### Step 2: Soft Logic Training
```python
class SoftLogicTrainer:
    def __init__(self, core_axioms: List[Axiom]):
        self.core_axioms = core_axioms
        self.embedding_learner = EmbeddingLearner()
        
    def train_context(self, context_name: str, soft_axioms: List[Axiom]):
        """Train embeddings for a specific context"""
        all_axioms = self.core_axioms + soft_axioms
        
        # Initialize embeddings from core axioms
        self.embedding_learner.initialize_from_core(self.core_axioms)
        
        # Train with soft axioms
        model = self.embedding_learner.train(all_axioms)
        
        # Save context-specific model
        self.save_context_model(context_name, model)
        
        return model
```

### 5. Storage and Exploration Abstractions

#### Model Storage Structure
```
models/
├── contexts/
│   ├── default/
│   │   ├── embeddings.pt
│   │   ├── metadata.json
│   │   └── training_history.json
│   ├── fantasy_literature/
│   │   ├── embeddings.pt
│   │   ├── metadata.json
│   │   └── training_history.json
│   └── scientific/
│       ├── embeddings.pt
│       ├── metadata.json
│       └── training_history.json
├── axioms/
│   ├── core/
│   │   ├── basic_analogies.yaml
│   │   ├── wordnet_synonyms.yaml
│   │   └── logical_operators.yaml
│   └── soft/
│       ├── domain_specific.yaml
│       └── experimental.yaml
└── concepts/
    ├── registry.json
    └── synset_mappings.json
```

#### Model Exploration Interface
```python
class ModelExplorer:
    def __init__(self, context_name: str):
        self.context = self.load_context(context_name)
        self.embeddings = self.load_embeddings(context_name)
    
    def find_similar_concepts(self, concept: str, top_k: int = 5) -> List[str]:
        """Find most similar concepts to given concept"""
        
    def test_analogy(self, a: str, b: str, c: str) -> Tuple[str, float]:
        """Test analogy: a is to b as c is to ?"""
        
    def visualize_concept_space(self, concepts: List[str] = None):
        """Generate PCA visualization of concept space"""
        
    def export_embeddings(self, format: str = "json") -> str:
        """Export embeddings in specified format"""
```

### 6. Context Management System

#### Context Inheritance
```python
class ContextManager:
    def __init__(self):
        self.contexts = {}
        self.inheritance_graph = {}
    
    def create_context(self, name: str, parent: str = "default"):
        """Create new context inheriting from parent"""
        if parent not in self.contexts:
            raise ContextNotFoundError(f"Parent context {parent} not found")
            
        # Inherit core axioms and embeddings from parent
        parent_model = self.contexts[parent]
        new_context = self.inherit_context(parent_model)
        self.contexts[name] = new_context
        
    def add_context_axioms(self, context_name: str, axioms: List[Axiom]):
        """Add axioms specific to context"""
        context = self.contexts[context_name]
        
        # Separate core vs soft axioms
        core_axioms = [a for a in axioms if a.classification == "core"]
        soft_axioms = [a for a in axioms if a.classification == "soft"]
        
        # Verify core axioms don't conflict
        for axiom in core_axioms:
            self.verify_against_context(axiom, context)
            
        # Add all axioms to context
        context.axioms.extend(axioms)
        
        # Retrain embeddings
        self.retrain_context(context_name)
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

#### Core Endpoints
```python
# Context Management
POST /contexts/{context_name}           # Create context
GET /contexts                          # List contexts
GET /contexts/{context_name}           # Get context details
DELETE /contexts/{context_name}        # Delete context

# Axiom Management  
POST /axioms                          # Add axiom
GET /axioms                           # List axioms
GET /axioms/{axiom_id}                # Get axiom details
PUT /axioms/{axiom_id}                # Update axiom
DELETE /axioms/{axiom_id}             # Delete axiom

# Model Operations
POST /contexts/{context_name}/train   # Train context model
GET /contexts/{context_name}/model    # Get model info
POST /contexts/{context_name}/query   # Query model (analogy, similarity)

# Exploration
GET /contexts/{context_name}/concepts # List concepts
GET /contexts/{context_name}/similar/{concept}  # Find similar concepts
POST /contexts/{context_name}/analogy # Test analogy
GET /contexts/{context_name}/visualization     # Generate visualization
```

#### WebSocket for Training Progress
```python
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

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Core abstractions (Concept, Axiom, Context)
- [ ] Axiom file format and parser
- [ ] Basic test suite
- [ ] WordNet integration

### Phase 2: Logic Systems (Weeks 3-6)
- [ ] SMT verifier implementation
- [ ] LTN training wrapper
- [ ] Model persistence
- [ ] Context inheritance

### Phase 3: Service Layer (Weeks 7-8)
- [ ] FastAPI endpoints
- [ ] WebSocket streaming
- [ ] Model serving
- [ ] Visualization endpoints

### Phase 4: Advanced Features (Weeks 9-12)
- [ ] Batch axiom processing
- [ ] Model comparison tools
- [ ] Performance optimization
- [ ] Documentation and examples

## Testing Strategy

### Unit Tests
- Individual component functionality
- Axiom parsing and validation
- SMT consistency checking
- LTN training convergence

### Integration Tests
- End-to-end axiom processing
- Context inheritance behavior
- API endpoint functionality
- Model persistence/loading

### Performance Tests
- Training time benchmarks
- Memory usage profiling
- API response times
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

This design provides a robust, scalable foundation for the soft logic microservice while maintaining the flexibility needed for experimental development and research applications.
