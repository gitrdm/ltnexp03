# Soft Logic Microservice: Complete Tutorial Guide

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Getting Started with Concepts](#getting-started-with-concepts)
4. [Basic Analogical Reasoning](#basic-analogical-reasoning)
5. [Advanced Semantic Reasoning](#advanced-semantic-reasoning)
6. [SMT Verification with Z3](#smt-verification-with-z3)
7. [Neural-Symbolic Training with LTNtorch](#neural-symbolic-training-with-ltntorch)
8. [Persistence Layer and Workflow Management](#persistence-layer-and-workflow-management)
9. [API Reference Guide](#api-reference-guide)
10. [WebSocket Streaming](#websocket-streaming)
11. [Production Deployment](#production-deployment)
12. [Troubleshooting](#troubleshooting)

---

## System Overview

The Soft Logic Microservice is a production-ready neural-symbolic AI platform that combines:

- **Symbolic Reasoning**: FrameNet-style semantic frames with clustering-based concept organization
- **Neural Learning**: LTNtorch integration for end-to-end differentiable training
- **Hard Logic Verification**: Z3 SMT solver for axiom consistency checking
- **Persistence**: Enterprise-grade storage with batch operations and workflow management
- **API Layer**: Complete FastAPI service with WebSocket streaming

### Key Features

✅ **Hybrid Semantic System**: Combines frame-based and cluster-based reasoning  
✅ **Neural-Symbolic Training**: LTNtorch integration with real-time progress monitoring  
✅ **SMT Verification**: Z3 solver for hard logic constraints  
✅ **Enterprise Persistence**: Multi-format storage with batch workflows  
✅ **REST API**: Complete FastAPI service with 30+ endpoints  
✅ **WebSocket Streaming**: Real-time training and workflow monitoring  
✅ **Type Safety**: Full mypy compliance with Design by Contract validation  

---

## Installation and Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Conda (recommended) or virtualenv

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/ltnexp03.git
cd ltnexp03
```

### Step 2: Create Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate ltnexp03
```

#### Option B: Using Python venv

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Linux/macOS)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install project and dependencies with Poetry
poetry install

# Or install directly with pip
pip install -e .
```

### Step 4: Verify Installation

```bash
# Run basic tests to verify installation
poetry run pytest tests/ -v

# Check that all components are working
python -c "from app.core import EnhancedHybridRegistry; print('✅ Installation successful!')"
```

### Step 5: Start the Service

```bash
# Start the FastAPI service
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8321 --reload

# Or use the convenience script
poetry run start
```

The service will be available at:
- **API Documentation**: http://localhost:8321/api/docs
- **Health Check**: http://localhost:8321/health
- **Service Info**: http://localhost:8321/service-info

---

## Getting Started with Concepts

Concepts are the fundamental building blocks of the soft logic system. Let's start by creating and managing concepts.

### Basic Concept Creation

#### Using Python API

```python
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry

# Initialize the registry
registry = EnhancedHybridRegistry(
    download_wordnet=False,  # Set to True for WordNet integration
    n_clusters=8,
    enable_cross_domain=True,
    embedding_provider="semantic"  # or "random" for testing
)

# Create a basic concept
concept = registry.create_frame_aware_concept_with_advanced_embedding(
    name="king",
    context="royalty",
    synset_id="king.n.01",  # WordNet synset ID
    disambiguation="monarch ruler",
    use_semantic_embedding=True
)

print(f"Created concept: {concept.name}")
print(f"Unique ID: {concept.unique_id}")
print(f"Context: {concept.context}")
```

#### Using REST API

```bash
# Create a concept via API
curl -X POST "http://localhost:8321/api/concepts" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "king",
       "context": "royalty",
       "synset_id": "king.n.01",
       "disambiguation": "monarch ruler",
       "metadata": {"domain": "political_hierarchy"},
       "auto_disambiguate": true
     }'
```

**Response:**
```json
{
  "concept_id": "royalty:king:king.n.01",
  "name": "king",
  "synset_id": "king.n.01",
  "disambiguation": "monarch ruler",
  "context": "royalty",
  "created_at": "2025-07-05T10:00:00Z",
  "metadata": {"domain": "political_hierarchy"},
  "embedding_size": 300
}
```

### Searching for Concepts

#### Python API

```python
# Search for concepts
search_results = []
for concept_id, concept in registry.concepts.items():
    if "royal" in concept.name.lower() or "royal" in concept.context.lower():
        search_results.append(concept)

for concept in search_results:
    print(f"Found: {concept.name} in {concept.context}")
```

#### REST API

```bash
# Search for concepts
curl -X POST "http://localhost:8321/api/concepts/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "royal",
       "context": "royalty",
       "similarity_threshold": 0.7,
       "max_results": 10,
       "include_metadata": true
     }'
```

### Computing Concept Similarity

#### Python API

```python
# Create two related concepts
queen = registry.create_frame_aware_concept_with_advanced_embedding(
    name="queen",
    context="royalty",
    synset_id="queen.n.01",
    disambiguation="female monarch"
)

# Compute similarity (if implemented in registry)
# similarity = registry.compute_similarity(concept, queen)
```

#### REST API

```bash
# Compute similarity between concepts
curl -X POST "http://localhost:8321/api/concepts/similarity" \
     -H "Content-Type: application/json" \
     -d '{
       "concept1": "king",
       "concept2": "queen",
       "context1": "royalty",
       "context2": "royalty",
       "similarity_method": "hybrid"
     }'
```

---

## Basic Analogical Reasoning

The system excels at analogical reasoning - finding patterns like "king is to man as queen is to woman".

### Creating Analogies

#### Python API

```python
# Create concepts for analogy
king = registry.create_frame_aware_concept_with_advanced_embedding(
    name="king", context="royalty", synset_id="king.n.01"
)
man = registry.create_frame_aware_concept_with_advanced_embedding(
    name="man", context="default", synset_id="man.n.01"
)
woman = registry.create_frame_aware_concept_with_advanced_embedding(
    name="woman", context="default", synset_id="woman.n.01"
)

# Complete analogy: king is to man as ? is to woman
partial_analogy = {
    "king": "man",
    "?": "woman"
}

completions = registry.complete_analogy(partial_analogy, max_completions=5)
for completion in completions:
    print(f"Completion: {completion}")
```

#### REST API

```bash
# Complete an analogy
curl -X POST "http://localhost:8321/api/analogies/complete" \
     -H "Content-Type: application/json" \
     -d '{
       "source_a": "king",
       "source_b": "man",
       "target_a": "queen",
       "context": "royalty",
       "max_completions": 5,
       "min_confidence": 0.5
     }'
```

**Response:**
```json
{
  "completions": [
    {
      "target_b": "woman",
      "confidence": 0.92,
      "reasoning": "Gender-based analogy in royal hierarchy"
    },
    {
      "target_b": "female",
      "confidence": 0.78,
      "reasoning": "Biological gender mapping"
    }
  ],
  "reasoning_trace": [
    "Analogy pattern: king:man :: queen:?",
    "Found 2 completions above confidence 0.5"
  ],
  "metadata": {
    "context": "royalty",
    "method": "enhanced_hybrid_reasoning"
  }
}
```

### Batch Analogy Processing

For large-scale analogy processing, use the batch operations:

#### REST API

```bash
# Create a batch of analogies
curl -X POST "http://localhost:8321/api/batch/analogies" \
     -H "Content-Type: application/json" \
     -d '{
       "workflow_id": "royal_analogies_001",
       "analogies": [
         {"king": "man", "queen": "woman"},
         {"prince": "boy", "princess": "girl"},
         {"lord": "master", "lady": "mistress"}
       ]
     }'
```

**Response:**
```json
{
  "workflow_id": "royal_analogies_001",
  "workflow_type": "analogy_batch",
  "status": "processing",
  "created_at": "2025-07-05T10:00:00Z",
  "updated_at": "2025-07-05T10:00:00Z",
  "items_total": 3,
  "items_processed": 0,
  "error_count": 0,
  "metadata": {},
  "error_log": []
}
```

---

## Advanced Semantic Reasoning

The system provides advanced reasoning capabilities including semantic field discovery and cross-domain analogies.

### Semantic Field Discovery

Semantic fields are coherent groups of related concepts. The system can automatically discover these fields:

#### Python API

```python
# Discover semantic fields
fields = registry.discover_semantic_fields(min_coherence=0.6)

for field in fields:
    print(f"Field: {field.get('name', 'Unnamed')}")
    print(f"Coherence: {field.get('coherence', 0.0):.2f}")
    print(f"Concepts: {field.get('core_concepts', [])}")
    print("---")
```

#### REST API

```bash
# Discover semantic fields
curl -X POST "http://localhost:8321/api/semantic-fields/discover" \
     -H "Content-Type: application/json" \
     -d '{
       "domain": "royalty",
       "min_coherence": 0.6,
       "max_fields": 10,
       "clustering_method": "kmeans"
     }'
```

**Response:**
```json
{
  "semantic_fields": [
    {
      "field_id": "field_0",
      "name": "Royal Hierarchy",
      "concepts": ["king", "queen", "prince", "princess"],
      "coherence": 0.87,
      "domain": "royalty"
    },
    {
      "field_id": "field_1",
      "name": "Gender Relations",
      "concepts": ["man", "woman", "boy", "girl"],
      "coherence": 0.82,
      "domain": "royalty"
    }
  ],
  "discovery_metadata": {
    "method": "kmeans",
    "domain": "royalty",
    "total_concepts_analyzed": 12
  },
  "coherence_scores": {
    "field_0": 0.87,
    "field_1": 0.82
  }
}
```

### Cross-Domain Analogies

The system can discover analogical patterns that span different domains:

#### Python API

```python
# Discover cross-domain analogies
analogies = registry.discover_cross_domain_analogies(min_quality=0.3)

for analogy in analogies:
    print(f"Cross-domain analogy: {analogy}")
    if hasattr(analogy, 'compute_overall_quality'):
        print(f"Quality: {analogy.compute_overall_quality():.2f}")
```

#### REST API

```bash
# Discover cross-domain analogies
curl -X POST "http://localhost:8321/api/analogies/cross-domain" \
     -H "Content-Type: application/json" \
     -d '{
       "source_domain": "royalty",
       "target_domain": "business",
       "min_quality": 0.3,
       "max_analogies": 5
     }'
```

---

## SMT Verification with Z3

The system integrates with Z3 SMT solver for hard logic verification, ensuring axiom consistency.

### Basic SMT Verification

#### Python API

```python
from app.core.neural_symbolic_integration import Z3SMTVerifier
from app.core.abstractions import Axiom, FormulaNode

# Initialize SMT verifier
verifier = Z3SMTVerifier()

# Create axioms for verification
axioms = [
    Axiom(
        axiom_id="gender_consistency",
        axiom_type="consistency",
        classification="core",
        context="default",
        description="Gender consistency rule",
        formula=FormulaNode("and", [
            FormulaNode("implies", [
                FormulaNode("male", ["x"]),
                FormulaNode("not", [FormulaNode("female", ["x"])])
            ])
        ])
    )
]

# Verify axiom consistency
result = verifier.verify_axiom_consistency(axioms)
print(f"Verification result: {result.is_consistent}")
print(f"Verification time: {result.verification_time_ms}ms")
```

#### REST API

```bash
# Verify axioms using neural-symbolic service
curl -X POST "http://localhost:8321/api/neural-symbolic/verify" \
     -H "Content-Type: application/json" \
     -d '{
       "axioms": [
         {
           "axiom_id": "gender_consistency",
           "formula": "forall x: male(x) -> not female(x)",
           "description": "Gender exclusivity axiom"
         }
       ],
       "verification_timeout": 10000
     }'
```

### Advanced SMT Usage

#### Consistency Checking with Multiple Axioms

```python
# Create multiple related axioms
axioms = [
    # Basic gender axioms
    Axiom(
        axiom_id="male_female_exclusive",
        formula=FormulaNode("forall", ["x"], 
            FormulaNode("implies", [
                FormulaNode("male", ["x"]),
                FormulaNode("not", [FormulaNode("female", ["x"])])
            ])
        )
    ),
    # Royal hierarchy axioms
    Axiom(
        axiom_id="king_is_male",
        formula=FormulaNode("forall", ["x"],
            FormulaNode("implies", [
                FormulaNode("king", ["x"]),
                FormulaNode("male", ["x"])
            ])
        )
    ),
    Axiom(
        axiom_id="queen_is_female",
        formula=FormulaNode("forall", ["x"],
            FormulaNode("implies", [
                FormulaNode("queen", ["x"]),
                FormulaNode("female", ["x"])
            ])
        )
    )
]

# Verify consistency of the entire axiom set
result = verifier.verify_axiom_consistency(axioms)
if result.is_consistent:
    print("✅ All axioms are consistent")
else:
    print("❌ Axiom set contains inconsistencies")
    print(f"Counterexample: {result.counterexample}")
```

---

## Neural-Symbolic Training with LTNtorch

The system integrates LTNtorch for end-to-end neural-symbolic learning, combining symbolic constraints with neural optimization.

### Basic Neural Training

#### Python API

```python
from app.core.neural_symbolic_integration import (
    NeuralSymbolicTrainingManager,
    TrainingConfiguration,
    LTNTrainingProvider
)

# Initialize training components
training_manager = NeuralSymbolicTrainingManager()

# Create training configuration
config = TrainingConfiguration(
    epochs=10,
    learning_rate=0.01,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Prepare concepts for training
concepts = [
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="king", context="royalty", synset_id="king.n.01"
    ),
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="queen", context="royalty", synset_id="queen.n.01"
    ),
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="man", context="default", synset_id="man.n.01"
    ),
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="woman", context="default", synset_id="woman.n.01"
    )
]

# Start training
async def run_training():
    async for progress in training_manager.train_with_progress_streaming(
        concepts=concepts,
        config=config
    ):
        print(f"Epoch {progress.epoch}: loss={progress.loss:.4f}, "
              f"satisfiability={progress.satisfiability:.4f}")

# Run the training
import asyncio
asyncio.run(run_training())
```

#### REST API

```bash
# Start neural-symbolic training
curl -X POST "http://localhost:8321/api/neural-symbolic/train" \
     -H "Content-Type: application/json" \
     -d '{
       "training_config": {
         "epochs": 10,
         "learning_rate": 0.01,
         "batch_size": 32,
         "device": "auto"
       },
       "concepts": [
         "royalty:king:king.n.01",
         "royalty:queen:queen.n.01",
         "default:man:man.n.01",
         "default:woman:woman.n.01"
       ],
       "axioms": [
         "gender_consistency",
         "royal_hierarchy"
       ]
     }'
```

**Response:**
```json
{
  "training_session_id": "train_20250705_100000",
  "status": "started",
  "config": {
    "epochs": 10,
    "learning_rate": 0.01,
    "batch_size": 32,
    "device": "cuda"
  },
  "progress": {
    "current_epoch": 0,
    "total_epochs": 10,
    "current_loss": null,
    "satisfiability": null
  },
  "created_at": "2025-07-05T10:00:00Z"
}
```

### Real-Time Training Monitoring

#### WebSocket Monitoring

```javascript
// Connect to training progress WebSocket
const ws = new WebSocket('ws://localhost:8321/api/ws/neural-symbolic/train_20250705_100000');

ws.onmessage = function(event) {
    const progress = JSON.parse(event.data);
    console.log(`Epoch ${progress.epoch}: loss=${progress.loss}, satisfiability=${progress.satisfiability}`);
    
    // Update UI with training progress
    updateProgressBar(progress.epoch / progress.total_epochs);
    updateMetrics(progress.loss, progress.satisfiability);
};

ws.onopen = function() {
    console.log('Connected to training progress stream');
};

ws.onclose = function() {
    console.log('Training monitoring disconnected');
};
```

#### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def monitor_training(session_id):
    uri = f"ws://localhost:8321/api/ws/neural-symbolic/{session_id}"
    
    async with websockets.connect(uri) as websocket:
        print(f"Connected to training session: {session_id}")
        
        async for message in websocket:
            progress = json.loads(message)
            print(f"Epoch {progress['epoch']}: "
                  f"loss={progress['loss']:.4f}, "
                  f"satisfiability={progress['satisfiability']:.4f}")
            
            # Stop monitoring when training completes
            if progress.get('status') == 'completed':
                break

# Monitor training progress
asyncio.run(monitor_training("train_20250705_100000"))
```

### Advanced Training Features

#### Training with Custom Axioms

```python
# Define custom axioms for training
axioms = [
    {
        "axiom_id": "gender_analogy",
        "formula": "forall x,y: (king(x) ∧ male(x) ∧ queen(y) ∧ female(y)) → analogous(x,y)",
        "weight": 1.0
    },
    {
        "axiom_id": "hierarchy_consistency",
        "formula": "forall x: (king(x) ∨ queen(x)) → royal(x)",
        "weight": 0.8
    }
]

# Training with axiom constraints
training_result = await training_manager.train_with_axioms(
    concepts=concepts,
    axioms=axioms,
    config=config
)

print(f"Final loss: {training_result.final_loss}")
print(f"Axiom satisfaction: {training_result.final_satisfiability}")
```

#### Multi-Domain Training

```python
# Create concepts from multiple domains
royal_concepts = [
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="king", context="royalty"
    ),
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="queen", context="royalty"
    )
]

business_concepts = [
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="ceo", context="business"
    ),
    registry.create_frame_aware_concept_with_advanced_embedding(
        name="manager", context="business"
    )
]

# Train across domains
all_concepts = royal_concepts + business_concepts
cross_domain_result = await training_manager.train_cross_domain(
    concepts=all_concepts,
    domain_mappings={
        "royalty": royal_concepts,
        "business": business_concepts
    },
    config=config
)
```

---

## Persistence Layer and Workflow Management

The system provides enterprise-grade persistence with multi-format storage and workflow management.

### Basic Persistence Operations

#### Python API

```python
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from pathlib import Path

# Initialize persistence manager
storage_path = Path("demo_storage")
persistence_manager = ContractEnhancedPersistenceManager(storage_path)

# Save a concept
concept_data = {
    "concept_id": "royalty:king:king.n.01",
    "name": "king",
    "context": "royalty",
    "synset_id": "king.n.01",
    "disambiguation": "monarch ruler",
    "metadata": {"domain": "political_hierarchy"}
}

persistence_manager.save_concept(concept_data)

# Load the concept
loaded_concept = persistence_manager.load_concept("royalty:king:king.n.01")
print(f"Loaded: {loaded_concept['name']}")
```

### Batch Operations

#### Creating and Processing Batches

```python
from app.core.batch_persistence import BatchPersistenceManager

# Initialize batch manager
batch_manager = BatchPersistenceManager(storage_path)

# Create a batch of analogies
analogies = [
    {"king": "man", "queen": "woman"},
    {"prince": "boy", "princess": "girl"},
    {"lord": "master", "lady": "mistress"}
]

# Create batch workflow
workflow = batch_manager.create_analogy_batch(
    analogies=analogies,
    workflow_id="royal_gender_analogies"
)

print(f"Created workflow: {workflow.workflow_id}")
print(f"Status: {workflow.status}")
print(f"Items: {workflow.items_total}")

# Process the batch
processed_workflow = batch_manager.process_analogy_batch(workflow.workflow_id)
print(f"Processing complete. Status: {processed_workflow.status}")
```

#### REST API Batch Operations

```bash
# List all workflows
curl -X GET "http://localhost:8321/api/batch/workflows"

# Get specific workflow status
curl -X GET "http://localhost:8321/api/batch/workflows/royal_gender_analogies"

# Create new batch
curl -X POST "http://localhost:8321/api/batch/analogies" \
     -H "Content-Type: application/json" \
     -d '{
       "workflow_id": "medieval_batch_001",
       "analogies": [
         {"knight": "warrior", "squire": "apprentice"},
         {"castle": "fortress", "village": "settlement"}
       ]
     }'
```

### Streaming Large Datasets

For large datasets, use streaming operations:

#### Python API

```python
# Stream analogies with filtering
def process_large_dataset():
    count = 0
    for analogy in batch_manager.stream_analogies(
        domain_filter="medieval",
        min_quality=0.7
    ):
        count += 1
        print(f"Processing analogy {count}: {analogy}")
        
        # Process in chunks to avoid memory issues
        if count % 1000 == 0:
            print(f"Processed {count} analogies...")

# Stream with custom criteria
for analogy in batch_manager.stream_analogies_filtered(
    criteria=lambda x: x.get('quality', 0) > 0.8
):
    print(f"High-quality analogy: {analogy}")
```

### Storage Maintenance

#### Cleanup and Compaction

```python
# Perform storage maintenance
from app.core.batch_persistence import DeleteCriteria

# Clean up old workflows
cleanup_criteria = DeleteCriteria(
    older_than_days=30,
    status_filter="completed"
)

deleted_count = batch_manager.cleanup_old_workflows(cleanup_criteria)
print(f"Cleaned up {deleted_count} old workflows")

# Compact storage for better performance
batch_manager.compact_storage()
print("Storage compaction complete")

# Validate storage integrity
integrity_report = batch_manager.validate_storage_integrity()
if integrity_report['is_valid']:
    print("✅ Storage integrity validated")
else:
    print(f"❌ Storage issues found: {integrity_report['issues']}")
```

---

## API Reference Guide

### Core Endpoints

#### Concepts

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/concepts` | POST | Create a new concept |
| `/api/concepts/{id}` | GET | Retrieve concept by ID |
| `/api/concepts/search` | POST | Search for concepts |
| `/api/concepts/similarity` | POST | Compute concept similarity |

#### Reasoning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analogies/complete` | POST | Complete analogical reasoning |
| `/api/semantic-fields/discover` | POST | Discover semantic fields |
| `/api/analogies/cross-domain` | POST | Find cross-domain analogies |

#### Frames

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/frames` | POST | Create semantic frame |
| `/api/frames/{id}/instances` | POST | Create frame instance |
| `/api/frames/query` | POST | Query frames and instances |

#### Batch Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/batch/analogies` | POST | Create analogy batch |
| `/api/batch/workflows` | GET | List workflows |
| `/api/batch/workflows/{id}` | GET | Get workflow status |

#### Neural-Symbolic

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/neural-symbolic/train` | POST | Start neural training |
| `/api/neural-symbolic/status/{id}` | GET | Get training status |
| `/api/neural-symbolic/verify` | POST | SMT verification |

#### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/status` | GET | Detailed system status |
| `/api/docs-overview` | GET | API documentation overview |

### Request/Response Examples

#### Create Concept Request

```json
{
  "name": "knight",
  "context": "medieval",
  "synset_id": "knight.n.01",
  "disambiguation": "armored warrior",
  "metadata": {
    "domain": "military",
    "era": "medieval"
  },
  "auto_disambiguate": true
}
```

#### Analogy Completion Request

```json
{
  "source_a": "king",
  "source_b": "man",
  "target_a": "queen",
  "context": "royalty",
  "max_completions": 5,
  "min_confidence": 0.6
}
```

#### Neural Training Request

```json
{
  "training_config": {
    "epochs": 20,
    "learning_rate": 0.001,
    "batch_size": 64,
    "device": "cuda"
  },
  "concepts": [
    "royalty:king:king.n.01",
    "royalty:queen:queen.n.01"
  ],
  "axioms": [
    {
      "axiom_id": "gender_consistency",
      "formula": "forall x: male(x) -> not female(x)",
      "weight": 1.0
    }
  ]
}
```

---

## WebSocket Streaming

The system provides real-time streaming capabilities for long-running operations.

### Training Progress Streaming

```javascript
// Connect to training progress stream
const trainingWs = new WebSocket('ws://localhost:8321/api/ws/neural-symbolic/train_session_123');

trainingWs.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'training_progress':
            updateTrainingUI(data.progress);
            break;
        case 'epoch_complete':
            logEpochResults(data.epoch, data.loss, data.satisfiability);
            break;
        case 'training_complete':
            onTrainingComplete(data.results);
            break;
        case 'error':
            handleTrainingError(data.error);
            break;
    }
};

function updateTrainingUI(progress) {
    document.getElementById('epoch').textContent = progress.epoch;
    document.getElementById('loss').textContent = progress.loss.toFixed(4);
    document.getElementById('satisfiability').textContent = progress.satisfiability.toFixed(4);
    
    const progressBar = document.getElementById('progress');
    progressBar.value = (progress.epoch / progress.total_epochs) * 100;
}
```

### Workflow Status Streaming

```python
import asyncio
import websockets
import json

async def monitor_workflow(workflow_id):
    uri = f"ws://localhost:8321/api/ws/workflows/{workflow_id}/status"
    
    async with websockets.connect(uri) as websocket:
        print(f"Monitoring workflow: {workflow_id}")
        
        async for message in websocket:
            status = json.loads(message)
            
            print(f"Workflow {workflow_id}:")
            print(f"  Status: {status['status']}")
            print(f"  Progress: {status['items_processed']}/{status['items_total']}")
            print(f"  Errors: {status['error_count']}")
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                print("Workflow finished")
                break

# Monitor a specific workflow
asyncio.run(monitor_workflow("royal_analogies_001"))
```

### Analogy Streaming

```bash
# Stream analogies with filtering
curl -N -H "Connection: Upgrade" \
     -H "Upgrade: websocket" \
     -H "Sec-WebSocket-Key: x3JJHMbDL1EzLkh9GBhXDw==" \
     -H "Sec-WebSocket-Version: 13" \
     "ws://localhost:8321/api/ws/analogies/stream?domain=royalty&min_quality=0.7"
```

---

## Production Deployment

### Docker Deployment

#### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application code
COPY . .

# Expose port
EXPOSE 8321

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8321/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8321"]
```

#### Build and Run

```bash
# Build Docker image
docker build -t soft-logic-microservice .

# Run container
docker run -d \
    --name soft-logic \
    -p 8321:8321 \
    -v $(pwd)/storage:/app/storage \
    -e ENVIRONMENT=production \
    soft-logic-microservice

# Check logs
docker logs soft-logic

# Health check
curl http://localhost:8321/health
```

### Kubernetes Deployment

#### Create Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: soft-logic-microservice
  labels:
    app: soft-logic
spec:
  replicas: 3
  selector:
    matchLabels:
      app: soft-logic
  template:
    metadata:
      labels:
        app: soft-logic
    spec:
      containers:
      - name: soft-logic
        image: soft-logic-microservice:latest
        ports:
        - containerPort: 8321
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8321
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8321
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: storage
          mountPath: /app/storage
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: soft-logic-storage

---
apiVersion: v1
kind: Service
metadata:
  name: soft-logic-service
spec:
  selector:
    app: soft-logic
  ports:
  - port: 80
    targetPort: 8321
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: soft-logic-storage
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

#### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -l app=soft-logic

# Port forward for testing
kubectl port-forward service/soft-logic-service 8321:80
```

### Environment Configuration

#### Production Settings

```python
# app/config.py
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    log_level: str = "INFO"
    
    # Database settings
    storage_path: str = "./storage"
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    
    # Neural training settings
    default_device: str = "cpu"
    max_training_sessions: int = 10
    training_timeout_minutes: int = 60
    
    # API settings
    cors_origins: list = ["*"]
    rate_limit_per_minute: int = 1000
    
    # SMT verification settings
    smt_timeout_seconds: int = 30
    z3_solver_timeout: int = 10000
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### Environment Variables

```bash
# .env file for production
ENVIRONMENT=production
LOG_LEVEL=INFO
STORAGE_PATH=/app/storage
BACKUP_ENABLED=true
DEFAULT_DEVICE=cuda
MAX_TRAINING_SESSIONS=20
CORS_ORIGINS=["https://yourdomain.com"]
RATE_LIMIT_PER_MINUTE=2000
```

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Issues

**Problem**: Poetry installation fails
```bash
# Solution: Install poetry directly
curl -sSL https://install.python-poetry.org | python3 -
```

**Problem**: CUDA not detected for neural training
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Runtime Issues

**Problem**: Service fails to start
```bash
# Check logs
poetry run uvicorn app.main:app --log-level debug

# Verify dependencies
poetry run python -c "from app.core import EnhancedHybridRegistry; print('OK')"
```

**Problem**: Neural training fails
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in training config
{
  "training_config": {
    "batch_size": 16,  # Reduce from 32
    "device": "cpu"    # Fallback to CPU
  }
}
```

**Problem**: SMT verification times out
```bash
# Increase timeout in request
{
  "verification_timeout": 30000,  # 30 seconds
  "axioms": [...]
}
```

#### Performance Issues

**Problem**: Slow API responses
```bash
# Enable performance monitoring
curl -X GET "http://localhost:8321/api/status"

# Check storage performance
curl -X POST "http://localhost:8321/api/batch/analogies" \
     -d '{"analogies": [{"test": "performance"}]}'
```

**Problem**: Memory usage too high
```bash
# Monitor memory usage
docker stats soft-logic

# Reduce clustering parameters
registry = EnhancedHybridRegistry(
    n_clusters=4,  # Reduce from 8
    enable_cross_domain=False  # Disable if not needed
)
```

#### Storage Issues

**Problem**: Storage corruption
```python
# Validate and repair storage
from app.core.batch_persistence import BatchPersistenceManager

batch_manager = BatchPersistenceManager("./storage")
integrity_report = batch_manager.validate_storage_integrity()

if not integrity_report['is_valid']:
    print("Repairing storage...")
    batch_manager.repair_storage()
```

**Problem**: Disk space issues
```bash
# Clean up old workflows
curl -X DELETE "http://localhost:8321/api/batch/cleanup" \
     -d '{"older_than_days": 7, "status_filter": "completed"}'

# Compact storage
curl -X POST "http://localhost:8321/api/batch/compact"
```

### Debug Mode

Enable debug mode for detailed logging:

```python
# app/main.py
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Start with debug logging
uvicorn app.main:app --log-level debug --reload
```

### Health Monitoring

Set up monitoring endpoints:

```bash
# Basic health check
curl http://localhost:8321/health

# Detailed system status
curl http://localhost:8321/api/status

# Component-specific health
curl http://localhost:8321/api/neural-symbolic/health
curl http://localhost:8321/api/batch/health
```

### Performance Profiling

```python
# Profile API performance
import cProfile
import pstats

def profile_api_call():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Make API call
    response = requests.post("http://localhost:8321/api/concepts", 
                           json={"name": "test", "context": "benchmark"})
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)

profile_api_call()
```

---

## Advanced Examples

### Complete Workflow Example

This example demonstrates a complete workflow from concept creation through neural training:

```python
#!/usr/bin/env python3
"""
Complete Soft Logic Workflow Example
===================================

This script demonstrates a full workflow using all major system components.
"""

import asyncio
import json
from pathlib import Path

from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry
from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.neural_symbolic_integration import (
    NeuralSymbolicTrainingManager,
    TrainingConfiguration,
    Z3SMTVerifier
)
from app.core.abstractions import Axiom, FormulaNode

async def complete_workflow_example():
    """Demonstrate complete workflow."""
    
    print("🚀 Starting Complete Soft Logic Workflow")
    print("=" * 50)
    
    # Step 1: Initialize components
    print("\n1️⃣ Initializing System Components")
    
    storage_path = Path("workflow_demo_storage")
    storage_path.mkdir(exist_ok=True)
    
    registry = EnhancedHybridRegistry(
        download_wordnet=False,
        n_clusters=6,
        enable_cross_domain=True,
        embedding_provider="semantic"
    )
    
    persistence_manager = ContractEnhancedPersistenceManager(storage_path)
    training_manager = NeuralSymbolicTrainingManager()
    smt_verifier = Z3SMTVerifier()
    
    # Step 2: Create knowledge base
    print("\n2️⃣ Creating Knowledge Base")
    
    # Create royal hierarchy concepts
    royal_concepts = []
    royal_data = [
        ("king", "male monarch", "king.n.01"),
        ("queen", "female monarch", "queen.n.01"),
        ("prince", "male heir", "prince.n.01"),
        ("princess", "female heir", "princess.n.01")
    ]
    
    for name, desc, synset in royal_data:
        concept = registry.create_frame_aware_concept_with_advanced_embedding(
            name=name,
            context="royalty",
            synset_id=synset,
            disambiguation=desc,
            use_semantic_embedding=True
        )
        royal_concepts.append(concept)
        
        # Persist concept
        concept_data = {
            "concept_id": concept.unique_id,
            "name": concept.name,
            "context": concept.context,
            "synset_id": concept.synset_id,
            "disambiguation": concept.disambiguation
        }
        persistence_manager.save_concept(concept_data)
        print(f"  ✅ Created and saved: {name}")
    
    # Create gender concepts
    gender_concepts = []
    gender_data = [
        ("man", "adult male human", "man.n.01"),
        ("woman", "adult female human", "woman.n.01"),
        ("boy", "male child", "boy.n.01"),
        ("girl", "female child", "girl.n.01")
    ]
    
    for name, desc, synset in gender_data:
        concept = registry.create_frame_aware_concept_with_advanced_embedding(
            name=name,
            context="default",
            synset_id=synset,
            disambiguation=desc,
            use_semantic_embedding=True
        )
        gender_concepts.append(concept)
        print(f"  ✅ Created: {name}")
    
    all_concepts = royal_concepts + gender_concepts
    
    # Step 3: Define and verify axioms
    print("\n3️⃣ Defining and Verifying Axioms")
    
    axioms = [
        Axiom(
            axiom_id="gender_exclusivity",
            axiom_type="consistency",
            classification="core",
            context="default",
            description="Male and female are mutually exclusive",
            formula=FormulaNode("forall", ["x"], 
                FormulaNode("not", [
                    FormulaNode("and", [
                        FormulaNode("male", ["x"]),
                        FormulaNode("female", ["x"])
                    ])
                ])
            )
        ),
        Axiom(
            axiom_id="royal_gender_consistency",
            axiom_type="consistency",
            classification="core",
            context="royalty",
            description="Kings are male, queens are female",
            formula=FormulaNode("and", [
                FormulaNode("forall", ["x"],
                    FormulaNode("implies", [
                        FormulaNode("king", ["x"]),
                        FormulaNode("male", ["x"])
                    ])
                ),
                FormulaNode("forall", ["x"],
                    FormulaNode("implies", [
                        FormulaNode("queen", ["x"]),
                        FormulaNode("female", ["x"])
                    ])
                )
            ])
        )
    ]
    
    # Verify axiom consistency
    verification_result = smt_verifier.verify_axiom_consistency(axioms)
    if verification_result.is_consistent:
        print("  ✅ All axioms are consistent")
    else:
        print(f"  ❌ Axiom inconsistency detected: {verification_result.counterexample}")
        return
    
    # Step 4: Discover semantic patterns
    print("\n4️⃣ Discovering Semantic Patterns")
    
    # Update clusters
    registry.update_clusters()
    print(f"  ✅ Updated concept clusters")
    
    # Discover semantic fields
    semantic_fields = registry.discover_semantic_fields(min_coherence=0.5)
    print(f"  ✅ Discovered {len(semantic_fields)} semantic fields")
    
    for i, field in enumerate(semantic_fields[:3]):
        field_name = field.get('name', f'Field {i}')
        coherence = field.get('coherence', 0.0)
        concepts = field.get('core_concepts', [])
        print(f"    Field '{field_name}': {coherence:.2f} coherence, {len(concepts)} concepts")
    
    # Find cross-domain analogies
    cross_domain_analogies = registry.discover_cross_domain_analogies(min_quality=0.3)
    print(f"  ✅ Found {len(cross_domain_analogies)} cross-domain analogies")
    
    # Step 5: Create and process analogy batches
    print("\n5️⃣ Processing Analogy Batches")
    
    from app.core.batch_persistence import BatchPersistenceManager
    batch_manager = BatchPersistenceManager(storage_path)
    
    # Create batch of analogies
    analogies = [
        {"king": "man", "queen": "woman"},
        {"prince": "boy", "princess": "girl"},
        {"king": "ruler", "queen": "ruler"},  # Different type of analogy
    ]
    
    workflow = batch_manager.create_analogy_batch(
        analogies=analogies,
        workflow_id="royal_gender_workflow"
    )
    print(f"  ✅ Created workflow: {workflow.workflow_id}")
    
    # Process the batch
    processed_workflow = batch_manager.process_analogy_batch(workflow.workflow_id)
    print(f"  ✅ Processed {processed_workflow.items_processed} analogies")
    
    # Step 6: Neural-symbolic training
    print("\n6️⃣ Neural-Symbolic Training")
    
    # Configure training
    config = TrainingConfiguration(
        epochs=5,  # Small number for demo
        learning_rate=0.01,
        batch_size=16,
        device="cpu"  # Use CPU for demo reliability
    )
    
    print(f"  🏋️ Starting training: {config.epochs} epochs")
    
    # Train with progress monitoring
    epoch_count = 0
    async for progress in training_manager.train_with_progress_streaming(
        concepts=all_concepts,
        config=config
    ):
        epoch_count += 1
        print(f"    Epoch {progress.epoch}: "
              f"loss={progress.loss:.4f}, "
              f"satisfiability={progress.satisfiability:.4f}")
        
        # Early stopping for demo
        if epoch_count >= 3:
            break
    
    print("  ✅ Training completed")
    
    # Step 7: Evaluate results
    print("\n7️⃣ Evaluating Results")
    
    # Test analogical reasoning
    test_analogies = [
        {"king": "man", "target": "queen"},
        {"prince": "boy", "target": "princess"}
    ]
    
    for analogy in test_analogies:
        source_a = analogy["king"] if "king" in analogy else analogy["prince"]
        source_b = analogy.get("man", analogy.get("boy", ""))
        target_a = analogy["target"]
        
        partial = {source_a: source_b, target_a: "?"}
        completions = registry.complete_analogy(partial, max_completions=3)
        
        print(f"  🧠 Analogy '{source_a}:{source_b} :: {target_a}:?'")
        for completion in completions[:2]:
            print(f"    → {completion}")
    
    # Step 8: Generate final report
    print("\n8️⃣ Final System Report")
    
    stats = {
        "concepts_created": len(all_concepts),
        "axioms_verified": len(axioms),
        "semantic_fields": len(semantic_fields),
        "cross_domain_analogies": len(cross_domain_analogies),
        "workflows_processed": 1,
        "training_epochs": epoch_count
    }
    
    print(f"  📊 System Statistics:")
    for key, value in stats.items():
        print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Save final report
    report_path = storage_path / "workflow_report.json"
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n🎉 Workflow Complete! Report saved to: {report_path}")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(complete_workflow_example())
```

Save this example as `complete_workflow_demo.py` and run it to see the entire system in action.

---

## Conclusion

This tutorial has covered the complete Soft Logic Microservice system, from basic installation through advanced neural-symbolic training. The system provides a robust foundation for building intelligent applications that combine symbolic reasoning with neural learning.

### Key Takeaways

1. **Modular Architecture**: The system is built with clear separation between components
2. **Type Safety**: Full mypy compliance ensures robust development
3. **Enterprise Ready**: Production-grade persistence and monitoring capabilities
4. **Neural-Symbolic Integration**: Seamless combination of symbolic and neural approaches
5. **Comprehensive API**: Complete REST API with WebSocket streaming

### Next Steps

- Explore the interactive documentation at `/api/docs`
- Run the demonstration scripts to see specific features
- Experiment with your own domain-specific concepts and analogies
- Deploy to production using the provided Docker and Kubernetes configurations

For additional support and advanced usage patterns, refer to the extensive documentation in the `documentation/` directory.

Happy reasoning! 🧠✨
