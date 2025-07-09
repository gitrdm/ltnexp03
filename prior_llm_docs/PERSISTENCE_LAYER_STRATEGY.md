# Persistence Layer Strategy for Soft Logic Microservice

## Current State Analysis

### ✅ Existing Persistence (Limited)
- **Vector Embeddings**: Basic caching in `embeddings_cache/` directory
- **Axiom Files**: YAML/JSON parsing for axiom definitions
- **Demo State**: No persistence - rebuilt on each run

### ❌ Missing Persistence Components
- **Semantic Frames**: No persistence for FrameNet-style structures
- **Frame Instances**: No storage for story/domain-specific instances  
- **Concept Clusters**: No persistence for trained clustering models
- **Cross-Domain Analogies**: No storage for discovered analogical patterns
- **Semantic Fields**: No persistence for discovered coherent regions
- **Registry State**: No full system state serialization/deserialization
- **Context Inheritance**: No persistence for context hierarchies
- **Contract Validation Results**: No audit trail for contract violations

## Comprehensive Persistence Layer Design

### 1. **Batch-Optimized Storage Strategy**

**Problem Analysis:**
- Current JSON approach doesn't support efficient batch operations
- Adding single analogies requires rewriting entire files
- No support for incremental updates or deletions
- Memory inefficient for large knowledge bases

**Recommended Multi-Modal Approach:**

```
storage/
├── contexts/                    # Context-specific storage
│   ├── default/
│   │   ├── analogies.jsonl     # 🆕 Line-delimited for batch ops
│   │   ├── frames.jsonl        # 🆕 Incremental frame updates
│   │   ├── concepts.sqlite     # 🆕 SQLite for complex queries
│   │   ├── clusters.json       # Stable cluster configurations
│   │   └── embeddings/         # Vector embeddings cache
│   │       ├── embeddings.npz  # Compressed NumPy arrays
│   │       └── metadata.json   # Embedding metadata
│   └── batch_operations/       # 🆕 Batch processing workspace
│       ├── pending_analogies.jsonl
│       ├── batch_metadata.json
│       └── processing_log.jsonl
├── models/                     # Trained model persistence
│   ├── ltn_models/            # LTN PyTorch models (.pth)
│   ├── clustering_models/      # Scikit-learn models (.joblib)
│   └── vector_indexes/        # 🆕 Vector search indexes
│       ├── analogy_index.faiss
│       └── concept_index.annoy
└── workflows/                  # 🆕 Batch workflow management
    ├── analogy_workflows/
    │   ├── royal_military_batch_001.jsonl
    │   └── batch_001_metadata.json
    └── frame_workflows/
        ├── medieval_frames_batch_001.jsonl
        └── batch_001_metadata.json
```

### 2. **Format-Specific Strategy by Use Case**

#### 2.1 **JSONL for Batch Operations** 
```python
# Efficient batch analogy creation
def add_analogies_batch(analogies: List[Dict]) -> None:
    """Add analogies in batch - append-only operation."""
    with open('contexts/default/analogies.jsonl', 'a') as f:
        for analogy in analogies:
            f.write(json.dumps(analogy) + '\n')

# Efficient batch processing
def process_analogies_stream():
    """Process analogies one-by-one without loading entire file."""
    with open('contexts/default/analogies.jsonl', 'r') as f:
        for line in f:
            analogy = json.loads(line)
            yield analogy
```

#### 2.2 **SQLite for Complex Operations**
```python
# Efficient updates and deletes
def delete_analogies_by_domain(domain: str) -> int:
    """Delete all analogies from specific domain."""
    with sqlite3.connect('contexts/default/concepts.sqlite') as conn:
        cursor = conn.execute(
            "DELETE FROM analogies WHERE source_domain = ? OR target_domain = ?",
            (domain, domain)
        )
        return cursor.rowcount

# Complex queries
def find_analogies_by_quality(min_quality: float) -> List[Dict]:
    """Find high-quality analogies across all domains."""
    with sqlite3.connect('contexts/default/concepts.sqlite') as conn:
        return conn.execute(
            "SELECT * FROM analogies WHERE quality_score >= ? ORDER BY quality_score DESC",
            (min_quality,)
        ).fetchall()
```

#### 2.3 **Vector Indexes for Similarity Search**
```python
import faiss
import numpy as np

class VectorIndexManager:
    """Manage vector indexes for efficient similarity search."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.analogy_index = faiss.IndexFlatIP(embedding_dim)
        self.concept_index = faiss.IndexFlatIP(embedding_dim)
    
    def add_analogies_to_index(self, analogies: List[Dict], embeddings: np.ndarray):
        """Add batch of analogies to vector index."""
        self.analogy_index.add(embeddings)
        
    def find_similar_analogies(self, query_embedding: np.ndarray, k: int = 10):
        """Find k most similar analogies."""
        distances, indices = self.analogy_index.search(query_embedding.reshape(1, -1), k)
        return indices[0], distances[0]
```

### 3. **Workflow-Aware Persistence Architecture**

```python
@dataclass
class BatchWorkflow:
    """Represents a batch operation workflow."""
    workflow_id: str
    workflow_type: str  # 'analogy_creation', 'frame_expansion', 'concept_batch'
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: datetime
    items_total: int
    items_processed: int
    error_count: int
    metadata: Dict[str, Any]

class WorkflowAwarePersistenceManager:
    """Persistence manager with batch workflow support."""
    
    def create_analogy_batch(self, analogies: List[Dict], workflow_id: str) -> BatchWorkflow:
        """Create batch workflow for analogy creation."""
        
    def process_batch_workflow(self, workflow_id: str) -> BatchWorkflow:
        """Process pending batch workflow."""
        
    def rollback_batch_workflow(self, workflow_id: str) -> bool:
        """Rollback failed batch workflow."""
        
    def get_workflow_status(self, workflow_id: str) -> Optional[BatchWorkflow]:
        """Get current workflow status."""
```

### 4. **Delete/Update Strategy**

#### 4.1 **Soft Deletes with Versioning**
```python
# Instead of physical deletion
def delete_analogy(analogy_id: str) -> bool:
    """Soft delete analogy with versioning."""
    with sqlite3.connect('contexts/default/concepts.sqlite') as conn:
        conn.execute(
            "UPDATE analogies SET deleted_at = ?, version = version + 1 WHERE id = ?",
            (datetime.now(timezone.utc), analogy_id)
        )
        
# Append tombstone record to JSONL
def delete_analogy_jsonl(analogy_id: str) -> bool:
    """Add deletion record to JSONL."""
    tombstone = {
        "type": "deletion",
        "target_id": analogy_id,
        "deleted_at": datetime.now(timezone.utc).isoformat()
    }
    with open('contexts/default/analogies.jsonl', 'a') as f:
        f.write(json.dumps(tombstone) + '\n')
```

#### 4.2 **Compaction and Cleanup**
```python
def compact_analogies_jsonl() -> None:
    """Remove deleted records and compact file."""
    active_analogies = {}
    deleted_ids = set()
    
    # Read all records
    with open('contexts/default/analogies.jsonl', 'r') as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "deletion":
                deleted_ids.add(record["target_id"])
            elif record["type"] == "analogy":
                active_analogies[record["id"]] = record
    
    # Remove deleted analogies
    for deleted_id in deleted_ids:
        active_analogies.pop(deleted_id, None)
    
    # Rewrite file
    with open('contexts/default/analogies.jsonl', 'w') as f:
        for analogy in active_analogies.values():
            f.write(json.dumps(analogy) + '\n')
```

### 5. **Production Workflow Integration**

#### 5.1 **API Endpoints for Batch Operations**
```python
@app.post("/analogies/batch")
async def create_analogy_batch(batch: AnalogiesBatch) -> BatchWorkflow:
    """Create batch of analogies with workflow tracking."""
    
@app.get("/workflows/{workflow_id}/status")
async def get_batch_status(workflow_id: str) -> BatchWorkflow:
    """Get batch workflow status."""
    
@app.delete("/analogies/batch")
async def delete_analogies_batch(criteria: DeleteCriteria) -> BatchWorkflow:
    """Delete analogies matching criteria."""
    
@app.post("/analogies/compact")
async def compact_analogies() -> CompactionResult:
    """Compact analogy storage by removing deleted records."""
```

#### 5.2 **Background Processing**
```python
class BatchProcessor:
    """Background processor for batch operations."""
    
    async def process_pending_workflows(self):
        """Process all pending workflows."""
        
    async def cleanup_completed_workflows(self):
        """Clean up old completed workflows."""
        
    async def monitor_workflow_health(self):
        """Monitor workflow processing health."""
```

### 6. **Recommendation Summary**

**For Your Batch Workflow:**

1. **✅ Use JSONL** for analogies, frames, and incremental data
2. **✅ Use SQLite** for complex queries and transactional operations  
3. **✅ Keep .npz** for vector embeddings (already optimal)
4. **✅ Add Vector Indexes** (FAISS/Annoy) for similarity search
5. **✅ Implement Workflow Management** for batch operations
6. **✅ Use Soft Deletes** with compaction for data safety

This hybrid approach gives you:
- **Efficient batch operations** (JSONL append-only)
- **Complex queries** (SQLite with indexes)
- **Fast similarity search** (Vector indexes)
- **Workflow tracking** (Batch operation management)
- **Data safety** (Soft deletes with rollback capability)
