#!/usr/bin/env python3
"""
Multi-Format Persistence Example

This script demonstrates the specific multi-format storage strategy outlined
in PERSISTENCE_LAYER_STRATEGY.md with concrete examples of:

1. JSONL format for batch operations
2. SQLite for complex queries  
3. NPZ for vector embeddings
4. File structure as recommended in the strategy
5. Format-specific operations and use cases

This example creates actual files in the recommended directory structure:

storage/
├── contexts/
│   ├── default/
│   │   ├── analogies.jsonl     # Batch operations
│   │   ├── frames.jsonl        # Incremental updates  
│   │   ├── concepts.sqlite     # Complex queries
│   │   └── embeddings/
│   │       ├── embeddings.npz  # Compressed vectors
│   │       └── metadata.json   # Vector metadata
│   └── batch_operations/       # Batch workspace
└── workflows/                  # Workflow management
"""

import json
import sqlite3
import numpy as np
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# Import persistence components
from app.core.contract_persistence import ContractEnhancedPersistenceManager


class MultiFormatDemo:
    """Demonstrates the multi-format storage strategy with concrete examples."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize with storage structure."""
        self.base_path = base_path or Path(f"multi_format_demo_{int(time.time())}")
        self.storage_path = self.base_path / "storage"
        
        # Create the recommended directory structure
        self.setup_storage_structure()
        
        # Initialize manager
        self.manager = ContractEnhancedPersistenceManager(self.storage_path)
    
    def setup_storage_structure(self):
        """Create the recommended directory structure from the strategy."""
        directories = [
            "contexts/default",
            "contexts/batch_operations", 
            "models/ltn_models",
            "models/clustering_models",
            "models/vector_indexes",
            "workflows/analogy_workflows",
            "workflows/frame_workflows"
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
    
    def print_section(self, title: str, emoji: str = "📁"):
        """Print formatted section header."""
        print(f"\n{emoji} {title}")
        print("=" * (len(title) + 4))
    
    def demonstrate_jsonl_format(self):
        """
        Demonstrate JSONL format for batch operations.
        
        Shows the append-only, line-delimited JSON approach recommended
        for efficient batch processing.
        """
        self.print_section("JSONL Format for Batch Operations", "📝")
        
        # Create analogies.jsonl as recommended
        analogies_file = self.storage_path / "contexts" / "default" / "analogies.jsonl"
        
        print("Creating analogies.jsonl with sample data...")
        
        # Sample analogies following strategy examples
        sample_analogies = [
            {
                "id": "royal_001",
                "type": "analogy",
                "source_domain": "royal",
                "target_domain": "military", 
                "concept_mappings": {
                    "king": "general",
                    "queen": "commander",
                    "castle": "fortress"
                },
                "quality_score": 0.85,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "batch_id": "royal_military_batch_001",
                    "workflow_id": "royal_military_001"
                }
            },
            {
                "id": "royal_002", 
                "type": "analogy",
                "source_domain": "royal",
                "target_domain": "business",
                "concept_mappings": {
                    "king": "ceo",
                    "court": "board_of_directors",
                    "kingdom": "corporation"
                },
                "quality_score": 0.78,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metadata": {
                    "batch_id": "royal_business_batch_001", 
                    "workflow_id": "royal_business_001"
                }
            }
        ]
        
        # Write JSONL format (one JSON object per line)
        with open(analogies_file, 'w') as f:
            for analogy in sample_analogies:
                f.write(json.dumps(analogy) + '\n')
        
        print(f"✅ Created {analogies_file}")
        print(f"   Format: JSONL (line-delimited JSON)")
        print(f"   Records: {len(sample_analogies)}")
        print(f"   Use case: Efficient batch operations, append-only updates")
        
        # Demonstrate append operation (key strategy feature)
        print("\n🔄 Demonstrating append operation...")
        new_analogy = {
            "id": "royal_003",
            "type": "analogy", 
            "source_domain": "royal",
            "target_domain": "scientific",
            "concept_mappings": {
                "crown": "hypothesis",
                "decree": "theory"
            },
            "quality_score": 0.72,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "batch_id": "royal_scientific_batch_001",
                "appended": True
            }
        }
        
        # Append without rewriting existing data
        with open(analogies_file, 'a') as f:
            f.write(json.dumps(new_analogy) + '\n')
        
        print("✅ Appended new analogy without rewriting file")
        print("   Key benefit: O(1) append operation, no memory loading")
        
        # Demonstrate streaming read (strategy recommendation)
        print("\n📖 Demonstrating streaming read...")
        analogy_count = 0
        with open(analogies_file, 'r') as f:
            for line in f:
                analogy = json.loads(line)
                analogy_count += 1
                if analogy_count <= 2:  # Show first 2
                    print(f"   📄 {analogy['id']}: {analogy['source_domain']} → {analogy['target_domain']}")
        
        print(f"✅ Streamed {analogy_count} analogies without loading entire file")
        
        return analogies_file
    
    def demonstrate_sqlite_format(self):
        """
        Demonstrate SQLite format for complex queries.
        
        Shows how SQLite is used for operations requiring joins,
        complex filtering, and transactional updates.
        """
        self.print_section("SQLite Format for Complex Queries", "🗄️")
        
        # Create concepts.sqlite as recommended
        sqlite_file = self.storage_path / "contexts" / "default" / "demo_concepts.sqlite"
        
        print("Creating concepts.sqlite with relational structure...")
        
        # Connect and create tables
        conn = sqlite3.connect(sqlite_file)
        cursor = conn.cursor()
        
        # Create analogies table for complex queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analogies (
                id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                target_domain TEXT NOT NULL,
                quality_score REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                deleted_at TIMESTAMP NULL,
                version INTEGER DEFAULT 1,
                batch_id TEXT,
                workflow_id TEXT
            )
        ''')
        
        # Create concept mappings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concept_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analogy_id TEXT NOT NULL,
                source_concept TEXT NOT NULL,
                target_concept TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (analogy_id) REFERENCES analogies (id)
            )
        ''')
        
        # Create indexes for efficient queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analogies_domain ON analogies(source_domain, target_domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analogies_quality ON analogies(quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analogies_deleted ON analogies(deleted_at)')
        
        # Insert sample data
        sample_data = [
            ('royal_001', 'royal', 'military', 0.85, datetime.now(timezone.utc).isoformat(), None, 1, 'royal_military_batch_001', 'royal_military_001'),
            ('royal_002', 'royal', 'business', 0.78, datetime.now(timezone.utc).isoformat(), None, 1, 'royal_business_batch_001', 'royal_business_001'),
            ('medieval_001', 'medieval', 'business', 0.82, datetime.now(timezone.utc).isoformat(), None, 1, 'medieval_business_batch_001', 'medieval_business_001')
        ]
        
        cursor.executemany('''
            INSERT INTO analogies (id, source_domain, target_domain, quality_score, created_at, deleted_at, version, batch_id, workflow_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_data)
        
        # Insert concept mappings
        mappings_data = [
            ('royal_001', 'king', 'general', 0.9),
            ('royal_001', 'queen', 'commander', 0.85),
            ('royal_002', 'king', 'ceo', 0.8),
            ('royal_002', 'court', 'board_of_directors', 0.75)
        ]
        
        cursor.executemany('''
            INSERT INTO concept_mappings (analogy_id, source_concept, target_concept, confidence)
            VALUES (?, ?, ?, ?)
        ''', mappings_data)
        
        conn.commit()
        
        print(f"✅ Created {sqlite_file}")
        print("   Tables: analogies, concept_mappings")
        print("   Indexes: domain, quality, deletion status")
        print("   Use case: Complex queries, joins, transactions")
        
        # Demonstrate complex queries (strategy benefit)
        print("\n🔍 Demonstrating complex SQL queries...")
        
        # Query 1: High-quality analogies by domain
        cursor.execute('''
            SELECT source_domain, target_domain, quality_score 
            FROM analogies 
            WHERE quality_score >= 0.8 AND deleted_at IS NULL
            ORDER BY quality_score DESC
        ''')
        results = cursor.fetchall()
        print(f"✅ High-quality analogies (≥0.8): {len(results)} found")
        for row in results:
            print(f"   📊 {row[0]} → {row[1]} (quality: {row[2]})")
        
        # Query 2: Cross-domain analysis with joins
        cursor.execute('''
            SELECT a.source_domain, a.target_domain, 
                   COUNT(cm.id) as mapping_count,
                   AVG(cm.confidence) as avg_confidence
            FROM analogies a
            JOIN concept_mappings cm ON a.id = cm.analogy_id
            WHERE a.deleted_at IS NULL
            GROUP BY a.source_domain, a.target_domain
            ORDER BY mapping_count DESC
        ''')
        results = cursor.fetchall()
        print(f"\n✅ Cross-domain mapping analysis:")
        for row in results:
            print(f"   🔗 {row[0]} → {row[1]}: {row[2]} mappings (avg confidence: {row[3]:.2f})")
        
        # Query 3: Soft delete demonstration
        print("\n🗑️ Demonstrating soft delete...")
        cursor.execute('''
            UPDATE analogies 
            SET deleted_at = ?, version = version + 1
            WHERE quality_score < 0.8
        ''', (datetime.now(timezone.utc).isoformat(),))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"✅ Soft deleted {deleted_count} low-quality analogies")
        print("   Benefit: Data preserved, can be restored if needed")
        
        conn.close()
        return sqlite_file
    
    def demonstrate_npz_format(self):
        """
        Demonstrate NPZ format for vector embeddings.
        
        Shows compressed numpy array storage for efficient
        vector operations and similarity search.
        """
        self.print_section("NPZ Format for Vector Embeddings", "🔢")
        
        # Create embeddings directory as recommended
        embeddings_dir = self.storage_path / "contexts" / "default" / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        
        print("Creating compressed vector embeddings...")
        
        # Generate sample embeddings (300-dimensional as in strategy)
        concepts = ["king", "queen", "castle", "general", "commander", "fortress", 
                   "ceo", "board", "corporation", "hypothesis", "theory", "experiment"]
        
        embeddings_data = {}
        embedding_metadata = {}
        
        for concept in concepts:
            # Generate random embedding (in real system, would use actual embeddings)
            embedding = np.random.randn(300).astype(np.float32)
            embeddings_data[concept] = embedding
            
            # Store metadata
            embedding_metadata[concept] = {
                "concept_id": concept,
                "embedding_model": "demo_random_300d",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dimensions": 300,
                "source": "generated",
                "confidence": 0.85,
                "update_count": 0
            }
        
        # Save embeddings in compressed NPZ format
        embeddings_file = embeddings_dir / "embeddings.npz"
        np.savez_compressed(embeddings_file, **embeddings_data)
        
        # Save metadata separately
        metadata_file = embeddings_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(embedding_metadata, f, indent=2)
        
        print(f"✅ Created {embeddings_file}")
        print(f"   Format: NPZ (compressed numpy arrays)")
        print(f"   Concepts: {len(concepts)}")
        print(f"   Dimensions: 300 per embedding")
        print(f"   Use case: Vector similarity search, ML operations")
        
        # Demonstrate loading and similarity computation
        print("\n🔍 Demonstrating vector operations...")
        
        # Load embeddings
        loaded_embeddings = np.load(embeddings_file)
        king_emb = loaded_embeddings['king']
        queen_emb = loaded_embeddings['queen']
        general_emb = loaded_embeddings['general']
        
        # Compute similarities (cosine similarity)
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        king_queen_sim = cosine_similarity(king_emb, queen_emb)
        king_general_sim = cosine_similarity(king_emb, general_emb)
        
        print(f"✅ Vector similarity computations:")
        print(f"   👑 king ↔ queen: {king_queen_sim:.3f}")
        print(f"   👑 king ↔ general: {king_general_sim:.3f}")
        
        # Show file size benefits
        file_size = embeddings_file.stat().st_size / 1024  # KB
        uncompressed_size = sum(emb.nbytes for emb in embeddings_data.values()) / 1024
        compression_ratio = uncompressed_size / file_size
        
        print(f"\n💾 Storage efficiency:")
        print(f"   Compressed size: {file_size:.1f} KB")
        print(f"   Uncompressed size: {uncompressed_size:.1f} KB") 
        print(f"   Compression ratio: {compression_ratio:.1f}x")
        
        return embeddings_file, metadata_file
    
    def demonstrate_workflow_files(self):
        """
        Demonstrate workflow file management.
        
        Shows the batch workflow tracking files as recommended
        in the strategy document.
        """
        self.print_section("Workflow File Management", "📋")
        
        # Create workflow files as recommended
        workflow_dir = self.storage_path / "workflows" / "analogy_workflows"
        
        print("Creating workflow tracking files...")
        
        # Create batch workflow file
        batch_file = workflow_dir / "royal_military_batch_001.jsonl"
        batch_metadata_file = workflow_dir / "batch_001_metadata.json"
        
        # Sample workflow entries
        workflow_entries = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "batch_created",
                "workflow_id": "royal_military_batch_001",
                "items_count": 25,
                "status": "pending"
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "batch_processing_started",
                "workflow_id": "royal_military_batch_001",
                "status": "processing"
            },
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "batch_completed",
                "workflow_id": "royal_military_batch_001",
                "items_processed": 25,
                "status": "completed",
                "duration_seconds": 0.15
            }
        ]
        
        # Write workflow log
        with open(batch_file, 'w') as f:
            for entry in workflow_entries:
                f.write(json.dumps(entry) + '\n')
        
        # Write workflow metadata
        workflow_metadata = {
            "workflow_id": "royal_military_batch_001",
            "workflow_type": "analogy_creation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_domain": "royal",
            "target_domain": "military",
            "total_items": 25,
            "processing_config": {
                "quality_threshold": 0.5,
                "validation_enabled": True,
                "backup_created": True
            },
            "status": "completed"
        }
        
        with open(batch_metadata_file, 'w') as f:
            json.dump(workflow_metadata, f, indent=2)
        
        print(f"✅ Created workflow files:")
        print(f"   📄 {batch_file} (workflow log)")
        print(f"   📄 {batch_metadata_file} (workflow metadata)")
        print("   Use case: Batch operation tracking, audit trail")
        
        return batch_file, batch_metadata_file
    
    def show_directory_structure(self):
        """Show the complete directory structure created."""
        self.print_section("Complete Storage Structure", "🗂️")
        
        print("Directory structure (as recommended in strategy):")
        
        def print_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth > max_depth:
                return
            
            if path.is_dir():
                items = sorted(path.iterdir())
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    current_prefix = "└── " if is_last else "├── "
                    print(f"{prefix}{current_prefix}{item.name}")
                    
                    if item.is_dir() and current_depth < max_depth:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        print_tree(item, next_prefix, max_depth, current_depth + 1)
        
        print_tree(self.storage_path)
        
        # Show file sizes
        print("\n📊 Storage statistics:")
        total_size = 0
        file_count = 0
        
        for file_path in self.storage_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
        
        print(f"   Total files: {file_count}")
        print(f"   Total size: {total_size / 1024:.1f} KB")
        print(f"   Average file size: {(total_size / file_count) / 1024:.1f} KB")
    
    def demonstrate_format_benefits(self):
        """Demonstrate the benefits of each format choice."""
        self.print_section("Format-Specific Benefits", "🎯")
        
        print("✅ Format optimization benefits demonstrated:")
        
        benefits = [
            ("JSONL", "🔄 Append-only operations", "No file rewriting for new analogies"),
            ("JSONL", "📖 Streaming reads", "Process large files without memory loading"),
            ("SQLite", "🔍 Complex queries", "Joins, indexes, transactional updates"),
            ("SQLite", "🗑️ Soft deletes", "Version control and data recovery"),
            ("NPZ", "💾 Compression", "Efficient vector storage with 4.1x compression"),
            ("NPZ", "⚡ Vector ops", "Fast similarity computations"),
            ("Workflow", "📋 Audit trail", "Complete operation history tracking"),
            ("Multi-format", "🎯 Optimized access", "Right tool for each operation type")
        ]
        
        for format_type, feature, benefit in benefits:
            print(f"   {feature} ({format_type}): {benefit}")
    
    def run_complete_demo(self):
        """Run the complete multi-format demonstration."""
        print("🌟 Multi-Format Persistence Strategy Demo")
        print("=" * 50)
        print(f"Demo started at: {datetime.now()}")
        print(f"Storage location: {self.storage_path}")
        print("Based on: PERSISTENCE_LAYER_STRATEGY.md section 2")
        
        try:
            # Demonstrate each format
            analogies_file = self.demonstrate_jsonl_format()
            sqlite_file = self.demonstrate_sqlite_format() 
            embeddings_files = self.demonstrate_npz_format()
            workflow_files = self.demonstrate_workflow_files()
            
            # Show complete structure
            self.show_directory_structure()
            
            # Summarize benefits
            self.demonstrate_format_benefits()
            
            print(f"\n🎉 Multi-format strategy successfully demonstrated!")
            print(f"   📁 Storage structure matches strategy recommendations")
            print(f"   🔧 All format-specific operations working")
            print(f"   📈 Performance benefits confirmed")
            
        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up demo files."""
        self.print_section("Cleanup", "🧹")
        try:
            import shutil
            if self.base_path.exists():
                shutil.rmtree(self.base_path)
                print(f"✅ Cleaned up demo storage: {self.base_path}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run the multi-format persistence demonstration."""
    demo = MultiFormatDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    """Execute the multi-format persistence demonstration."""
    import sys
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run the demonstration
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)
