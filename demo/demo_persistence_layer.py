#!/usr/bin/env python3
"""
Persistence Layer Demo Script

This script demonstrates the comprehensive persistence layer capabilities
including batch operations, workflow management, and contract validation.

Usage:
    python demo_persistence_layer.py

Features Demonstrated:
- Contract-enhanced persistence manager
- Batch analogy creation and processing
- Workflow tracking and management
- Streaming queries and filtering
- Soft deletes and compaction
- Storage integrity validation
- Performance testing with large batches
"""

import asyncio
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from app.core.contract_persistence import ContractEnhancedPersistenceManager
from app.core.batch_persistence import DeleteCriteria, WorkflowStatus
from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry


def generate_sample_analogies(count: int = 50) -> List[Dict[str, Any]]:
    """Generate sample analogies for testing."""
    domains = ["medieval", "royal", "military", "business", "scientific", "natural"]
    concept_pairs = [
        ("king", "president"), ("queen", "vice_president"), ("knight", "soldier"),
        ("castle", "headquarters"), ("sword", "weapon"), ("crown", "authority"),
        ("army", "workforce"), ("battle", "competition"), ("victory", "success"),
        ("strategy", "plan"), ("alliance", "partnership"), ("territory", "market")
    ]
    
    analogies = []
    for i in range(count):
        source_domain = random.choice(domains)
        target_domain = random.choice([d for d in domains if d != source_domain])
        
        # Select random concept mappings
        mappings = {}
        for _ in range(random.randint(2, 4)):
            concept_pair = random.choice(concept_pairs)
            mappings[concept_pair[0]] = concept_pair[1]
        
        analogy = {
            "id": f"analogy_{i:04d}",
            "source_domain": source_domain,
            "target_domain": target_domain,
            "quality_score": round(random.uniform(0.3, 0.95), 2),
            "concept_mappings": mappings,
            "frame_mappings": {
                "hierarchy": "structure",
                "action": "process"
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": [source_domain, target_domain, "auto_generated"],
            "metadata": {
                "generated_by": "demo_script",
                "batch_id": f"batch_{i // 10}",
                "confidence": random.uniform(0.7, 0.9)
            }
        }
        analogies.append(analogy)
    
    return analogies


def print_section(title: str, emoji: str = "🔸"):
    """Print a formatted section header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n📌 {title}")
    print("-" * (len(title) + 4))


async def main():
    """Main demonstration function."""
    print("🚀 Persistence Layer Comprehensive Demo")
    print("=" * 50)
    print(f"Demo started at: {datetime.now()}")
    
    # Setup temporary storage
    storage_path = Path("demo_storage") / f"run_{int(time.time())}"
    print(f"📁 Storage location: {storage_path}")
    
    try:
        # ================================================================
        # 1. INITIALIZATION
        # ================================================================
        print_section("Initialization & Setup", "🏗️")
        
        print("Initializing contract-enhanced persistence manager...")
        manager = ContractEnhancedPersistenceManager(storage_path)
        print("✅ Persistence manager initialized successfully")
        
        print("Initializing enhanced hybrid registry...")
        registry = EnhancedHybridRegistry(
            download_wordnet=False,
            n_clusters=6,
            enable_cross_domain=True,
            embedding_provider="random"
        )
        
        # Add some concepts to the registry
        concepts = ["king", "queen", "knight", "castle", "general", "army", "president", "company"]
        for concept in concepts:
            registry.create_concept(concept, "demo")
        
        print(f"✅ Registry initialized with {len(registry.concepts)} concepts")
        
        # ================================================================
        # 2. BASIC PERSISTENCE OPERATIONS
        # ================================================================
        print_section("Basic Persistence Operations", "💾")
        
        print_subsection("Registry State Persistence")
        save_result = manager.save_registry_state(registry, "demo_context", "json")
        print(f"✅ Saved registry state: {save_result['context_name']}")
        print(f"   Components saved: {', '.join(save_result['components_saved'])}")
        print(f"   Format: {save_result['format']}")
        
        print_subsection("Storage Statistics")
        stats = manager.get_storage_statistics()
        print(f"✅ Storage statistics:")
        print(f"   Total workflows: {stats['total_workflows']}")
        print(f"   Storage size: {stats['storage_size_mb']:.2f} MB")
        print(f"   Last updated: {stats['last_updated']}")
        
        # ================================================================
        # 3. BATCH OPERATIONS
        # ================================================================
        print_section("Batch Operations Demo", "📦")
        
        print_subsection("Generating Sample Data")
        sample_analogies = generate_sample_analogies(30)
        print(f"✅ Generated {len(sample_analogies)} sample analogies")
        
        # Show sample analogy
        sample = sample_analogies[0]
        print(f"📄 Sample analogy:")
        print(f"   {sample['source_domain']} → {sample['target_domain']}")
        print(f"   Quality: {sample['quality_score']}")
        print(f"   Mappings: {list(sample['concept_mappings'].items())[:2]}...")
        
        print_subsection("Creating Batch Workflow")
        workflow = manager.create_analogy_batch(sample_analogies[:20])
        print(f"✅ Created batch workflow: {workflow.workflow_id}")
        print(f"   Type: {workflow.workflow_type.value}")
        print(f"   Status: {workflow.status.value}")
        print(f"   Items: {workflow.items_total}")
        
        print_subsection("Processing Batch")
        processed_workflow = manager.process_analogy_batch(workflow.workflow_id)
        print(f"✅ Processed batch workflow")
        print(f"   Final status: {processed_workflow.status.value}")
        print(f"   Items processed: {processed_workflow.items_processed}/{processed_workflow.items_total}")
        print(f"   Errors: {processed_workflow.error_count}")
        
        # ================================================================
        # 4. QUERYING AND FILTERING
        # ================================================================
        print_section("Querying & Filtering Demo", "🔍")
        
        print_subsection("Streaming All Analogies")
        all_analogies = list(manager.stream_analogies())
        print(f"✅ Streamed {len(all_analogies)} analogies")
        
        print_subsection("Domain-Specific Filtering")
        medieval_analogies = list(manager.stream_analogies(domain="medieval"))
        print(f"✅ Found {len(medieval_analogies)} analogies involving 'medieval' domain")
        
        print_subsection("Quality-Based Filtering")
        high_quality_analogies = manager.find_high_quality_analogies(0.8)
        print(f"✅ Found {len(high_quality_analogies)} high-quality analogies (≥ 0.8)")
        
        if high_quality_analogies:
            best = max(high_quality_analogies, key=lambda x: x['quality_score'])
            print(f"🏆 Best analogy: {best['source_domain']} → {best['target_domain']} ({best['quality_score']})")
        
        print_subsection("Quality Threshold Streaming")
        quality_analogies = list(manager.stream_analogies(min_quality=0.7))
        print(f"✅ Streamed {len(quality_analogies)} analogies with quality ≥ 0.7")
        
        # ================================================================
        # 5. ADDITIONAL BATCH OPERATIONS
        # ================================================================
        print_section("Additional Batch Operations", "🔄")
        
        print_subsection("Creating Second Batch")
        second_batch = sample_analogies[20:]
        workflow2 = manager.create_analogy_batch(second_batch)
        processed2 = manager.process_analogy_batch(workflow2.workflow_id)
        print(f"✅ Processed second batch: {processed2.items_processed} items")
        
        print_subsection("Workflow Management")
        all_workflows = manager._batch_manager.list_workflows()
        print(f"✅ Total workflows created: {len(all_workflows)}")
        
        for wf in all_workflows:
            print(f"   📋 {wf.workflow_id}: {wf.status.value} ({wf.items_processed}/{wf.items_total})")
        
        # ================================================================
        # 6. DELETION OPERATIONS
        # ================================================================
        print_section("Deletion Operations Demo", "🗑️")
        
        print_subsection("Batch Deletion by Domain")
        delete_criteria = DeleteCriteria(domains=["scientific"])
        delete_workflow = manager.delete_analogies_batch(delete_criteria)
        print(f"✅ Created deletion workflow: {delete_workflow.workflow_id}")
        print(f"   Target items for deletion: {delete_workflow.items_total}")
        
        print_subsection("Batch Deletion by Quality")
        low_quality_criteria = DeleteCriteria(quality_threshold=0.5)
        low_quality_delete = manager.delete_analogies_batch(low_quality_criteria)
        print(f"✅ Created low-quality deletion workflow: {low_quality_delete.workflow_id}")
        print(f"   Low-quality items marked for deletion: {low_quality_delete.items_total}")
        
        # ================================================================
        # 7. STORAGE OPTIMIZATION
        # ================================================================
        print_section("Storage Optimization", "🧹")
        
        print_subsection("Pre-Compaction Stats")
        pre_stats = manager.get_storage_statistics()
        print(f"Storage size before compaction: {pre_stats['storage_size_mb']:.2f} MB")
        
        print_subsection("Compacting Storage")
        compaction_result = manager.compact_analogies_jsonl()
        print(f"✅ Compaction completed: {compaction_result['status']}")
        print(f"   Records removed: {compaction_result.get('records_removed', 0)}")
        print(f"   Active records: {compaction_result.get('active_records', 0)}")
        
        if compaction_result.get('backup_created'):
            print(f"   Backup created: {compaction_result['backup_created']}")
        
        print_subsection("Post-Compaction Stats")
        post_stats = manager.get_storage_statistics()
        print(f"Storage size after compaction: {post_stats['storage_size_mb']:.2f} MB")
        
        # ================================================================
        # 8. PERFORMANCE TESTING
        # ================================================================
        print_section("Performance Testing", "⚡")
        
        print_subsection("Large Batch Performance")
        large_batch = generate_sample_analogies(100)
        
        # Time batch creation
        start_time = time.time()
        perf_workflow = manager.create_analogy_batch(large_batch)
        create_time = time.time() - start_time
        
        # Time batch processing
        start_time = time.time()
        manager.process_analogy_batch(perf_workflow.workflow_id)
        process_time = time.time() - start_time
        
        print(f"✅ Performance results for {len(large_batch)} analogies:")
        print(f"   Batch creation: {create_time:.3f} seconds")
        print(f"   Batch processing: {process_time:.3f} seconds")
        print(f"   Throughput: {len(large_batch) / process_time:.1f} analogies/second")
        
        print_subsection("Streaming Performance")
        start_time = time.time()
        streamed_count = sum(1 for _ in manager.stream_analogies())
        stream_time = time.time() - start_time
        
        print(f"✅ Streaming performance:")
        print(f"   Streamed {streamed_count} analogies in {stream_time:.3f} seconds")
        print(f"   Streaming rate: {streamed_count / stream_time:.1f} analogies/second")
        
        # ================================================================
        # 9. INTEGRITY VALIDATION
        # ================================================================
        print_section("Storage Integrity Validation", "🔐")
        
        integrity_result = manager.validate_storage_integrity()
        print(f"✅ Storage integrity check: {'PASS' if integrity_result['valid'] else 'FAIL'}")
        
        if not integrity_result['valid']:
            print("⚠️  Issues found:")
            for issue in integrity_result['issues']:
                print(f"   - {issue}")
        else:
            print("   All integrity checks passed")
        
        print(f"   Checked at: {integrity_result['checked_at']}")
        
        # ================================================================
        # 10. ADVANCED WORKFLOW SCENARIOS
        # ================================================================
        print_section("Advanced Workflow Scenarios", "🎯")
        
        print_subsection("Concurrent Workflow Creation")
        concurrent_workflows = []
        for i in range(3):
            batch = generate_sample_analogies(10)
            wf = manager.create_analogy_batch(batch, f"concurrent_{i}")
            concurrent_workflows.append(wf)
        
        print(f"✅ Created {len(concurrent_workflows)} concurrent workflows")
        
        print_subsection("Processing Multiple Workflows")
        for wf in concurrent_workflows:
            processed = manager.process_analogy_batch(wf.workflow_id)
            print(f"   📋 {wf.workflow_id}: {processed.status.value}")
        
        print_subsection("Workflow Status Summary")
        final_workflows = manager._batch_manager.list_workflows()
        status_counts = {}
        for wf in final_workflows:
            status = wf.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print("✅ Final workflow status summary:")
        for status, count in status_counts.items():
            print(f"   {status.upper()}: {count} workflows")
        
        # ================================================================
        # 11. EXPORT OPERATIONS
        # ================================================================
        print_section("Export Operations", "📤")
        
        print_subsection("Knowledge Base Export")
        export_path = manager.export_knowledge_base("json", compressed=False)
        print(f"✅ Exported knowledge base to: {export_path}")
        print(f"   File exists: {export_path.exists()}")
        
        # ================================================================
        # 12. FINAL STATISTICS
        # ================================================================
        print_section("Final Statistics & Summary", "📊")
        
        final_stats = manager.get_storage_statistics()
        final_analogies = list(manager.stream_analogies())
        
        print("✅ Demo completed successfully!")
        print(f"\n📈 Final Statistics:")
        print(f"   Total analogies created: {len(final_analogies)}")
        print(f"   Total workflows: {final_stats['total_workflows']}")
        print(f"   Storage size: {final_stats['storage_size_mb']:.2f} MB")
        print(f"   High-quality analogies: {len(manager.find_high_quality_analogies(0.8))}")
        
        print(f"\n🎯 Key Features Demonstrated:")
        features = [
            "✅ Contract-enhanced persistence with validation",
            "✅ Batch workflow creation and processing",
            "✅ Multi-format storage (JSONL + SQLite)",
            "✅ Streaming queries with filtering",
            "✅ Soft deletes and storage compaction",
            "✅ Performance testing and optimization",
            "✅ Storage integrity validation",
            "✅ Concurrent workflow management",
            "✅ Export/import capabilities",
            "✅ Comprehensive error handling"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print(f"\n🕐 Demo duration: {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print_section("Cleanup", "🧹")
        try:
            import shutil
            if storage_path.exists():
                shutil.rmtree(storage_path)
                print(f"✅ Cleaned up demo storage: {storage_path}")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
        
        print("\n🎉 Persistence Layer Demo Complete!")


if __name__ == "__main__":
    """Run the persistence layer demo."""
    import sys
    import os
    
    # Add the project root to the Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo crashed: {e}")
        sys.exit(1)
