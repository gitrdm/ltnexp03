#!/usr/bin/env python3
"""
Enhanced Hybrid System Demonstration

This script demonstrates the advanced capabilities of the enhanced hybrid registry,
including semantic field discovery, cross-domain analogy discovery, and analogical
completion tasks.
"""

import numpy as np
import sys
from pathlib import Path

# Add the app directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry


def create_rich_embeddings():
    """Create rich embeddings for different semantic domains."""
    np.random.seed(42)
    
    embeddings = {}
    
    # ANIMAL DOMAIN
    animal_base = np.array([1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 295)
    embeddings["lion"] = animal_base + np.random.normal(0, 0.05, 300)
    embeddings["tiger"] = animal_base + np.random.normal(0, 0.05, 300)
    embeddings["wolf"] = animal_base + np.random.normal(0, 0.05, 300)
    embeddings["eagle"] = animal_base + np.random.normal(0, 0.05, 300)
    embeddings["shark"] = animal_base + np.random.normal(0, 0.05, 300)
    
    # ROYALTY/LEADERSHIP DOMAIN
    royalty_base = np.array([0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 295)
    embeddings["king"] = royalty_base + np.random.normal(0, 0.05, 300)
    embeddings["queen"] = royalty_base + np.random.normal(0, 0.05, 300)
    embeddings["prince"] = royalty_base + np.random.normal(0, 0.05, 300)
    embeddings["princess"] = royalty_base + np.random.normal(0, 0.05, 300)
    embeddings["emperor"] = royalty_base + np.random.normal(0, 0.05, 300)
    
    # MILITARY DOMAIN
    military_base = np.array([0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 295)
    embeddings["general"] = military_base + np.random.normal(0, 0.05, 300)
    embeddings["colonel"] = military_base + np.random.normal(0, 0.05, 300)
    embeddings["captain"] = military_base + np.random.normal(0, 0.05, 300)
    embeddings["sergeant"] = military_base + np.random.normal(0, 0.05, 300)
    embeddings["soldier"] = military_base + np.random.normal(0, 0.05, 300)
    
    # BUSINESS DOMAIN
    business_base = np.array([0.0, 0.0, 0.0, 1.0, 0.0] + [0.0] * 295)
    embeddings["ceo"] = business_base + np.random.normal(0, 0.05, 300)
    embeddings["manager"] = business_base + np.random.normal(0, 0.05, 300)
    embeddings["director"] = business_base + np.random.normal(0, 0.05, 300)
    embeddings["supervisor"] = business_base + np.random.normal(0, 0.05, 300)
    embeddings["employee"] = business_base + np.random.normal(0, 0.05, 300)
    
    # SPORTS DOMAIN
    sports_base = np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 295)
    embeddings["quarterback"] = sports_base + np.random.normal(0, 0.05, 300)
    embeddings["striker"] = sports_base + np.random.normal(0, 0.05, 300)
    embeddings["captain_sports"] = sports_base + np.random.normal(0, 0.05, 300)
    embeddings["coach"] = sports_base + np.random.normal(0, 0.05, 300)
    embeddings["player"] = sports_base + np.random.normal(0, 0.05, 300)
    
    return embeddings


def demonstrate_enhanced_system():
    """Demonstrate the enhanced hybrid registry capabilities."""
    print("=" * 80)
    print("ENHANCED HYBRID SEMANTIC REASONING SYSTEM")
    print("=" * 80)
    
    # Create enhanced registry
    print("\n1. Creating Enhanced Hybrid Registry...")
    registry = EnhancedHybridRegistry(
        download_wordnet=False, 
        n_clusters=8, 
        enable_cross_domain=True,
        embedding_provider="semantic"  # Use semantic embeddings
    )
    
    # Create rich concept set with embeddings
    print("\n2. Creating Rich Concept Set...")
    embeddings = create_rich_embeddings()
    
    concepts = {}
    contexts = {
        "animal": ["lion", "tiger", "wolf", "eagle", "shark"],
        "royalty": ["king", "queen", "prince", "princess", "emperor"],
        "military": ["general", "colonel", "captain", "sergeant", "soldier"],
        "business": ["ceo", "manager", "director", "supervisor", "employee"],
        "sports": ["quarterback", "striker", "captain_sports", "coach", "player"]
    }
    
    for context, concept_names in contexts.items():
        for name in concept_names:
            concept = registry.create_frame_aware_concept_with_advanced_embedding(
                name=name,
                context=context,
                use_semantic_embedding=True
            )
            concepts[name] = concept
            print(f"   Created: {concept.unique_id}")
    
    # Update clusters
    print("\n3. Training Concept Clusters...")
    registry.update_clusters()
    
    return registry, concepts


def demonstrate_semantic_fields():
    """Demonstrate semantic field discovery."""
    registry, concepts = demonstrate_enhanced_system()
    
    print("\n" + "=" * 80)
    print("SEMANTIC FIELD DISCOVERY")
    print("=" * 80)
    
    # Discover semantic fields
    print("\n1. Discovering Semantic Fields...")
    fields = registry.discover_semantic_fields(min_coherence=0.3)  # Lower threshold
    
    print(f"\n2. Discovered {len(fields)} semantic fields:")
    for field in fields:
        print(f"\n   Field: {field['name']}")
        print(f"   Description: {field['description']}")
        print(f"   Core concepts: {list(field['core_concepts'])}")
        print(f"   Related concepts: {len(field['related_concepts'])}")
        print(f"   Associated frames: {list(field['associated_frames'])}")
    
    return registry, concepts, fields


def demonstrate_cross_domain_analogies():
    """Demonstrate cross-domain analogy discovery."""
    registry, concepts, fields = demonstrate_semantic_fields()
    
    print("\n" + "=" * 80)
    print("CROSS-DOMAIN ANALOGY DISCOVERY")
    print("=" * 80)
    
    # Create semantic frames for different domains
    print("\n1. Creating Domain-Specific Frames...")
    
    # Hierarchy frame for royalty
    royalty_frame = registry.create_semantic_frame(
        name="Royalty_Hierarchy",
        definition="Hierarchical structure in royal systems",
        core_elements=["Ruler", "Heir", "Subject"],
        peripheral_elements=["Territory", "Authority"],
        lexical_units=["rule", "reign", "govern"]
    )
    
    # Hierarchy frame for military
    military_frame = registry.create_semantic_frame(
        name="Military_Hierarchy",
        definition="Command structure in military organizations",
        core_elements=["Commander", "Subordinate", "Unit"],
        peripheral_elements=["Mission", "Authority"],
        lexical_units=["command", "order", "deploy"]
    )
    
    # Hierarchy frame for business
    business_frame = registry.create_semantic_frame(
        name="Business_Hierarchy",
        definition="Organizational structure in business",
        core_elements=["Executive", "Manager", "Employee"],
        peripheral_elements=["Department", "Responsibility"],
        lexical_units=["manage", "direct", "supervise"]
    )
    
    # Create frame instances
    print("\n2. Creating Frame Instances...")
    
    # Royalty hierarchy instance
    royalty_instance = registry.create_frame_instance(
        frame_name="Royalty_Hierarchy",
        instance_id="medieval_kingdom",
        concept_bindings={
            "Ruler": "king",
            "Heir": "prince",
            "Subject": "princess"
        },
        context="royalty"
    )
    
    # Military hierarchy instance
    military_instance = registry.create_frame_instance(
        frame_name="Military_Hierarchy",
        instance_id="army_command",
        concept_bindings={
            "Commander": "general",
            "Subordinate": "colonel",
            "Unit": "soldier"
        },
        context="military"
    )
    
    # Business hierarchy instance
    business_instance = registry.create_frame_instance(
        frame_name="Business_Hierarchy",
        instance_id="corporate_structure",
        concept_bindings={
            "Executive": "ceo",
            "Manager": "director",
            "Employee": "employee"
        },
        context="business"
    )
    
    # Discover cross-domain analogies
    print("\n3. Discovering Cross-Domain Analogies...")
    analogies = registry.discover_cross_domain_analogies(min_quality=0.2)  # Lower threshold
    
    print(f"\n4. Found {len(analogies)} cross-domain analogies:")
    for i, analogy in enumerate(analogies[:5]):  # Show top 5
        print(f"\n   Analogy {i+1}: {analogy.source_domain} ↔ {analogy.target_domain}")
        print(f"   Quality: {analogy.compute_overall_quality():.3f}")
        print(f"   Structural coherence: {analogy.structural_coherence:.3f}")
        print(f"   Semantic coherence: {analogy.semantic_coherence:.3f}")
        print(f"   Productivity: {analogy.productivity:.3f}")
        
        if analogy.concept_mappings:
            print("   Concept mappings:")
            for source, target in list(analogy.concept_mappings.items())[:3]:
                print(f"     {source} ↔ {target}")
        
        if analogy.frame_mappings:
            print("   Frame mappings:")
            for source, target in analogy.frame_mappings.items():
                print(f"     {source} ↔ {target}")
    
    return registry, concepts, fields, analogies


def demonstrate_analogical_completion():
    """Demonstrate analogical completion tasks."""
    registry, concepts, fields, analogies = demonstrate_cross_domain_analogies()
    
    print("\n" + "=" * 80)
    print("ANALOGICAL COMPLETION TASKS")
    print("=" * 80)
    
    # Test analogical completion
    print("\n1. Testing Analogical Completions...")
    
    test_analogies = [
        {"king": "queen", "prince": "?"},
        {"general": "colonel", "captain": "?"},
        {"ceo": "director", "manager": "?"},
        {"lion": "tiger", "eagle": "?"},
    ]
    
    for i, partial_analogy in enumerate(test_analogies):
        print(f"\n   Test {i+1}: {partial_analogy}")
        
        completions = registry.find_analogical_completions(
            partial_analogy, 
            max_completions=3
        )
        
        print("   Possible completions:")
        for j, completion in enumerate(completions):
            missing_key = [k for k, v in partial_analogy.items() if v == "?"][0]
            suggested_value = completion[missing_key]
            print(f"     {j+1}. {missing_key} → {suggested_value}")
    
    return registry


def demonstrate_advanced_queries():
    """Demonstrate advanced querying capabilities."""
    registry = demonstrate_analogical_completion()
    
    print("\n" + "=" * 80)
    print("ADVANCED QUERYING CAPABILITIES")
    print("=" * 80)
    
    # Test advanced analogical queries
    print("\n1. Finding Analogous Concepts...")
    
    test_concepts = ["king", "general", "ceo", "lion"]
    
    for concept_name in test_concepts:
        print(f"\n   Concepts analogous to '{concept_name}':")
        
        analogous = registry.find_analogous_concepts(
            concept_name,
            cluster_threshold=0.6,
            frame_threshold=0.5
        )
        
        for target_concept, similarity, basis in analogous[:3]:
            print(f"     {target_concept.name} (similarity: {similarity:.3f}, basis: {basis})")
    
    # Show system statistics
    print("\n2. Enhanced System Statistics:")
    stats = registry.get_enhanced_statistics()
    
    key_stats = [
        "total_concepts", "frame_aware_concepts", "semantic_frames", 
        "concept_clusters", "semantic_fields", "cross_domain_analogies",
        "avg_cluster_size", "avg_field_size", "avg_analogy_quality"
    ]
    
    for stat in key_stats:
        if stat in stats:
            print(f"   {stat}: {stats[stat]}")
    
    return registry


def main():
    """Main demonstration function."""
    print("Starting Enhanced Hybrid Semantic Reasoning Demonstration...\n")
    
    try:
        registry = demonstrate_advanced_queries()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        
        print("\nThe enhanced hybrid system successfully demonstrates:")
        print("• Rich semantic field discovery")
        print("• Cross-domain analogy identification")
        print("• Analogical completion tasks")
        print("• Advanced concept similarity queries")
        print("• Multi-level semantic reasoning")
        print("• Integration of frames, clusters, and embeddings")
        
        print(f"\nSystem contains:")
        stats = registry.get_enhanced_statistics()
        print(f"• {stats.get('frame_aware_concepts', 0)} frame-aware concepts")
        print(f"• {stats.get('semantic_frames', 0)} semantic frames")
        print(f"• {stats.get('semantic_fields', 0)} semantic fields")
        print(f"• {stats.get('cross_domain_analogies', 0)} cross-domain analogies")
        print(f"• {stats.get('concept_clusters', 0)} concept clusters")
        
        return True
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
