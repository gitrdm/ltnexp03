#!/usr/bin/env python3
"""
Demonstration of Hybrid FrameNet + Clustering System

This script showcases the advanced capabilities of the hybrid concept registry
that combines basic concept management with semantic frames and clustering.
"""

import numpy as np
import sys
from pathlib import Path

# Add the app directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from app.core.hybrid_registry import HybridConceptRegistry
from app.core.frame_cluster_abstractions import FrameElementType, FrameElement


def create_sample_embeddings():
    """Create sample embeddings for demonstration."""
    # Simple embeddings based on semantic categories
    np.random.seed(42)
    
    embeddings = {}
    
    # Animals cluster
    animal_base = np.array([1.0, 0.0, 0.0, 0.5, 0.3] + [0.0] * 295)
    embeddings["dog"] = animal_base + np.random.normal(0, 0.1, 300)
    embeddings["cat"] = animal_base + np.random.normal(0, 0.1, 300)
    embeddings["lion"] = animal_base + np.random.normal(0, 0.1, 300)
    
    # People cluster
    people_base = np.array([0.0, 1.0, 0.0, 0.8, 0.9] + [0.0] * 295)
    embeddings["king"] = people_base + np.random.normal(0, 0.1, 300)
    embeddings["queen"] = people_base + np.random.normal(0, 0.1, 300)
    embeddings["man"] = people_base + np.random.normal(0, 0.1, 300)
    embeddings["woman"] = people_base + np.random.normal(0, 0.1, 300)
    
    # Vehicles cluster
    vehicle_base = np.array([0.0, 0.0, 1.0, 0.2, 0.1] + [0.0] * 295)
    embeddings["car"] = vehicle_base + np.random.normal(0, 0.1, 300)
    embeddings["bike"] = vehicle_base + np.random.normal(0, 0.1, 300)
    embeddings["plane"] = vehicle_base + np.random.normal(0, 0.1, 300)
    
    # Money/Transaction cluster
    money_base = np.array([0.0, 0.0, 0.0, 1.0, 0.5] + [0.0] * 295)
    embeddings["money"] = money_base + np.random.normal(0, 0.1, 300)
    embeddings["dollar"] = money_base + np.random.normal(0, 0.1, 300)
    embeddings["price"] = money_base + np.random.normal(0, 0.1, 300)
    
    return embeddings


def demonstrate_basic_functionality():
    """Demonstrate basic hybrid registry functionality."""
    print("=" * 60)
    print("HYBRID CONCEPT REGISTRY DEMONSTRATION")
    print("=" * 60)
    
    # Create hybrid registry
    print("\n1. Creating hybrid concept registry...")
    registry = HybridConceptRegistry(download_wordnet=False, n_clusters=5)
    
    # Create sample embeddings
    embeddings = create_sample_embeddings()
    
    # Create frame-aware concepts with embeddings
    print("\n2. Creating frame-aware concepts with embeddings...")
    concepts = {}
    for name, embedding in embeddings.items():
        concept = registry.create_frame_aware_concept(name, embedding=embedding)
        concepts[name] = concept
        print(f"   Created: {concept.unique_id}")
    
    # Update clusters
    print("\n3. Training concept clusters...")
    registry.update_clusters()
    
    # Show cluster memberships
    print("\n4. Cluster memberships:")
    for name, concept in concepts.items():
        top_clusters = concept.get_primary_clusters(top_k=2)
        print(f"   {name}: {top_clusters}")
    
    return registry, concepts


def demonstrate_frame_creation():
    """Demonstrate semantic frame creation and usage."""
    registry, concepts = demonstrate_basic_functionality()
    
    print("\n" + "=" * 60)
    print("SEMANTIC FRAME DEMONSTRATION")
    print("=" * 60)
    
    # Create a commercial transaction frame
    print("\n1. Creating Commercial_transaction frame...")
    commercial_frame = registry.create_semantic_frame(
        name="Commercial_transaction",
        definition="A transaction involving the exchange of goods for money",
        core_elements=["Buyer", "Seller", "Goods", "Money"],
        peripheral_elements=["Time", "Place"],
        lexical_units=["buy", "sell", "purchase", "trade"]
    )
    print(f"   Created frame: {commercial_frame.name}")
    print(f"   Core elements: {[elem.name for elem in commercial_frame.core_elements]}")
    
    # Create a gift-giving frame
    print("\n2. Creating Gift_giving frame...")
    gift_frame = registry.create_semantic_frame(
        name="Gift_giving",
        definition="An event where someone gives something to another person",
        core_elements=["Giver", "Recipient", "Gift"],
        peripheral_elements=["Occasion", "Time", "Place"],
        lexical_units=["give", "present", "donate"]
    )
    print(f"   Created frame: {gift_frame.name}")
    
    # Create frame instances
    print("\n3. Creating frame instances...")
    
    # Commercial transaction: John buys a car
    transaction_instance = registry.create_frame_instance(
        frame_name="Commercial_transaction",
        instance_id="john_buys_car",
        concept_bindings={
            "Buyer": concepts["man"],    # John (represented as man)
            "Seller": concepts["woman"], # Mary (represented as woman)
            "Goods": concepts["car"],
            "Money": concepts["dollar"]
        }
    )
    print(f"   Created instance: {transaction_instance.instance_id}")
    
    # Gift giving: Queen gives gift
    gift_instance = registry.create_frame_instance(
        frame_name="Gift_giving", 
        instance_id="queen_gives_gift",
        concept_bindings={
            "Giver": concepts["queen"],
            "Recipient": concepts["king"],
            "Gift": concepts["money"]
        }
    )
    print(f"   Created instance: {gift_instance.instance_id}")
    
    return registry, concepts


def demonstrate_analogical_reasoning():
    """Demonstrate analogical reasoning capabilities."""
    registry, concepts = demonstrate_frame_creation()
    
    print("\n" + "=" * 60)
    print("ANALOGICAL REASONING DEMONSTRATION")
    print("=" * 60)
    
    # Find analogous concepts
    print("\n1. Finding concepts analogous to 'king'...")
    analogies = registry.find_analogous_concepts(
        concepts["king"], 
        cluster_threshold=0.5,
        frame_threshold=0.5
    )
    
    for concept, score, basis in analogies[:3]:  # Top 3
        print(f"   {concept.name}: similarity={score:.3f}, basis={basis}")
    
    # Find cluster-based similarities
    print("\n2. Finding cluster-based similar concepts to 'car'...")
    similar_concepts = registry.cluster_registry.find_similar_concepts(
        concepts["car"].unique_id, threshold=0.6
    )
    
    for concept_id, similarity in similar_concepts[:3]:  # Top 3
        concept_name = concept_id.split(":")[-1] if ":" in concept_id else concept_id
        print(f"   {concept_name}: similarity={similarity:.3f}")
    
    # Frame-based analogies
    print("\n3. Finding frame instances analogous to 'john_buys_car'...")
    frame_analogies = registry.frame_registry.find_analogous_instances(
        "john_buys_car", threshold=0.3
    )
    
    for analogy in frame_analogies:
        print(f"   {analogy.target_instance}: structural={analogy.structural_similarity:.3f}, "
              f"semantic={analogy.semantic_similarity:.3f}, "
              f"overall={analogy.overall_quality:.3f}")
    
    return registry


def demonstrate_statistics():
    """Show comprehensive registry statistics."""
    registry = demonstrate_analogical_reasoning()
    
    print("\n" + "=" * 60)
    print("SYSTEM STATISTICS")
    print("=" * 60)
    
    stats = registry.get_hybrid_statistics()
    
    print(f"\nBasic Concepts:")
    print(f"   Total concepts: {stats['total_concepts']}")
    print(f"   Frame-aware concepts: {stats['frame_aware_concepts']}")
    print(f"   Contexts: {stats['contexts']}")
    print(f"   With synsets: {stats['with_synsets']}")
    
    print(f"\nSemantic Frames:")
    print(f"   Semantic frames: {stats['semantic_frames']}")
    print(f"   Frame instances: {stats['frame_instances']}")
    
    print(f"\nClustering:")
    print(f"   Concept clusters: {stats['concept_clusters']}")
    print(f"   Concepts with embeddings: {stats['concepts_with_embeddings']}")
    print(f"   Clustering trained: {stats['clustering_trained']}")
    
    if stats['clustering_trained']:
        print(f"   Average cluster size: {stats['avg_cluster_size']:.1f}")
        print(f"   Max cluster size: {stats['max_cluster_size']}")
        print(f"   Min cluster size: {stats['min_cluster_size']}")


def demonstrate_advanced_analogies():
    """Demonstrate advanced analogical reasoning patterns."""
    print("\n" + "=" * 60)
    print("ADVANCED ANALOGY PATTERNS")
    print("=" * 60)
    
    registry = HybridConceptRegistry(download_wordnet=False, n_clusters=3)
    
    # Create a more complex set of concepts with rich embeddings
    print("\n1. Creating rich concept set...")
    
    # Monarchy concepts
    king_base = np.array([1, 0, 0, 1, 0.8] + [0.5] * 295)
    king_emb = king_base + np.random.normal(0, 0.1, 300)
    king_emb = np.clip(king_emb, 0, 1)
    
    queen_base = np.array([1, 0, 0, 0.9, 0.8] + [0.5] * 295)
    queen_emb = queen_base + np.random.normal(0, 0.1, 300)
    queen_emb = np.clip(queen_emb, 0, 1)
    
    prince_base = np.array([0.8, 0, 0, 0.7, 0.6] + [0.4] * 295)
    prince_emb = prince_base + np.random.normal(0, 0.1, 300)
    prince_emb = np.clip(prince_emb, 0, 1)
    
    # Animals (for analogical reasoning)
    lion_base = np.array([0, 1, 0, 1, 0.9] + [0.3] * 295)
    lion_emb = lion_base + np.random.normal(0, 0.1, 300)
    lion_emb = np.clip(lion_emb, 0, 1)
    
    lioness_base = np.array([0, 1, 0, 0.9, 0.9] + [0.3] * 295)
    lioness_emb = lioness_base + np.random.normal(0, 0.1, 300)
    lioness_emb = np.clip(lioness_emb, 0, 1)
    
    cub_base = np.array([0, 0.8, 0, 0.7, 0.7] + [0.2] * 295)
    cub_emb = cub_base + np.random.normal(0, 0.1, 300)
    cub_emb = np.clip(cub_emb, 0, 1)
    
    concepts = {}
    concept_embeddings = {
        "king": king_emb, "queen": queen_emb, "prince": prince_emb,
        "lion": lion_emb, "lioness": lioness_emb, "cub": cub_emb
    }
    
    for name, embedding in concept_embeddings.items():
        concept = registry.create_frame_aware_concept(name, embedding=embedding)
        concepts[name] = concept
    
    # Create domain-specific frames
    print("\n2. Creating domain frames...")
    
    # Monarchy frame
    monarchy_frame = registry.create_semantic_frame(
        name="Monarchy",
        definition="Royal family structure and hierarchy",
        core_elements=["Ruler", "Consort", "Heir"],
        peripheral_elements=["Kingdom", "Crown"]
    )
    
    # Animal family frame
    animal_frame = registry.create_semantic_frame(
        name="Animal_family",
        definition="Animal family structure and relationships",
        core_elements=["Alpha_male", "Alpha_female", "Offspring"],
        peripheral_elements=["Territory", "Pack"]
    )
    
    # Create frame instances
    print("\n3. Creating analogical frame instances...")
    
    # Royal family instance
    royal_instance = registry.create_frame_instance(
        frame_name="Monarchy",
        instance_id="british_royalty",
        concept_bindings={
            "Ruler": concepts["king"],
            "Consort": concepts["queen"], 
            "Heir": concepts["prince"]
        }
    )
    
    # Lion pride instance
    pride_instance = registry.create_frame_instance(
        frame_name="Animal_family",
        instance_id="lion_pride",
        concept_bindings={
            "Alpha_male": concepts["lion"],
            "Alpha_female": concepts["lioness"],
            "Offspring": concepts["cub"]
        }
    )
    
    registry.update_clusters()
    
    print("\n4. Cross-domain analogical reasoning:")
    print("   King:Queen :: Lion:Lioness")
    
    # Find analogies between domains
    king_analogies = registry.find_analogous_concepts(
        concepts["king"], cluster_threshold=0.4, frame_threshold=0.3
    )
    
    for concept, score, basis in king_analogies:
        if concept.name in ["lion", "lioness", "cub"]:
            print(f"   {concepts['king'].name} -> {concept.name}: "
                  f"similarity={score:.3f}, basis={basis}")
    
    print("\n   Cross-frame instance analogies:")
    frame_analogies = registry.frame_registry.find_analogous_instances(
        "british_royalty", threshold=0.1
    )
    
    for analogy in frame_analogies:
        print(f"   {analogy.source_instance} -> {analogy.target_instance}: "
              f"quality={analogy.overall_quality:.3f}")


if __name__ == "__main__":
    try:
        demonstrate_basic_functionality()
        demonstrate_frame_creation()
        demonstrate_analogical_reasoning()
        demonstrate_statistics()
        demonstrate_advanced_analogies()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nThe hybrid system successfully demonstrates:")
        print("• Frame-aware concept management")
        print("• Clustering-based similarity")
        print("• Multi-level analogical reasoning") 
        print("• Cross-domain concept mapping")
        print("• Integrated semantic understanding")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
