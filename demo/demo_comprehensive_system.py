#!/usr/bin/env python3
"""
Comprehensive Hybrid FrameNet-Clustering Demonstration

This script showcases the full power of our hybrid semantic reasoning system
with a realistic knowledge domain: Medieval Fantasy World Building.

We'll demonstrate:
1. Rich concept creation across multiple domains
2. Frame-based structural knowledge representation  
3. Clustering-based semantic organization
4. Cross-domain analogical reasoning
5. Practical applications for story generation and world building
"""

import numpy as np
import sys
from pathlib import Path

# Add the app directory to the path for imports
sys.path.append(str(Path(__file__).parent))

from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry


def create_medieval_fantasy_world():
    """Create a rich medieval fantasy knowledge base."""
    print("=" * 80)
    print("BUILDING MEDIEVAL FANTASY KNOWLEDGE BASE")
    print("=" * 80)
    
    # Create enhanced registry
    print("\n1. Creating Enhanced Semantic Registry...")
    registry = EnhancedHybridRegistry(
        download_wordnet=False,
        n_clusters=6,  # Reduced to accommodate sample size
        enable_cross_domain=True,
        embedding_provider="semantic"
    )
    
    # Define rich concept domains
    domains = {
        "royalty": {
            "concepts": ["king", "queen", "prince", "princess", "duke", "duchess", "baron", "knight"],
            "description": "Royal hierarchy and noble titles"
        },
        "magic": {
            "concepts": ["wizard", "sorcerer", "witch", "enchanter", "alchemist", "druid", "cleric", "paladin"],
            "description": "Magical practitioners and their arts"
        },
        "creatures": {
            "concepts": ["dragon", "griffin", "unicorn", "phoenix", "basilisk", "hydra", "wyvern", "pegasus"],
            "description": "Mythical and legendary creatures"
        },
        "military": {
            "concepts": ["general", "captain", "sergeant", "soldier", "archer", "cavalry", "infantry", "guard"],
            "description": "Military ranks and units"
        },
        "locations": {
            "concepts": ["castle", "tower", "dungeon", "fortress", "temple", "monastery", "village", "city"],
            "description": "Places and structures in the world"
        },
        "artifacts": {
            "concepts": ["sword", "shield", "crown", "scepter", "orb", "tome", "staff", "amulet"],
            "description": "Magical and royal artifacts"
        }
    }
    
    # Create concepts with semantic embeddings
    print("\n2. Creating Rich Concept Set...")
    all_concepts = {}
    
    # Temporarily disable auto-clustering for performance during bulk creation
    registry.auto_cluster = False
    
    for domain, info in domains.items():
        print(f"\n   Creating {domain} domain:")
        for concept_name in info["concepts"]:
            concept = registry.create_frame_aware_concept_with_advanced_embedding(
                name=concept_name,
                context=domain,
                use_semantic_embedding=True
            )
            all_concepts[f"{domain}:{concept_name}"] = concept
            print(f"     • {concept.unique_id}")
    
    # Re-enable auto-clustering and update clusters once at the end
    registry.auto_cluster = True
    print("\n   Updating clusters with all concepts...")
    registry.update_clusters()
    
    # Train clustering system
    print("\n3. Training Semantic Clustering...")
    registry.update_clusters()
    
    return registry, all_concepts, domains


def create_story_frames(registry):
    """Create narrative and world-building frames."""
    print("\n" + "=" * 80)
    print("CREATING STORY FRAMES")
    print("=" * 80)
    
    frames = []
    
    # Quest frame
    print("\n1. Creating Quest narrative frame...")
    quest_frame = registry.create_semantic_frame(
        name="Quest",
        definition="A heroic journey with hero, goal, obstacles, and resolution",
        core_elements=["Hero", "Quest_Goal", "Obstacle", "Helper", "Reward"],
        peripheral_elements=["Starting_Location", "Destination", "Duration"],
        lexical_units=["quest", "journey", "adventure", "mission"]
    )
    frames.append(quest_frame)
    
    # Magical_Conflict frame  
    print("2. Creating Magical_Conflict frame...")
    conflict_frame = registry.create_semantic_frame(
        name="Magical_Conflict",
        definition="Conflict involving magical forces and practitioners",
        core_elements=["Protagonist", "Antagonist", "Magical_Power", "Conflict_Reason"],
        peripheral_elements=["Witnesses", "Magical_Artifacts", "Consequences"],
        lexical_units=["battle", "duel", "confrontation", "curse"]
    )
    frames.append(conflict_frame)
    
    # Royal_Court frame
    print("3. Creating Royal_Court political frame...")
    court_frame = registry.create_semantic_frame(
        name="Royal_Court",
        definition="Political interactions within royal hierarchy",
        core_elements=["Ruler", "Courtier", "Political_Action", "Royal_Decision"],
        peripheral_elements=["Court_Location", "Ceremony", "Consequences"],
        lexical_units=["rule", "decree", "court", "ceremony"]
    )
    frames.append(court_frame)
    
    # Dragon_Encounter frame
    print("4. Creating Dragon_Encounter frame...")
    dragon_frame = registry.create_semantic_frame(
        name="Dragon_Encounter", 
        definition="Interactions between humans and dragons",
        core_elements=["Human", "Dragon", "Encounter_Type", "Dragon_Response"],
        peripheral_elements=["Location", "Treasure", "Outcome"],
        lexical_units=["encounter", "confront", "negotiate", "flee"]
    )
    frames.append(dragon_frame)
    
    return frames


def create_story_instances(registry, concepts):
    """Create specific story instances using the frames."""
    print("\n" + "=" * 80)  
    print("CREATING STORY INSTANCES")
    print("=" * 80)
    
    instances = []
    
    # Quest instance: Knight's Dragon Quest
    print("\n1. Creating 'Knight's Dragon Quest' instance...")
    quest_instance = registry.create_frame_instance(
        frame_name="Quest",
        instance_id="knights_dragon_quest",
        concept_bindings={
            "Hero": "knight",
            "Quest_Goal": "dragon",
            "Obstacle": "basilisk", 
            "Helper": "wizard",
            "Reward": "crown"
        },
        context="story"
    )
    instances.append(quest_instance)
    print(f"   Created: {quest_instance.instance_id}")
    
    # Magical conflict: Wizard vs Sorcerer
    print("\n2. Creating 'Wizard vs Sorcerer' magical conflict...")
    conflict_instance = registry.create_frame_instance(
        frame_name="Magical_Conflict",
        instance_id="wizard_sorcerer_duel",
        concept_bindings={
            "Protagonist": "wizard",
            "Antagonist": "sorcerer", 
            "Magical_Power": "staff",
            "Conflict_Reason": "tome"
        },
        context="story"
    )
    instances.append(conflict_instance)
    print(f"   Created: {conflict_instance.instance_id}")
    
    # Royal court: King's Decision
    print("\n3. Creating 'King's Royal Decree' court instance...")
    court_instance = registry.create_frame_instance(
        frame_name="Royal_Court",
        instance_id="kings_decree",
        concept_bindings={
            "Ruler": "king",
            "Courtier": "duke",
            "Political_Action": "scepter",
            "Royal_Decision": "crown"
        },
        context="story"
    )
    instances.append(court_instance)
    print(f"   Created: {court_instance.instance_id}")
    
    # Dragon encounter: Princess meets Dragon
    print("\n4. Creating 'Princess and Dragon' encounter...")
    dragon_instance = registry.create_frame_instance(
        frame_name="Dragon_Encounter",
        instance_id="princess_dragon_encounter", 
        concept_bindings={
            "Human": "princess",
            "Dragon": "dragon",
            "Encounter_Type": "castle",
            "Dragon_Response": "unicorn"
        },
        context="story"
    )
    instances.append(dragon_instance)
    print(f"   Created: {dragon_instance.instance_id}")
    
    return instances


def demonstrate_semantic_reasoning(registry):
    """Demonstrate advanced semantic reasoning capabilities."""
    print("\n" + "=" * 80)
    print("ADVANCED SEMANTIC REASONING")
    print("=" * 80)
    
    # Discover semantic fields
    print("\n1. Discovering Semantic Fields...")
    fields = registry.discover_semantic_fields(min_coherence=0.4)
    
    print(f"\n   Found {len(fields)} semantic fields:")
    for field in fields[:5]:  # Show top 5
        core_concepts = list(field.core_concepts)[:3]
        related_count = len(field.related_concepts)
        print(f"   • {field.name}: {core_concepts} + {related_count} related")
    
    # Cross-domain analogies
    print("\n2. Discovering Cross-Domain Analogies...")
    analogies = registry.discover_cross_domain_analogies(min_quality=0.3)
    
    print(f"\n   Found {len(analogies)} cross-domain analogies:")
    for analogy in analogies[:3]:  # Show top 3
        quality = analogy.compute_overall_quality()
        print(f"   • {analogy.source_domain} ↔ {analogy.target_domain} (quality: {quality:.3f})")
        
        if analogy.concept_mappings:
            mappings = list(analogy.concept_mappings.items())[:2]
            for source, target in mappings:
                source_name = source.split(':')[-1] if ':' in source else source
                target_name = target.split(':')[-1] if ':' in target else target
                print(f"     - {source_name} ↔ {target_name}")
    
    # Analogical completion
    print("\n3. Testing Analogical Completion...")
    test_analogies = [
        {"king": "queen", "prince": "?"},
        {"wizard": "staff", "knight": "?"},
        {"dragon": "castle", "unicorn": "?"},
    ]
    
    for analogy in test_analogies:
        print(f"\n   Testing: {analogy}")
        completions = registry.find_analogical_completions(analogy, max_completions=2)
        
        for completion in completions:
            missing_key = [k for k, v in analogy.items() if v == "?"][0]
            suggestion = completion[missing_key].split(':')[-1] if ':' in completion[missing_key] else completion[missing_key]
            print(f"     → {missing_key}: {suggestion}")


def demonstrate_story_applications(registry):
    """Demonstrate practical story generation applications."""
    print("\n" + "=" * 80)
    print("STORY GENERATION APPLICATIONS")
    print("=" * 80)
    
    print("\n1. Finding Character Relationships...")
    
    # Find concepts similar to key character types
    character_types = ["knight", "wizard", "dragon", "princess"]
    
    for char_type in character_types:
        print(f"\n   Characters similar to '{char_type}':")
        similar = registry.find_analogous_concepts(
            f"royalty:{char_type}" if char_type in ["knight", "princess"] else f"magic:{char_type}" if char_type == "wizard" else f"creatures:{char_type}",
            cluster_threshold=0.5,
            frame_threshold=0.4
        )
        
        for concept, similarity, basis in similar[:3]:
            name = concept.name
            print(f"     • {name} (similarity: {similarity:.3f}, basis: {basis})")
    
    print("\n2. Generating Story Variants...")
    
    # Use frame instances to generate story variants
    base_story = {
        "hero": "knight",
        "quest_object": "dragon", 
        "helper": "wizard",
        "obstacle": "basilisk",
        "reward": "crown"
    }
    
    print(f"\n   Base story: {base_story}")
    
    # Generate variants by finding analogous concepts
    variants = []
    for key, value in base_story.items():
        if key == "hero":
            similar = registry.find_analogous_concepts(f"royalty:{value}", cluster_threshold=0.6)
            if similar:
                variant = base_story.copy()
                variant[key] = similar[0][0].name
                variants.append(variant)
    
    print("\n   Story variants:")
    for i, variant in enumerate(variants[:2]):
        print(f"     Variant {i+1}: {variant}")


def main():
    """Main demonstration function."""
    print("COMPREHENSIVE HYBRID FRAMENET-CLUSTERING DEMONSTRATION")
    print("Building Medieval Fantasy Knowledge Base with Advanced Semantic Reasoning")
    print()
    
    try:
        # Build knowledge base
        registry, concepts, domains = create_medieval_fantasy_world()
        
        # Create narrative structures
        frames = create_story_frames(registry)
        
        # Create story instances  
        instances = create_story_instances(registry, concepts)
        
        # Demonstrate reasoning
        demonstrate_semantic_reasoning(registry)
        
        # Show applications
        demonstrate_story_applications(registry)
        
        # Final statistics
        print("\n" + "=" * 80)
        print("FINAL SYSTEM STATISTICS")
        print("=" * 80)
        
        stats = registry.get_enhanced_statistics()
        
        print(f"\nKnowledge Base:")
        print(f"• Concepts: {stats.get('frame_aware_concepts', 0)}")
        print(f"• Semantic frames: {stats.get('semantic_frames', 0)}")
        print(f"• Frame instances: {stats.get('frame_instances', 0)}")
        print(f"• Concept clusters: {stats.get('concept_clusters', 0)}")
        print(f"• Semantic fields: {stats.get('semantic_fields', 0)}")
        print(f"• Cross-domain analogies: {stats.get('cross_domain_analogies', 0)}")
        
        print(f"\nQuality Metrics:")
        print(f"• Average cluster size: {stats.get('avg_cluster_size', 0):.1f}")
        print(f"• Average field size: {stats.get('avg_field_size', 0):.1f}")  
        print(f"• Average analogy quality: {stats.get('avg_analogy_quality', 0):.3f}")
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        
        print("\nThis demonstration showcases a complete hybrid semantic reasoning system that:")
        print("✓ Integrates FrameNet-style structural knowledge with clustering-based organization")
        print("✓ Discovers semantic fields and cross-domain analogies automatically")
        print("✓ Supports rich analogical reasoning and completion tasks")
        print("✓ Provides practical applications for story generation and world building")
        print("✓ Scales to complex knowledge domains with sophisticated reasoning")
        
        return True
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
