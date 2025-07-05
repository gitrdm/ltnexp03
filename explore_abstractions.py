"""Test script to explore the core abstractions."""

from app.core import (
    Concept, Axiom, Context, FormulaNode,
    AxiomType, AxiomClassification, OperationType,
    AxiomParser, ConceptRegistry
)


def explore_concepts():
    """Explore the Concept class."""
    print("=== EXPLORING CONCEPTS ===")
    
    # Create some basic concepts
    king = Concept(
        name="king",
        synset_id="king.n.01",
        disambiguation="monarch ruler"
    )
    
    man = Concept(name="man", synset_id="man.n.01")
    woman = Concept(name="woman")
    
    print(f"King concept: {king}")
    print(f"King unique ID: {king.unique_id}")
    print(f"Man concept: {man}")
    print(f"Woman concept: {woman}")
    
    # Test equality
    king2 = Concept(name="king", synset_id="king.n.01")
    print(f"king == king2: {king == king2}")
    
    king_different = Concept(name="king", synset_id="king.n.02")
    print(f"king == king_different: {king == king_different}")
    
    print()


def explore_formulas():
    """Explore formula construction."""
    print("=== EXPLORING FORMULAS ===")
    
    # Create concepts
    king = Concept("king")
    man = Concept("man")
    woman = Concept("woman")
    queen = Concept("queen")
    
    # Build formula: similar(king - man + woman, queen)
    # This represents: similar(add(subtract(king, man), woman), queen)
    
    subtract_node = FormulaNode(
        operation=OperationType.SUBTRACT,
        args=[king, man]
    )
    
    add_node = FormulaNode(
        operation=OperationType.ADD,
        args=[subtract_node, woman]
    )
    
    similarity_node = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[add_node, queen]
    )
    
    print(f"Formula: {similarity_node}")
    print(f"Concepts in formula: {[str(c) for c in similarity_node.get_concepts()]}")
    
    print()


def explore_axioms():
    """Explore axiom creation."""
    print("=== EXPLORING AXIOMS ===")
    
    # Create concepts
    concepts = [
        Concept("king", "king.n.01", "monarch ruler"),
        Concept("man", "man.n.01", "adult male"),
        Concept("woman", "woman.n.01", "adult female"),
        Concept("queen", "queen.n.01", "female monarch")
    ]
    
    # Create formula
    formula = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[
            FormulaNode(
                operation=OperationType.ADD,
                args=[
                    FormulaNode(OperationType.SUBTRACT, [concepts[0], concepts[1]]),
                    concepts[2]
                ]
            ),
            concepts[3]
        ]
    )
    
    # Create axiom
    axiom = Axiom(
        axiom_id="test_analogy",
        axiom_type=AxiomType.ANALOGY,
        classification=AxiomClassification.CORE,
        description="Test analogy axiom",
        formula=formula,
        concepts=concepts
    )
    
    print(f"Axiom: {axiom}")
    print(f"Is core axiom: {axiom.is_core_axiom()}")
    print(f"Concept names: {axiom.get_concept_names()}")
    
    print()


def explore_contexts():
    """Explore context management."""
    print("=== EXPLORING CONTEXTS ===")
    
    # Create a context
    context = Context(
        name="test_context",
        description="A test context for exploration"
    )
    
    print(f"Empty context: {context}")
    
    # Create and add an axiom
    car = Concept("car", "car.n.01", "motor vehicle")
    automobile = Concept("automobile", "car.n.01", "motor vehicle")
    
    formula = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[car, automobile]
    )
    
    axiom = Axiom(
        axiom_id="car_synonym",
        axiom_type=AxiomType.SYNONYM,
        classification=AxiomClassification.CORE,
        description="Car and automobile are synonyms",
        formula=formula,
        concepts=[car, automobile]
    )
    
    context.add_axiom(axiom)
    
    print(f"Context with axiom: {context}")
    print(f"Core axioms: {len(context.get_core_axioms())}")
    print(f"Soft axioms: {len(context.get_soft_axioms())}")
    print(f"Get car concept: {context.get_concept('car')}")
    
    print()


def explore_parsing():
    """Explore axiom parsing."""
    print("=== EXPLORING PARSING ===")
    
    parser = AxiomParser()
    yaml_axioms = []
    json_axioms = []
    
    # Parse YAML file
    try:
        yaml_axioms = parser.parse_file("examples/basic_analogy.yaml")
        print(f"Parsed YAML axiom: {yaml_axioms[0]}")
        print(f"Formula: {yaml_axioms[0].formula}")
        print()
    except Exception as e:
        print(f"Error parsing YAML: {e}")
    
    # Parse JSON file
    try:
        json_axioms = parser.parse_file("examples/core_axioms.json")
        print(f"Parsed {len(json_axioms)} JSON axioms:")
        for axiom in json_axioms:
            print(f"  - {axiom.axiom_id}: {axiom.description}")
        print()
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    
    # Create context from axioms
    try:
        all_axioms = yaml_axioms + json_axioms
        if all_axioms:
            context = parser.create_context_from_axioms(
                all_axioms, 
                "parsed_context", 
                "Context created from parsed axioms"
            )
            print(f"Created context: {context}")
        else:
            print("No axioms to create context from")
    except Exception as e:
        print(f"Error creating context: {e}")
    
    print()


def explore_concept_registry():
    """Explore concept registry."""
    print("=== EXPLORING CONCEPT REGISTRY ===")
    
    registry = ConceptRegistry(download_wordnet=False)  # Skip WordNet for demo
    
    # Create and register concepts
    king = registry.create_concept(
        "king", 
        context="medieval",
        disambiguation="royal ruler"
    )
    
    queen = registry.create_concept(
        "queen",
        context="medieval", 
        disambiguation="royal ruler female"
    )
    
    # Register a homonym
    bank_financial = registry.create_concept(
        "bank",
        context="finance",
        synset_id="bank.n.01",
        disambiguation="financial institution"
    )
    
    bank_river = registry.create_concept(
        "bank", 
        context="geography",
        synset_id="bank.n.09",
        disambiguation="land beside water"
    )
    
    print(f"Registered concepts:")
    for concept in registry.list_concepts():
        print(f"  - {concept}")
    
    print(f"\nRegistry stats: {registry.get_concept_stats()}")
    
    # Test retrieval
    print(f"\nRetrieve 'king' from medieval context: {registry.get_concept('king', 'medieval')}")
    print(f"Retrieve 'bank' from finance context: {registry.get_concept('bank', 'finance')}")
    print(f"Retrieve 'bank' from geography context: {registry.get_concept('bank', 'geography')}")
    
    # Test pattern search
    bank_concepts = registry.find_concepts_by_pattern("bank")
    print(f"\nConcepts matching 'bank': {[str(c) for c in bank_concepts]}")
    
    print()


def main():
    """Run all exploration functions."""
    print("🔍 EXPLORING CORE ABSTRACTIONS\n")
    
    explore_concepts()
    explore_formulas()
    explore_axioms()
    explore_contexts()
    explore_parsing()
    explore_concept_registry()
    
    print("✅ Exploration complete!")


if __name__ == "__main__":
    main()
