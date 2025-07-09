"""
Demo: Core Abstractions in Action

This script demonstrates how to use the core abstractions to build
a knowledge base with concepts, axioms, and contexts.
"""

from app.core import (
    Concept, Axiom, Context, FormulaNode,
    AxiomType, AxiomClassification, OperationType,
    AxiomParser, ConceptRegistry
)


def create_word_analogy_knowledge_base():
    """Create a knowledge base with word analogies."""
    print("🏗️  Building Word Analogy Knowledge Base")
    print("="*50)
    
    # Create concept registry
    registry = ConceptRegistry(download_wordnet=False)
    
    # Create concepts for classic analogies
    print("\n📝 Creating concepts...")
    
    # Gender analogy: king:man :: queen:woman
    king = registry.create_concept("king", "royal", synset_id="king.n.01", disambiguation="male monarch")
    man = registry.create_concept("man", "royal", synset_id="man.n.01", disambiguation="adult male")
    woman = registry.create_concept("woman", "royal", synset_id="woman.n.01", disambiguation="adult female")
    queen = registry.create_concept("queen", "royal", synset_id="queen.n.01", disambiguation="female monarch")
    
    # Vehicle synonyms
    car = registry.create_concept("car", "transport", synset_id="car.n.01", disambiguation="motor vehicle")
    automobile = registry.create_concept("automobile", "transport", synset_id="car.n.01", disambiguation="motor vehicle")
    
    # Temperature antonyms
    hot = registry.create_concept("hot", "weather", synset_id="hot.a.01", disambiguation="high temperature")
    cold = registry.create_concept("cold", "weather", synset_id="cold.a.01", disambiguation="low temperature")
    
    # Bank homonyms
    bank_financial = registry.create_concept("bank", "finance", synset_id="bank.n.01", disambiguation="financial institution")
    bank_river = registry.create_concept("bank", "geography", synset_id="bank.n.09", disambiguation="land beside water")
    
    print(f"✅ Created {len(registry.list_concepts())} concepts")
    
    # Create axioms
    print("\n⚖️  Creating axioms...")
    
    axioms = []
    
    # 1. Gender analogy: king - man + woman ≈ queen
    analogy_formula = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[
            FormulaNode(
                operation=OperationType.ADD,
                args=[
                    FormulaNode(OperationType.SUBTRACT, [king, man]),
                    woman
                ]
            ),
            queen
        ]
    )
    
    gender_axiom = Axiom(
        axiom_id="gender_analogy",
        axiom_type=AxiomType.ANALOGY,
        classification=AxiomClassification.CORE,
        description="Gender analogy: king is to man as queen is to woman",
        formula=analogy_formula,
        concepts=[king, man, woman, queen]
    )
    axioms.append(gender_axiom)
    
    # 2. Vehicle synonym: car ≈ automobile
    synonym_formula = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[car, automobile]
    )
    
    vehicle_axiom = Axiom(
        axiom_id="vehicle_synonym",
        axiom_type=AxiomType.SYNONYM,
        classification=AxiomClassification.CORE,
        description="Car and automobile are synonyms",
        formula=synonym_formula,
        concepts=[car, automobile]
    )
    axioms.append(vehicle_axiom)
    
    # 3. Temperature antonym: hot ≉ cold
    antonym_formula = FormulaNode(
        operation=OperationType.DISSIMILARITY,
        args=[hot, cold]
    )
    
    temperature_axiom = Axiom(
        axiom_id="temperature_antonym",
        axiom_type=AxiomType.ANTONYM,
        classification=AxiomClassification.CORE,
        description="Hot and cold are antonyms",
        formula=antonym_formula,
        concepts=[hot, cold]
    )
    axioms.append(temperature_axiom)
    
    # 4. Bank disambiguation: bank_financial ≉ bank_river
    disambiguation_formula = FormulaNode(
        operation=OperationType.DISSIMILARITY,
        args=[bank_financial, bank_river]
    )
    
    bank_axiom = Axiom(
        axiom_id="bank_disambiguation",
        axiom_type=AxiomType.DISSIMILARITY,
        classification=AxiomClassification.CORE,
        description="Different meanings of bank should be distinct",
        formula=disambiguation_formula,
        concepts=[bank_financial, bank_river]
    )
    axioms.append(bank_axiom)
    
    print(f"✅ Created {len(axioms)} axioms")
    
    # Create context and add axioms
    print("\n🌍 Creating context...")
    
    context = Context(
        name="word_analogies",
        description="Knowledge base for word analogies, synonyms, and antonyms"
    )
    
    for axiom in axioms:
        context.add_axiom(axiom)
    
    print(f"✅ Created context with {len(context.axioms)} axioms and {len(context.concepts)} concepts")
    
    return context, registry


def analyze_knowledge_base(context, registry):
    """Analyze the knowledge base structure."""
    print("\n🔍 Analyzing Knowledge Base")
    print("="*50)
    
    # Context analysis
    print(f"\n📊 Context: {context.name}")
    print(f"   Description: {context.description}")
    print(f"   Total axioms: {len(context.axioms)}")
    print(f"   Core axioms: {len(context.get_core_axioms())}")
    print(f"   Soft axioms: {len(context.get_soft_axioms())}")
    print(f"   Total concepts: {len(context.concepts)}")
    
    # Axiom breakdown
    print(f"\n📜 Axioms by type:")
    axiom_types = {}
    for axiom in context.axioms:
        axiom_type = axiom.axiom_type.value
        if axiom_type not in axiom_types:
            axiom_types[axiom_type] = 0
        axiom_types[axiom_type] += 1
    
    for axiom_type, count in axiom_types.items():
        print(f"   {axiom_type}: {count}")
    
    # Concept breakdown by context
    print(f"\n🏷️  Concepts by context:")
    context_stats = registry.get_concept_stats()
    for key, value in context_stats.items():
        if key.startswith("context_"):
            context_name = key.replace("context_", "")
            print(f"   {context_name}: {value}")
    
    # Show some example concepts
    print(f"\n🎯 Example concepts:")
    for concept in list(context.concepts.values())[:5]:
        print(f"   {concept} - {concept.disambiguation}")
    
    # Show homonym handling
    print(f"\n🔄 Homonym examples:")
    bank_concepts = registry.find_concepts_by_pattern("bank")
    for concept in bank_concepts:
        print(f"   {concept} - {concept.disambiguation}")


def test_knowledge_queries(context, registry):
    """Test various knowledge queries."""
    print("\n🧪 Testing Knowledge Queries")
    print("="*50)
    
    # Test concept retrieval
    print("\n🔍 Concept retrieval tests:")
    
    # Test context-specific retrieval
    finance_bank = registry.get_concept("bank", "finance")
    geo_bank = registry.get_concept("bank", "geography")
    
    print(f"   Bank (finance): {finance_bank}")
    print(f"   Bank (geography): {geo_bank}")
    
    # Test pattern matching
    print(f"\n🎯 Pattern matching tests:")
    royal_concepts = registry.find_concepts_by_pattern(".*ing|.*een")  # king, queen
    print(f"   Royal concepts (king/queen): {[str(c) for c in royal_concepts]}")
    
    # Test axiom analysis
    print(f"\n⚖️  Axiom analysis:")
    for axiom in context.axioms:
        print(f"   {axiom.axiom_id}: {axiom.axiom_type.value} - {axiom.classification.value}")
        print(f"      Formula: {axiom.formula}")
        print(f"      Concepts: {axiom.get_concept_names()}")
        print()


def save_and_load_demo(context):
    """Demo saving and loading axioms."""
    print("💾 Save/Load Demo")
    print("="*50)
    
    # We'll create a simple serialization for demo
    print("\n📝 Axiom summaries:")
    for axiom in context.axioms:
        print(f"   {axiom.axiom_id}:")
        print(f"      Type: {axiom.axiom_type.value}")
        print(f"      Classification: {axiom.classification.value}")
        print(f"      Description: {axiom.description}")
        print(f"      Concepts: {len(axiom.concepts)}")
        print()


def demonstrate_generalized_vector_arithmetic():
    """Demonstrate that vector arithmetic is fully generalized for complex formulas."""
    print("\n🧮 Demonstrating Generalized Vector Arithmetic")
    print("="*60)
    
    # Create a registry for this demo
    registry = ConceptRegistry(download_wordnet=False)
    
    # Create concepts for complex arithmetic
    print("\n📝 Creating concepts for complex arithmetic...")
    
    # Royal hierarchy
    emperor = registry.create_concept("emperor", "royalty", disambiguation="highest monarch")
    king = registry.create_concept("king", "royalty", disambiguation="male monarch")
    queen = registry.create_concept("queen", "royalty", disambiguation="female monarch")
    prince = registry.create_concept("prince", "royalty", disambiguation="male heir")
    princess = registry.create_concept("princess", "royalty", disambiguation="female heir")
    
    # Gender concepts
    male = registry.create_concept("male", "gender", disambiguation="masculine")
    female = registry.create_concept("female", "gender", disambiguation="feminine")
    
    # Power/status concepts
    power = registry.create_concept("power", "abstract", disambiguation="authority")
    youth = registry.create_concept("youth", "abstract", disambiguation="young age")
    
    print(f"✅ Created {len(registry.list_concepts())} concepts")
    
    # Demonstrate complex nested formulas
    print("\n🔧 Building complex nested vector arithmetic formulas...")
    
    # Example 1: Multiple additions and subtractions
    # Formula: ((emperor - king) + queen) - male + female ≈ empress
    step1 = FormulaNode(OperationType.SUBTRACT, [emperor, king])
    step2 = FormulaNode(OperationType.ADD, [step1, queen])
    step3 = FormulaNode(OperationType.SUBTRACT, [step2, male])
    step4 = FormulaNode(OperationType.ADD, [step3, female])
    
    complex_formula_1 = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[
            step4,
            registry.create_concept("empress", "royalty", disambiguation="female emperor")
        ]
    )
    
    # Example 2: Deeply nested hierarchy
    # Formula: ((king + power) - (prince + youth)) + female ≈ mature_queen
    complex_formula_2 = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[
            FormulaNode(
                operation=OperationType.ADD,
                args=[
                    FormulaNode(
                        operation=OperationType.SUBTRACT,
                        args=[
                            FormulaNode(OperationType.ADD, [king, power]),
                            FormulaNode(OperationType.ADD, [prince, youth])
                        ]
                    ),
                    female
                ]
            ),
            registry.create_concept("mature_queen", "royalty", disambiguation="experienced female ruler")
        ]
    )
    
    # Example 3: Chain of operations with 5+ terms
    # Formula: king - male + female - youth + power ≈ wise_queen
    chain_step1 = FormulaNode(OperationType.SUBTRACT, [king, male])
    chain_step2 = FormulaNode(OperationType.ADD, [chain_step1, female])
    chain_step3 = FormulaNode(OperationType.SUBTRACT, [chain_step2, youth])
    chain_step4 = FormulaNode(OperationType.ADD, [chain_step3, power])
    
    chain_formula = FormulaNode(
        operation=OperationType.SIMILARITY,
        args=[
            chain_step4,
            registry.create_concept("wise_queen", "royalty", disambiguation="intelligent female ruler")
        ]
    )
    
    # Display the formulas
    formulas = [
        ("Multi-term transformation", complex_formula_1),
        ("Hierarchical composition", complex_formula_2),
        ("Linear chain (5+ terms)", chain_formula)
    ]
    
    print("\n📐 Generated formulas:")
    for name, formula in formulas:
        print(f"\n   {name}:")
        print(f"   Formula: {formula}")
        concepts = formula.get_concepts()
        print(f"   Involves {len(concepts)} concepts: {[c.name for c in concepts]}")
    
    # Create axioms from these formulas
    axioms = []
    for i, (name, formula) in enumerate(formulas):
        axiom = Axiom(
            axiom_id=f"complex_arithmetic_{i+1}",
            axiom_type=AxiomType.ANALOGY,
            classification=AxiomClassification.CORE,
            description=f"Complex vector arithmetic: {name}",
            formula=formula,
            concepts=formula.get_concepts()
        )
        axioms.append(axiom)
    
    print(f"\n✅ Created {len(axioms)} complex arithmetic axioms")
    
    # Demonstrate formula parsing capabilities
    print("\n🔍 Demonstrating arbitrary nesting depth...")
    
    # Create an extremely nested formula (depth 6)
    # Formula: ((((king - male) + female) - youth) + power) - complexity + wisdom ≈ ultimate_ruler
    deep_step1 = FormulaNode(OperationType.SUBTRACT, [king, male])
    deep_step2 = FormulaNode(OperationType.ADD, [deep_step1, female])
    deep_step3 = FormulaNode(OperationType.SUBTRACT, [deep_step2, youth])
    deep_step4 = FormulaNode(OperationType.ADD, [deep_step3, power])
    deep_step5 = FormulaNode(OperationType.SUBTRACT, [deep_step4, registry.create_concept("complexity", "abstract", disambiguation="intricacy")])
    deep_step6 = FormulaNode(OperationType.ADD, [deep_step5, registry.create_concept("wisdom", "abstract", disambiguation="deep knowledge")])
    
    deep_formula = FormulaNode(
        OperationType.SIMILARITY,
        [
            deep_step6,
            registry.create_concept("ultimate_ruler", "royalty", disambiguation="supreme leader")
        ]
    )
    
    print(f"   Deep nested formula (6 levels): {deep_formula}")
    print(f"   Total concepts involved: {len(deep_formula.get_concepts())}")
    
    return registry, axioms


def main():
    """Main demo function."""
    print("🚀 SOFT LOGIC CORE ABSTRACTIONS DEMO")
    print("="*60)
    
    # Build knowledge base
    context, registry = create_word_analogy_knowledge_base()
    
    # Analyze structure
    analyze_knowledge_base(context, registry)
    
    # Test queries
    test_knowledge_queries(context, registry)
    
    # Save/load demo
    save_and_load_demo(context)
    
    # Generalized vector arithmetic demo
    demonstrate_generalized_vector_arithmetic()
    
    print("\n🎉 Demo complete!")
    print("\nNext steps:")
    print("- Phase 2: Add SMT verification for core axioms")
    print("- Phase 3: Add LTN training for soft axioms")
    print("- Phase 4: Add REST API endpoints")


if __name__ == "__main__":
    main()
