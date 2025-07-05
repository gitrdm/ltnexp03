"""Tests for core abstractions."""

import pytest

from app.core import (
    Concept, Axiom, Context, FormulaNode,
    AxiomType, AxiomClassification, OperationType,
    AxiomParser, ConceptRegistry
)


class TestConcept:
    """Test the Concept class."""
    
    def test_concept_creation(self):
        """Test basic concept creation."""
        concept = Concept("king", "king.n.01", "royal ruler")
        assert concept.name == "king"
        assert concept.synset_id == "king.n.01"
        assert concept.disambiguation == "royal ruler"
        assert concept.context == "default"
    
    def test_concept_normalization(self):
        """Test that concept names are normalized."""
        concept = Concept("  KING  ")
        assert concept.name == "king"
    
    def test_concept_unique_id(self):
        """Test unique ID generation."""
        concept1 = Concept("king", "king.n.01", context="medieval")
        concept2 = Concept("king", "king.n.02", context="medieval")
        concept3 = Concept("king", context="medieval")
        
        assert concept1.unique_id == "medieval:king:king.n.01"
        assert concept2.unique_id == "medieval:king:king.n.02"
        assert concept3.unique_id == "medieval:king"
    
    def test_concept_equality(self):
        """Test concept equality."""
        concept1 = Concept("king", "king.n.01")
        concept2 = Concept("king", "king.n.01")
        concept3 = Concept("king", "king.n.02")
        
        assert concept1 == concept2
        assert concept1 != concept3
    
    def test_concept_validation(self):
        """Test concept validation."""
        with pytest.raises(ValueError):
            Concept("")  # Empty name should raise error


class TestFormulaNode:
    """Test the FormulaNode class."""
    
    def test_formula_creation(self):
        """Test formula node creation."""
        king = Concept("king")
        man = Concept("man")
        
        formula = FormulaNode(OperationType.SIMILARITY, [king, man])
        assert formula.operation == OperationType.SIMILARITY
        assert len(formula.args) == 2
    
    def test_formula_validation(self):
        """Test formula validation."""
        with pytest.raises(ValueError):
            FormulaNode(OperationType.SIMILARITY, [Concept("king")])  # Need 2 args
        
        with pytest.raises(ValueError):
            FormulaNode(OperationType.CONSTANT, [Concept("king"), Concept("man")])  # Need 1 arg
    
    def test_get_concepts(self):
        """Test concept extraction from formula."""
        king = Concept("king")
        man = Concept("man")
        woman = Concept("woman")
        
        # Create nested formula: similar(king - man, woman)
        subtract = FormulaNode(OperationType.SUBTRACT, [king, man])
        formula = FormulaNode(OperationType.SIMILARITY, [subtract, woman])
        
        concepts = formula.get_concepts()
        concept_names = [c.name for c in concepts]
        assert "king" in concept_names
        assert "man" in concept_names
        assert "woman" in concept_names


class TestAxiom:
    """Test the Axiom class."""
    
    def test_axiom_creation(self):
        """Test axiom creation."""
        concepts = [Concept("king"), Concept("man")]
        formula = FormulaNode(OperationType.SIMILARITY, concepts)
        
        axiom = Axiom(
            axiom_id="test_axiom",
            axiom_type=AxiomType.SIMILARITY,
            classification=AxiomClassification.CORE,
            description="Test axiom",
            formula=formula,
            concepts=concepts
        )
        
        assert axiom.axiom_id == "test_axiom"
        assert axiom.is_core_axiom()
        assert len(axiom.concepts) == 2
    
    def test_axiom_concept_extraction(self):
        """Test automatic concept extraction from formula."""
        king = Concept("king")
        man = Concept("man")
        formula = FormulaNode(OperationType.SIMILARITY, [king, man])
        
        axiom = Axiom(
            axiom_id="test_axiom",
            axiom_type=AxiomType.SIMILARITY,
            classification=AxiomClassification.CORE,
            description="Test axiom",
            formula=formula
            # No concepts provided - should extract from formula
        )
        
        assert len(axiom.concepts) == 2
        concept_names = axiom.get_concept_names()
        assert "king" in concept_names
        assert "man" in concept_names


class TestContext:
    """Test the Context class."""
    
    def test_context_creation(self):
        """Test context creation."""
        context = Context("test_context", description="Test context")
        assert context.name == "test_context"
        assert context.description == "Test context"
        assert len(context.axioms) == 0
    
    def test_context_axiom_management(self):
        """Test adding axioms to context."""
        context = Context("test_context")
        
        # Create axiom
        concepts = [Concept("king"), Concept("man")]
        formula = FormulaNode(OperationType.SIMILARITY, concepts)
        axiom = Axiom(
            axiom_id="test_axiom",
            axiom_type=AxiomType.SIMILARITY,
            classification=AxiomClassification.CORE,
            description="Test axiom",
            formula=formula,
            concepts=concepts
        )
        
        context.add_axiom(axiom)
        
        assert len(context.axioms) == 1
        assert len(context.concepts) == 2
        assert len(context.get_core_axioms()) == 1
        assert len(context.get_soft_axioms()) == 0
    
    def test_context_concept_retrieval(self):
        """Test concept retrieval from context."""
        context = Context("test_context")
        
        king = Concept("king")
        concepts = [king, Concept("man")]
        formula = FormulaNode(OperationType.SIMILARITY, concepts)
        axiom = Axiom(
            axiom_id="test_axiom",
            axiom_type=AxiomType.SIMILARITY,
            classification=AxiomClassification.CORE,
            description="Test axiom",
            formula=formula,
            concepts=concepts
        )
        
        context.add_axiom(axiom)
        
        retrieved_king = context.get_concept("king")
        assert retrieved_king is not None
        assert retrieved_king.name == "king"
        
        # Test case insensitive retrieval
        retrieved_king2 = context.get_concept("KING")
        assert retrieved_king2 is not None
        assert retrieved_king2.name == "king"


class TestConceptRegistry:
    """Test the ConceptRegistry class."""
    
    def test_concept_registration(self):
        """Test concept registration."""
        registry = ConceptRegistry(download_wordnet=False)
        
        concept = registry.create_concept("king", "medieval", disambiguation="royal ruler")
        assert concept.name == "king"
        assert concept.context == "medieval"
        
        # Test retrieval
        retrieved = registry.get_concept("king", "medieval")
        assert retrieved is not None
        assert retrieved.name == "king"
    
    def test_homonym_handling(self):
        """Test handling of homonyms."""
        registry = ConceptRegistry(download_wordnet=False)
        
        bank1 = registry.create_concept(
            "bank", "finance", synset_id="bank.n.01", disambiguation="financial institution"
        )
        bank2 = registry.create_concept(
            "bank", "geography", synset_id="bank.n.09", disambiguation="land beside water"
        )
        
        # Should be different concepts
        assert bank1 != bank2
        assert bank1.unique_id != bank2.unique_id
        
        # Should retrieve correctly by context
        finance_bank = registry.get_concept("bank", "finance")
        geo_bank = registry.get_concept("bank", "geography")
        
        assert finance_bank is not None
        assert geo_bank is not None
        assert finance_bank.synset_id == "bank.n.01"
        assert geo_bank.synset_id == "bank.n.09"
    
    def test_pattern_search(self):
        """Test pattern-based concept search."""
        registry = ConceptRegistry(download_wordnet=False)
        
        registry.create_concept("king", disambiguation="royal ruler")
        registry.create_concept("kingdom", disambiguation="royal domain")
        registry.create_concept("queen", disambiguation="royal ruler female")
        
        # Search for concepts containing "king"
        matches = registry.find_concepts_by_pattern("king")
        match_names = [c.name for c in matches]
        
        assert "king" in match_names
        assert "kingdom" in match_names
        assert "queen" not in match_names  # Should not match "king" pattern


class TestAxiomParser:
    """Test the AxiomParser class."""
    
    def test_yaml_parsing(self):
        """Test YAML axiom parsing."""
        parser = AxiomParser()
        
        yaml_content = """
axiom_id: test_axiom
type: similarity
classification: core
context: default
description: "Test axiom"
formula:
  similarity:
    left: king
    right: man
concepts:
  - name: king
    synset: king.n.01
  - name: man
    synset: man.n.01
"""
        
        axioms = parser.parse_yaml(yaml_content)
        assert len(axioms) == 1
        
        axiom = axioms[0]
        assert axiom.axiom_id == "test_axiom"
        assert axiom.axiom_type == AxiomType.SIMILARITY
        assert axiom.classification == AxiomClassification.CORE
        assert len(axiom.concepts) == 2
    
    def test_json_parsing(self):
        """Test JSON axiom parsing."""
        parser = AxiomParser()
        
        json_content = """
{
  "axiom_id": "test_axiom",
  "type": "similarity",
  "classification": "core",
  "context": "default",
  "description": "Test axiom",
  "formula": {
    "similarity": {
      "left": "king",
      "right": "man"
    }
  },
  "concepts": [
    {"name": "king", "synset": "king.n.01"},
    {"name": "man", "synset": "man.n.01"}
  ]
}
"""
        
        axioms = parser.parse_json(json_content)
        assert len(axioms) == 1
        
        axiom = axioms[0]
        assert axiom.axiom_id == "test_axiom"
        assert axiom.axiom_type == AxiomType.SIMILARITY
        assert axiom.classification == AxiomClassification.CORE
        assert len(axiom.concepts) == 2
