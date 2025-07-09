import pytest
from types import SimpleNamespace
from app.core.abstractions import Concept, concept_from_pydantic, concept_to_pydantic

class DummyPydantic:
    def __init__(self, name, context="default", synset_id=None, disambiguation=None, metadata=None):
        self.name = name
        self.context = context
        self.synset_id = synset_id
        self.disambiguation = disambiguation
        self.metadata = metadata or {}

def test_concept_from_pydantic():
    p = DummyPydantic("queen", context="royal", synset_id="queen.n.01", disambiguation="female ruler", metadata={"domain": "monarchy"})
    c = concept_from_pydantic(p)
    assert isinstance(c, Concept)
    assert c.name == "queen"
    assert c.context == "royal"
    assert c.synset_id == "queen.n.01"
    assert c.disambiguation == "female ruler"
    assert c.metadata["domain"] == "monarchy"

def test_concept_to_pydantic():
    c = Concept("bishop", synset_id="bishop.n.01", disambiguation="chess piece", context="chess", metadata={"role": "diagonal"})
    class DummyPydanticModel:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    p = concept_to_pydantic(c, DummyPydanticModel)
    assert isinstance(p, DummyPydanticModel)
    assert p.name == "bishop"
    assert p.context == "chess"
    assert p.synset_id == "bishop.n.01"
    assert p.disambiguation == "chess piece"
    assert p.metadata["role"] == "diagonal"
