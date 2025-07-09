"""
Concept Registry: Centralized Concept Management with WordNet Integration

This module implements a sophisticated concept management system that solves 
key challenges in soft logic systems:

1. DISAMBIGUATION PROBLEM: Words like "bank" have multiple meanings 
   (financial institution vs. river edge). We solve this using WordNet 
   synset IDs to uniquely identify concepts.

2. CONTEXT ISOLATION: Different domains need different concept interpretations.
   We provide context-aware storage allowing the same word to have different 
   meanings per domain.

3. SCALABLE LOOKUP: Fast retrieval across thousands of concepts using 
   multi-level indexing (global + context-specific + pattern matching).

DESIGN PHILOSOPHY:
- Graceful degradation: Works with or without WordNet
- Performance-oriented: Caching and O(1) lookups
- Research-friendly: Rich metadata and pattern search
- Type-safe: Proper error handling and validation

ARCHITECTURE:
    ConceptRegistry
    ├── concepts: Dict[unique_id, Concept]           # Global concept store
    ├── context_mappings: Dict[context, Dict[name, Concept]]  # Fast context lookup  
    ├── synset_cache: Dict[synset_id, SynsetInfo]   # WordNet cache
    └── wordnet_enabled: bool                       # Feature flag

UNIQUE ID STRATEGY:
    "default:king"                    # Simple concept
    "default:king:king.n.01"         # With synset
    "finance:bank:bank.n.01"         # Context + synset
    "geography:bank:bank.n.09"       # Different context, different synset
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Design by Contract support
from icontract import require, ensure, invariant, ViolationError

# WordNet Integration with Graceful Degradation
# ==============================================
# We attempt to import NLTK/WordNet but gracefully handle cases where 
# they're not available. This allows the system to work in minimal 
# environments while providing rich features when dependencies are present.

try:
    import nltk
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False

from .abstractions import Concept
from .protocols import ConceptRegistryProtocol


@dataclass
class SynsetInfo:
    """
    WordNet Synset Information Container
    
    This dataclass encapsulates all relevant information from a WordNet synset,
    providing a clean interface that abstracts away NLTK implementation details.
    We cache these to avoid repeated API calls to WordNet.
    
    Fields:
        id: WordNet synset identifier (e.g., "bank.n.01")
        definition: Human-readable definition
        examples: List of usage examples from WordNet
        lemmas: Alternative word forms with same meaning
        pos: Part of speech (noun, verb, adjective, etc.)
    """
    id: str
    definition: str
    examples: List[str]
    lemmas: List[str]
    pos: str  # Part of speech


@invariant(lambda self: hasattr(self, 'concepts') and isinstance(self.concepts, dict),
           "concepts must be a dictionary")
@invariant(lambda self: hasattr(self, 'context_mappings') and isinstance(self.context_mappings, dict),
           "context_mappings must be a dictionary")
class ConceptRegistry(ConceptRegistryProtocol):
    """
    Centralized Concept Management System
    
    The ConceptRegistry is the heart of our concept management system. It provides:
    
    1. UNIQUE CONCEPT IDENTIFICATION: Each concept gets a globally unique ID
       combining context, name, and optional synset ID
       
    2. FAST LOOKUPS: Multiple indexing strategies for different access patterns:
       - Global index for uniqueness guarantees
       - Context-specific index for domain isolation  
       - Pattern matching for discovery
       
    3. WORDNET INTEGRATION: Optional semantic enrichment with:
       - Automatic disambiguation using most common senses
       - Synset validation to prevent invalid references
       - Caching to avoid repeated API calls
       
    4. GRACEFUL DEGRADATION: Full functionality even without WordNet
    
    USAGE PATTERNS:
        # Basic usage
        registry = ConceptRegistry()
        king = registry.create_concept("king", "medieval")
        
        # With disambiguation  
        bank = registry.create_concept("bank", "finance", 
                                     synset_id="bank.n.01",
                                     disambiguation="financial institution")
        
        # Context-aware retrieval
        finance_bank = registry.get_concept("bank", "finance")
        geo_bank = registry.get_concept("bank", "geography")
    """
    
    def __init__(self, download_wordnet: bool = True):
        """
        Initialize the Concept Registry
        
        STORAGE ARCHITECTURE:
        - concepts: Global store ensuring uniqueness across all contexts
        - context_mappings: Fast lookup by context + name (O(1) access)
        - synset_cache: Cached WordNet data to avoid repeated API calls
        
        WORDNET INITIALIZATION:
        We attempt to download WordNet data if available, but gracefully
        handle cases where NLTK is not installed or download fails.
        
        Args:
            download_wordnet: Whether to attempt WordNet initialization
        """
        # DUAL STORAGE STRATEGY
        # =====================
        # We maintain two complementary data structures:
        # 1. Global concepts dict for uniqueness and complete access
        # 2. Context mappings for fast context-specific lookup
        self.concepts: Dict[str, Concept] = {}
        self.synset_cache: Dict[str, SynsetInfo] = {}
        self.context_mappings: Dict[str, Dict[str, Concept]] = {}
        
        # WORDNET INITIALIZATION WITH GRACEFUL DEGRADATION
        # ================================================
        # We try to enable WordNet features but don't fail if unavailable.
        # This allows the system to work in minimal environments while
        # providing rich semantic features when possible.
        if WORDNET_AVAILABLE and download_wordnet:
            try:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                self.wordnet_enabled = True
            except Exception:
                self.wordnet_enabled = False
        else:
            self.wordnet_enabled = False
    
    @require(lambda concept: isinstance(concept, Concept),
             "concept must be a Concept instance")
    @require(lambda concept: hasattr(concept, 'unique_id') and len(concept.unique_id) > 0,
             "concept must have non-empty unique_id")
    @ensure(lambda result: isinstance(result, Concept),
            "result must be a Concept instance")
    @ensure(lambda result, concept: result.unique_id == concept.unique_id,
            "result must have same unique_id as input")
    def register_concept(self, concept: Concept) -> Concept:
        """
        Register a Concept in the Registry
        
        This is the core registration method that implements our dual storage
        strategy. Every concept must go through this method to ensure consistency.
        
        VALIDATION STRATEGY:
        1. If synset provided and WordNet available, validate synset exists
        2. Store in global registry using unique_id for global uniqueness
        3. Store in context mapping for fast context-specific lookup
        
        DESIGN RATIONALE:
        - Global storage prevents duplicate concepts across contexts
        - Context mapping enables O(1) lookup within specific domains
        - Synset validation prevents invalid WordNet references
        
        Args:
            concept: The concept to register
            
        Returns:
            The registered concept (for method chaining)
            
        Raises:
            ValueError: If synset_id is invalid when WordNet is enabled
        """
        # SYNSET VALIDATION
        # =================
        # If a synset ID is provided and WordNet is available, we validate
        # that the synset actually exists. This prevents invalid references
        # from polluting our concept space.
        if concept.synset_id and self.wordnet_enabled:
            synset_info = self.get_synset_info(concept.synset_id)
            if not synset_info:
                raise ValueError(f"Invalid synset ID: {concept.synset_id}")
        
        # GLOBAL STORAGE FOR UNIQUENESS
        # ==============================
        # The global concepts dict uses unique_id as key, ensuring that
        # each concept can be uniquely identified across all contexts.
        # Format: "context:name" or "context:name:synset_id"
        self.concepts[concept.unique_id] = concept
        
        # CONTEXT-SPECIFIC STORAGE FOR FAST LOOKUP
        # =========================================
        # We also maintain context-specific mappings for O(1) lookup
        # within a specific domain. This enables fast queries like
        # "get me the 'bank' concept in the 'finance' context"
        if concept.context not in self.context_mappings:
            self.context_mappings[concept.context] = {}
        self.context_mappings[concept.context][concept.name] = concept
        
        return concept
    
    @require(lambda name: isinstance(name, str) and len(name.strip()) > 0,
             "name must be non-empty string")
    @require(lambda context: isinstance(context, str) and len(context.strip()) > 0,
             "context must be non-empty string")
    @ensure(lambda result: result is None or isinstance(result, Concept),
            "result must be None or Concept instance")
    def get_concept(self, name: str, context: str = "default", 
                   synset_id: Optional[str] = None) -> Optional[Concept]:
        """
        Retrieve a Concept by Name and Context
        
        This method implements a two-tier lookup strategy:
        1. Fast context-specific lookup for common queries
        2. Fallback to global lookup for precise synset matching
        
        LOOKUP STRATEGY:
        - First: Check context mapping for O(1) access
        - Second: Construct unique_id and check global registry
        - Always: Respect synset_id constraints if provided
        
        DESIGN RATIONALE:
        The dual lookup strategy optimizes for the most common case
        (context + name) while still supporting precise synset-based
        retrieval when disambiguation is needed.
        
        Args:
            name: Concept name (case-insensitive)
            context: Context to search within
            synset_id: Optional synset constraint
            
        Returns:
            Matching concept or None if not found
        """
        name = name.lower().strip()
        
        # FAST CONTEXT-SPECIFIC LOOKUP
        # =============================
        # For most queries, users want "the 'bank' concept in 'finance' context".
        # Context mappings provide O(1) access for this common pattern.
        if context in self.context_mappings:
            if name in self.context_mappings[context]:
                concept = self.context_mappings[context][name]
                # Respect synset constraint if provided
                if synset_id is None or concept.synset_id == synset_id:
                    return concept
        
        # FALLBACK TO GLOBAL LOOKUP
        # =========================
        # If context lookup fails or synset constraint not met,
        # try constructing the exact unique_id for global lookup.
        if synset_id:
            unique_id = f"{context}:{name}:{synset_id}"
        else:
            unique_id = f"{context}:{name}"
        
        return self.concepts.get(unique_id)
    
    def find_concepts_by_pattern(self, pattern: str, context: Optional[str] = None) -> List[Concept]:
        """
        Pattern-Based Concept Discovery
        
        Enables flexible concept discovery using regex patterns. This is
        particularly useful for research and exploration where you want to
        find concepts matching certain criteria.
        
        SEARCH STRATEGY:
        - Uses regex for flexible pattern matching
        - Searches both concept names and disambiguation text
        - Can be limited to specific context for focused discovery
        - Case-insensitive by default
        
        PERFORMANCE CONSIDERATIONS:
        - Context-limited search reduces search space
        - Regex compilation done once per call
        - Linear search through relevant concepts
        
        Args:
            pattern: Regex pattern to match
            context: Optional context to limit search
            
        Returns:
            List of matching concepts
        """
        regex = re.compile(pattern, re.IGNORECASE)
        matches = []
        
        # CONTEXT-AWARE SEARCH SPACE SELECTION
        # ====================================
        # If context specified, only search within that context.
        # Otherwise, search all concepts globally.
        search_space = (
            self.context_mappings.get(context, {}).values() if context 
            else self.concepts.values()
        )
        
        # PATTERN MATCHING ON NAME AND DISAMBIGUATION
        # ===========================================
        # We search both the concept name and disambiguation text,
        # allowing rich discovery based on semantic content.
        for concept in search_space:
            if regex.search(concept.name) or regex.search(concept.disambiguation or ""):
                matches.append(concept)
        
        return matches
    
    def get_synset_info(self, synset_id: str) -> Optional[SynsetInfo]:
        """
        WordNet Synset Information Retrieval with Caching
        
        This method provides cached access to WordNet synset information.
        We cache results to avoid repeated API calls to WordNet, which
        can be expensive for large concept sets.
        
        CACHING STRATEGY:
        - Check cache first for O(1) retrieval of previously accessed synsets
        - On cache miss, fetch from WordNet and cache the result
        - Handle WordNet API failures gracefully
        
        PERFORMANCE BENEFITS:
        - Eliminates repeated WordNet API calls
        - Provides consistent interface even when WordNet unavailable
        - Graceful degradation for invalid synset IDs
        
        Args:
            synset_id: WordNet synset identifier (e.g., "bank.n.01")
            
        Returns:
            SynsetInfo object or None if not available/invalid
        """
        if not self.wordnet_enabled:
            return None
        
        # CACHE LOOKUP FIRST
        # ==================
        # Check if we've already fetched this synset to avoid
        # repeated API calls to WordNet.
        if synset_id in self.synset_cache:
            return self.synset_cache[synset_id]
        
        # FETCH FROM WORDNET AND CACHE
        # ============================
        # On cache miss, fetch from WordNet API and cache the result.
        # We handle exceptions gracefully to avoid crashes on invalid synsets.
        try:
            synset = wn.synset(synset_id)
            info = SynsetInfo(
                id=synset_id,
                definition=synset.definition(),
                examples=synset.examples(),
                lemmas=[lemma.name() for lemma in synset.lemmas()],
                pos=synset.pos()
            )
            # Cache for future use
            self.synset_cache[synset_id] = info
            return info
        except Exception:
            # Invalid synset ID or WordNet API error
            return None
    
    def find_synsets(self, word: str, pos: Optional[str] = None) -> List[SynsetInfo]:
        """
        WordNet Synset Discovery
        
        Find all possible synsets (word senses) for a given word. This is
        used for automatic disambiguation and concept discovery.
        
        DISAMBIGUATION USE CASE:
        When a user creates a concept without specifying a synset, we can
        use this method to find all possible meanings and either:
        1. Auto-select the most common meaning (first synset)
        2. Present options to the user for manual selection
        
        PERFORMANCE CONSIDERATIONS:
        - Results are not cached (unlike get_synset_info) since this is
          typically used for discovery rather than repeated access
        - Part-of-speech filtering can reduce result set size
        
        Args:
            word: Word to find synsets for
            pos: Optional part-of-speech filter
            
        Returns:
            List of SynsetInfo objects representing all word senses
        """
        if not self.wordnet_enabled:
            return []
        
        try:
            synsets = wn.synsets(word, pos=pos)
            return [
                SynsetInfo(
                    id=synset.name(),
                    definition=synset.definition(),
                    examples=synset.examples(),
                    lemmas=[lemma.name() for lemma in synset.lemmas()],
                    pos=synset.pos()
                )
                for synset in synsets
            ]
        except Exception:
            return []
    
    def disambiguate_concept(self, name: str, context_hints: Optional[List[str]] = None) -> List[Concept]:
        """Suggest concept disambiguations based on context hints."""
        candidates = []
        
        # Find existing concepts with this name
        for concept in self.concepts.values():
            if concept.name == name.lower().strip():
                candidates.append(concept)
        
        # Find WordNet synsets if enabled
        if self.wordnet_enabled:
            synsets = self.find_synsets(name)
            for synset_info in synsets:
                # Check if we already have a concept for this synset
                existing = self.get_concept(name, synset_id=synset_info.id)
                if not existing:
                    # Create new concept suggestion
                    concept = Concept(
                        name=name,
                        synset_id=synset_info.id,
                        disambiguation=synset_info.definition
                    )
                    candidates.append(concept)
        
        # Score candidates based on context hints
        if context_hints:
            scored_candidates = []
            for concept in candidates:
                score = self._score_concept_context(concept, context_hints)
                scored_candidates.append((score, concept))
            
            # Sort by score (descending)
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return [concept for _, concept in scored_candidates]
        
        return candidates
    
    def _score_concept_context(self, concept: Concept, context_hints: List[str]) -> float:
        """Score how well a concept matches context hints."""
        score = 0.0
        context_text = " ".join(context_hints).lower()
        
        # Check disambiguation text
        if concept.disambiguation:
            for word in concept.disambiguation.lower().split():
                if word in context_text:
                    score += 1.0
        
        # Check synset definition and examples
        if concept.synset_id and self.wordnet_enabled:
            synset_info = self.get_synset_info(concept.synset_id)
            if synset_info:
                # Check definition
                for word in synset_info.definition.lower().split():
                    if word in context_text:
                        score += 0.5
                
                # Check examples
                for example in synset_info.examples:
                    for word in example.lower().split():
                        if word in context_text:
                            score += 0.3
        
        return score
    
    @require(lambda name: isinstance(name, str) and len(name.strip()) > 0,
             "name must be non-empty string")
    @require(lambda context: isinstance(context, str) and len(context.strip()) > 0,
             "context must be non-empty string")
    @require(lambda synset_id: synset_id is None or (isinstance(synset_id, str) and len(synset_id.strip()) > 0),
             "synset_id must be None or non-empty string")
    @ensure(lambda result: isinstance(result, Concept),
            "result must be a Concept instance")
    @ensure(lambda result, name: result.name == name.lower().strip(),
            "result name must match normalized input name")
    @ensure(lambda result, context: result.context == context,
            "result context must match input context")
    def create_concept(self, name: str, context: str = "default",
                      synset_id: Optional[str] = None,
                      disambiguation: Optional[str] = None,
                      auto_disambiguate: bool = True) -> Concept:
        """Create and register a new concept."""
        name = name.lower().strip()
        
        # Auto-disambiguate if requested and no synset provided
        if auto_disambiguate and not synset_id and self.wordnet_enabled:
            synsets = self.find_synsets(name)
            if synsets:
                # Use the most common synset (first one)
                synset_id = synsets[0].id
                if not disambiguation:
                    disambiguation = synsets[0].definition
        
        concept = Concept(
            name=name,
            synset_id=synset_id,
            disambiguation=disambiguation,
            context=context
        )
        
        return self.register_concept(concept)
    
    def list_concepts(self, context: Optional[str] = None) -> List[Concept]:
        """List all concepts, optionally filtered by context."""
        if context:
            return list(self.context_mappings.get(context, {}).values())
        return list(self.concepts.values())
    
    def find_similar_concepts(
        self, 
        concept: Concept, 
        threshold: float = 0.7
    ) -> List[Tuple[Concept, float]]:
        """Find concepts with similar embeddings using cosine similarity."""
        if concept.embedding is None:
            return []

        similar = []
        
        # Prepare source embedding
        embedding1 = concept.embedding.reshape(1, -1)

        for other_concept in self.concepts.values():
            if other_concept.unique_id == concept.unique_id or other_concept.embedding is None:
                continue
            
            # Prepare target embedding
            embedding2 = other_concept.embedding.reshape(1, -1)
            
            # Compute cosine similarity
            sim = cosine_similarity(embedding1, embedding2)[0][0]
            
            if sim >= threshold:
                similar.append((other_concept, float(sim)))
        
        # Sort by similarity score in descending order
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    @property
    def concept_count(self) -> int:
        """Return total number of registered concepts."""
        return len(self.concepts)

    def get_concept_stats(self) -> Dict[str, int]:
        """Get statistics about registered concepts."""
        stats = {
            "total_concepts": len(self.concepts),
            "contexts": len(self.context_mappings),
            "with_synsets": len([c for c in self.concepts.values() if c.synset_id]),
            "wordnet_enabled": self.wordnet_enabled
        }
        
        # Count concepts per context
        for context, concepts in self.context_mappings.items():
            stats[f"context_{context}"] = len(concepts)
        
        return stats
