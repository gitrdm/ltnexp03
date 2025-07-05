"""
Persistence Layer: Comprehensive State Management

This module provides comprehensive persistence capabilities for the soft logic
microservice, including save/load functionality for all semantic structures,
LTN models, clustering models, and embeddings with format-specific optimizations.
"""

import json
import pickle
import gzip
import shutil
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone
from dataclasses import asdict
import logging

from icontract import require, ensure, invariant, ViolationError
import numpy as np
import torch

from .abstractions import Concept, Axiom, Context
from .frame_cluster_abstractions import SemanticFrame, FrameInstance, ConceptCluster

# Type hint for registry - avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .enhanced_semantic_reasoning import EnhancedHybridRegistry


class StorageFormat:
    """Storage format specifications and validation."""
    
    SUPPORTED_FORMATS = ["json", "pickle", "compressed", "pytorch", "joblib", "numpy"]
    SUPPORTED_VERSIONS = ["1.0"]
    
    @staticmethod
    def validate_format(format_type: str) -> bool:
        """Validate storage format."""
        return format_type in StorageFormat.SUPPORTED_FORMATS
    
    @staticmethod
    def get_file_extension(format_type: str) -> str:
        """Get file extension for format type."""
        extensions = {
            "json": ".json",
            "pickle": ".pkl", 
            "compressed": ".json.gz",
            "pytorch": ".pth",
            "joblib": ".joblib", 
            "numpy": ".npz"
        }
        return extensions.get(format_type, ".json")
    
    @staticmethod
    def get_optimal_format(data_type: str) -> str:
        """Get optimal storage format for data type."""
        format_mapping = {
            "semantic_structures": "json",      # Frames, concepts, analogies
            "embeddings": "numpy",              # Vector embeddings
            "clustering_models": "joblib",      # Scikit-learn models
            "ltn_models": "pytorch",           # LTN constants and neural networks
            "metadata": "json",                # Configuration and metadata
            "large_data": "compressed"         # Large JSON with compression
        }
        return format_mapping.get(data_type, "json")


@invariant(lambda self: self.storage_path.exists())
@invariant(lambda self: self.storage_path.is_dir())
class PersistenceManager:
    """
    Comprehensive persistence management for soft logic system.
    
    Provides contract-validated save/load functionality for all semantic
    structures including concepts, frames, clusters, and analogical mappings.
    """
    
    @require(lambda storage_path: isinstance(storage_path, (str, Path)))
    def __init__(self, storage_path: Union[str, Path]):
        """Initialize persistence manager with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage structure
        self._init_storage_structure()
    
    def _init_storage_structure(self) -> None:
        """Initialize the storage directory structure."""
        directories = [
            "contexts",
            "models", 
            "models/clustering_models",
            "models/embedding_models",
            "exports",
            "exports/compressed",
            "audit"
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized storage structure at {self.storage_path}")
    
    @require(lambda context_name: len(context_name.strip()) > 0)
    @ensure(lambda result: "save_metadata" in result)
    def save_registry_state(self, registry: 'EnhancedHybridRegistry', 
                           context_name: str = "default",
                           format_type: str = "json") -> Dict[str, Any]:
        """
        Save complete registry state with versioning.
        
        :param registry: The registry to save
        :param context_name: Context identifier
        :param format_type: Storage format (json, pickle, compressed)
        :return: Save operation metadata
        """
        if not StorageFormat.validate_format(format_type):
            raise ViolationError(f"Unsupported format: {format_type}")
        
        context_dir = self.storage_path / "contexts" / context_name
        context_dir.mkdir(parents=True, exist_ok=True)
        
        save_metadata = {
            "context_name": context_name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "format": format_type,
            "version": "1.0",
            "components_saved": []
        }
        
        # Save concepts
        concepts_data = self._serialize_concepts(registry)
        self._save_component(concepts_data, context_dir / f"concepts{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("concepts")
        
        # Save semantic frames
        frames_data = self._serialize_frames(registry)
        self._save_component(frames_data, context_dir / f"frames{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("frames")
        
        # Save frame instances
        instances_data = self._serialize_frame_instances(registry)
        self._save_component(instances_data, context_dir / f"instances{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("instances")
        
        # Save clusters
        clusters_data = self._serialize_clusters(registry)
        self._save_component(clusters_data, context_dir / f"clusters{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("clusters")
        
        # Save semantic fields
        fields_data = self._serialize_semantic_fields(registry)
        self._save_component(fields_data, context_dir / f"fields{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("fields")
        
        # Save cross-domain analogies
        analogies_data = self._serialize_analogies(registry)
        self._save_component(analogies_data, context_dir / f"analogies{StorageFormat.get_file_extension(format_type)}", format_type)
        save_metadata["components_saved"].append("analogies")
        
        # Save embeddings (always use specialized format)
        embeddings_data = self._serialize_embeddings(registry)
        embeddings_dir = context_dir / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        self._save_embeddings(embeddings_data, embeddings_dir)
        save_metadata["components_saved"].append("embeddings")
        
        # Save metadata
        self._save_component(save_metadata, context_dir / "metadata.json", "json")
        
        self.logger.info(f"Saved registry state for context '{context_name}' with {len(save_metadata['components_saved'])} components")
        return save_metadata
    
    @require(lambda context_name: len(context_name.strip()) > 0)
    def load_registry_state(self, context_name: str = "default",
                           format_type: str = "json") -> Optional[Dict[str, Any]]:
        """
        Load complete registry state.
        
        :param context_name: Context identifier
        :param format_type: Expected storage format
        :return: Registry state data or None if not found
        """
        context_dir = self.storage_path / "contexts" / context_name
        
        if not context_dir.exists():
            self.logger.warning(f"Context directory not found: {context_dir}")
            return None
        
        metadata_file = context_dir / "metadata.json"
        if not metadata_file.exists():
            self.logger.warning(f"Metadata file not found: {metadata_file}")
            return None
        
        # Load metadata
        metadata = self._load_component(metadata_file, "json")
        if not metadata:
            return None
        
        # Load all components
        registry_data = {
            "metadata": metadata,
            "concepts": self._load_component(context_dir / f"concepts{StorageFormat.get_file_extension(format_type)}", format_type),
            "frames": self._load_component(context_dir / f"frames{StorageFormat.get_file_extension(format_type)}", format_type),
            "instances": self._load_component(context_dir / f"instances{StorageFormat.get_file_extension(format_type)}", format_type),
            "clusters": self._load_component(context_dir / f"clusters{StorageFormat.get_file_extension(format_type)}", format_type),
            "fields": self._load_component(context_dir / f"fields{StorageFormat.get_file_extension(format_type)}", format_type),
            "analogies": self._load_component(context_dir / f"analogies{StorageFormat.get_file_extension(format_type)}", format_type),
            "embeddings": self._load_embeddings(context_dir / "embeddings")
        }
        
        self.logger.info(f"Loaded registry state for context '{context_name}'")
        return registry_data
    
    @require(lambda format_type: StorageFormat.validate_format(format_type))
    @ensure(lambda result: result.exists())
    def export_knowledge_base(self, context_name: str = "default",
                             format_type: str = "json", 
                             compressed: bool = False) -> Path:
        """
        Export complete knowledge base in specified format.
        
        :param context_name: Context to export
        :param format_type: Export format
        :param compressed: Whether to compress the export
        :return: Path to exported file
        """
        registry_data = self.load_registry_state(context_name, format_type)
        if not registry_data:
            raise ValueError(f"No data found for context: {context_name}")
        
        # Create export filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"knowledge_base_{context_name}_{timestamp}"
        
        if compressed:
            export_path = self.storage_path / "exports" / "compressed" / f"{filename}.json.gz"
            with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, default=self._json_serializer)
        else:
            export_path = self.storage_path / "exports" / f"{filename}.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, default=self._json_serializer)
        
        self.logger.info(f"Exported knowledge base to {export_path}")
        return export_path
    
    def import_knowledge_base(self, source_path: Path, 
                            target_context: str = "imported",
                            merge_strategy: str = "overwrite") -> bool:
        """
        Import knowledge base with conflict resolution.
        
        :param source_path: Path to import file
        :param target_context: Target context name
        :param merge_strategy: How to handle conflicts (overwrite, merge, skip)
        :return: Success status
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Import file not found: {source_path}")
        
        try:
            # Load import data
            if source_path.suffix == '.gz':
                with gzip.open(source_path, 'rt', encoding='utf-8') as f:
                    import_data = json.load(f)
            else:
                with open(source_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            
            # Validate import data structure
            required_components = ["metadata", "concepts", "frames", "instances", "clusters"]
            for component in required_components:
                if component not in import_data:
                    raise ValueError(f"Missing required component: {component}")
            
            # Save imported data to target context
            context_dir = self.storage_path / "contexts" / target_context
            context_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle merge strategy
            if merge_strategy == "overwrite" or not (context_dir / "metadata.json").exists():
                # Direct save
                for component, data in import_data.items():
                    if component == "embeddings":
                        embeddings_dir = context_dir / "embeddings"
                        embeddings_dir.mkdir(exist_ok=True)
                        self._save_embeddings(data, embeddings_dir)
                    else:
                        file_path = context_dir / f"{component}.json"
                        self._save_component(data, file_path, "json")
            else:
                # TODO: Implement merge strategy
                raise NotImplementedError("Merge strategy not yet implemented")
            
            self.logger.info(f"Imported knowledge base from {source_path} to context '{target_context}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import knowledge base: {e}")
            return False
    
    def create_backup(self, context_name: str = "default") -> Path:
        """Create versioned backup of context."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{context_name}_{timestamp}"
        
        return self.export_knowledge_base(
            context_name=context_name,
            format_type="compressed", 
            compressed=True
        )
    
    def list_contexts(self) -> List[str]:
        """List all available contexts."""
        contexts_dir = self.storage_path / "contexts"
        if not contexts_dir.exists():
            return []
        
        return [d.name for d in contexts_dir.iterdir() if d.is_dir()]
    
    def get_context_info(self, context_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific context."""
        context_dir = self.storage_path / "contexts" / context_name
        metadata_file = context_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        return self._load_component(metadata_file, "json")
    
    # Serialization methods
    def _serialize_concepts(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize concepts to dictionary format."""
        concepts_data = {
            "concepts": {},
            "storage_metadata": {
                "version": "1.0",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_concepts": len(registry.frame_aware_concepts)
            }
        }
        
        for concept_id, concept in registry.frame_aware_concepts.items():
            concepts_data["concepts"][concept_id] = {
                "unique_id": concept.unique_id,
                "name": concept.name,
                "context": concept.context,
                "synset_id": concept.synset_id,
                "disambiguation": concept.disambiguation,
                "frame_roles": dict(concept.frame_roles),
                "frame_instances": list(concept.frame_instances),
                "cluster_memberships": getattr(concept, 'cluster_memberships', {}),
                "semantic_fields": getattr(concept, 'semantic_fields', []),
                "created_at": getattr(concept, 'created_at', datetime.now(timezone.utc).isoformat()),
                "metadata": getattr(concept, 'metadata', {})
            }
        
        return concepts_data
    
    def _serialize_frames(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize semantic frames to dictionary format."""
        frames_data = {
            "frames": {},
            "frame_instances": {}
        }
        
        for frame_name, frame in registry.frame_registry.frames.items():
            frames_data["frames"][frame_name] = {
                "name": frame.name,
                "definition": frame.definition,
                "core_elements": [asdict(elem) for elem in frame.core_elements],
                "peripheral_elements": [asdict(elem) for elem in frame.peripheral_elements],
                "lexical_units": list(frame.lexical_units),
                "inherits_from": frame.inherits_from,
                "created_at": getattr(frame, 'created_at', datetime.now(timezone.utc).isoformat())
            }
        
        for instance_id, instance in registry.frame_registry.frame_instances.items():
            frames_data["frame_instances"][instance_id] = {
                "instance_id": instance.instance_id,
                "frame_name": instance.frame_name,
                "element_bindings": {k: v.unique_id if hasattr(v, 'unique_id') else str(v) 
                                   for k, v in instance.element_bindings.items()},
                "context": instance.context,
                "confidence": getattr(instance, 'confidence', 1.0),
                "created_at": getattr(instance, 'created_at', datetime.now(timezone.utc).isoformat())
            }
        
        return frames_data
    
    def _serialize_frame_instances(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize frame instances - included in _serialize_frames for now."""
        return {"instances": {}}  # Placeholder - handled in _serialize_frames
    
    def _serialize_clusters(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize clustering data to dictionary format."""
        clusters_data = {
            "clustering_model": {
                "model_type": "kmeans",
                "n_clusters": registry.cluster_registry.n_clusters,
                "is_trained": registry.cluster_registry.is_trained
            },
            "clusters": {},
            "training_metadata": {
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "num_concepts": len(registry.cluster_registry.concept_embeddings),
                "embedding_dimension": registry.cluster_registry.embedding_dim
            }
        }
        
        if registry.cluster_registry.is_trained and registry.cluster_registry.cluster_centroids is not None:
            clusters_data["clustering_model"]["centroids"] = registry.cluster_registry.cluster_centroids.tolist()
            
            for cluster_id, cluster in registry.cluster_registry.clusters.items():
                clusters_data["clusters"][str(cluster_id)] = {
                    "cluster_id": cluster.cluster_id,
                    "members": dict(cluster.members),
                    "centroid": cluster.centroid.tolist() if cluster.centroid is not None else None,
                    "coherence_score": cluster.coherence_score
                }
        
        return clusters_data
    
    def _serialize_semantic_fields(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize semantic fields to dictionary format."""
        if hasattr(registry, 'semantic_fields') and registry.semantic_fields:
            return {
                "semantic_fields": {
                    field_name: {
                        "name": getattr(field_data, 'name', field_name),
                        "description": getattr(field_data, 'description', ''),
                        "coherence": getattr(field_data, 'coherence', 0.0),
                        "core_concepts": getattr(field_data, 'core_concepts', []),
                        "related_concepts": getattr(field_data, 'related_concepts', {}),
                        "associated_frames": getattr(field_data, 'associated_frames', [])
                    }
                    for field_name, field_data in registry.semantic_fields.items()
                }
            }
        return {"semantic_fields": {}}
    
    def _serialize_analogies(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize cross-domain analogies to dictionary format."""
        if hasattr(registry, 'cross_domain_analogies') and registry.cross_domain_analogies:
            analogies_data = []
            for analogy in registry.cross_domain_analogies:
                analogies_data.append({
                    "source_domain": getattr(analogy, 'source_domain', ''),
                    "target_domain": getattr(analogy, 'target_domain', ''),
                    "concept_mappings": getattr(analogy, 'concept_mappings', {}),
                    "frame_mappings": getattr(analogy, 'frame_mappings', {}),
                    "structural_coherence": getattr(analogy, 'structural_coherence', 0.0),
                    "semantic_coherence": getattr(analogy, 'semantic_coherence', 0.0),
                    "productivity": getattr(analogy, 'productivity', 0.0),
                    "overall_quality": getattr(analogy, 'compute_overall_quality', lambda: 0.0)()
                })
            return {"cross_domain_analogies": analogies_data}
        return {"cross_domain_analogies": []}
    
    def _serialize_embeddings(self, registry: 'EnhancedHybridRegistry') -> Dict[str, Any]:
        """Serialize embeddings to dictionary format."""
        embeddings_data = {
            "embeddings": {},
            "metadata": {}
        }
        
        if hasattr(registry, 'cluster_registry') and registry.cluster_registry.concept_embeddings:
            for concept_id, embedding in registry.cluster_registry.concept_embeddings.items():
                embeddings_data["embeddings"][concept_id] = embedding.tolist()
                embeddings_data["metadata"][concept_id] = {
                    "concept_id": concept_id,
                    "dimensions": len(embedding),
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
        
        return embeddings_data
    
    # Helper methods for file operations
    def _save_component(self, data: Any, file_path: Path, format_type: str) -> None:
        """Save component data in specified format."""
        try:
            if format_type == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            elif format_type == "pickle":
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            elif format_type == "compressed":
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            elif format_type == "joblib":
                joblib.dump(data, file_path)
            elif format_type == "pytorch":
                torch.save(data, file_path)
            elif format_type == "numpy":
                # Convert to numpy arrays for efficient storage
                if isinstance(data, dict) and "embeddings" in data:
                    embeddings = data["embeddings"]
                    if embeddings:
                        concept_ids = list(embeddings.keys())
                        embeddings_array = np.array([embeddings[cid] for cid in concept_ids])
                        np.savez_compressed(file_path, 
                                          embeddings=embeddings_array,
                                          concept_ids=concept_ids)
                else:
                    raise ValueError(f"Unsupported data format for numpy: {type(data)}")
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            self.logger.error(f"Failed to save component to {file_path}: {e}")
            raise
    
    def _load_component(self, file_path: Path, format_type: str) -> Optional[Any]:
        """Load component data from specified format."""
        if not file_path.exists():
            return None
        
        try:
            if format_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif format_type == "pickle":
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif format_type == "compressed":
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            elif format_type == "joblib":
                return joblib.load(file_path)
            elif format_type == "pytorch":
                return torch.load(file_path)
            elif format_type == "numpy":
                with np.load(file_path) as data:
                    embeddings_array = data['embeddings']
                    concept_ids = data['concept_ids']
                    
                    return {
                        "embeddings": {
                            concept_id: embedding.tolist() 
                            for concept_id, embedding in zip(concept_ids, embeddings_array)
                        },
                        "metadata": {}  # Metadata not stored in numpy format
                    }
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            self.logger.error(f"Failed to load component from {file_path}: {e}")
            return None
    
    def _save_embeddings(self, embeddings_data: Dict[str, Any], embeddings_dir: Path) -> None:
        """Save embeddings with optimized format."""
        # Save embeddings as numpy arrays
        embeddings_file = embeddings_dir / "embeddings.npz"
        metadata_file = embeddings_dir / "metadata.json"
        
        if embeddings_data["embeddings"]:
            # Convert to numpy arrays for efficient storage
            concept_ids = list(embeddings_data["embeddings"].keys())
            embeddings_array = np.array([embeddings_data["embeddings"][cid] for cid in concept_ids])
            
            np.savez_compressed(embeddings_file, 
                              embeddings=embeddings_array,
                              concept_ids=concept_ids)
        
        # Save metadata separately
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data["metadata"], f, indent=2, default=self._json_serializer)
    
    def _load_embeddings(self, embeddings_dir: Path) -> Optional[Dict[str, Any]]:
        """Load embeddings from optimized format."""
        embeddings_file = embeddings_dir / "embeddings.npz"
        metadata_file = embeddings_dir / "metadata.json"
        
        if not embeddings_file.exists() or not metadata_file.exists():
            return None
        
        try:
            # Load embeddings
            npz_data = np.load(embeddings_file)
            embeddings_array = npz_data['embeddings']
            concept_ids = npz_data['concept_ids']
            
            embeddings_dict = {
                concept_id: embedding.tolist() 
                for concept_id, embedding in zip(concept_ids, embeddings_array)
            }
            
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return {
                "embeddings": embeddings_dict,
                "metadata": metadata
            }
        except Exception as e:
            self.logger.error(f"Failed to load embeddings from {embeddings_dir}: {e}")
            return None
    
    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
