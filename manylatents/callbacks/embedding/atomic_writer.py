"""
Atomic file writing utilities for EmbeddingOutputs.

Ensures multi-process writes don't corrupt data on cluster nodes.
"""

import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any


def serialize_embedding_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize EmbeddingOutputs dict for JSON storage.
    
    Separates numpy arrays (saved as .npy) from metadata (saved as JSON).
    
    Args:
        outputs: EmbeddingOutputs dict containing embeddings, scores, metadata, config
        
    Returns:
        Serializable dict with embeddings metadata
    """
    serialized = {
        'embeddings_info': {
            'shape': outputs['embeddings'].shape,
            'dtype': str(outputs['embeddings'].dtype),
            'saved_as': 'embeddings.npy'
        },
        'scores': outputs.get('scores', {}),
        'metadata': outputs.get('metadata', {}),
        'config': outputs.get('config', {})
    }
    
    return serialized


def write_embedding_outputs_atomic(
    outputs: Dict[str, Any],
    output_path: Path,
    save_embeddings: bool = True
) -> None:
    """
    Atomically write EmbeddingOutputs to avoid corruption from concurrent processes.
    
    Uses the atomic rename pattern (write to temp, then rename) which is
    atomic on POSIX filesystems (cluster nodes).
    
    Args:
        outputs: EmbeddingOutputs dict containing:
            - embeddings: np.ndarray
            - scores: dict of metrics
            - metadata: dict of timing/shape info
            - config: dict of algorithm hyperparameters
        output_path: Path to write JSON metadata (e.g., outputs.json)
        save_embeddings: Whether to save embeddings array as .npy file
        
    Raises:
        ValueError: If outputs missing required keys
    """
    if 'embeddings' not in outputs:
        raise ValueError("EmbeddingOutputs must contain 'embeddings' key")
    
    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Serialize metadata
    serialized = serialize_embedding_outputs(outputs)
    
    # Write metadata to temp file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=output_path.parent,
        delete=False,
        suffix='.tmp',
        prefix='.tmp_'
    ) as tmp:
        json.dump(serialized, tmp, indent=2)
        tmp_path = Path(tmp.name)
    
    # Atomic rename (POSIX guarantees atomicity)
    tmp_path.rename(output_path)
    
    # Save embeddings separately as binary (also atomic)
    if save_embeddings:
        embeddings_path = output_path.with_suffix('.npy')
        
        # Write to temp, then rename
        with tempfile.NamedTemporaryFile(
            dir=embeddings_path.parent,
            delete=False,
            suffix='.tmp',
            prefix='.tmp_embeddings_'
        ) as tmp:
            tmp_embeddings_path = Path(tmp.name)
        
        np.save(tmp_embeddings_path, outputs['embeddings'])
        tmp_embeddings_path.rename(embeddings_path)


def load_embedding_outputs(output_path: Path) -> Dict[str, Any]:
    """
    Load EmbeddingOutputs from atomically written files.
    
    Args:
        output_path: Path to JSON metadata file (outputs.json)
        
    Returns:
        Dict with embeddings (if .npy exists), scores, metadata, config
        
    Raises:
        FileNotFoundError: If output_path doesn't exist
    """
    if not output_path.exists():
        raise FileNotFoundError(f"Output file not found: {output_path}")
    
    # Load metadata
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Load embeddings if available
    embeddings_path = output_path.with_suffix('.npy')
    if embeddings_path.exists():
        data['embeddings'] = np.load(embeddings_path)
    else:
        data['embeddings'] = None
    
    return data


def write_step_outputs(
    outputs: Dict[str, Any],
    step_dir: Path,
    step_idx: int,
    step_name: str
) -> Path:
    """
    Write outputs for a workflow step with standardized naming.
    
    Args:
        outputs: EmbeddingOutputs dict
        step_dir: Directory for this step's outputs
        step_idx: Step index
        step_name: Step name
        
    Returns:
        Path to written outputs.json file
    """
    step_dir.mkdir(parents=True, exist_ok=True)
    
    # Add step context to metadata
    if 'metadata' not in outputs:
        outputs['metadata'] = {}
    outputs['metadata'].update({
        'step_idx': step_idx,
        'step_name': step_name
    })
    
    # Write outputs
    output_path = step_dir / "outputs.json"
    write_embedding_outputs_atomic(outputs, output_path)
    
    return output_path
