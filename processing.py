import numpy as np
from typing import Optional, Tuple

def normalize_data(projections: np.ndarray,
                  flat_field: Optional[np.ndarray] = None,
                  dark_field: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize projection data using flat and dark fields."""
    # Convert inputs to float32
    projections = projections.astype(np.float32)
    if flat_field is not None:
        flat_field = flat_field.astype(np.float32)
    if dark_field is not None:
        dark_field = dark_field.astype(np.float32)
    
    if flat_field is None:
        # Use simple min-max normalization if no flat/dark fields
        return (projections - projections.min()) / (projections.max() - projections.min())
    
    # Apply flat/dark field correction
    norm = (projections - dark_field) / (flat_field - dark_field)
    return np.clip(norm, 0, None)  # Ensure non-negative values

def remove_ring_artifacts(data: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Remove ring artifacts using median filtering."""
    # Simple ring removal using median filter
    filtered = np.copy(data).astype(np.float32)
    for i in range(data.shape[0]):
        filtered[i] = np.median(data[max(0, i-int(level)):min(data.shape[0], i+int(level)+1)], axis=0).astype(np.float32)
    return filtered

def find_center_of_rotation(data: np.ndarray) -> np.float32:
    """Estimate center of rotation using image symmetry."""
    # Simple center estimation using the middle of the image
    return np.float32(data.shape[2] / 2.0)

def reconstruct_slice(data: np.ndarray,
                     theta: np.ndarray,
                     center: float,
                     algorithm: str = 'simple') -> np.ndarray:
    """Simple backprojection reconstruction."""
    num_angles = data.shape[0]
    img_size = data.shape[2]
    reconstruction = np.zeros((img_size, img_size), dtype=np.float32)
    
    # Simple backprojection
    for i, angle in enumerate(theta):
        projection = data[i].astype(np.float32)
        rotated = np.rot90(np.tile(projection, (img_size, 1)), k=int(angle * 2/np.pi))
        reconstruction += rotated
    
    return (reconstruction / num_angles).astype(np.float32)

def process_pipeline(data: np.ndarray,
                    normalize: bool = True,
                    remove_rings: bool = True,
                    ring_level: float = 1.0) -> dict:
    """Complete processing pipeline with intermediate results."""
    results = {
        'original': data.astype(np.float32),
        'normalized': None,
        'ring_corrected': None,
        'reconstructed': None,
        'center': None,
        'theta': np.linspace(0, np.pi, data.shape[0], dtype=np.float32)
    }
    
    # Normalization
    if normalize:
        results['normalized'] = normalize_data(data)
        data = results['normalized']
    else:
        results['normalized'] = data
    
    # Ring artifact removal
    if remove_rings:
        results['ring_corrected'] = remove_ring_artifacts(data, ring_level)
        data = results['ring_corrected']
    else:
        results['ring_corrected'] = data
    
    # Find center of rotation
    results['center'] = find_center_of_rotation(data)
    
    # Reconstruction
    reconstructed = np.zeros((data.shape[1], data.shape[2], data.shape[2]))
    for i in range(data.shape[1]):
        slice_data = data[:, i:i+1, :]
        reconstructed[i] = reconstruct_slice(slice_data, results['theta'], results['center'])
    
    results['reconstructed'] = reconstructed
    return results
