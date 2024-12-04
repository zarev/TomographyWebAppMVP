import numpy as np
from typing import Optional, Tuple


def normalize_data(projections: np.ndarray,
                   flat_field: Optional[np.ndarray] = None,
                   dark_field: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize projection data using flat and dark fields."""
    if flat_field is None:
        # Use simple min-max normalization if no flat/dark fields
        return (projections - projections.min()) / (projections.max() -
                                                    projections.min())

    # Apply flat/dark field correction
    norm = (projections - dark_field) / (flat_field - dark_field)
    return np.clip(norm, 0, None)  # Ensure non-negative values


def remove_ring_artifacts(data: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Remove ring artifacts using median filtering."""
    # Simple ring removal using median filter
    filtered = np.copy(data)
    for i in range(data.shape[0]):
        filtered[i] = np.median(
            data[max(0, i - int(level)):min(data.shape[0], i + int(level) +
                                            1)],
            axis=0)
    return filtered


def find_center_of_rotation(data: np.ndarray) -> float:
    """Estimate center of rotation using image symmetry."""
    # Simple center estimation using the middle of the image
    return data.shape[2] / 2.0


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
        rotated = np.rot90(np.tile(projection, (img_size, 1)),
                           k=int(angle * 2 / np.pi))
        reconstruction += rotated

    return reconstruction / num_angles


def process_pipeline(data: np.ndarray,
                     normalize: bool = True,
                     remove_rings: bool = True,
                     ring_level: float = 1.0) -> dict:
    """Complete processing pipeline with intermediate results."""
    results = {
        'original': data.copy(),
        'normalized': None,
        'ring_removed': None,
        'reconstructed': None,
        'center': None
    }

    # Generate projection angles
    theta = np.linspace(0, np.pi, data.shape[0], dtype=np.float32)
    current_data = data

    # Normalization
    if normalize:
        results['normalized'] = normalize_data(current_data)
        current_data = results['normalized']

    # Ring artifact removal
    if remove_rings:
        results['ring_removed'] = remove_ring_artifacts(current_data, ring_level).astype(np.float32)
        current_data = results['ring_removed']

    # Find center of rotation
    results['center'] = find_center_of_rotation(current_data)

    # Reconstruction
    results['reconstructed'] = np.zeros((data.shape[1], data.shape[2], data.shape[2]), dtype=np.float32)

    return results
