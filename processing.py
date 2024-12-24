import numpy as np
from typing import Optional, Tuple

def normalize_data(projections: np.ndarray,
                  flat_field: Optional[np.ndarray] = None,
                  dark_field: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize projection data using flat and dark fields."""
    if flat_field is None:
        # Use simple min-max normalization if no flat/dark fields
        return (projections - projections.min()) / (projections.max() - projections.min())
    
    # Apply flat/dark field correction
    norm = (projections - dark_field) / (flat_field - dark_field)
    return np.clip(norm, 0, None)  # Ensure non-negative values

def remove_ring_artifacts(data: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Remove ring artifacts using median filtering."""
    # Simple ring removal using median filter
    filtered = np.copy(data)
    for i in range(data.shape[0]):
        filtered[i] = np.median(data[max(0, i-int(level)):min(data.shape[0], i+int(level)+1)], axis=0)
    return filtered

def find_center_of_rotation(data: np.ndarray) -> float:
    """Estimate center of rotation using image symmetry."""
    # Simple center estimation using the middle of the image
    return data.shape[2] / 2.0

def reconstruct_slice(data: np.ndarray,
                     theta: np.ndarray,
                     center: float,
                     algorithm: str = 'simple') -> np.ndarray:
    """Reconstruction using simple backprojection or ASTRA."""
    if algorithm == 'astra':
        import astra
        vol_geom = astra.create_vol_geom(data.shape[2], data.shape[2])
        proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[2], theta)
        
        # Create sinogram and volume
        sino_id = astra.data2d.create('-sino', proj_geom, data[0])
        rec_id = astra.data2d.create('-vol', vol_geom)
        
        # Create configuration and algorithm
        cfg = astra.astra_dict('FBP')
        cfg['ProjectorId'] = astra.create_projector('line', proj_geom, vol_geom)
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = rec_id
        
        # Run reconstruction
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        reconstruction = astra.data2d.get(rec_id)
        
        # Cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sino_id)
        
        return reconstruction
    
    # Simple backprojection
    num_angles = data.shape[0]
    img_size = data.shape[2]
    reconstruction = np.zeros((img_size, img_size))
    for i, angle in enumerate(theta):
        projection = data[i]
        rotated = np.rot90(np.tile(projection, (img_size, 1)), k=int(angle * 2/np.pi))
        reconstruction += rotated
    
    return reconstruction / num_angles

def process_pipeline(data: np.ndarray,
                    normalize: bool = True,
                    remove_rings: bool = True,
                    ring_level: float = 1.0,
                    algorithm: str = 'simple') -> Tuple[np.ndarray, float]:
    """Complete processing pipeline."""
    # Generate projection angles
    theta = np.linspace(0, np.pi, data.shape[0])
    
    # Normalization
    if normalize:
        data = normalize_data(data)
    
    # Ring artifact removal
    if remove_rings:
        data = remove_ring_artifacts(data, ring_level)
    
    # Find center of rotation
    center = find_center_of_rotation(data)
    
    # Reconstruction
    reconstructed = np.zeros((data.shape[1], data.shape[2], data.shape[2]))
    for i in range(data.shape[1]):
        slice_data = data[:, i:i+1, :]
        reconstructed[i] = reconstruct_slice(slice_data, theta, center, algorithm=algorithm)
    
    return reconstructed, center
