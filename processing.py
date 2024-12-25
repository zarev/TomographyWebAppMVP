import numpy as np
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_data(projections: np.ndarray,
                  flat_field: Optional[np.ndarray] = None,
                  dark_field: Optional[np.ndarray] = None) -> np.ndarray:
    """Normalize projection data using flat and dark fields."""
    logger.info(f"Normalizing data with shape: {projections.shape}")
    if flat_field is None:
        # Use simple min-max normalization if no flat/dark fields
        return (projections - projections.min()) / (projections.max() - projections.min())

    # Apply flat/dark field correction
    norm = (projections - dark_field) / (flat_field - dark_field)
    return np.clip(norm, 0, None)  # Ensure non-negative values

def remove_ring_artifacts(data: np.ndarray, level: float = 1.0) -> np.ndarray:
    """Remove ring artifacts using median filtering."""
    logger.info(f"Removing ring artifacts with level: {level}")
    # Simple ring removal using median filter
    filtered = np.copy(data)
    for i in range(data.shape[0]):
        filtered[i] = np.median(data[max(0, i-int(level)):min(data.shape[0], i+int(level)+1)], axis=0)
    return filtered

def find_center_of_rotation(data: np.ndarray) -> float:
    """Estimate center of rotation using image symmetry."""
    logger.info("Estimating center of rotation")
    # Simple center estimation using the middle of the image
    return data.shape[2] / 2.0

def reconstruct_slice(data: np.ndarray,
                     theta: np.ndarray,
                     center: float,
                     algorithm: str = 'simple') -> np.ndarray:
    """Reconstruction using simple backprojection or ASTRA."""
    logger.info(f"Starting reconstruction with algorithm: {algorithm}")

    # Validate input data
    if data is None or len(data.shape) < 2:
        raise ValueError("Invalid input data dimensions")

    if algorithm == 'astra':
        try:
            import astra
            logger.info("Successfully imported ASTRA")

            # Validate data dimensions
            if len(data.shape) != 3 or data.shape[0] != len(theta):
                raise ValueError(f"Invalid data shape for ASTRA: {data.shape}")

            vol_geom = astra.create_vol_geom(data.shape[2], data.shape[2])
            proj_geom = astra.create_proj_geom('parallel', 1.0, data.shape[2], theta)

            logger.info("Created ASTRA geometries")

            # Create sinogram and volume
            sino_id = astra.data2d.create('-sino', proj_geom, data[0])
            rec_id = astra.data2d.create('-vol', vol_geom)

            # Create configuration and algorithm
            # Try CUDA first, fall back to CPU if not available
            try:
                cfg = astra.astra_dict('FBP_CUDA')
                logger.info("Using CUDA-accelerated reconstruction")
            except Exception:
                cfg = astra.astra_dict('FBP')
                logger.info("Using CPU-based reconstruction")

            # Create CPU-compatible projector
            cfg['ProjectorId'] = astra.create_projector('linear', proj_geom, vol_geom)
            cfg['ProjectionDataId'] = sino_id
            cfg['ReconstructionDataId'] = rec_id

            logger.info("Starting ASTRA reconstruction")

            # Run reconstruction
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            reconstruction = astra.data2d.get(rec_id)

            logger.info("ASTRA reconstruction completed")

            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sino_id)

            return reconstruction

        except ImportError:
            logger.error("Failed to import ASTRA, falling back to simple reconstruction")
            algorithm = 'simple'
        except Exception as e:
            logger.error(f"ASTRA reconstruction failed: {str(e)}")
            logger.info("Falling back to simple reconstruction")
            algorithm = 'simple'

    if algorithm == 'simple':
        logger.info("Using simple backprojection")
        # Simple backprojection
        num_angles = data.shape[0]
        img_size = data.shape[2]
        reconstruction = np.zeros((img_size, img_size))

        try:
            for i, angle in enumerate(theta):
                projection = data[i]
                rotated = np.rot90(np.tile(projection, (img_size, 1)), k=int(angle * 2/np.pi))
                reconstruction += rotated

            return reconstruction / num_angles
        except Exception as e:
            logger.error(f"Simple reconstruction failed: {str(e)}")
            raise

def process_pipeline(data: np.ndarray,
                    normalize: bool = True,
                    remove_rings: bool = True,
                    ring_level: float = 1.0,
                    algorithm: str = 'simple') -> Tuple[np.ndarray, float]:
    """Complete processing pipeline."""
    logger.info(f"Starting processing pipeline with algorithm: {algorithm}")

    try:
        # Validate input data
        if data is None or len(data.shape) < 2:
            raise ValueError("Invalid input data dimensions")

        logger.info(f"Input data shape: {data.shape}")

        # Generate projection angles
        theta = np.linspace(0, np.pi, data.shape[0])

        # Normalization
        if normalize:
            data = normalize_data(data)
            logger.info("Normalization completed")

        # Ring artifact removal
        if remove_rings:
            data = remove_ring_artifacts(data, ring_level)
            logger.info("Ring artifact removal completed")

        # Find center of rotation
        center = find_center_of_rotation(data)
        logger.info(f"Center of rotation: {center}")

        # Reconstruction
        reconstructed = np.zeros((data.shape[1], data.shape[2], data.shape[2]))
        logger.info("Starting slice-by-slice reconstruction")

        for i in range(data.shape[1]):
            slice_data = data[:, i:i+1, :]
            reconstructed[i] = reconstruct_slice(slice_data, theta, center, algorithm=algorithm)
            if i % 10 == 0:  # Log progress every 10 slices
                logger.info(f"Reconstructed slice {i+1}/{data.shape[1]}")

        logger.info("Reconstruction completed successfully")
        return reconstructed, center

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise