import numpy as np
from typing import Optional, Tuple
import logging

import sys

print("Python executable:", sys.executable)
print("Python path:", sys.path)
try:
    import tomopy
    print(f"TomoPy version: {tomopy.__version__}")
except ImportError as e:
    print(f"ImportError: {e}")

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
                     algorithm: str = 'gridrec') -> np.ndarray:
    """
    Reconstruction using TomoPy or simple backprojection.

    Parameters:
    - data (np.ndarray): The projection data.
    - theta (np.ndarray): The projection angles.
    - center (float): The center of rotation.
    - algorithm (str): Reconstruction algorithm ('gridrec', 'art', etc., or 'simple').

    Returns:
    - np.ndarray: The reconstructed slice.
    """
    logger.info(f"Starting reconstruction with algorithm: {algorithm}")

    
    # Ensure input data has the correct shape: (projections, slices, pixels)
    if data.ndim != 3 or data.shape[0] != len(theta):
        raise ValueError(f"Invalid data shape: {data.shape}. "
                            f"Expected shape (projections, slices, pixels).")

    # Validate TomoPy algorithm
    supported_tomopy_algorithms = ['gridrec', 'art', 'mlem', 'osem', 'sirt', 'pml_hybrid']
    
    if algorithm.lower() != 'simple':
        try:
            import tomopy
            logger.info("Successfully imported TomoPy")

            if algorithm not in supported_tomopy_algorithms:
                raise ValueError(f"Unsupported TomoPy algorithm: '{algorithm}'. "
                                 f"Supported algorithms are: {supported_tomopy_algorithms}")

            # If 'center' is not provided accurately, TomoPy can estimate it
            logger.info(f"Reconstructing with center: {center}")

            # Perform TomoPy reconstruction
            reconstruction = tomopy.recon(data, theta, center=center, algorithm=algorithm)
            logger.info("Reconstruction completed using TomoPy")

            # TomoPy returns an array with shape (slices, pixels, pixels)
            # Since we're reconstructing a single slice, return the first (and only) slice
            return reconstruction[0]

        except ImportError:
            logger.error("Failed to import TomoPy")
        except Exception as e:
            logger.error(f"TomoPy reconstruction failed: {str(e)}")

    if algorithm.lower() == 'simple':
        logger.info("Using simple backprojection")

        try:
            num_angles = data.shape[0]
            img_size = data.shape[2]
            reconstruction = np.zeros((img_size, img_size))

            for i, angle in enumerate(theta):
                projection = data[i, 0, :]  # Assuming single slice
                rotated = np.rot90(np.tile(projection, (img_size, 1)), k=int(np.round(angle * 2 / np.pi)))
                reconstruction += rotated

                if (i + 1) % 10 == 0 or (i + 1) == num_angles:
                    logger.info(f"Reconstructed slice {i+1}/{num_angles}")

            reconstruction /= num_angles
            logger.info("Simple backprojection reconstruction completed successfully")
            return reconstruction

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