import h5py
import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional
import tifffile

def read_hdf5(file) -> Tuple[np.ndarray, dict]:
    """Read HDF5 file and return data with metadata."""
    with h5py.File(file, 'r') as f:
        # Assuming standard tomography data structure
        projections = np.array(f['exchange/data'])
        metadata = {
            'shape': projections.shape,
            'dtype': str(projections.dtype),
            'n_projections': projections.shape[0]
        }
    return projections, metadata

def read_tiff_stack(file) -> Tuple[np.ndarray, dict]:
    """Read TIFF stack and return data with metadata."""
    data = tifffile.imread(file)
    if len(data.shape) == 2:
        data = np.expand_dims(data, 0)
    
    metadata = {
        'shape': data.shape,
        'dtype': str(data.dtype),
        'n_projections': data.shape[0]
    }
    return data, metadata

def save_tiff(data: np.ndarray) -> bytes:
    """Convert numpy array to TIFF bytes."""
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, data, format='tiff')
    buffer.seek(0)
    return buffer.getvalue()

def validate_file(file) -> bool:
    """Validate uploaded file format."""
    if file is None:
        return False
    
    filename = file.name.lower()
    return filename.endswith(('.h5', '.hdf5', '.tif', '.tiff'))
