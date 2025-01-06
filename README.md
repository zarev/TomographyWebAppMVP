
# Tomographic Reconstruction Web App

A Streamlit-based web application for processing and reconstructing tomographic data.

## Features

- Upload and process HDF5 or TIFF files
- Visualization of projection data and reconstructions 
- Support for multiple reconstruction algorithms (Simple backprojection, ASTRA)
- Ring artifact removal
- Data normalization
- Interactive slice navigation
- Export reconstructions as TIFF files

## Installation

1. Fork this Repl from the Replit interface
2. The dependencies will be automatically installed based on the pyproject.toml file

Required packages:
- numpy
- streamlit 
- h5py
- pillow
- tifffile
- astra-toolbox

## Usage

1. Run the application:
   - Click the "Run" button in Replit
   - The app will start on port 5000

2. Upload Data:
   - Use the file uploader to select HDF5 (.h5, .hdf5) or TIFF (.tif, .tiff) files
   - Multiple TIFF files will be automatically combined into a single dataset

3. Process Data:
   - Select a dataset from the dropdown menu
   - Configure processing parameters:
     - Normalization
     - Ring artifact removal (adjustable strength)
     - Reconstruction algorithm selection
   - Click "Process Selected Dataset" or "Process All Datasets"

4. View Results:
   - Navigate through slices using the slider
   - View original projections and reconstructed data
   - Examine intensity distributions through histograms

5. Export Results:
   - Download individual reconstructions or all results as TIFF files

## Notes

- For large datasets, processing may take several minutes
- The ASTRA reconstruction algorithm runs in CPU-only mode
- Data is processed in float32 format to optimize memory usage
