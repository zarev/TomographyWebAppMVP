import streamlit as st
import numpy as np
from utils import read_hdf5, read_tiff_stack, save_tiff, validate_file
from processing import process_pipeline
from visualization import display_slice, create_slice_navigator

# Page config
st.set_page_config(
    page_title="Tomographic Reconstruction",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'reconstructed' not in st.session_state:
    st.session_state.reconstructed = None

# Title and description
st.title("Tomographic Reconstruction")
st.markdown("Upload your tomographic data and process it through the reconstruction pipeline.")

# File upload
uploaded_file = st.file_uploader(
    "Upload HDF5 or TIFF file",
    type=['h5', 'hdf5', 'tif', 'tiff'],
    help="Supported formats: HDF5 (.h5, .hdf5) and TIFF (.tif, .tiff)"
)

if uploaded_file is not None and validate_file(uploaded_file):
    # Read data
    try:
        if uploaded_file.name.lower().endswith(('.h5', '.hdf5')):
            data, metadata = read_hdf5(uploaded_file)
        else:
            data, metadata = read_tiff_stack(uploaded_file)
        
        st.session_state.data = data
        
        # Display metadata
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Projections", metadata['shape'][0])
        with col2:
            st.metric("Height", metadata['shape'][1])
        with col3:
            st.metric("Width", metadata['shape'][2])
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.session_state.data = None

# Processing parameters
if st.session_state.data is not None:
    st.subheader("Processing Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        normalize = st.checkbox("Apply Normalization", value=True)
        remove_rings = st.checkbox("Remove Ring Artifacts", value=True)
    
    with col2:
        ring_level = st.slider(
            "Ring Removal Strength",
            0.1, 5.0, 1.0,
            disabled=not remove_rings
        )
    
    # Process button
    if st.button("Run Processing Pipeline"):
        with st.spinner("Processing data..."):
            try:
                reconstructed, center = process_pipeline(
                    st.session_state.data,
                    normalize=normalize,
                    remove_rings=remove_rings,
                    ring_level=ring_level
                )
                st.session_state.reconstructed = reconstructed
                st.success(f"Processing complete! Center of rotation: {center:.2f}")
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")

# Visualization
if st.session_state.data is not None and st.session_state.reconstructed is not None:
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sinogram")
        sino_idx = create_slice_navigator(st.session_state.data, "sino")
        display_slice(st.session_state.data, sino_idx, "")
        
    with col2:
        st.markdown("### Reconstruction")
        recon_idx = create_slice_navigator(st.session_state.reconstructed, "recon")
        display_slice(st.session_state.reconstructed, recon_idx, "")
    
    # Export results
    st.subheader("Export Results")
    if st.button("Download Reconstruction"):
        try:
            tiff_bytes = save_tiff(st.session_state.reconstructed)
            st.download_button(
                label="Download TIFF",
                data=tiff_bytes,
                file_name="reconstruction.tiff",
                mime="image/tiff"
            )
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
