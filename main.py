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
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'reconstructed' not in st.session_state:
    st.session_state.reconstructed = {}
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'configurations' not in st.session_state:
    st.session_state.configurations = {}
if 'current_config' not in st.session_state:
    st.session_state.current_config = None

# Title and description
st.title("Tomographic Reconstruction")
st.markdown("Upload your tomographic data and process it through the reconstruction pipeline.")

# File upload
uploaded_files = st.file_uploader(
    "Upload HDF5 or TIFF files",
    type=['h5', 'hdf5', 'tif', 'tiff'],
    accept_multiple_files=True,
    help="Supported formats: HDF5 (.h5, .hdf5) and TIFF (.tif, .tiff)"
)

# Process uploaded files
for uploaded_file in uploaded_files:
    if validate_file(uploaded_file):
        # Read data
        try:
            if uploaded_file.name.lower().endswith(('.h5', '.hdf5')):
                data, metadata = read_hdf5(uploaded_file)
            else:
                data, metadata = read_tiff_stack(uploaded_file)
            
            # Store dataset in session state
            st.session_state.datasets[uploaded_file.name] = {
                'data': data,
                'metadata': metadata
            }
                
        except Exception as e:
            st.error(f"Error reading file {uploaded_file.name}: {str(e)}")

# Dataset selection
if st.session_state.datasets:
    st.subheader("Available Datasets")
    dataset_names = list(st.session_state.datasets.keys())
    selected_dataset = st.selectbox(
        "Select dataset to view/process",
        dataset_names,
        index=0 if st.session_state.current_dataset is None else dataset_names.index(st.session_state.current_dataset)
    )
    st.session_state.current_dataset = selected_dataset
    
    # Display metadata for selected dataset
    if selected_dataset:
        dataset = st.session_state.datasets[selected_dataset]
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Projections", dataset['metadata']['shape'][0])
        with col2:
            st.metric("Height", dataset['metadata']['shape'][1])
        with col3:
            st.metric("Width", dataset['metadata']['shape'][2])

# Processing parameters
if st.session_state.datasets:
    st.subheader("Processing Parameters")
    
    # Configuration management
    col1, col2 = st.columns(2)
    with col1:
        # Save current configuration
        config_name = st.text_input("Configuration Name", key="config_name")
        if st.button("Save Configuration"):
            if config_name:
                st.session_state.configurations[config_name] = {
                    'normalize': st.session_state.get('normalize', True),
                    'remove_rings': st.session_state.get('remove_rings', True),
                    'ring_level': st.session_state.get('ring_level', 1.0)
                }
                st.success(f"Configuration '{config_name}' saved!")
    
    with col2:
        # Load configuration
        if st.session_state.configurations:
            selected_config = st.selectbox(
                "Select Configuration",
                ['Default'] + list(st.session_state.configurations.keys()),
                key="selected_config"
            )
            
            if selected_config != 'Default' and selected_config in st.session_state.configurations:
                config = st.session_state.configurations[selected_config]
                st.session_state.normalize = config['normalize']
                st.session_state.remove_rings = config['remove_rings']
                st.session_state.ring_level = config['ring_level']
                st.session_state.current_config = selected_config
    
    st.markdown("---")
    
    # Processing parameters
    col1, col2 = st.columns(2)
    with col1:
        normalize = st.checkbox("Apply Normalization", 
                              value=st.session_state.get('normalize', True),
                              key='normalize')
        remove_rings = st.checkbox("Remove Ring Artifacts", 
                                 value=st.session_state.get('remove_rings', True),
                                 key='remove_rings')
    
    with col2:
        ring_level = st.slider(
            "Ring Removal Strength",
            0.1, 5.0, 
            value=st.session_state.get('ring_level', 1.0),
            disabled=not remove_rings,
            key='ring_level'
        )
    
    # Process buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Selected Dataset"):
            with st.spinner(f"Processing {st.session_state.current_dataset}..."):
                try:
                    data = st.session_state.datasets[st.session_state.current_dataset]['data']
                    reconstructed, center = process_pipeline(
                        data,
                        normalize=normalize,
                        remove_rings=remove_rings,
                        ring_level=ring_level
                    )
                    st.session_state.reconstructed[st.session_state.current_dataset] = reconstructed
                    st.success(f"Processing complete! Center of rotation: {center:.2f}")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
    
    with col2:
        if st.button("Process All Datasets"):
            for dataset_name in st.session_state.datasets:
                with st.spinner(f"Processing {dataset_name}..."):
                    try:
                        data = st.session_state.datasets[dataset_name]['data']
                        reconstructed, center = process_pipeline(
                            data,
                            normalize=normalize,
                            remove_rings=remove_rings,
                            ring_level=ring_level
                        )
                        st.session_state.reconstructed[dataset_name] = reconstructed
                        st.success(f"Processed {dataset_name}! Center of rotation: {center:.2f}")
                    except Exception as e:
                        st.error(f"Processing {dataset_name} failed: {str(e)}")

# Visualization
if st.session_state.current_dataset and st.session_state.current_dataset in st.session_state.reconstructed:
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Sinogram")
        data = st.session_state.datasets[st.session_state.current_dataset]['data']
        sino_idx = create_slice_navigator(data, "sino")
        display_slice(data, sino_idx, "")
        
    with col2:
        st.markdown("### Reconstruction")
        reconstructed = st.session_state.reconstructed[st.session_state.current_dataset]
        recon_idx = create_slice_navigator(reconstructed, "recon")
        display_slice(reconstructed, recon_idx, "")
    
    # Export results
    st.subheader("Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Current Reconstruction"):
            try:
                current_reconstructed = st.session_state.reconstructed[st.session_state.current_dataset]
                tiff_bytes = save_tiff(current_reconstructed)
                st.download_button(
                    label="Download TIFF",
                    data=tiff_bytes,
                    file_name=f"reconstruction_{st.session_state.current_dataset}.tiff",
                    mime="image/tiff"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        if st.button("Download All Reconstructions"):
            try:
                for dataset_name, reconstructed in st.session_state.reconstructed.items():
                    tiff_bytes = save_tiff(reconstructed)
                    st.download_button(
                        label=f"Download {dataset_name}",
                        data=tiff_bytes,
                        file_name=f"reconstruction_{dataset_name}.tiff",
                        mime="image/tiff"
                    )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
