import numpy as np
import streamlit as st
from PIL import Image
import io

def display_slice(data: np.ndarray, slice_idx: int, title: str):
    """Display a single slice with controls."""
    if data is None or len(data.shape) < 2:
        return
    
    # Normalize for display
    slice_data = data[slice_idx]
    normalized = ((slice_data - slice_data.min()) / 
                 (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(normalized)
    
    # Display with title
    st.subheader(title)
    st.image(img, use_column_width=True)

def create_slice_navigator(data: np.ndarray, key_prefix: str) -> int:
    """Create a slider for navigating through slices."""
    if data is None or len(data.shape) < 2:
        return 0
    
    max_slice = data.shape[0] - 1
    slice_idx = st.slider(
        "Select slice",
        0, max_slice, max_slice // 2,
        key=f"{key_prefix}_slider"
    )
    return slice_idx


def create_histogram(data: np.ndarray, bins: int = 100):
    """Create a histogram of the data values."""
    if data is None:
        return
    
    # Calculate histogram
    hist, bins = np.histogram(data.flatten(), bins=bins)
    
    # Create figure using streamlit
    st.subheader("Intensity Distribution")
    
    # Plot histogram
    st.bar_chart(
        data={"intensity": hist},
        width=0,
        height=200
    )
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{data.min():.2f}")
    with col2:
        st.metric("Max", f"{data.max():.2f}")
    with col3:
        st.metric("Mean", f"{data.mean():.2f}")
    with col4:
        st.metric("Std", f"{data.std():.2f}")