import streamlit as st
from contextlib import contextmanager
import time

def validate_file(uploaded_file):
    """Validate uploaded file size and format."""
    # Check file size (limit to 100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes
    
    if uploaded_file.size > MAX_FILE_SIZE:
        return False
    
    # Check file extension
    valid_extensions = ['.csv', '.xlsx']
    file_extension = uploaded_file.name.lower()[-4:]
    if not any(file_extension.endswith(ext) for ext in valid_extensions):
        return False
    
    return True

@contextmanager
def show_progress(message):
    """Context manager for showing progress spinner."""
    with st.spinner(message):
        yield
