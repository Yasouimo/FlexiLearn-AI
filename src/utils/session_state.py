import streamlit as st

def initialize_session_state():
    """Initialize all session state variables"""
    if 'runs' not in st.session_state:
        st.session_state.runs = []
    if 'custom_ann_layers' not in st.session_state:
        st.session_state.custom_ann_layers = [{'units': 64, 'activation': 'ReLU', 'dropout': 0.2}]
    if 'custom_cnn_layers' not in st.session_state:
        st.session_state.custom_cnn_layers = [
            {'type': 'Conv2d', 'filters': 16, 'kernel': 3},
            {'type': 'MaxPool2d', 'kernel': 2},
            {'type': 'Conv2d', 'filters': 32, 'kernel': 3},
            {'type': 'MaxPool2d', 'kernel': 2},
        ]