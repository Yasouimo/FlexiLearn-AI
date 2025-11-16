import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO messages (0=all, 1=filter INFO, 2=filter WARNING, 3=filter ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN (optional, might be slower)

import streamlit as st
from ui.sidebar import show_sidebar
from ui.data_preview import show_data_preview
from ui.model_configuration import show_model_configuration
from ui.results_display import show_results
from utils.session_state import initialize_session_state

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 FlexiLearn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
initialize_session_state()

# --- Main App ---
st.title("🧠 FlexiLearn AI Playground")

# Show sidebar and get configuration
sidebar_config = show_sidebar()

# Extract configuration
uploaded_file = sidebar_config['uploaded_file']
df = sidebar_config['df']
data_type = sidebar_config['data_type']
problem_type = sidebar_config['problem_type']
model_family = sidebar_config['model_family']

# Main content flow
if uploaded_file is None and df is None:
    st.info("Welcome! Please choose your data source in the sidebar to get started.")
else:
    # Step 1: Data Preview & Preprocessing
    show_data_preview(uploaded_file, df, data_type, problem_type)
    
    # Step 2: Model Configuration & Training
    # For time series, we need df in session state; for others, we need processed 'data'
    if data_type == "Time Series":
        # Time series uses df directly, show config if df is available
        if df is not None or 'df' in st.session_state:
            show_model_configuration(data_type, problem_type, model_family)
    else:
        # Classical ML and Deep Learning need processed data
        if 'data' in st.session_state:
            show_model_configuration(data_type, problem_type, model_family)
    
    # Step 3: Results & Evaluation
    if st.session_state.runs:
        show_results(data_type, problem_type)