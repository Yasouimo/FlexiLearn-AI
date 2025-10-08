# File: /Streamlit_app_deep/Streamlit_app_deep/src/ui/sidebar.py

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing

def show_architecture_guide(data_type, problem_type):
    """Show architecture guide in sidebar"""
    with st.sidebar.expander("🧠 Architecture Guide & Suggestions"):
        st.markdown("#### General Principles")
        st.info("Start simple! A smaller network trains faster and is less likely to 'memorize' your data (overfit). Add complexity only if needed.")
        
        if data_type == "Tabular (CSV/Excel)":
            st.markdown("---")
            st.markdown("#### Suggested Starting Points (ANN)")
            if problem_type == "Classification":
                st.success("**For Tabular Classification**: A 'funnel' shape with 2-3 layers is a great start. Use `ReLU` activation and add `Dropout` to prevent overfitting.\n* **Example**: `Input -> 128 neurons (ReLU) -> Dropout(0.3) -> 64 neurons (ReLU) -> Output`")
            else:
                st.success("**For Tabular Regression**: Similar to classification, but the **final layer must have 1 neuron and no activation function**.\n* **Example**: `Input -> 64 neurons (ReLU) -> 32 neurons (ReLU) -> 1 Neuron`")
            st.markdown("---")
            st.markdown("#### Layer Explanations (ANN)")
            st.markdown("**Dense (Linear)**: The standard 'brain cell' layer. It connects every input to every output neuron.")
            st.markdown("**Activation (ReLU)**: The 'switch'. It decides if a neuron should pass on information, adding non-linearity to the network.")
            st.markdown("**Dropout**: The 'forgetfulness' mechanism. It randomly ignores some neurons during training, forcing the network to learn more robust features.")
        
        elif data_type == "Image (ZIP)":
            st.markdown("---")
            st.markdown("#### Suggested Starting Points (CNN)")
            st.success("**For Image Classification**: The classic pattern is to stack `Conv2d -> Pool` blocks. Start with fewer filters and increase them as the network gets deeper.\n* **Example**: `Input -> Conv(16 filters) -> Pool -> Conv(32 filters) -> Pool -> Flatten -> Dense -> Output`")
            st.markdown("---")
            st.markdown("#### Layer Explanations (CNN)")
            st.markdown("**Conv2d**: The 'Pattern Detector'. It's like a flashlight scanning the image for features like edges, corners, and textures.")
            st.markdown("**MaxPool2d**: The 'Summarizer'. It shrinks the image, keeping only the most important features found by the convolutional layer. This makes the model faster and more efficient.")
            st.markdown("**Dense (Linear)**: After features are extracted, these layers perform the final classification, just like in an ANN.")

def show_sidebar():
    """Display and handle all sidebar interactions"""
    st.sidebar.title("FlexiLearn AI Configuration")
    st.sidebar.markdown("Your personal ML/DL playground. 🚀")

    st.sidebar.header("1. Choose Your Data")
    data_source = st.sidebar.selectbox("Data Source", ["Upload File", "Load Classic Dataset"])

    uploaded_file, df = None, None
    data_type = None
    
    if data_source == "Upload File":
        data_type = st.sidebar.selectbox("Data Type", ("Tabular (CSV/Excel)", "Image (ZIP)"))
        if data_type == "Tabular (CSV/Excel)":
            allowed_types = ['csv', 'xlsx', 'txt']
        else:
            allowed_types = ['zip', 'tgz']
        uploaded_file = st.sidebar.file_uploader("Upload your file", type=allowed_types)
    else:
        dataset_name = st.sidebar.selectbox("Choose a classic dataset", ["Iris (Classification)", "California Housing (Regression)"])
        data_type = "Tabular (CSV/Excel)"
        if dataset_name == "Iris (Classification)":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
        else:
            data = fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target

    st.sidebar.header("2. Configure Your Model")
    if data_type == "Tabular (CSV/Excel)":
        problem_type = st.sidebar.selectbox("Problem Type", ("Classification", "Regression"))
    else:
        problem_type = "Classification"
    
    model_family = st.sidebar.selectbox("Model Family", ("Classical ML (Scikit-learn)", "Deep Learning (PyTorch)"))

    st.sidebar.header("Actions")
    if st.sidebar.button("🗑️ Start New Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    return {
        'data_source': data_source,
        'data_type': data_type,
        'uploaded_file': uploaded_file,
        'df': df,
        'problem_type': problem_type,
        'model_family': model_family,
        'show_architecture_guide': show_architecture_guide
    }