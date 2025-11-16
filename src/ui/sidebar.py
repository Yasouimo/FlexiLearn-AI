# File: /Streamlit_app_deep/Streamlit_app_deep/src/ui/sidebar.py

import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, fetch_california_housing
from utils.timeseries_utils import generate_synthetic_sales_data

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
    data_source = st.sidebar.selectbox("Data Source", ["Upload File", "Load Classic Dataset", "Generate Synthetic Time Series"])

    uploaded_file, df = None, None
    data_type = None
    
    if data_source == "Upload File":
        data_type = st.sidebar.selectbox("Data Type", ("Tabular (CSV/Excel)", "Image (ZIP)", "Time Series"))
        
        if data_type == "Time Series":
            allowed_types = ['csv', 'xlsx']
            st.sidebar.info("📁 Max file size: 500MB")
            uploaded_file = st.sidebar.file_uploader("Upload your time series file", type=allowed_types)
            
            if uploaded_file:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Let user select date and target columns
                    st.sidebar.subheader("Column Selection")
                    date_col = st.sidebar.selectbox("Date Column", df.columns)
                    target_col = st.sidebar.selectbox("Target Column", df.columns, index=1 if len(df.columns) > 1 else 0)
                    
                    # Convert to datetime and set as index
                    df[date_col] = pd.to_datetime(df[date_col])
                    df.set_index(date_col, inplace=True)
                    df = df[[target_col]]  # Keep only target column
                    # Store in session state for RNN training
                    st.session_state.df = df

                except Exception as e:
                    st.sidebar.error(f"Error reading file: {e}")
        
        elif data_type == "Tabular (CSV/Excel)":
            allowed_types = ['csv', 'xlsx', 'txt']
            st.sidebar.info("📁 Max file size: 500MB")
            uploaded_file = st.sidebar.file_uploader("Upload your file", type=allowed_types)
        else:
            allowed_types = ['zip']
            st.sidebar.info("📁 Max file size: 500MB")
            uploaded_file = st.sidebar.file_uploader("Upload your file", type=allowed_types)
    
    elif data_source == "Generate Synthetic Time Series":
        st.sidebar.success("Generating synthetic sales data...")
        num_points = st.sidebar.slider("Number of Days", 365, 365*5, 365*3)
        df = generate_synthetic_sales_data(num_points)
        data_type = "Time Series"
        st.sidebar.info(f"Generated {len(df)} days of synthetic sales data")
        # Store in session state for RNN training
        st.session_state.df = df
    
    else:  # Load Classic Dataset
        dataset_name = st.sidebar.selectbox("Choose Dataset", ("Iris (Classification)", "California Housing (Regression)"))
        if dataset_name == "Iris (Classification)":
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target
            data_type = "Tabular (CSV/Excel)"
        else:
            housing = fetch_california_housing()
            df = pd.DataFrame(housing.data, columns=housing.feature_names)
            df['price'] = housing.target
            data_type = "Tabular (CSV/Excel)"

    # Problem Type Selection
    st.sidebar.header("2. Define Your Problem")
    if data_type == "Time Series":
        problem_type = "Forecasting"
        st.sidebar.info("Time Series → Forecasting")
    elif data_type == "Image (ZIP)":
        problem_type = "Classification"
        st.sidebar.info("Images → Classification Only")
    else:
        problem_type = st.sidebar.selectbox("Problem Type", ("Classification", "Regression"))

    # Model Family Selection
    st.sidebar.header("3. Choose Model Family")
    if data_type == "Time Series":
        model_family = st.sidebar.selectbox("Model Family", ("RNN (Recurrent Neural Networks)",))
    elif data_type == "Tabular (CSV/Excel)":
        model_family = st.sidebar.selectbox("Model Family", ("Classical ML (Scikit-learn)", "Deep Learning (PyTorch)"))
    else:
        model_family = "Deep Learning (PyTorch)"
        st.sidebar.info("Images → Deep Learning (CNNs)")

    return {
        'uploaded_file': uploaded_file,
        'df': df,
        'data_type': data_type,
        'problem_type': problem_type,
        'model_family': model_family
    }