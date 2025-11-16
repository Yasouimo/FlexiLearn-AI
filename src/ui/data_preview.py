from collections import OrderedDict
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from models.custom_datasets import ImageZipDataset
from statsmodels.tsa.seasonal import seasonal_decompose

def show_data_preview(uploaded_file, df, data_type, problem_type):
    """Display data preview and preprocessing section"""
    st.header("Step 1: Data Preview & Preprocessing")
    
    if data_type == "Tabular (CSV/Excel)":
        _show_tabular_preview(uploaded_file, df, problem_type)
    elif data_type == "Image (ZIP)":
        _show_image_preview(uploaded_file)
    elif data_type == "Time Series":
        _show_timeseries_preview(df)

def _show_tabular_preview(uploaded_file, df, problem_type):
    """Handle tabular data preview and preprocessing"""
    # Load data if from uploaded file
    if df is None:
        try:
            if uploaded_file.name.endswith(('.csv', '.txt')):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    
    # Display preview
    st.dataframe(df.head())
    st.write("Data Shape:", df.shape)
    
    # Preprocessing controls in sidebar
    st.sidebar.subheader("Preprocessing Options")
    target_column = st.sidebar.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
    
    if st.sidebar.button("🚀 Quick Process Data"):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        num_features = X.select_dtypes(include=np.number).columns.tolist()
        cat_features = X.select_dtypes(include=object).columns.tolist()
        
        # Build preprocessing pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
        
        # Encode target
        if problem_type == "Classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            st.session_state.target_encoder = le
        else:
            y_encoded = y.values
        
        # Transform and split
        X_processed = preprocessor.fit_transform(X)
        st.session_state.preprocessor = preprocessor
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42
        )
        
        # Store in session state
        st.session_state.data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'n_features': X_processed.shape[1],
            'n_classes': len(np.unique(y_encoded)) if problem_type == "Classification" else 1,
            'class_names': list(le.classes_) if problem_type == "Classification" and hasattr(le, 'classes_') else None
        }
        st.success("Data processed and split successfully!")

def _show_image_preview(uploaded_file):
    """Handle image data preview and preprocessing"""
    st.sidebar.subheader("Image Preprocessing")
    
    # Get root folder options
    try:
        zip_buffer = io.BytesIO(uploaded_file.getvalue())
        with zipfile.ZipFile(zip_buffer, 'r') as z:
            potential_roots = set([''] + [
                os.path.dirname(os.path.dirname(p)) for p in z.namelist() 
                if '/' in p and not p.startswith('__MACOSX') and len(p.split('/')) > 2
            ])
        image_root_folder = st.sidebar.selectbox(
            "Select folder containing classes", 
            sorted(list(potential_roots))
        )
    except Exception as e:
        st.error(f"Invalid ZIP file. Error: {e}")
        st.stop()
    
    # Image preprocessing options
    img_size = st.sidebar.slider("Resize Images to (pixels)", 32, 256, 64)
    test_split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.5, 0.2)
    
    if st.sidebar.button("🖼️ Process Images"):
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        full_dataset = ImageZipDataset(
            io.BytesIO(uploaded_file.getvalue()), 
            transform=transform, 
            root_folder=image_root_folder
        )
        
        if len(full_dataset) == 0:
            st.error(f"No images found in folder '{image_root_folder}'. Please select another.")
        else:
            st.session_state.class_names = list(full_dataset.idx_to_class.values())
            
            # Split dataset
            train_size = int((1 - test_split_ratio) * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size]
            )
            
            # Store in session state
            st.session_state.data = {
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'img_size': img_size,  # FIX: Store img_size here
                'n_classes': len(full_dataset.class_to_idx),
                'class_names': st.session_state.class_names
            }
            
            st.success(f"Processed {len(full_dataset)} images into {len(st.session_state.class_names)} classes.")
            
            # Display sample images
            st.subheader("Sample Images")
            sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            images, labels = next(iter(sample_loader))
            
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            for i, ax in enumerate(axes):
                img = images[i].permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                ax.set_title(f"Class: {st.session_state.class_names[labels[i]]}")
                ax.axis('off')
            st.pyplot(fig)

def _show_timeseries_preview(df):
    """Handle time series data preview and analysis"""
    if df is None:
        return
    
    st.subheader("📈 Time Series Data")
    
    # Display basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")
    with col3:
        st.metric("Target Column", df.columns[0])
    
    # Display data
    st.dataframe(df.head(20))
    
    # Time series plot
    st.subheader("📊 Time Series Visualization")
    st.line_chart(df)
    
    # Decomposition analysis
    st.subheader("🔍 Seasonal Decomposition")
    decomp_model = st.selectbox(
        "Decomposition Model",
        ["Additive", "Multiplicative"],
        help="Additive: Seasonality is constant. Multiplicative: Seasonality grows with trend."
    )
    
    if st.button("Perform Decomposition"):
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(
                df[df.columns[0]], 
                model=decomp_model.lower(),
                period=min(365, len(df) // 2)  # Use yearly seasonality or half the data
            )
            
            # Plot components
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            # Original
            axes[0].plot(df.index, df[df.columns[0]], label='Original', color='blue')
            axes[0].set_ylabel('Original')
            axes[0].legend()
            axes[0].grid(True)
            
            # Trend
            axes[1].plot(df.index, decomposition.trend, label='Trend', color='orange')
            axes[1].set_ylabel('Trend')
            axes[1].legend()
            axes[1].grid(True)
            
            # Seasonal
            axes[2].plot(df.index, decomposition.seasonal, label='Seasonal', color='green')
            axes[2].set_ylabel('Seasonal')
            axes[2].legend()
            axes[2].grid(True)
            
            # Residual
            axes[3].plot(df.index, decomposition.resid, label='Residual', color='red')
            axes[3].set_ylabel('Residual')
            axes[3].legend()
            axes[3].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success("Decomposition complete! This helps identify trend, seasonality, and noise in your data.")
            
        except Exception as e:
            st.error(f"Decomposition failed: {e}. Try using more data points or a different period.")