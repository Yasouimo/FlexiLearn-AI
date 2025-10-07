import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import os
import base64
from PIL import Image
from collections import OrderedDict

# Plotting Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn for data and classical models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score, classification_report
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Deep Learning (PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms

# --- Page Configuration ---
st.set_page_config(
    page_title="🧠 FlexiLearn AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions & Classes ---

def get_model_download_link(model, model_name):
    # This function is unchanged
    if isinstance(model, nn.Module):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pth">Download Model</a>'
    else:
        import pickle
        b = io.BytesIO()
        pickle.dump(model, b)
        b_val = b.getvalue()
        b64 = base64.b64encode(b_val).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{model_name}.pkl">Download Model</a>'

# All plotting functions are unchanged
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
    st.pyplot(fig)

def plot_roc_curve(y_true, y_prob, n_classes, class_names):
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fig, ax = plt.subplots(figsize=(8, 6))
    if n_classes <= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_prob[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve'); plt.legend(loc="lower right")
    st.pyplot(fig)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history['train_acc']); ax1.plot(history['val_acc'])
    ax1.set_title('Model accuracy'); ax1.set_ylabel('Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend(['Train', 'Validation'], loc='upper left')
    ax2.plot(history['train_loss']); ax2.plot(history['val_loss'])
    ax2.set_title('Model loss'); ax2.set_ylabel('Loss'); ax2.set_xlabel('Epoch'); ax2.legend(['Train', 'Validation'], loc='upper left')
    st.pyplot(fig)

class ImageZipDataset(Dataset):
    # This class is unchanged
    def __init__(self, zip_file_buffer, transform=None, root_folder=''):
        self.zip_file_buffer = zip_file_buffer; self.transform = transform
        self.image_paths = []; self.labels = []; self.class_to_idx = {}
        if root_folder and not root_folder.endswith('/'): root_folder += '/'
        with zipfile.ZipFile(self.zip_file_buffer, 'r') as zf:
            class_names = set()
            for file_path in zf.namelist():
                if file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                    relative_path = file_path[len(root_folder):]
                    if '/' in relative_path: class_names.add(relative_path.split('/')[0])
            self.class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
            self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
            for file_path in zf.namelist():
                if file_path.endswith(('.png', '.jpg', '.jpeg')) and file_path.startswith(root_folder) and not file_path.startswith('__MACOSX'):
                    relative_path = file_path[len(root_folder):]
                    if '/' in relative_path:
                        class_name = relative_path.split('/')[0]
                        if class_name in self.class_to_idx:
                            self.image_paths.append(file_path); self.labels.append(self.class_to_idx[class_name])
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_file_buffer, 'r') as zf:
            with zf.open(self.image_paths[idx]) as f: image = Image.open(f).convert('RGB')
        label = self.labels[idx]
        if self.transform: image = self.transform(image)
        return image, label

def display_model_architecture(model):
    """Creates a simple text representation of a PyTorch model."""
    st.subheader("Model Architecture")
    st.code(str(model))

# --- NEW: Enhanced Architecture Guide ---
def show_architecture_guide(data_type, problem_type):
    with st.sidebar.expander("🧠 Architecture Guide & Suggestions"):
        st.markdown("#### General Principles")
        st.info("Start simple! A smaller network trains faster and is less likely to 'memorize' your data (overfit). Add complexity only if needed.")
        
        if data_type == "Tabular (CSV/Excel)":
            st.markdown("---")
            st.markdown("#### Suggested Starting Points (ANN)")
            if problem_type == "Classification":
                st.success("**For Tabular Classification**: A 'funnel' shape with 2-3 layers is a great start. Use `ReLU` activation and add `Dropout` to prevent overfitting.\n* **Example**: `Input -> 128 neurons (ReLU) -> Dropout(0.3) -> 64 neurons (ReLU) -> Output`")
            else: # Regression
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

# --- Session State Initialization ---
if 'runs' not in st.session_state: st.session_state.runs = []
if 'custom_ann_layers' not in st.session_state: st.session_state.custom_ann_layers = [{'units': 64, 'activation': 'ReLU', 'dropout': 0.2}]
if 'custom_cnn_layers' not in st.session_state:
    st.session_state.custom_cnn_layers = [
        {'type': 'Conv2d', 'filters': 16, 'kernel': 3},
        {'type': 'MaxPool2d', 'kernel': 2},
        {'type': 'Conv2d', 'filters': 32, 'kernel': 3},
        {'type': 'MaxPool2d', 'kernel': 2},
    ]

# --- Sidebar ---
st.sidebar.title("FlexiLearn AI Configuration")
st.sidebar.markdown("Your personal ML/DL playground. 🚀")

st.sidebar.header("1. Choose Your Data")
data_source = st.sidebar.selectbox("Data Source", ["Upload File", "Load Classic Dataset"])

uploaded_file, df = None, None
if data_source == "Upload File":
    data_type = st.sidebar.selectbox("Data Type", ("Tabular (CSV/Excel)", "Image (ZIP)"))
    if data_type == "Tabular (CSV/Excel)": allowed_types = ['csv', 'xlsx', 'txt']
    else: allowed_types = ['zip', 'tgz']
    uploaded_file = st.sidebar.file_uploader("Upload your file", type=allowed_types)
else:
    dataset_name = st.sidebar.selectbox("Choose a classic dataset", ["Iris (Classification)", "California Housing (Regression)"])
    data_type = "Tabular (CSV/Excel)"
    if dataset_name == "Iris (Classification)": data = load_iris(); df = pd.DataFrame(data.data, columns=data.feature_names); df['target'] = data.target
    else: data = fetch_california_housing(); df = pd.DataFrame(data.data, columns=data.feature_names); df['target'] = data.target

st.sidebar.header("2. Configure Your Model")
if data_type == "Tabular (CSV/Excel)": problem_type = st.sidebar.selectbox("Problem Type", ("Classification", "Regression"))
else: problem_type = "Classification"
model_family = st.sidebar.selectbox("Model Family", ("Classical ML (Scikit-learn)", "Deep Learning (PyTorch)"))

st.sidebar.header("Actions")
if st.sidebar.button("🗑️ Start New Session"):
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# --- Main App Body ---
st.title("🧠 FlexiLearn AI Playground")

if uploaded_file is None and df is None:
    st.info("Welcome! Please choose your data source in the sidebar to get started.")
else:
    st.header("Step 1: Data Preview & Preprocessing")
    
    if data_type == "Tabular (CSV/Excel)":
        if df is None:
            try:
                if uploaded_file.name.endswith(('.csv', '.txt')): df = pd.read_csv(uploaded_file)
                else: df = pd.read_excel(uploaded_file)
            except Exception as e: st.error(f"Error reading file: {e}"); st.stop()
        st.dataframe(df.head()); st.write("Data Shape:", df.shape)
        st.sidebar.subheader("Preprocessing Options")
        target_column = st.sidebar.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)
        if st.sidebar.button("🚀 Quick Process Data"):
            X = df.drop(columns=[target_column]); y = df[target_column]
            num_features = X.select_dtypes(include=np.number).columns.tolist()
            cat_features = X.select_dtypes(include=object).columns.tolist()
            num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])
            if problem_type == "Classification": le = LabelEncoder(); y_encoded = le.fit_transform(y); st.session_state.target_encoder = le
            else: y_encoded = y.values
            X_processed = preprocessor.fit_transform(X); st.session_state.preprocessor = preprocessor
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)
            st.session_state.data = {
                'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                'n_features': X_processed.shape[1],
                'n_classes': len(np.unique(y_encoded)) if problem_type == "Classification" else 1,
                'class_names': list(le.classes_) if problem_type == "Classification" and hasattr(le, 'classes_') else None
            }
            st.success("Data processed and split successfully!")

    elif data_type == "Image (ZIP)":
        st.sidebar.subheader("Image Preprocessing")
        try:
            zip_buffer = io.BytesIO(uploaded_file.getvalue())
            with zipfile.ZipFile(zip_buffer, 'r') as z:
                potential_roots = set([''] + [os.path.dirname(os.path.dirname(p)) for p in z.namelist() if '/' in p and not p.startswith('__MACOSX') and len(p.split('/')) > 2])
            image_root_folder = st.sidebar.selectbox("Select folder containing classes", sorted(list(potential_roots)))
        except Exception as e: st.error(f"Invalid ZIP file. Error: {e}"); st.stop()
        img_size = st.sidebar.slider("Resize Images to (pixels)", 32, 256, 64)
        test_split_ratio = st.sidebar.slider("Train-Test Split Ratio", 0.1, 0.5, 0.2)
        if st.sidebar.button("🖼️ Process Images"):
            transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            full_dataset = ImageZipDataset(io.BytesIO(uploaded_file.getvalue()), transform=transform, root_folder=image_root_folder)
            if len(full_dataset) == 0:
                st.error(f"No images found in folder '{image_root_folder}'. Please select another.")
            else:
                st.session_state.class_names = list(full_dataset.idx_to_class.values())
                train_size = int((1 - test_split_ratio) * len(full_dataset)); test_size = len(full_dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
                st.session_state.data = {
                    'train_dataset': train_dataset, 'test_dataset': test_dataset, 'img_size': img_size,
                    'n_classes': len(full_dataset.class_to_idx), 'class_names': st.session_state.class_names
                }
                st.success(f"Processed {len(full_dataset)} images into {len(st.session_state.class_names)} classes.")
                st.subheader("Sample Images")
                sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
                images, labels = next(iter(sample_loader))
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                for i, ax in enumerate(axes):
                    img = images[i].permute(1, 2, 0).numpy()
                    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
                    img = std * img + mean; img = np.clip(img, 0, 1)
                    ax.imshow(img); ax.set_title(f"Class: {st.session_state.class_names[labels[i]]}"); ax.axis('off')
                st.pyplot(fig)

    if 'data' in st.session_state:
        st.header("Step 2: Model Configuration & Training")
        model = None
        if model_family == "Classical ML (Scikit-learn)":
            # This logic is unchanged
            if data_type != "Tabular (CSV/Excel)": st.warning("Classical ML models are for tabular data only.")
            else:
                model_type = "Unknown"
                if problem_type == "Classification":
                    model_type = st.selectbox("Select Model", ("Logistic Regression", "Random Forest Classifier", "Support Vector Machine (SVM)", "K-Nearest Neighbors", "Decision Tree Classifier"))
                    if model_type == "Logistic Regression": model = LogisticRegression(C=st.slider("Regularization (C)", 0.01, 10.0, 1.0), max_iter=500, random_state=42)
                    elif model_type == "Random Forest Classifier": model = RandomForestClassifier(n_estimators=st.slider("Number of Trees", 10, 500, 100), max_depth=st.slider("Max Depth", 2, 32, 10), random_state=42)
                    elif model_type == "Support Vector Machine (SVM)": model = SVC(C=st.slider("Regularization (C)", 0.1, 10.0, 1.0), kernel=st.selectbox("Kernel", ('rbf', 'linear', 'poly')), probability=True, random_state=42)
                    elif model_type == "K-Nearest Neighbors": model = KNeighborsClassifier(n_neighbors=st.slider("Number of Neighbors (K)", 1, 20, 5))
                    elif model_type == "Decision Tree Classifier": model = DecisionTreeClassifier(max_depth=st.slider("Max Depth", 2, 32, 10), random_state=42)
                elif problem_type == "Regression":
                    model_type = st.selectbox("Select Model", ("Linear Regression", "Random Forest Regressor", "Support Vector Regressor (SVR)"))
                    if model_type == "Linear Regression": model = LinearRegression()
                    elif model_type == "Random Forest Regressor": model = RandomForestRegressor(n_estimators=st.slider("Number of Trees", 10, 500, 100), max_depth=st.slider("Max Depth", 2, 32, 10), random_state=42)
                    elif model_type == "Support Vector Regressor (SVR)": model = SVR(C=st.slider("Regularization (C)", 0.1, 10.0, 1.0), kernel=st.selectbox("Kernel", ('rbf', 'linear', 'poly')))
                if model is not None and st.button(f"💪 Train {model_type}"):
                    with st.spinner("Training in progress..."):
                        data = st.session_state.data; model.fit(data['X_train'], data['y_train']); y_pred = model.predict(data['X_test'])
                        run_info = {'model_name': model_type, 'model': model, 'family': 'Classical'}
                        if problem_type == "Classification":
                            accuracy = accuracy_score(data['y_test'], y_pred); y_prob = model.predict_proba(data['X_test']) if hasattr(model, 'predict_proba') else None
                            run_info.update({'Accuracy': f"{accuracy:.4f}", 'y_pred': y_pred, 'y_prob': y_prob})
                        else:
                            mse = mean_squared_error(data['y_test'], y_pred); r2 = r2_score(data['y_test'], y_pred)
                            run_info.update({'MSE': f"{mse:.4f}", 'R2 Score': f"{r2:.4f}", 'y_pred': y_pred})
                        st.session_state.runs.append(run_info)
                    st.success("Training complete!")

        elif model_family == "Deep Learning (PyTorch)":
            st.sidebar.subheader("Network Architecture")
            show_architecture_guide(data_type, problem_type) # Call the guide function
            arch_mode = st.sidebar.selectbox("Architecture Mode", ["Suggested", "Custom"])
            
            if data_type == "Tabular (CSV/Excel)":
                layers = OrderedDict(); input_size, output_size = st.session_state.data['n_features'], st.session_state.data['n_classes'] if problem_type == "Classification" else 1
                if arch_mode == "Suggested":
                    template = st.sidebar.selectbox("Choose ANN template", ["Simple (2 layers)", "Medium (3 layers + Dropout)"])
                    if template == "Simple (2 layers)": layers['layer_1'] = nn.Linear(input_size, 128); layers['act_1'] = nn.ReLU(); layers['layer_2'] = nn.Linear(128, 64); layers['act_2'] = nn.ReLU(); layers['output'] = nn.Linear(64, output_size)
                    else: layers['layer_1'] = nn.Linear(input_size, 128); layers['act_1'] = nn.ReLU(); layers['dropout_1'] = nn.Dropout(0.3); layers['layer_2'] = nn.Linear(128, 64); layers['act_2'] = nn.ReLU(); layers['dropout_2'] = nn.Dropout(0.3); layers['layer_3'] = nn.Linear(64, 32); layers['act_3'] = nn.ReLU(); layers['output'] = nn.Linear(32, output_size)
                else:
                    st.sidebar.markdown("Define your Dense layers:"); last_size = input_size
                    for i, layer_params in enumerate(st.session_state.custom_ann_layers):
                        with st.sidebar.expander(f"Layer {i+1}", expanded=True):
                            layer_params['units'] = st.number_input("Neurons", 1, 2048, layer_params['units'], key=f"ann_units_{i}")
                            layer_params['activation'] = st.selectbox("Activation", ['ReLU', 'LeakyReLU', 'Tanh'], key=f"ann_act_{i}")
                            layer_params['dropout'] = st.slider("Dropout", 0.0, 0.9, layer_params['dropout'], key=f"ann_drop_{i}")
                        layers[f'layer_{i}'] = nn.Linear(last_size, layer_params['units']); layers[f'act_{i}'] = getattr(nn, layer_params['activation'])()
                        if layer_params['dropout'] > 0: layers[f'dropout_{i}'] = nn.Dropout(layer_params['dropout'])
                        last_size = layer_params['units']
                    cols = st.sidebar.columns(2)
                    if cols[0].button("Add Layer"): st.session_state.custom_ann_layers.append({'units': 32, 'activation': 'ReLU', 'dropout': 0.2}); st.rerun()
                    if cols[1].button("Remove Last") and len(st.session_state.custom_ann_layers) > 1: st.session_state.custom_ann_layers.pop(); st.rerun()
                    # Final layer logic for regression
                    if problem_type == "Regression":
                        layers['output'] = nn.Linear(last_size, 1)
                    else:
                        layers['output'] = nn.Linear(last_size, output_size)
                model = nn.Sequential(layers)

            elif data_type == "Image (ZIP)":
                conv_layers = OrderedDict(); last_channels = 3; current_dim = st.session_state.data['img_size']
                if arch_mode == "Suggested":
                     st.sidebar.info("Using a standard CNN: Conv -> Pool -> Conv -> Pool -> FC -> Output")
                     cnn_layer_configs = st.session_state.custom_cnn_layers
                else:
                    st.sidebar.markdown("Define your Conv/Pool layers:")
                    cnn_layer_configs = st.session_state.custom_cnn_layers
                    for i, p in enumerate(cnn_layer_configs):
                        with st.sidebar.expander(f"Layer {i+1}: {p['type']}", expanded=True):
                            if p['type'] == 'Conv2d':
                                p['filters'] = st.number_input("Filters", 4, 512, p['filters'], step=4, key=f"cnn_filters_{i}")
                                p['kernel'] = st.slider("Kernel Size", 3, 7, p['kernel'], step=2, key=f"cnn_kernel_{i}")
                            elif p['type'] == 'MaxPool2d':
                                p['kernel'] = st.slider("Pool Size/Stride", 2, 4, p['kernel'], key=f"cnn_pool_{i}")
                    cols = st.sidebar.columns(3)
                    if cols[0].button("Add Conv"): st.session_state.custom_cnn_layers.append({'type': 'Conv2d', 'filters': 64, 'kernel': 3}); st.rerun()
                    if cols[1].button("Add Pool"): st.session_state.custom_cnn_layers.append({'type': 'MaxPool2d', 'kernel': 2}); st.rerun()
                    if cols[2].button("Remove Last") and len(st.session_state.custom_cnn_layers) > 1: st.session_state.custom_cnn_layers.pop(); st.rerun()
                for i, p in enumerate(cnn_layer_configs):
                    if p['type'] == 'Conv2d':
                        conv_layers[f'conv_{i}'] = nn.Conv2d(last_channels, p['filters'], kernel_size=p['kernel'], padding='same')
                        conv_layers[f'relu_{i}'] = nn.ReLU(); last_channels = p['filters']
                    elif p['type'] == 'MaxPool2d':
                        conv_layers[f'pool_{i}'] = nn.MaxPool2d(kernel_size=p['kernel'], stride=p['kernel'])
                        current_dim //= p['kernel']
                class DynamicCNN(nn.Module):
                    def __init__(self, conv_part, flattened_size, num_classes):
                        super().__init__(); self.conv_part = conv_part; self.flatten = nn.Flatten()
                        self.classifier = nn.Sequential(nn.Linear(flattened_size, 128), nn.ReLU(), nn.Linear(128, num_classes))
                    def forward(self, x): return self.classifier(self.flatten(self.conv_part(x)))
                flattened_size = last_channels * current_dim * current_dim
                model = DynamicCNN(nn.Sequential(conv_layers), flattened_size, st.session_state.data['n_classes'])
            
            if model: display_model_architecture(model)

            st.sidebar.subheader("Training Parameters")
            lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
            epochs = st.sidebar.number_input("Epochs", 1, 500, 10)
            batch_size = st.sidebar.number_input("Batch Size", 8, 512, 32)
            if st.button("🧠 Train Deep Learning Model"):
                data = st.session_state.data
                if data_type == "Tabular (CSV/Excel)":
                    train_ds = TensorDataset(torch.Tensor(data['X_train']), torch.Tensor(data['y_train']).long() if problem_type == "Classification" else torch.Tensor(data['y_train']).view(-1, 1))
                    test_ds = TensorDataset(torch.Tensor(data['X_test']), torch.Tensor(data['y_test']).long() if problem_type == "Classification" else torch.Tensor(data['y_test']).view(-1, 1))
                else: train_ds, test_ds = data['train_dataset'], data['test_dataset']
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True); test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss() if problem_type == "Classification" else nn.MSELoss()
                history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                progress_bar = st.progress(0); status_text = st.empty()
                for epoch in range(epochs):
                    model.train(); train_loss, train_correct, train_total = 0, 0, 0
                    for inputs, labels in train_loader:
                        optimizer.zero_grad(); outputs = model(inputs); loss = criterion(outputs, labels); loss.backward(); optimizer.step(); train_loss += loss.item()
                        if problem_type == "Classification": _, pred = torch.max(outputs.data, 1); train_total += labels.size(0); train_correct += (pred == labels).sum().item()
                    model.eval(); val_loss, val_correct, val_total = 0, 0, 0; all_preds, all_labels, all_probs = [], [], []
                    with torch.no_grad():
                        for inputs, labels in test_loader:
                            outputs = model(inputs); loss = criterion(outputs, labels); val_loss += loss.item()
                            if problem_type == "Classification":
                                _, pred = torch.max(outputs.data, 1); val_total += labels.size(0); val_correct += (pred == labels).sum().item()
                                all_preds.extend(pred.numpy()); all_labels.extend(labels.numpy()); all_probs.extend(torch.softmax(outputs, dim=1).numpy())
                            else: all_preds.extend(outputs.numpy()); all_labels.extend(labels.numpy())
                    avg_train_loss = train_loss/len(train_loader); avg_val_loss = val_loss/len(test_loader)
                    train_acc = train_correct/train_total if train_total > 0 else 0; val_acc = val_correct/val_total if val_total > 0 else 0
                    history['train_loss'].append(avg_train_loss); history['val_loss'].append(avg_val_loss); history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
                    status_text.text(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
                    progress_bar.progress((epoch + 1) / epochs)
                st.success("Training complete!")
                run_info = {'model_name': 'PyTorch ' + ('CNN' if data_type == 'Image (ZIP)' else 'ANN'), 'model': model, 'family': 'Deep Learning', 'history': history}
                if problem_type == "Classification": run_info.update({'Accuracy': f"{val_acc:.4f}", 'y_pred': np.array(all_preds), 'y_prob': np.array(all_probs)})
                else: 
                    all_preds = np.array(all_preds).flatten(); all_labels = np.array(all_labels).flatten()
                    mse = mean_squared_error(all_labels, all_preds); r2 = r2_score(all_labels, all_preds)
                    run_info.update({'MSE': f"{mse:.4f}", 'R2 Score': f"{r2:.4f}"})
                st.session_state.runs.append(run_info)

    if st.session_state.runs:
        st.header("Step 3: Evaluation & Comparison")
        results_data = []
        for i, run in enumerate(st.session_state.runs):
            res = {'Run': i, 'Model': run['model_name'], 'Family': run['family']}
            if 'Accuracy' in run: res['Accuracy'] = run['Accuracy']
            if 'MSE' in run: res['MSE'] = run['MSE']
            if 'R2 Score' in run: res['R2 Score'] = run['R2 Score']
            results_data.append(res)
        st.dataframe(pd.DataFrame(results_data).set_index('Run'))
        selected_run_idx = st.selectbox("Select a run to inspect in detail:", range(len(st.session_state.runs)), format_func=lambda x: f"Run {x}: {st.session_state.runs[x]['model_name']}")
        run = st.session_state.runs[selected_run_idx]
        data = st.session_state.data
        y_test_data = data['y_test'] if data_type == "Tabular (CSV/Excel)" else [label for _, label in data['test_dataset']]
        st.subheader(f"Detailed Results for Run {selected_run_idx}: {run['model_name']}")
        
        is_classification_run = 'Accuracy' in run
        is_regression_run = 'MSE' in run

        if problem_type == "Classification" and is_classification_run:
            class_names_raw = data.get('class_names', [str(i) for i in range(data['n_classes'])])
            class_names = [str(c) for c in class_names_raw if c is not None]
            tab1, tab2, tab3 = st.tabs(["📊 Classification Report", "📈 ROC Curve", "📉 Loss/Accuracy Curves"])
            with tab1:
                st.text("Classification Report:"); st.code(classification_report(y_test_data, run['y_pred'], target_names=class_names, zero_division=0))
                st.text("Confusion Matrix:"); plot_confusion_matrix(y_test_data, run['y_pred'], class_names)
            with tab2:
                if run.get('y_prob') is not None: plot_roc_curve(y_test_data, run['y_prob'], data['n_classes'], class_names)
                else: st.warning("ROC curve not available for this model.")
            with tab3:
                if run['family'] == "Deep Learning": plot_training_history(run['history'])
                else: st.info("Training history plots are only available for Deep Learning models.")
        
        elif problem_type == "Regression" and is_regression_run:
            st.metric("Mean Squared Error (MSE)", run['MSE'])
            st.metric("R² Score", run['R2 Score'])
        
        else:
            st.warning(f"The selected run '{run['model_name']}' does not match the current problem type '{problem_type}'. Please select a compatible run or change the problem type in the sidebar.")

        st.subheader("Export Model")
        st.markdown(get_model_download_link(run['model'], f"model_run_{selected_run_idx}"), unsafe_allow_html=True)