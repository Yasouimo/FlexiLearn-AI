from collections import OrderedDict
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from evaluation.visualization import display_model_architecture
from ui.sidebar import show_architecture_guide

def show_model_configuration(data_type, problem_type, model_family):
    """Display model configuration and training section"""
    st.header("Step 2: Model Configuration & Training")
    
    if model_family == "Classical ML (Scikit-learn)":
        _configure_classical_ml(data_type, problem_type)
    elif model_family == "Deep Learning (PyTorch)":
        _configure_deep_learning(data_type, problem_type)

def _configure_classical_ml(data_type, problem_type):
    """Configure and train classical ML models"""
    if data_type != "Tabular (CSV/Excel)":
        st.warning("Classical ML models are for tabular data only.")
        return
    
    model = None
    model_type = "Unknown"
    
    if problem_type == "Classification":
        model_type = st.selectbox("Select Model", (
            "Logistic Regression",
            "Random Forest Classifier",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors",
            "Decision Tree Classifier"
        ))
        
        if model_type == "Logistic Regression":
            model = LogisticRegression(
                C=st.slider("Regularization (C)", 0.01, 10.0, 1.0),
                max_iter=500,
                random_state=42
            )
        elif model_type == "Random Forest Classifier":
            model = RandomForestClassifier(
                n_estimators=st.slider("Number of Trees", 10, 500, 100),
                max_depth=st.slider("Max Depth", 2, 32, 10),
                random_state=42
            )
        elif model_type == "Support Vector Machine (SVM)":
            model = SVC(
                C=st.slider("Regularization (C)", 0.1, 10.0, 1.0),
                kernel=st.selectbox("Kernel", ('rbf', 'linear', 'poly')),
                probability=True,
                random_state=42
            )
        elif model_type == "K-Nearest Neighbors":
            model = KNeighborsClassifier(
                n_neighbors=st.slider("Number of Neighbors (K)", 1, 20, 5)
            )
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier(
                max_depth=st.slider("Max Depth", 2, 32, 10),
                random_state=42
            )
    
    elif problem_type == "Regression":
        model_type = st.selectbox("Select Model", (
            "Linear Regression",
            "Random Forest Regressor",
            "Support Vector Regressor (SVR)"
        ))
        
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Random Forest Regressor":
            model = RandomForestRegressor(
                n_estimators=st.slider("Number of Trees", 10, 500, 100),
                max_depth=st.slider("Max Depth", 2, 32, 10),
                random_state=42
            )
        elif model_type == "Support Vector Regressor (SVR)":
            model = SVR(
                C=st.slider("Regularization (C)", 0.1, 10.0, 1.0),
                kernel=st.selectbox("Kernel", ('rbf', 'linear', 'poly'))
            )
    
    # Train button
    if model is not None and st.button(f"💪 Train {model_type}"):
        _train_classical_model(model, model_type, problem_type)

def _train_classical_model(model, model_type, problem_type):
    """Train a classical ML model"""
    with st.spinner("Training in progress..."):
        data = st.session_state.data
        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_test'])
        
        run_info = {
            'model_name': model_type,
            'model': model,
            'family': 'Classical'
        }
        
        if problem_type == "Classification":
            accuracy = accuracy_score(data['y_test'], y_pred)
            y_prob = model.predict_proba(data['X_test']) if hasattr(model, 'predict_proba') else None
            run_info.update({
                'Accuracy': f"{accuracy:.4f}",
                'y_pred': y_pred,
                'y_prob': y_prob
            })
        else:
            mse = mean_squared_error(data['y_test'], y_pred)
            r2 = r2_score(data['y_test'], y_pred)
            run_info.update({
                'MSE': f"{mse:.4f}",
                'R2 Score': f"{r2:.4f}",
                'y_pred': y_pred
            })
        
        st.session_state.runs.append(run_info)
    st.success("Training complete!")

def _configure_deep_learning(data_type, problem_type):
    """Configure and train deep learning models"""
    st.sidebar.subheader("Network Architecture")
    show_architecture_guide(data_type, problem_type)
    arch_mode = st.sidebar.selectbox("Architecture Mode", ["Suggested", "Custom"])
    
    model = None
    
    if data_type == "Tabular (CSV/Excel)":
        model = _build_ann_model(arch_mode, problem_type)
    elif data_type == "Image (ZIP)":
        model = _build_cnn_model(arch_mode)
    
    if model:
        display_model_architecture(model)
        
        # Training parameters
        st.sidebar.subheader("Training Parameters")
        lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")
        epochs = st.sidebar.number_input("Epochs", 1, 500, 10)
        batch_size = st.sidebar.number_input("Batch Size", 8, 512, 32)
        
        if st.button("🧠 Train Deep Learning Model"):
            _train_deep_learning_model(model, data_type, problem_type, lr, epochs, batch_size)

def _build_ann_model(arch_mode, problem_type):
    """Build ANN model for tabular data"""
    layers = OrderedDict()
    input_size = st.session_state.data['n_features']
    output_size = st.session_state.data['n_classes'] if problem_type == "Classification" else 1
    
    if arch_mode == "Suggested":
        template = st.sidebar.selectbox("Choose ANN template", ["Simple (2 layers)", "Medium (3 layers + Dropout)"])
        if template == "Simple (2 layers)":
            layers['layer_1'] = nn.Linear(input_size, 128)
            layers['act_1'] = nn.ReLU()
            layers['layer_2'] = nn.Linear(128, 64)
            layers['act_2'] = nn.ReLU()
            layers['output'] = nn.Linear(64, output_size)
        else:
            layers['layer_1'] = nn.Linear(input_size, 128)
            layers['act_1'] = nn.ReLU()
            layers['dropout_1'] = nn.Dropout(0.3)
            layers['layer_2'] = nn.Linear(128, 64)
            layers['act_2'] = nn.ReLU()
            layers['dropout_2'] = nn.Dropout(0.3)
            layers['layer_3'] = nn.Linear(64, 32)
            layers['act_3'] = nn.ReLU()
            layers['output'] = nn.Linear(32, output_size)
    else:
        st.sidebar.markdown("Define your Dense layers:")
        last_size = input_size
        for i, layer_params in enumerate(st.session_state.custom_ann_layers):
            with st.sidebar.expander(f"Layer {i+1}", expanded=True):
                layer_params['units'] = st.number_input("Neurons", 1, 2048, layer_params['units'], key=f"ann_units_{i}")
                layer_params['activation'] = st.selectbox("Activation", ['ReLU', 'LeakyReLU', 'Tanh'], key=f"ann_act_{i}")
                layer_params['dropout'] = st.slider("Dropout", 0.0, 0.9, layer_params['dropout'], key=f"ann_drop_{i}")
            
            layers[f'layer_{i}'] = nn.Linear(last_size, layer_params['units'])
            layers[f'act_{i}'] = getattr(nn, layer_params['activation'])()
            if layer_params['dropout'] > 0:
                layers[f'dropout_{i}'] = nn.Dropout(layer_params['dropout'])
            last_size = layer_params['units']
        
        cols = st.sidebar.columns(2)
        if cols[0].button("Add Layer"):
            st.session_state.custom_ann_layers.append({'units': 32, 'activation': 'ReLU', 'dropout': 0.2})
            st.rerun()
        if cols[1].button("Remove Last") and len(st.session_state.custom_ann_layers) > 1:
            st.session_state.custom_ann_layers.pop()
            st.rerun()
        
        layers['output'] = nn.Linear(last_size, output_size)
    
    return nn.Sequential(layers)

def _build_cnn_model(arch_mode):
    """Build CNN model for images"""
    conv_layers = OrderedDict()
    last_channels = 3
    current_dim = st.session_state.data['img_size']
    
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
        if cols[0].button("Add Conv"):
            st.session_state.custom_cnn_layers.append({'type': 'Conv2d', 'filters': 64, 'kernel': 3})
            st.rerun()
        if cols[1].button("Add Pool"):
            st.session_state.custom_cnn_layers.append({'type': 'MaxPool2d', 'kernel': 2})
            st.rerun()
        if cols[2].button("Remove Last") and len(st.session_state.custom_cnn_layers) > 1:
            st.session_state.custom_cnn_layers.pop()
            st.rerun()
    
    for i, p in enumerate(cnn_layer_configs):
        if p['type'] == 'Conv2d':
            conv_layers[f'conv_{i}'] = nn.Conv2d(last_channels, p['filters'], kernel_size=p['kernel'], padding='same')
            conv_layers[f'relu_{i}'] = nn.ReLU()
            last_channels = p['filters']
        elif p['type'] == 'MaxPool2d':
            conv_layers[f'pool_{i}'] = nn.MaxPool2d(kernel_size=p['kernel'], stride=p['kernel'])
            current_dim //= p['kernel']
    
    class DynamicCNN(nn.Module):
        def __init__(self, conv_part, flattened_size, num_classes):
            super().__init__()
            self.conv_part = conv_part
            self.flatten = nn.Flatten()
            self.classifier = nn.Sequential(
                nn.Linear(flattened_size, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            return self.classifier(self.flatten(self.conv_part(x)))
    
    flattened_size = last_channels * current_dim * current_dim
    return DynamicCNN(nn.Sequential(conv_layers), flattened_size, st.session_state.data['n_classes'])

def _train_deep_learning_model(model, data_type, problem_type, lr, epochs, batch_size):
    """Train a deep learning model"""
    data = st.session_state.data
    
    # Prepare data loaders
    if data_type == "Tabular (CSV/Excel)":
        train_ds = TensorDataset(
            torch.Tensor(data['X_train']),
            torch.Tensor(data['y_train']).long() if problem_type == "Classification" else torch.Tensor(data['y_train']).view(-1, 1)
        )
        test_ds = TensorDataset(
            torch.Tensor(data['X_test']),
            torch.Tensor(data['y_test']).long() if problem_type == "Classification" else torch.Tensor(data['y_test']).view(-1, 1)
        )
    else:
        train_ds, test_ds = data['train_dataset'], data['test_dataset']
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if problem_type == "Classification" else nn.MSELoss()
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if problem_type == "Classification":
                _, pred = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (pred == labels).sum().item()
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                if problem_type == "Classification":
                    _, pred = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (pred == labels).sum().item()
                    all_preds.extend(pred.numpy())
                    all_labels.extend(labels.numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1).numpy())
                else:
                    all_preds.extend(outputs.numpy())
                    all_labels.extend(labels.numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        status_text.text(f"Epoch {epoch+1}/{epochs} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        progress_bar.progress((epoch + 1) / epochs)
    
    st.success("Training complete!")
    
    run_info = {
        'model_name': 'PyTorch ' + ('CNN' if data_type == 'Image (ZIP)' else 'ANN'),
        'model': model,
        'family': 'Deep Learning',
        'history': history
    }
    
    if problem_type == "Classification":
        run_info.update({
            'Accuracy': f"{val_acc:.4f}",
            'y_pred': np.array(all_preds),
            'y_prob': np.array(all_probs)
        })
    else:
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        mse = mean_squared_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        run_info.update({
            'MSE': f"{mse:.4f}",
            'R2 Score': f"{r2:.4f}"
        })
    
    st.session_state.runs.append(run_info)