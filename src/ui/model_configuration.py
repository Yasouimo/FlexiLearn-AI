import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from evaluation.visualization import display_model_architecture
from ui.sidebar import show_architecture_guide
from utils.timeseries_utils import prepare_timeseries_data

# Simple fix: ONLY import keras (no tensorflow)
import keras
from keras import layers, optimizers, callbacks

def show_model_configuration(data_type, problem_type, model_family):
    """Display model configuration and training section"""
    st.header("Step 2: Model Configuration & Training")
    
    if model_family == "Classical ML (Scikit-learn)":
        _configure_classical_ml(data_type, problem_type)
    elif model_family == "Deep Learning (PyTorch)":
        _configure_deep_learning(data_type, problem_type)
    elif model_family == "RNN (Recurrent Neural Networks)":
        _configure_rnn_forecasting(data_type, problem_type)

def _configure_rnn_forecasting(data_type, problem_type):
    """Configure and train RNN models for time series forecasting"""
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ Please load time series data first from the sidebar.")
        return
    
    df = st.session_state.df
    target_column = df.columns[0]
    
    st.subheader("🔧 RNN Configuration")
    
    # Architecture mode selection
    st.markdown("#### Architecture Mode")
    arch_mode = st.radio(
        "Choose Configuration Mode",
        ["Suggested Architecture", "Custom Architecture"],
        help="Suggested: Pre-configured optimal settings. Custom: Build your own network."
    )
    
    # Data Preparation Settings
    st.markdown("#### 📊 Data Preparation")
    col1, col2 = st.columns(2)
    
    with col1:
        train_split = st.slider(
            "Train/Test Split (%)",
            min_value=50,
            max_value=95,
            value=80,
            step=5,
            help="Percentage of data to use for training. Remaining is for testing."
        )
    
    with col2:
        window_size = st.number_input(
            "Window Size (Lookback Period)",
            min_value=5,
            max_value=365,
            value=30,
            step=5,
            help="Number of previous time steps to use for prediction. E.g., 30 = use last 30 days to predict next day."
        )
    
    if arch_mode == "Suggested Architecture":
        st.markdown("#### 🎯 Suggested Configuration")
        st.info("""
        **Recommended Settings for Time Series:**
        - **LSTM**: Best for long-term dependencies (default)
        - **2 Layers**: Good balance between complexity and overfitting
        - **64 Units**: Sufficient for most time series patterns
        - **50 Epochs**: Enough for convergence without overfitting
        """)
        
        rnn_type = "LSTM"
        num_layers = 2
        units = 64
        epochs = 50
        batch_size = 32
        optimizer_name = "Adam"
        learning_rate = 0.001
        
        # Show the suggested config
        st.markdown("**Using:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", rnn_type)
            st.metric("Layers", num_layers)
        with col2:
            st.metric("Units", units)
            st.metric("Epochs", epochs)
        with col3:
            st.metric("Batch Size", batch_size)
            st.metric("Learning Rate", learning_rate)
        
    else:  # Custom Architecture
        st.markdown("#### 🛠️ Custom Architecture")
        
        # Show architecture tips
        with st.expander("💡 RNN Architecture Guide", expanded=True):
            st.markdown("""
            ### RNN Model Selection:
            
            **Simple RNN:**
            - ✅ Fastest training
            - ✅ Good for short sequences
            - ❌ Struggles with long-term dependencies
            - 📌 Use for: Quick experiments, short patterns (<10 steps)
            
            **LSTM (Long Short-Term Memory):**
            - ✅ Best for long-term dependencies
            - ✅ Remembers important patterns over time
            - ✅ Most popular for time series
            - 📌 Use for: Most time series problems (recommended!)
            
            **GRU (Gated Recurrent Unit):**
            - ✅ Faster than LSTM (fewer parameters)
            - ✅ Similar performance to LSTM
            - ✅ Good balance of speed and accuracy
            - 📌 Use for: When training time matters
            
            ### Layer Configuration Tips:
            
            **Number of Layers:**
            - 1 layer: Simple patterns, fast training
            - 2 layers: Most common (recommended)
            - 3+ layers: Complex patterns, risk of overfitting
            
            **Units per Layer:**
            - 32: Small datasets (<1000 points)
            - 64: Medium datasets (1000-10000 points) ✅ Recommended
            - 128+: Large datasets (>10000 points)
            
            **Training Tips:**
            - Start with 50 epochs, increase if needed
            - Batch size 32 is a good default
            - If loss is unstable, reduce learning rate to 0.0001
            """)
        
        # Model Architecture
        st.markdown("#### Model Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            rnn_type = st.selectbox(
                "RNN Type",
                ["LSTM", "GRU", "Simple RNN"],
                help="LSTM and GRU are better at capturing long-term dependencies"
            )
        
        with col2:
            num_layers = st.number_input(
                "Number of Layers",
                min_value=1,
                max_value=5,
                value=2,
                help="More layers can capture more complex patterns but may overfit"
            )
        
        units = st.number_input(
            "Units per Layer",
            min_value=16,
            max_value=256,
            value=64,
            step=16,
            help="Number of neurons in each RNN layer. Higher = more capacity but slower training."
        )
        
        # Training Parameters
        st.markdown("#### ⚙️ Training Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            epochs = st.number_input(
                "Epochs", 
                min_value=10, 
                max_value=500, 
                value=50, 
                step=10,
                help="Number of complete passes through the training data"
            )
        
        with col2:
            batch_size = st.number_input(
                "Batch Size", 
                min_value=8, 
                max_value=128, 
                value=32, 
                step=8,
                help="Number of samples processed before updating weights"
            )
        
        with col3:
            optimizer_name = st.selectbox(
                "Optimizer", 
                ["Adam", "RMSprop", "SGD"],
                help="Adam is the best default choice"
            )
        
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.00001,
            max_value=0.1,
            value=0.001,
            format="%.5f",
            help="Controls how much to adjust weights. 0.001 is a good default."
        )
    
    # Train button
    st.markdown("---")
    if st.button(f"🚀 Train {rnn_type} Model", type="primary", use_container_width=True):
        _train_rnn_model(
            df=df,
            target_column=target_column,
            window_size=window_size,
            train_split=train_split / 100.0,  # Convert to decimal
            rnn_type=rnn_type,
            num_layers=num_layers,
            units=units,
            epochs=epochs,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate
        )

def _train_rnn_model(df, target_column, window_size, train_split, rnn_type, 
                     num_layers, units, epochs, batch_size, optimizer_name, learning_rate):
    """Train an RNN model for time series forecasting"""
    
    with st.spinner("📊 Preparing data..."):
        # Prepare data
        data_dict = prepare_timeseries_data(df, target_column, window_size, train_split)
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        scaler = data_dict['scaler']
        
        st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Build model
    with st.spinner(f"🏗️ Building {rnn_type} model..."):
        model = keras.Sequential()
        
        # Add RNN layers
        for i in range(num_layers):
            return_sequences = (i < num_layers - 1)  # Only last layer doesn't return sequences
            
            if rnn_type == "LSTM":
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    input_shape=(window_size, 1) if i == 0 else None
                ))
            elif rnn_type == "GRU":
                model.add(layers.GRU(
                    units,
                    return_sequences=return_sequences,
                    input_shape=(window_size, 1) if i == 0 else None
                ))
            else:  # Simple RNN
                model.add(layers.SimpleRNN(
                    units,
                    return_sequences=return_sequences,
                    input_shape=(window_size, 1) if i == 0 else None
                ))
            
            model.add(layers.Dropout(0.2))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compile model
        if optimizer_name == "Adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = optimizers.SGD(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        # Display model summary
        st.text("📋 Model Architecture:")
        summary_string = []
        model.summary(print_fn=lambda x: summary_string.append(x))
        st.code('\n'.join(summary_string))
    
    # Train model
    st.subheader("🎯 Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        loss_metric = st.empty()
    with metrics_col2:
        val_loss_metric = st.empty()
    
    # Custom callback for progress updates
    class StreamlitCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {logs['loss']:.4f} | "
                f"Val Loss: {logs['val_loss']:.4f} | "
                f"MAE: {logs['mae']:.4f}"
            )
            loss_metric.metric("Training Loss", f"{logs['loss']:.4f}")
            val_loss_metric.metric("Validation Loss", f"{logs['val_loss']:.4f}")
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[StreamlitCallback()]
    )
    
    st.success("✅ Training complete!")
    
    # Make predictions
    with st.spinner("🔮 Generating predictions..."):
        train_predictions = model.predict(X_train, verbose=0)
        test_predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_predictions = scaler.inverse_transform(train_predictions)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_train_actual = scaler.inverse_transform(y_train)
        y_test_actual = scaler.inverse_transform(y_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    test_mae = mean_absolute_error(y_test_actual, test_predictions)
    
    # Calculate R2 score
    from sklearn.metrics import r2_score
    train_r2 = r2_score(y_train_actual, train_predictions)
    test_r2 = r2_score(y_test_actual, test_predictions)
    
    # Store results
    run_info = {
        'model_name': f'{rnn_type} ({num_layers} layers, {units} units)',
        'model': model,
        'family': 'RNN',
        'history': history.history,
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'y_train_actual': y_train_actual,
        'y_test_actual': y_test_actual,
        'scaler': scaler,
        'window_size': window_size,
        'train_size': data_dict['train_size'],
        'original_data': data_dict['original_data'],
        'Train RMSE': f"{train_rmse:.4f}",
        'Test RMSE': f"{test_rmse:.4f}",
        'Train MAE': f"{train_mae:.4f}",
        'Test MAE': f"{test_mae:.4f}",
        'Train R²': f"{train_r2:.4f}",
        'Test R²': f"{test_r2:.4f}"
    }
    
    st.session_state.runs.append(run_info)
    
    # Display metrics
    st.subheader("📊 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train RMSE", f"{train_rmse:.2f}")
    with col2:
        st.metric("Test RMSE", f"{test_rmse:.2f}", 
                 delta=f"{test_rmse - train_rmse:.2f}",
                 delta_color="inverse")
    with col3:
        st.metric("Train MAE", f"{train_mae:.2f}")
    with col4:
        st.metric("Test MAE", f"{test_mae:.2f}",
                 delta=f"{test_mae - train_mae:.2f}",
                 delta_color="inverse")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Train R²", f"{train_r2:.4f}")
    with col2:
        st.metric("Test R²", f"{test_r2:.4f}")
    
    st.info("💡 Lower RMSE/MAE and higher R² (closer to 1.0) indicate better predictions!")
    
    st.balloons()

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