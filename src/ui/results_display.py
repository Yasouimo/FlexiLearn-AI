from collections import OrderedDict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from evaluation.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history
from utils.export_utils import get_model_download_link
import plotly.graph_objects as go

def display_results(run, data, problem_type):
    st.subheader(f"Detailed Results for {run['model_name']}")
    
    if problem_type == "Classification":
        class_names_raw = data.get('class_names', [str(i) for i in range(data['n_classes'])])
        class_names = [str(c) for c in class_names_raw if c is not None]
        
        tab1, tab2, tab3 = st.tabs(["📊 Classification Report", "📈 ROC Curve", "📉 Loss/Accuracy Curves"])
        
        with tab1:
            st.text("Classification Report:")
            st.code(classification_report(data['y_test'], run['y_pred'], target_names=class_names, zero_division=0))
            st.text("Confusion Matrix:")
            plot_confusion_matrix(data['y_test'], run['y_pred'], class_names)
        
        with tab2:
            if run.get('y_prob') is not None:
                plot_roc_curve(data['y_test'], run['y_prob'], data['n_classes'], class_names)
            else:
                st.warning("ROC curve not available for this model.")
        
        with tab3:
            if run['family'] == "Deep Learning":
                plot_training_history(run['history'])
            else:
                st.info("Training history plots are only available for Deep Learning models.")
    
    elif problem_type == "Regression":
        st.metric("Mean Squared Error (MSE)", run['MSE'])
        st.metric("R² Score", run['R2 Score'])

def show_results(data_type, problem_type):
    """Display results and comparison section"""
    st.header("Step 3: Evaluation & Comparison")
    
    # Summary table
    results_data = []
    for i, run in enumerate(st.session_state.runs):
        res = {
            'Run': i,
            'Model': run['model_name'],
            'Family': run['family']
        }
        if 'Accuracy' in run:
            res['Accuracy'] = run['Accuracy']
        if 'MSE' in run:
            res['MSE'] = run['MSE']
        if 'R2 Score' in run:
            res['R2 Score'] = run['R2 Score']
        if 'Test RMSE' in run:  # For forecasting
            res['Test RMSE'] = run['Test RMSE']
            res['Test MAE'] = run['Test MAE']
        results_data.append(res)
    
    st.dataframe(pd.DataFrame(results_data).set_index('Run'))
    
    # Detailed results for selected run
    selected_run_idx = st.selectbox(
        "Select a run to inspect in detail:",
        range(len(st.session_state.runs)),
        format_func=lambda x: f"Run {x}: {st.session_state.runs[x]['model_name']}"
    )
    
    run = st.session_state.runs[selected_run_idx]
    
    st.subheader(f"Detailed Results for Run {selected_run_idx}: {run['model_name']}")
    
    # Route to appropriate display function
    if run['family'] == 'RNN':
        _show_forecasting_results(run)
    else:
        # FIX: Check if data exists in session state before accessing it
        if 'data' not in st.session_state:
            st.warning("Data has been cleared. Please reload/process your data to view detailed results.")
            return
        
        data = st.session_state.data
        y_test_data = data['y_test'] if data_type == "Tabular (CSV/Excel)" else [label for _, label in data['test_dataset']]
        
        is_classification_run = 'Accuracy' in run
        is_regression_run = 'MSE' in run

        if problem_type == "Classification" and is_classification_run:
            class_names_raw = data.get('class_names', [str(i) for i in range(data['n_classes'])])
            class_names = [str(c) for c in class_names_raw if c is not None]
            
            tab1, tab2, tab3 = st.tabs(["📊 Classification Report", "📈 ROC Curve", "📉 Loss/Accuracy Curves"])
            
            with tab1:
                st.text("Classification Report:")
                st.code(classification_report(y_test_data, run['y_pred'], target_names=class_names, zero_division=0))
                st.text("Confusion Matrix:")
                plot_confusion_matrix(y_test_data, run['y_pred'], class_names)
            
            with tab2:
                if run.get('y_prob') is not None:
                    plot_roc_curve(y_test_data, run['y_prob'], data['n_classes'], class_names)
                else:
                    st.warning("ROC curve not available for this model.")
            
            with tab3:
                if run['family'] == "Deep Learning":
                    plot_training_history(run['history'])
                else:
                    st.info("Training history plots are only available for Deep Learning models.")
        
        elif problem_type == "Regression" and is_regression_run:
            st.metric("Mean Squared Error (MSE)", run['MSE'])
            st.metric("R² Score", run['R2 Score'])
        
        else:
            st.warning(f"The selected run '{run['model_name']}' does not match the current problem type '{problem_type}'. Please select a compatible run or change the problem type in the sidebar.")

        st.subheader("Export Model")
        st.markdown(get_model_download_link(run['model'], f"model_run_{selected_run_idx}"), unsafe_allow_html=True)

def _show_forecasting_results(run):
    """Display results for time series forecasting models"""
    
    tab1, tab2, tab3 = st.tabs(["📉 Loss Curves", "📊 Predictions", "📈 Metrics"])
    
    with tab1:
        st.subheader("Training & Validation Loss")
        history = run['history']
        
        # Create loss dataframe
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(history['loss']) + 1),
            'Training Loss': history['loss'],
            'Validation Loss': history['val_loss']
        })
        
        st.line_chart(loss_df.set_index('Epoch'))
        
        # MAE curve
        st.subheader("Mean Absolute Error")
        mae_df = pd.DataFrame({
            'Epoch': range(1, len(history['mae']) + 1),
            'Training MAE': history['mae'],
            'Validation MAE': history.get('val_mae', history['mae'])
        })
        
        st.line_chart(mae_df.set_index('Epoch'))
    
    with tab2:
        st.subheader("Predictions vs Actual Values")
        
        # Reconstruct full timeline
        window_size = run['window_size']
        train_size = run['train_size']
        original_data = run['original_data'].flatten()
        
        train_pred = run['train_predictions'].flatten()
        test_pred = run['test_predictions'].flatten()
        
        # Create indices for plotting
        train_indices = np.arange(window_size, window_size + len(train_pred))
        test_indices = np.arange(train_size + window_size, train_size + window_size + len(test_pred))
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=np.arange(len(original_data)),
            y=original_data,
            mode='lines',
            name='Original Data',
            line=dict(color='blue', width=1)
        ))
        
        # Add training predictions
        fig.add_trace(go.Scatter(
            x=train_indices,
            y=train_pred,
            mode='lines',
            name='Training Predictions',
            line=dict(color='green', width=2)
        ))
        
        # Add test predictions
        fig.add_trace(go.Scatter(
            x=test_indices,
            y=test_pred,
            mode='lines',
            name='Test Predictions',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add vertical line to separate train/test
        fig.add_vline(
            x=train_size,
            line_dash="dot",
            line_color="gray",
            annotation_text="Train/Test Split"
        )
        
        fig.update_layout(
            title="Time Series Forecast",
            xaxis_title="Time Step",
            yaxis_title="Value",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Zoomed test set view
        st.subheader("Test Set (Zoomed)")
        fig_test = go.Figure()
        
        fig_test.add_trace(go.Scatter(
            x=test_indices,
            y=run['y_test_actual'].flatten(),
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig_test.add_trace(go.Scatter(
            x=test_indices,
            y=test_pred,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_test.update_layout(
            title="Test Set: Actual vs Predicted",
            xaxis_title="Time Step",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_test, use_container_width=True)
    
    with tab3:
        st.subheader("Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Training Set")
            st.metric("RMSE", run['Train RMSE'])
            st.metric("MAE", run['Train MAE'])
        
        with col2:
            st.markdown("### Test Set")
            st.metric("RMSE", run['Test RMSE'])
            st.metric("MAE", run['Test MAE'])
        
        # Error distribution
        st.subheader("Prediction Error Distribution")
        test_errors = run['y_test_actual'].flatten() - run['test_predictions'].flatten()
        
        fig_error = go.Figure()
        fig_error.add_trace(go.Histogram(
            x=test_errors,
            nbinsx=30,
            name='Error Distribution'
        ))
        
        fig_error.update_layout(
            title="Test Set Prediction Errors",
            xaxis_title="Error (Actual - Predicted)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig_error, use_container_width=True)
        
        st.info(f"Mean Error: {np.mean(test_errors):.4f} | Std Error: {np.std(test_errors):.4f}")