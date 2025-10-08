from collections import OrderedDict
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from evaluation.visualization import plot_confusion_matrix, plot_roc_curve, plot_training_history
from utils.export_utils import get_model_download_link

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
        results_data.append(res)
    
    st.dataframe(pd.DataFrame(results_data).set_index('Run'))
    
    # Detailed results for selected run
    selected_run_idx = st.selectbox(
        "Select a run to inspect in detail:",
        range(len(st.session_state.runs)),
        format_func=lambda x: f"Run {x}: {st.session_state.runs[x]['model_name']}"
    )
    
    run = st.session_state.runs[selected_run_idx]
    
    # FIX: Check if data exists in session state before accessing it
    if 'data' not in st.session_state:
        st.warning("Data has been cleared. Please reload/process your data to view detailed results.")
        return
    
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