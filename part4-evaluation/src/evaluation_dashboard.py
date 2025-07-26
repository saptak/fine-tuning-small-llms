#!/usr/bin/env python3
"""
Interactive Evaluation Dashboard for Model Comparison
From: Fine-Tuning Small LLMs with Docker Desktop - Part 4
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LLM Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    
    .comparison-container {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .model-better {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    
    .model-worse {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_evaluation_results(results_dir: str = "./results") -> List[Dict]:
    """Load evaluation results from directory"""
    
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        st.warning(f"Results directory {results_dir} not found")
        return []
    
    # Load all JSON result files
    for file_path in results_path.glob("evaluation_results_*.json"):
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
                result['file_path'] = str(file_path)
                results.append(result)
        except Exception as e:
            st.error(f"Failed to load {file_path}: {e}")
    
    return sorted(results, key=lambda x: x.get('timestamp', ''), reverse=True)

def format_metric_name(metric: str) -> str:
    """Format metric names for display"""
    
    metric_names = {
        'exact_match': 'Exact Match',
        'bleu': 'BLEU Score',
        'rouge1': 'ROUGE-1',
        'rouge2': 'ROUGE-2',
        'rougeL': 'ROUGE-L',
        'meteor': 'METEOR',
        'bertscore_f1': 'BERTScore F1',
        'sql_syntax_validity': 'SQL Syntax Validity',
        'sql_keyword_accuracy': 'SQL Keyword Accuracy'
    }
    
    return metric_names.get(metric, metric.title())

def create_metrics_comparison_chart(results: List[Dict], selected_metrics: List[str]):
    """Create comparison chart for selected metrics"""
    
    # Prepare data for plotting
    chart_data = []
    
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        timestamp = result.get('timestamp', '')
        
        for metric in selected_metrics:
            if metric in result.get('metrics', {}):
                chart_data.append({
                    'Model': model_name,
                    'Metric': format_metric_name(metric),
                    'Value': result['metrics'][metric],
                    'Timestamp': timestamp
                })
    
    if not chart_data:
        st.warning("No data available for selected metrics")
        return
    
    df = pd.DataFrame(chart_data)
    
    # Create grouped bar chart
    fig = px.bar(
        df, 
        x='Model', 
        y='Value', 
        color='Metric',
        title='Model Performance Comparison',
        barmode='group',
        height=500
    )
    
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_radar_chart(results: List[Dict], selected_models: List[str]):
    """Create radar chart for model performance"""
    
    if len(selected_models) == 0:
        st.warning("Please select at least one model")
        return
    
    # Get common metrics across all selected models
    all_metrics = set()
    model_data = {}
    
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        if model_name in selected_models:
            metrics = result.get('metrics', {})
            all_metrics.update(metrics.keys())
            model_data[model_name] = metrics
    
    common_metrics = list(all_metrics)
    if not common_metrics:
        st.warning("No common metrics found")
        return
    
    # Create radar chart
    fig = go.Figure()
    
    for model_name in selected_models:
        if model_name in model_data:
            values = [model_data[model_name].get(metric, 0) for metric in common_metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[format_metric_name(m) for m in common_metrics],
                fill='toself',
                name=model_name
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_performance_timeline(results: List[Dict]):
    """Create timeline of model performance over time"""
    
    # Prepare data
    timeline_data = []
    
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        timestamp = result.get('timestamp', '')
        
        # Use overall quality score or average of main metrics
        main_metrics = ['exact_match', 'bleu', 'rouge1', 'meteor']
        available_metrics = [m for m in main_metrics if m in result.get('metrics', {})]
        
        if available_metrics:
            score = np.mean([result['metrics'][m] for m in available_metrics])
            timeline_data.append({
                'Model': model_name,
                'Timestamp': pd.to_datetime(timestamp),
                'Score': score,
                'Metrics_Count': len(available_metrics)
            })
    
    if not timeline_data:
        st.warning("No timeline data available")
        return
    
    df = pd.DataFrame(timeline_data)
    
    fig = px.line(
        df,
        x='Timestamp',
        y='Score',
        color='Model',
        title='Model Performance Over Time',
        markers=True,
        height=400
    )
    
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Average Score",
        legend_title="Model"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_detailed_comparison(results: List[Dict], model1: str, model2: str):
    """Display detailed side-by-side comparison of two models"""
    
    # Find results for selected models
    result1 = next((r for r in results if r.get('model_name') == model1), None)
    result2 = next((r for r in results if r.get('model_name') == model2), None)
    
    if not result1 or not result2:
        st.error("Selected models not found in results")
        return
    
    st.subheader(f"Detailed Comparison: {model1} vs {model2}")
    
    # Metrics comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.write(f"**{model1}**")
        for metric, value in result1.get('metrics', {}).items():
            st.metric(format_metric_name(metric), f"{value:.4f}")
    
    with col2:
        st.write("**Difference**")
        for metric in result1.get('metrics', {}):
            if metric in result2.get('metrics', {}):
                diff = result1['metrics'][metric] - result2['metrics'][metric]
                st.metric(
                    "Δ " + format_metric_name(metric), 
                    f"{diff:+.4f}",
                    delta=diff
                )
    
    with col3:
        st.write(f"**{model2}**")
        for metric, value in result2.get('metrics', {}).items():
            st.metric(format_metric_name(metric), f"{value:.4f}")
    
    # Performance stats comparison
    st.subheader("Performance Statistics")
    
    perf1 = result1.get('performance_stats', {})
    perf2 = result2.get('performance_stats', {})
    
    if perf1 and perf2:
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.write(f"**{model1} Performance**")
            for stat, value in perf1.items():
                if 'avg_' in stat:
                    st.write(f"{stat.replace('avg_', '').title()}: {value:.2f}")
        
        with perf_col2:
            st.write(f"**{model2} Performance**")
            for stat, value in perf2.items():
                if 'avg_' in stat:
                    st.write(f"{stat.replace('avg_', '').title()}: {value:.2f}")
    
    # Example comparisons
    st.subheader("Example Outputs Comparison")
    
    examples1 = result1.get('examples', [])
    examples2 = result2.get('examples', [])
    
    if examples1 and examples2:
        # Show first few examples
        for i in range(min(3, len(examples1), len(examples2))):
            with st.expander(f"Example {i+1}"):
                ex1 = examples1[i]
                ex2 = examples2[i] if i < len(examples2) else {}
                
                st.write("**Instruction:**", ex1.get('instruction', 'N/A'))
                if ex1.get('input'):
                    st.write("**Input:**", ex1.get('input', 'N/A'))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{model1} Output:**")
                    st.code(ex1.get('predicted_output', 'N/A'))
                
                with col2:
                    st.write(f"**{model2} Output:**")
                    st.code(ex2.get('predicted_output', 'N/A') if ex2 else 'N/A')
                
                st.write("**Expected Output:**")
                st.code(ex1.get('expected_output', 'N/A'))

def main():
    """Main dashboard application"""
    
    st.title("📊 LLM Evaluation Dashboard")
    st.markdown("Interactive dashboard for comparing and analyzing LLM model performance")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Results directory selector
        results_dir = st.text_input("Results Directory", value="./results")
        
        # Load results
        if st.button("🔄 Refresh Results"):
            st.cache_data.clear()
        
        results = load_evaluation_results(results_dir)
        
        if not results:
            st.error("No evaluation results found")
            st.stop()
        
        st.success(f"Loaded {len(results)} evaluation results")
        
        # Model selector
        model_names = list(set(r.get('model_name', 'Unknown') for r in results))
        selected_models = st.multiselect("Select Models", model_names, default=model_names[:2])
        
        # Metric selector
        all_metrics = set()
        for result in results:
            all_metrics.update(result.get('metrics', {}).keys())
        
        all_metrics = sorted(list(all_metrics))
        selected_metrics = st.multiselect(
            "Select Metrics", 
            all_metrics, 
            default=[m for m in ['exact_match', 'bleu', 'rouge1', 'meteor'] if m in all_metrics]
        )
    
    # Main content area
    if not results:
        st.warning("No evaluation results to display")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "📈 Metrics Comparison", 
        "🎯 Radar Chart", 
        "⏱️ Timeline", 
        "🔍 Detailed Comparison"
    ])
    
    with tab1:
        st.header("Evaluation Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", len(set(r.get('model_name') for r in results)))
        
        with col2:
            total_examples = sum(r.get('examples', []) for r in results)
            st.metric("Total Examples", len(total_examples) if total_examples else 0)
        
        with col3:
            avg_score = np.mean([
                np.mean(list(r.get('metrics', {}).values())) 
                for r in results if r.get('metrics')
            ]) if results else 0
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col4:
            recent_results = len([r for r in results if r.get('timestamp', '') >= datetime.now().strftime('%Y-%m-%d')])
            st.metric("Today's Evaluations", recent_results)
        
        # Recent results table
        st.subheader("Recent Evaluation Results")
        
        display_data = []
        for result in results[:10]:  # Show last 10 results
            metrics = result.get('metrics', {})
            row = {
                'Model': result.get('model_name', 'Unknown'),
                'Timestamp': result.get('timestamp', ''),
                'Dataset': result.get('dataset_name', 'Unknown')
            }
            
            # Add key metrics
            for metric in ['exact_match', 'bleu', 'rouge1']:
                if metric in metrics:
                    row[format_metric_name(metric)] = f"{metrics[metric]:.4f}"
            
            display_data.append(row)
        
        if display_data:
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
    
    with tab2:
        st.header("Metrics Comparison")
        
        if selected_metrics:
            create_metrics_comparison_chart(results, selected_metrics)
        else:
            st.warning("Please select metrics to compare")
    
    with tab3:
        st.header("Performance Radar Chart")
        
        if selected_models:
            create_performance_radar_chart(results, selected_models)
        else:
            st.warning("Please select models to compare")
    
    with tab4:
        st.header("Performance Timeline")
        create_performance_timeline(results)
    
    with tab5:
        st.header("Detailed Model Comparison")
        
        if len(model_names) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                model1 = st.selectbox("First Model", model_names, key="model1")
            
            with col2:
                model2 = st.selectbox("Second Model", model_names, index=1 if len(model_names) > 1 else 0, key="model2")
            
            if model1 != model2:
                display_detailed_comparison(results, model1, model2)
            else:
                st.warning("Please select different models for comparison")
        else:
            st.warning("Need at least 2 models for comparison")

if __name__ == "__main__":
    main()
