#!/usr/bin/env python3
"""
Streamlit Web Interface for LLM Fine-Tuning
From: Fine-Tuning Small LLMs with Docker Desktop - Part 5
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import os

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')
API_TOKEN = os.getenv('API_TOKEN', 'demo-token-12345')

# Page configuration
st.set_page_config(
    page_title="LLM Fine-Tuning Dashboard",
    page_icon="🤖",
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
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        background-color: #f8f9fa;
    }
    
    .user-message {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    
    .assistant-message {
        border-left-color: #007bff;
        background-color: #d1ecf1;
    }
</style>
""", unsafe_allow_html=True)

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request with authentication"""
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return {}

def get_service_status() -> Dict:
    """Get service status and health"""
    return make_api_request("/api/v1/health")

def get_available_models() -> List[Dict]:
    """Get list of available models"""
    result = make_api_request("/api/v1/models")
    return result if isinstance(result, list) else []

def generate_text(prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> Dict:
    """Generate text using the API"""
    
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    return make_api_request("/api/v1/generate", "POST", data)

def chat_completion(messages: List[Dict], max_tokens: int = 256, temperature: float = 0.7) -> Dict:
    """Get chat completion from API"""
    
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    return make_api_request("/api/v1/chat", "POST", data)

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 LLM Dashboard")
        
        # Service status
        status = get_service_status()
        if status:
            if status.get('status') == 'healthy':
                st.success("✅ Service Online")
            else:
                st.error("❌ Service Offline")
            
            st.metric("Model Status", status.get('model_status', 'Unknown'))
        
        # Model selection
        st.subheader("🔧 Configuration")
        
        models = get_available_models()
        if models:
            model_names = [m['name'] for m in models]
            selected_model = st.selectbox("Select Model", model_names)
        else:
            st.warning("No models available")
        
        # Generation parameters
        st.subheader("⚙️ Parameters")
        max_tokens = st.slider("Max Tokens", 1, 1024, 256)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Text Generation", "📊 Analytics", "⚙️ Admin"])
    
    with tab1:
        st.header("💬 Interactive Chat")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_completion(
                        st.session_state.messages,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                
                if response and 'message' in response:
                    assistant_message = response['message']['content']
                    st.markdown(assistant_message)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": assistant_message
                    })
                    
                    # Show response metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Response Time", f"{response.get('response_time_ms', 0):.0f} ms")
                    with col2:
                        st.metric("Input Tokens", response.get('usage', {}).get('prompt_tokens', 0))
                    with col3:
                        st.metric("Output Tokens", response.get('usage', {}).get('completion_tokens', 0))
                else:
                    st.error("Failed to get response from the model")
    
    with tab2:
        st.header("📝 Text Generation")
        
        # Input area
        prompt = st.text_area(
            "Enter your prompt:",
            height=150,
            placeholder="Write a SQL query to find all users who registered in the last 30 days..."
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            generate_button = st.button("🚀 Generate", type="primary")
        
        with col2:
            if st.button("📋 Example Prompts"):
                examples = [
                    "Write a SQL query to find the top 10 customers by total order value",
                    "Create a Python function to calculate the factorial of a number",
                    "Explain the concept of machine learning in simple terms",
                    "Write a SQL query to join customers and orders tables"
                ]
                st.session_state.example_prompts = examples
        
        # Show example prompts
        if hasattr(st.session_state, 'example_prompts'):
            st.subheader("💡 Example Prompts")
            for i, example in enumerate(st.session_state.example_prompts):
                if st.button(f"{i+1}. {example[:50]}...", key=f"example_{i}"):
                    st.session_state.selected_prompt = example
                    st.rerun()
        
        # Use selected example
        if hasattr(st.session_state, 'selected_prompt'):
            prompt = st.session_state.selected_prompt
            del st.session_state.selected_prompt
        
        # Generate text
        if generate_button and prompt:
            with st.spinner("Generating..."):
                start_time = time.time()
                response = generate_text(prompt, max_tokens, temperature)
                end_time = time.time()
            
            if response and 'text' in response:
                st.subheader("📄 Generated Text")
                st.code(response['text'], language='sql' if 'sql' in prompt.lower() else None)
                
                # Metrics
                st.subheader("📊 Generation Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Response Time", f"{response.get('response_time_ms', 0):.0f} ms")
                with col2:
                    st.metric("Input Tokens", response.get('usage', {}).get('prompt_tokens', 0))
                with col3:
                    st.metric("Output Tokens", response.get('usage', {}).get('completion_tokens', 0))
                with col4:
                    st.metric("Total Tokens", response.get('usage', {}).get('total_tokens', 0))
            else:
                st.error("Failed to generate text")
    
    with tab3:
        st.header("📊 Usage Analytics")
        
        # Mock analytics data (replace with actual metrics from your monitoring system)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", "1,234", "12%")
        with col2:
            st.metric("Avg Response Time", "850ms", "-5%")
        with col3:
            st.metric("Cache Hit Rate", "87%", "3%")
        with col4:
            st.metric("Error Rate", "0.2%", "-0.1%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Response Time Trend")
            
            # Mock data
            dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='H')
            response_times = [800 + (i % 24) * 20 + (i % 3) * 50 for i in range(len(dates))]
            
            df = pd.DataFrame({
                'timestamp': dates,
                'response_time_ms': response_times
            })
            
            fig = px.line(df, x='timestamp', y='response_time_ms', 
                         title='Response Time Over Time')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Request Distribution")
            
            # Mock data
            endpoints = ['/api/v1/generate', '/api/v1/chat', '/api/v1/models', '/api/v1/health']
            counts = [450, 380, 45, 120]
            
            fig = px.pie(values=counts, names=endpoints, title='Requests by Endpoint')
            st.plotly_chart(fig, use_container_width=True)
        
        # Usage table
        st.subheader("📝 Recent Activity")
        
        # Mock activity data
        activity_data = {
            'Timestamp': ['2024-01-07 10:30:00', '2024-01-07 10:29:45', '2024-01-07 10:29:30'],
            'Endpoint': ['/api/v1/generate', '/api/v1/chat', '/api/v1/generate'],
            'Status': ['200', '200', '200'],
            'Response Time (ms)': [850, 920, 780],
            'Tokens Used': [45, 67, 52]
        }
        
        df = pd.DataFrame(activity_data)
        st.dataframe(df, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Administration")
        
        # Service management
        st.subheader("🔧 Service Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Configuration"):
                st.info("Configuration reloaded successfully")
        
        with col2:
            if st.button("📊 View Metrics"):
                st.info("Redirecting to metrics dashboard...")
        
        with col3:
            if st.button("📋 View Logs"):
                st.info("Viewing recent logs...")
        
        # Model management
        st.subheader("🤖 Model Management")
        
        if models:
            for model in models:
                with st.expander(f"📦 {model['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Size:** {model.get('size', 'Unknown')}")
                        st.write(f"**Format:** {model.get('format', 'Unknown')}")
                        st.write(f"**Family:** {model.get('family', 'Unknown')}")
                    
                    with col2:
                        status_color = "🟢" if model.get('loaded') else "🔴"
                        st.write(f"**Status:** {status_color} {'Loaded' if model.get('loaded') else 'Not Loaded'}")
                        
                        if st.button(f"🔄 Reload {model['name']}", key=f"reload_{model['name']}"):
                            st.info(f"Reloading {model['name']}...")
        
        # System information
        st.subheader("💻 System Information")
        
        status = get_service_status()
        if status:
            st.json(status)

if __name__ == "__main__":
    main()
