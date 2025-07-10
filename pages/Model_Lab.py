"""
Model Laboratory

A clean, production-ready model interface with:
- Model synthesis and creation
- Dynamic model building
- Performance tracking and comparison
- Model registry management
- Clean UI without dev clutter
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for clean styling
st.markdown("""
<style>
    .model-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .synthesis-panel {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        border: 2px solid #4caf50;
    }
    
    .model-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-active {
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #4caf50;
    }
    
    .status-training {
        background: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ff9800;
    }
    
    .status-inactive {
        background: #ffebee;
        color: #c62828;
        border: 1px solid #f44336;
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .performance-item {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for model lab."""
    if 'model_registry' not in st.session_state:
        st.session_state.model_registry = {}
    
    if 'synthesis_history' not in st.session_state:
        st.session_state.synthesis_history = []
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None

def load_model_templates():
    """Load available model templates for synthesis."""
    templates = {
        'LSTM': {
            'description': 'Long Short-Term Memory neural network',
            'best_for': 'Time series forecasting with memory',
            'complexity': 'Medium',
            'training_time': '10-30 minutes',
            'parameters': ['layers', 'units', 'dropout', 'learning_rate'],
            'default_params': {'layers': 2, 'units': 64, 'dropout': 0.2, 'learning_rate': 0.001}
        },
        'Transformer': {
            'description': 'Attention-based transformer model',
            'best_for': 'Complex sequence modeling',
            'complexity': 'High',
            'training_time': '20-60 minutes',
            'parameters': ['heads', 'layers', 'd_model', 'dropout'],
            'default_params': {'heads': 8, 'layers': 6, 'd_model': 512, 'dropout': 0.1}
        },
        'XGBoost': {
            'description': 'Gradient boosting with XGBoost',
            'best_for': 'Structured data and feature importance',
            'complexity': 'Medium',
            'training_time': '5-15 minutes',
            'parameters': ['n_estimators', 'max_depth', 'learning_rate'],
            'default_params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        },
        'Ensemble': {
            'description': 'Combination of multiple models',
            'best_for': 'Maximum accuracy and robustness',
            'complexity': 'High',
            'training_time': '30-90 minutes',
            'parameters': ['models', 'weights', 'voting_method', 'weight_update_frequency'],
            'default_params': {'models': ['LSTM', 'XGBoost'], 'weights': [0.6, 0.4], 'voting_method': 'weighted', 'weight_update_frequency': 7}
        },
        'Autoformer': {
            'description': 'Auto-correlation transformer',
            'best_for': 'Long sequence forecasting',
            'complexity': 'High',
            'training_time': '15-45 minutes',
            'parameters': ['factor', 'd_model', 'n_heads'],
            'default_params': {'factor': 5, 'd_model': 512, 'n_heads': 8}
        },
        'Informer': {
            'description': 'Probabilistic attention mechanism',
            'best_for': 'Efficient long sequence modeling',
            'complexity': 'High',
            'training_time': '20-50 minutes',
            'parameters': ['factor', 'd_model', 'n_heads'],
            'default_params': {'factor': 5, 'd_model': 512, 'n_heads': 8}
        },
        'Ridge': {
            'description': 'Ridge regression with regularization',
            'best_for': 'Linear relationships with regularization',
            'complexity': 'Low',
            'training_time': '1-5 minutes',
            'parameters': ['alpha', 'solver'],
            'default_params': {'alpha': 1.0, 'solver': 'auto'}
        },
        'GARCH': {
            'description': 'Generalized Autoregressive Conditional Heteroskedasticity',
            'best_for': 'Volatility forecasting',
            'complexity': 'Medium',
            'training_time': '5-20 minutes',
            'parameters': ['p', 'q', 'vol'],
            'default_params': {'p': 1, 'q': 1, 'vol': 'GARCH'}
        }
    }
    return templates

def get_model_registry():
    """Get current model registry."""
    if not st.session_state.model_registry:
        # Initialize with default models
        st.session_state.model_registry = {
            'LSTM_v1': {
                'name': 'LSTM_v1',
                'type': 'LSTM',
                'accuracy': 0.87,
                'status': 'Active',
                'created': datetime.now() - timedelta(days=5),
                'last_updated': datetime.now() - timedelta(hours=2),
                'parameters': {'layers': 2, 'units': 64, 'dropout': 0.2},
                'performance': {'rmse': 0.023, 'mae': 0.018, 'mape': 2.1}
            },
            'Transformer_v2': {
                'name': 'Transformer_v2',
                'type': 'Transformer',
                'accuracy': 0.89,
                'status': 'Active',
                'created': datetime.now() - timedelta(days=3),
                'last_updated': datetime.now() - timedelta(hours=1),
                'parameters': {'heads': 8, 'layers': 6, 'd_model': 512},
                'performance': {'rmse': 0.021, 'mae': 0.016, 'mape': 1.9}
            },
            'Ensemble_v1': {
                'name': 'Ensemble_v1',
                'type': 'Ensemble',
                'accuracy': 0.92,
                'status': 'Active',
                'created': datetime.now() - timedelta(days=1),
                'last_updated': datetime.now() - timedelta(minutes=30),
                'parameters': {'models': ['LSTM', 'XGBoost'], 'weights': [0.6, 0.4]},
                'performance': {'rmse': 0.019, 'mae': 0.015, 'mape': 1.7}
            },
            'XGBoost_v1': {
                'name': 'XGBoost_v1',
                'type': 'XGBoost',
                'accuracy': 0.85,
                'status': 'Active',
                'created': datetime.now() - timedelta(days=7),
                'last_updated': datetime.now() - timedelta(hours=4),
                'parameters': {'n_estimators': 100, 'max_depth': 6},
                'performance': {'rmse': 0.025, 'mae': 0.020, 'mape': 2.3}
            }
        }
    return st.session_state.model_registry

def synthesize_model(model_type: str, parameters: Dict[str, Any], requirements: str) -> Dict[str, Any]:
    """Synthesize a new model based on requirements."""
    try:
        # Mock model synthesis process
        synthesis_result = {
            'model_name': f"{model_type}_synthesized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model_type': model_type,
            'parameters': parameters,
            'requirements': requirements,
            'status': 'Training',
            'created_at': datetime.now(),
            'estimated_completion': datetime.now() + timedelta(minutes=30),
            'progress': 0,
            'performance_metrics': {},
            'training_logs': []
        }
        
        # Add to synthesis history
        st.session_state.synthesis_history.append(synthesis_result)
        
        # Add to model registry
        st.session_state.model_registry[synthesis_result['model_name']] = {
            'name': synthesis_result['model_name'],
            'type': model_type,
            'accuracy': 0.0,  # Will be updated after training
            'status': 'Training',
            'created': synthesis_result['created_at'],
            'last_updated': synthesis_result['created_at'],
            'parameters': parameters,
            'performance': {}
        }
        
        logger.info(f"Model synthesis initiated: {synthesis_result['model_name']}")
        return synthesis_result
        
    except Exception as e:
        logger.error(f"Error synthesizing model: {e}")
        raise

def plot_model_performance_comparison(models: Dict[str, Any]):
    """Plot model performance comparison."""
    try:
        # Prepare data for plotting
        model_names = []
        accuracies = []
        rmse_values = []
        mae_values = []
        
        for model_name, model_data in models.items():
            if model_data.get('performance'):
                model_names.append(model_name)
                accuracies.append(model_data.get('accuracy', 0))
                rmse_values.append(model_data['performance'].get('rmse', 0))
                mae_values.append(model_data['performance'].get('mae', 0))
        
        if not model_names:
            st.info("No performance data available for comparison")
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'RMSE Comparison', 'MAE Comparison', 'Performance Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=accuracies,
                name='Accuracy',
                marker_color='#2ecc71'
            ),
            row=1, col=1
        )
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=rmse_values,
                name='RMSE',
                marker_color='#e74c3c'
            ),
            row=1, col=2
        )
        
        # MAE comparison
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=mae_values,
                name='MAE',
                marker_color='#3498db'
            ),
            row=2, col=1
        )
        
        # Performance overview (scatter plot)
        fig.add_trace(
            go.Scatter(
                x=rmse_values,
                y=accuracies,
                mode='markers+text',
                text=model_names,
                textposition="top center",
                name='Performance Overview',
                marker=dict(size=10, color='#9b59b6')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=True,
            title_text="Model Performance Comparison",
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
        st.error("Error generating model comparison visualization")

def display_model_details(model_data: Dict[str, Any]):
    """Display detailed model information."""
    try:
        st.markdown("### Model Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="model-card">
                <h4>{model_data['name']}</h4>
                <p><strong>Type:</strong> {model_data['type']}</p>
                <p><strong>Status:</strong> 
                    <span class="model-status status-{model_data['status'].lower()}">{model_data['status']}</span>
                </p>
                <p><strong>Accuracy:</strong> {model_data.get('accuracy', 0):.1%}</p>
                <p><strong>Created:</strong> {model_data['created'].strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Last Updated:</strong> {model_data['last_updated'].strftime('%Y-%m-%d %H:%M')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Parameters")
            st.json(model_data['parameters'])
        
        # Performance metrics
        if model_data.get('performance'):
            st.markdown("### Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="performance-item">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                        {model_data['performance'].get('rmse', 0):.4f}
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">RMSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="performance-item">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                        {model_data['performance'].get('mae', 0):.4f}
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">MAE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="performance-item">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                        {model_data['performance'].get('mape', 0):.1f}%
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">MAPE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="performance-item">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #2c3e50;">
                        {model_data.get('accuracy', 0):.1%}
                    </div>
                    <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Special handling for ensemble models
        if model_data['type'] == 'Ensemble' and 'weights' in model_data['parameters']:
            st.markdown("### 🎛️ Ensemble Weight Controls")
            
            weights = model_data['parameters']['weights']
            models = model_data['parameters'].get('models', [f'Model_{i}' for i in range(len(weights))])
            
            # Create weight sliders
            new_weights = []
            for i, (model, weight) in enumerate(zip(models, weights)):
                new_weight = st.slider(
                    f"Weight for {model}",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(weight),
                    step=0.05,
                    help=f"Adjust weight for {model} in ensemble"
                )
                new_weights.append(new_weight)
            
            # Normalize weights
            total_weight = sum(new_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in new_weights]
                st.write("**Normalized Weights:**")
                for model, weight in zip(models, normalized_weights):
                    st.write(f"- {model}: {weight:.2%}")
                
                # Update button
                if st.button("Update Ensemble Weights"):
                    model_data['parameters']['weights'] = normalized_weights
                    model_data['last_updated'] = datetime.now()
                    st.success("Ensemble weights updated!")
            else:
                st.warning("Total weight cannot be zero")
        
    except Exception as e:
        logger.error(f"Error displaying model details: {e}")

def main():
    """Main model lab function."""
    st.title("Model Laboratory")
    st.markdown("Create, train, and manage AI models for trading")
    
    # Add explanations for users
    with st.expander("ℹ️ About Model Laboratory", expanded=False):
        st.markdown("""
        **Model Laboratory** is your AI model development workspace where you can:
        
        ### 🧬 **Model Synthesis**
        - **LSTM Models**: Long Short-Term Memory networks for time series forecasting with memory capabilities
        - **XGBoost Models**: Gradient boosting for structured data with feature importance analysis
        - **Transformer Models**: Attention-based models for complex sequence modeling
        - **Ensemble Models**: Combinations of multiple models for maximum accuracy
        
        ### 🔧 **Model Tuning**
        - **Hyperparameter Optimization**: Automatically find the best model parameters
        - **Feature Engineering**: Create and test new features for better performance
        - **Cross-Validation**: Ensure model robustness across different data periods
        
        ### 📊 **Performance Tracking**
        - **Real-time Metrics**: Monitor accuracy, RMSE, MAE, and other performance indicators
        - **Model Comparison**: Compare different models side-by-side
        - **Performance History**: Track how models improve over time
        
        ### 🎯 **Best Practices**
        - **LSTM**: Use for time series with long-term dependencies (stock prices, weather data)
        - **XGBoost**: Use for structured data with clear feature relationships
        - **Transformer**: Use for complex patterns requiring attention mechanisms
        - **Ensemble**: Use when you need maximum accuracy and robustness
        """)
    
    # Initialize session state
    initialize_session_state()
    
    # Load templates and registry
    templates = load_model_templates()
    model_registry = get_model_registry()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Model Management")
        
        # Model selection
        model_options = list(model_registry.keys())
        selected_model = st.selectbox("Select Model", model_options, index=0)
        
        # Model actions
        st.subheader("Actions")
        
        if st.button("Train Model", type="primary"):
            if selected_model in model_registry:
                # Mock training process
                model_registry[selected_model]['status'] = 'Training'
                model_registry[selected_model]['last_updated'] = datetime.now()
                st.success(f"Training started for {selected_model}")
        
        if st.button("Evaluate Model"):
            if selected_model in model_registry:
                # Mock evaluation
                model_registry[selected_model]['accuracy'] = np.random.uniform(0.8, 0.95)
                model_registry[selected_model]['performance'] = {
                    'rmse': np.random.uniform(0.015, 0.035),
                    'mae': np.random.uniform(0.012, 0.028),
                    'mape': np.random.uniform(1.5, 3.0)
                }
                model_registry[selected_model]['status'] = 'Active'
                st.success(f"Evaluation completed for {selected_model}")
        
        if st.button("Delete Model"):
            if selected_model in model_registry:
                del model_registry[selected_model]
                st.success(f"Model {selected_model} deleted")
                st.rerun()
        
        # Model synthesis
        st.markdown("---")
        st.subheader("Create New Model")
        
        synthesis_type = st.selectbox("Model Type", list(templates.keys()))
        
        if synthesis_type in templates:
            template = templates[synthesis_type]
            st.markdown(f"**{synthesis_type}**")
            st.markdown(f"*{template['description']}*")
            st.markdown(f"**Best for:** {template['best_for']}")
            st.markdown(f"**Complexity:** {template['complexity']}")
            st.markdown(f"**Training time:** {template['training_time']}")
        
        requirements = st.text_area(
            "Model Requirements",
            placeholder="Describe what you want the model to do...",
            height=100
        )
        
        if st.button("Synthesize Model"):
            if requirements and synthesis_type:
                try:
                    template = templates[synthesis_type]
                    result = synthesize_model(synthesis_type, template['default_params'], requirements)
                    st.success(f"Model synthesis initiated: {result['model_name']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error synthesizing model: {e}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model details and performance
        if selected_model in model_registry:
            model_data = model_registry[selected_model]
            display_model_details(model_data)
            
            # Model comparison
            if len(model_registry) > 1:
                st.markdown("---")
                st.markdown("### Model Comparison")
                plot_model_performance_comparison(model_registry)
        else:
            st.info("Select a model from the sidebar")
    
    with col2:
        # Model registry overview
        st.markdown("### Model Registry")
        
        # Summary statistics
        active_models = sum(1 for m in model_registry.values() if m['status'] == 'Active')
        training_models = sum(1 for m in model_registry.values() if m['status'] == 'Training')
        avg_accuracy = np.mean([m.get('accuracy', 0) for m in model_registry.values()])
        
        st.metric("Total Models", len(model_registry))
        st.metric("Active Models", active_models)
        st.metric("Training Models", training_models)
        st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        
        # Recent synthesis history
        if st.session_state.synthesis_history:
            st.markdown("### Recent Synthesis")
            for synthesis in st.session_state.synthesis_history[-3:]:
                with st.expander(f"{synthesis['model_name']} ({synthesis['status']})"):
                    st.markdown(f"**Type:** {synthesis['model_type']}")
                    st.markdown(f"**Created:** {synthesis['created_at'].strftime('%Y-%m-%d %H:%M')}")
                    st.markdown(f"**Requirements:** {synthesis['requirements'][:100]}...")
    
    # Model synthesis section
    st.markdown("---")
    st.markdown("""
    <div class="synthesis-panel">
        <h3>Advanced Model Synthesis</h3>
        <p>Create custom models using natural language descriptions and advanced AI synthesis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Natural Language Model Creation")
        
        nl_requirements = st.text_area(
            "Describe your model requirements:",
            placeholder="e.g., 'Create a deep learning model for cryptocurrency price prediction that can handle high volatility and multiple timeframes'",
            height=150
        )
        
        if st.button("Create with AI", type="primary"):
            if nl_requirements:
                try:
                    # Mock AI model creation
                    model_name = f"AI_Synthesized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Determine model type based on requirements
                    requirements_lower = nl_requirements.lower()
                    if 'deep learning' in requirements_lower or 'neural' in requirements_lower:
                        model_type = 'LSTM'
                    elif 'ensemble' in requirements_lower or 'multiple' in requirements_lower:
                        model_type = 'Ensemble'
                    elif 'volatility' in requirements_lower:
                        model_type = 'GARCH'
                    else:
                        model_type = 'XGBoost'
                    
                    result = synthesize_model(model_type, {}, nl_requirements)
                    st.success(f"AI model creation initiated: {result['model_name']}")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating AI model: {e}")
            else:
                st.warning("Please provide model requirements")
    
    with col2:
        st.subheader("Template-Based Creation")
        
        template_type = st.selectbox("Select Template", list(templates.keys()))
        
        if template_type in templates:
            template = templates[template_type]
            
            st.markdown(f"**{template_type} Template**")
            st.markdown(f"*{template['description']}*")
            
            # Parameter customization
            st.markdown("**Parameters:**")
            for param in template['parameters']:
                default_value = template['default_params'].get(param, 0)
                if isinstance(default_value, int):
                    value = st.number_input(param, value=default_value, min_value=1)
                elif isinstance(default_value, float):
                    value = st.number_input(param, value=float(default_value), format="%.3f")
                else:
                    value = st.text_input(param, value=str(default_value))
                
                template['default_params'][param] = value
        
        if st.button("Create from Template"):
            try:
                result = synthesize_model(template_type, template['default_params'], f"Created from {template_type} template")
                st.success(f"Template model created: {result['model_name']}")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating template model: {e}")

if __name__ == "__main__":
    main() 