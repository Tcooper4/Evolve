"""
Integration Demo for Agentic Forecasting System

This script demonstrates the integration of all modules:
- Goal Status Management
- Optimizer Consolidation
- Market Analysis
- Data Pipeline
- Data Validation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_goal_status_integration():
    """Demonstrate goal status integration."""
    st.subheader("🎯 Goal Status Integration")
    
    try:
        from trading.memory.goals.status import (
            get_status_summary, 
            update_goal_progress, 
            log_agent_contribution,
            get_agent_contributions
        )
        
        # Display current goal status
        status_summary = get_status_summary()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Status", status_summary["current_status"])
        with col2:
            st.metric("Progress", f"{status_summary['progress']:.1%}")
        with col3:
            st.metric("Alerts", len(status_summary.get("alerts", [])))
        
        # Display recommendations
        if status_summary.get("recommendations"):
            st.markdown("**Recommendations:**")
            for rec in status_summary["recommendations"]:
                st.info(f"💡 {rec}")
        
        # Display recent agent contributions
        contributions = get_agent_contributions(limit=5)
        if contributions:
            st.markdown("**Recent Agent Contributions:**")
            for contrib in contributions:
                st.write(f"• **{contrib['agent']}**: {contrib['contribution']} ({contrib['impact']} impact)")
        
        # Interactive goal update
        with st.expander("Update Goal Progress"):
            new_progress = st.slider("Progress", 0.0, 1.0, status_summary["progress"], 0.1)
            new_status = st.selectbox("Status", ["on_track", "behind_schedule", "ahead_of_schedule", "completed"])
            new_message = st.text_input("Message", "Updated via demo")
            
            if st.button("Update Goal"):
                update_goal_progress(
                    progress=new_progress,
                    status=new_status,
                    message=new_message
                )
                st.success("Goal updated successfully!")
                st.experimental_rerun()
        
        return True
        
    except Exception as e:
        st.error(f"Goal status integration error: {str(e)}")
        return False


def demo_optimizer_consolidation():
    """Demonstrate optimizer consolidation."""
    st.subheader("🔧 Optimizer Consolidation")
    
    try:
        from trading.optimization.utils.consolidator import OptimizerConsolidator, get_optimizer_status
        
        # Get current optimizer status
        status = get_optimizer_status()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimize Dir Exists", "Yes" if status["optimize_dir_exists"] else "No")
        with col2:
            st.metric("Files in Optimize", len(status["files_in_optimize"]))
        with col3:
            st.metric("Consolidation Needed", "Yes" if status["consolidation_needed"] else "No")
        
        # Display file status
        if status["files_in_optimize"]:
            st.markdown("**Files in optimize directory:**")
            for file in status["files_in_optimize"]:
                st.write(f"• {file}")
        
        if status["duplicate_files"]:
            st.warning(f"⚠️ Duplicate files found: {status['duplicate_files']}")
        
        # Consolidation controls
        with st.expander("Run Consolidation"):
            create_backup = st.checkbox("Create backup before consolidation", value=True)
            
            if st.button("Run Optimizer Consolidation"):
                with st.spinner("Running consolidation..."):
                    consolidator = OptimizerConsolidator()
                    results = consolidator.run_optimizer_consolidation(create_backup=create_backup)
                    
                    if results["success"]:
                        st.success("✅ Consolidation completed successfully!")
                        st.json(results)
                    else:
                        st.error("❌ Consolidation failed!")
                        st.error(f"Errors: {results['errors']}")
        
        # Complete consolidation option
        with st.expander("Run Complete Consolidation"):
            st.info("This will run the complete optimization consolidation process including import fixes and validation.")
            
            if st.button("Run Complete Optimization Consolidation"):
                with st.spinner("Running complete consolidation..."):
                    try:
                        from complete_optimization_consolidation import main as run_complete_consolidation
                        run_complete_consolidation()
                        st.success("✅ Complete consolidation completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Complete consolidation failed: {str(e)}")
        
        return True
        
    except Exception as e:
        st.error(f"Optimizer consolidation error: {str(e)}")
        return False


def demo_market_analysis():
    """Demonstrate market analysis integration."""
    st.subheader("📊 Market Analysis Integration")
    
    try:
        from src.analysis.market_analysis import MarketAnalysis
        
        # Create sample market data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure price relationships are valid
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 10, len(dates))
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 10, len(dates))
        
        # Run market analysis
        market_analyzer = MarketAnalysis()
        analysis = market_analyzer.analyze_market(sample_data)
        
        # Display market regime
        if 'regime' in analysis:
            regime = analysis['regime']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Regime", regime.name)
            with col2:
                st.metric("Confidence", f"{regime.confidence:.1%}")
            with col3:
                st.metric("Trend Strength", f"{regime.metrics.get('trend_strength', 0):.2f}")
            
            st.info(f"**{regime.name}**: {regime.description}")
        
        # Display market conditions
        if 'conditions' in analysis and analysis['conditions']:
            st.markdown("**Market Conditions:**")
            for condition in analysis['conditions'][:3]:  # Show first 3 conditions
                with st.expander(f"{condition.name} (Strength: {condition.strength:.1%})"):
                    st.write(condition.description)
        
        # Display trading signals
        if 'signals' in analysis:
            signal_categories = ['trend', 'momentum', 'volatility']
            for category in signal_categories:
                if category in analysis['signals']:
                    signals = analysis['signals'][category]
                    if signals:
                        st.markdown(f"**{category.title()} Signals:**")
                        for signal_name, signal_data in list(signals.items())[:2]:  # Show first 2 signals
                            st.write(f"• {signal_name}: {signal_data}")
        
        return True
        
    except Exception as e:
        st.error(f"Market analysis error: {str(e)}")
        return False


def demo_data_pipeline():
    """Demonstrate data pipeline integration."""
    st.subheader("🔄 Data Pipeline Integration")
    
    try:
        from src.utils.data_pipeline import DataPipeline, run_data_pipeline
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Ensure price relationships are valid
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 10, len(dates))
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 10, len(dates))
        
        # Save sample data
        sample_file = "sample_market_data.csv"
        sample_data.to_csv(sample_file)
        
        # Configure pipeline
        pipeline_config = {
            'missing_data_method': 'ffill',
            'remove_outliers': True,
            'outlier_columns': ['close', 'volume'],
            'outlier_std': 3.0,
            'normalize': False
        }
        
        # Run pipeline
        with st.spinner("Running data pipeline..."):
            success, processed_data, stats = run_data_pipeline(
                sample_file, 
                config=pipeline_config
            )
        
        if success:
            st.success("✅ Data pipeline completed successfully!")
            
            # Display pipeline statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Shape", f"{stats['data_shape']}")
            with col2:
                st.metric("Processed Shape", f"{stats['processed_data_shape']}")
            with col3:
                st.metric("Duration", stats['pipeline_duration'])
            
            # Display processed data info
            if processed_data is not None:
                st.markdown("**Processed Data Info:**")
                st.write(f"• Shape: {processed_data.shape}")
                st.write(f"• Columns: {list(processed_data.columns)}")
                st.write(f"• Memory usage: {processed_data.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Show sample of processed data
                st.markdown("**Sample Processed Data:**")
                st.dataframe(processed_data.head())
        
        else:
            st.error("❌ Data pipeline failed!")
        
        # Clean up
        import os
        if os.path.exists(sample_file):
            os.remove(sample_file)
        
        return success
        
    except Exception as e:
        st.error(f"Data pipeline error: {str(e)}")
        return False


def demo_data_validation():
    """Demonstrate data validation integration."""
    st.subheader("✅ Data Validation Integration")
    
    try:
        from src.utils.data_validation import DataValidator, validate_data_for_training, validate_data_for_forecasting
        
        # Create sample data with some issues
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        # Create data with some validation issues
        sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, len(dates)),
            'high': np.random.uniform(200, 300, len(dates)),
            'low': np.random.uniform(50, 150, len(dates)),
            'close': np.random.uniform(100, 200, len(dates)),
            'volume': np.random.uniform(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Add some issues for demonstration
        sample_data.loc[10:15, 'close'] = np.nan  # Missing values
        sample_data.loc[20, 'high'] = 0  # Invalid price
        sample_data.loc[25, 'volume'] = -1000  # Negative volume
        
        # Ensure price relationships are valid for most data
        sample_data['high'] = sample_data[['open', 'close']].max(axis=1) + np.random.uniform(0, 10, len(dates))
        sample_data['low'] = sample_data[['open', 'close']].min(axis=1) - np.random.uniform(0, 10, len(dates))
        
        # Run validation
        validator = DataValidator()
        is_valid, error_message = validator.validate_dataframe(sample_data)
        validation_results = validator.get_validation_results()
        
        # Display validation results
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "success" if is_valid else "error"
            getattr(st, status_color)(f"Validation: {'Passed' if is_valid else 'Failed'}")
        with col2:
            st.metric("Total Checks", validation_results["summary"]["total_checks"])
        with col3:
            st.metric("Warnings", len(validation_results["warnings"]))
        
        # Display detailed results
        with st.expander("Validation Details"):
            st.json(validation_results)
        
        # Display warnings and errors
        if validation_results["warnings"]:
            st.warning("⚠️ Validation Warnings:")
            for warning in validation_results["warnings"]:
                st.write(f"• {warning}")
        
        if validation_results["errors"]:
            st.error("❌ Validation Errors:")
            for error in validation_results["errors"]:
                st.write(f"• {error}")
        
        # Test specific validation functions
        st.markdown("**Specific Validation Tests:**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Training Validation"):
                train_valid, train_summary = validate_data_for_training(sample_data)
                if train_valid:
                    st.success("✅ Training validation passed")
                else:
                    st.error("❌ Training validation failed")
                st.json(train_summary)
        
        with col2:
            if st.button("Test Forecasting Validation"):
                forecast_valid, forecast_summary = validate_data_for_forecasting(sample_data)
                if forecast_valid:
                    st.success("✅ Forecasting validation passed")
                else:
                    st.error("❌ Forecasting validation failed")
                st.json(forecast_summary)
        
        return True
        
    except Exception as e:
        st.error(f"Data validation error: {str(e)}")
        return False


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Integration Demo - Evolve",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Integration Demo - Agentic Forecasting System")
    st.markdown("""
    This demo showcases the integration of all system modules:
    
    - 🎯 **Goal Status Management**: Track and manage system goals
    - 🔧 **Optimizer Consolidation**: Organize and consolidate optimizer files
    - 📊 **Market Analysis**: Comprehensive market context analysis
    - 🔄 **Data Pipeline**: End-to-end data processing pipeline
    - ✅ **Data Validation**: Comprehensive data quality validation
    """)
    
    # Run all demos
    demo_results = {}
    
    demo_results["goal_status"] = demo_goal_status_integration()
    demo_results["optimizer_consolidation"] = demo_optimizer_consolidation()
    demo_results["market_analysis"] = demo_market_analysis()
    demo_results["data_pipeline"] = demo_data_pipeline()
    demo_results["data_validation"] = demo_data_validation()
    
    # Summary
    st.subheader("📋 Integration Summary")
    
    successful_demos = sum(demo_results.values())
    total_demos = len(demo_results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Successful Integrations", f"{successful_demos}/{total_demos}")
    with col2:
        success_rate = successful_demos / total_demos * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Display individual results
    st.markdown("**Module Integration Status:**")
    for module, success in demo_results.items():
        if success:
            st.success(f"✅ {module.replace('_', ' ').title()}")
        else:
            st.error(f"❌ {module.replace('_', ' ').title()}")
    
    if successful_demos == total_demos:
        st.success("🎉 All integrations successful! The system is ready for use.")
    else:
        st.warning(f"⚠️ {total_demos - successful_demos} integration(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main() 