#!/usr/bin/env python3
"""
Test script for the Enhanced Model Creator Agent

This script demonstrates all the new features:
- Automatic validation and compilation
- Full forecasting and backtesting
- Comprehensive evaluation metrics
- Centralized leaderboard
- Automatic model management
- Blueprint storage and reuse
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from trading.agents.model_creator_agent import get_model_creator_agent

def test_enhanced_model_creator():
    """Test the enhanced model creator agent."""
    print("🚀 Testing Enhanced Model Creator Agent")
    print("=" * 50)
    
    # Initialize the agent
    agent = get_model_creator_agent()
    
    # Test 1: Create and validate models
    print("\n1. Creating and Validating Models")
    print("-" * 30)
    
    requirements_list = [
        "Create a simple random forest model for price prediction",
        "Build a complex XGBoost model for forecasting with 200 trees",
        "Make a LightGBM model for regression with learning rate 0.05"
    ]
    
    created_models = []
    
    for i, requirements in enumerate(requirements_list):
        print(f"\nCreating model {i+1}: {requirements}")
        
        # Create and validate model
        spec, is_valid, errors = agent.create_and_validate_model(requirements)
        
        if is_valid:
            print(f"✅ Model '{spec.name}' created and validated successfully")
            print(f"   Framework: {spec.framework}")
            print(f"   Type: {spec.model_type}")
            print(f"   Validation Status: {spec.validation_status}")
            print(f"   Compilation Status: {spec.compilation_status}")
            created_models.append(spec.name)
        else:
            print(f"❌ Model creation failed: {errors}")
    
    # Test 2: Run full evaluation
    print("\n\n2. Running Full Evaluation")
    print("-" * 30)
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    
    evaluations = []
    
    for model_name in created_models:
        print(f"\nEvaluating model: {model_name}")
        
        try:
            evaluation = agent.run_full_evaluation(model_name, X, y)
            
            print(f"✅ Evaluation completed for {model_name}")
            print(f"   Performance Grade: {evaluation.performance_grade}")
            print(f"   Overall Score: {evaluation.validation_score:.3f}")
            print(f"   RMSE: {evaluation.metrics.get('RMSE', 0):.4f}")
            print(f"   MAPE: {evaluation.metrics.get('MAPE', 0):.2f}%")
            print(f"   MAE: {evaluation.metrics.get('MAE', 0):.4f}")
            print(f"   Sharpe Ratio: {evaluation.metrics.get('Sharpe_Ratio', 0):.3f}")
            print(f"   Win Rate: {evaluation.metrics.get('Win_Rate', 0):.3f}")
            print(f"   Approved: {'Yes' if evaluation.is_approved else 'No'}")
            
            if evaluation.recommendations:
                print(f"   Recommendations: {', '.join(evaluation.recommendations[:2])}")
            
            evaluations.append(evaluation)
            
        except Exception as e:
            print(f"❌ Evaluation failed for {model_name}: {e}")
    
    # Test 3: Leaderboard functionality
    print("\n\n3. Leaderboard Management")
    print("-" * 30)
    
    leaderboard = agent.get_leaderboard()
    print(f"Total models in leaderboard: {len(leaderboard)}")
    
    if leaderboard:
        print("\nTop 3 Models:")
        for i, entry in enumerate(leaderboard[:3]):
            print(f"   {i+1}. {entry.model_name} (Grade: {entry.performance_grade}, Score: {entry.overall_score:.3f})")
        
        approved_models = agent.get_leaderboard().get_approved_models()
        print(f"\nApproved models: {len(approved_models)}")
    
    # Test 4: Model suggestions
    print("\n\n4. Model Improvement Suggestions")
    print("-" * 30)
    
    suggestions = agent.get_model_suggestions("Create a simple sklearn model")
    print(f"Generated {len(suggestions)} suggestions:")
    
    for suggestion in suggestions:
        print(f"   - {suggestion['type']}: {suggestion['reason']}")
    
    # Test 5: Blueprint storage and reuse
    print("\n\n5. Blueprint Storage and Reuse")
    print("-" * 30)
    
    if created_models:
        model_name = created_models[0]
        blueprint_path = f"data/blueprints/{model_name}_blueprint.json"
        
        # Save blueprint
        success = agent.save_model_blueprint(model_name, blueprint_path)
        if success:
            print(f"✅ Saved blueprint for {model_name}")
            
            # Load blueprint
            loaded_spec = agent.load_model_blueprint(blueprint_path)
            if loaded_spec:
                print(f"✅ Loaded blueprint for {loaded_spec.name}")
                print(f"   Framework: {loaded_spec.framework}")
                print(f"   Architecture: {loaded_spec.architecture_blueprint.get('architecture_type', 'N/A')}")
        else:
            print(f"❌ Failed to save blueprint for {model_name}")
    
    # Test 6: Automatic cleanup
    print("\n\n6. Automatic Model Cleanup")
    print("-" * 30)
    
    removed_models = agent.auto_cleanup_poor_models()
    if removed_models:
        print(f"🗑️  Automatically removed {len(removed_models)} poor performing models:")
        for model in removed_models:
            print(f"   - {model}")
    else:
        print("✅ No poor performing models to remove")
    
    # Test 7: Framework status
    print("\n\n7. Framework Status")
    print("-" * 30)
    
    framework_status = agent.get_framework_status()
    for framework, available in framework_status.items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"   {framework}: {status}")
    
    # Summary
    print("\n\n📊 Summary")
    print("=" * 50)
    print(f"Models created: {len(created_models)}")
    print(f"Models evaluated: {len(evaluations)}")
    print(f"Models in leaderboard: {len(agent.get_leaderboard())}")
    print(f"Approved models: {len([e for e in evaluations if e.is_approved])}")
    
    # Performance distribution
    if evaluations:
        grades = [e.performance_grade for e in evaluations]
        grade_counts = pd.Series(grades).value_counts()
        print(f"\nPerformance Grade Distribution:")
        for grade, count in grade_counts.items():
            print(f"   {grade}: {count}")
    
    print("\n🎉 Enhanced Model Creator Agent test completed successfully!")

if __name__ == "__main__":
    test_enhanced_model_creator() 