"""
MetaControllerAgent Example

This example demonstrates how to use the MetaControllerAgent to:
1. Monitor system performance and health
2. Evaluate triggers and make intelligent decisions
3. Execute actions based on system conditions
4. Generate reports and recommendations
5. Maintain system logs and audit trails
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from meta.meta_controller import (
    MetaControllerAgent,
    ActionType,
    TriggerCondition,
    create_meta_controller,
    run_meta_controller
)


def main():
    """Main example function"""
    print("ðŸŽ›ï¸ MetaControllerAgent Example")
    print("=" * 50)
    
    # Initialize the meta controller
    print("Initializing MetaControllerAgent...")
    controller = create_meta_controller()
    
    # Example 1: System monitoring and metrics collection
    print("\nðŸ“Š System Monitoring and Metrics Collection")
    
    # Collect system metrics
    metrics = controller.collect_system_metrics()
    
    if metrics:
        print(f"System Metrics Collected:")
        print(f"  Timestamp: {metrics.timestamp}")
        print(f"  Model Performance: {len(metrics.model_performance)} models")
        print(f"  Strategy Performance: {len(metrics.strategy_performance)} strategies")
        print(f"  Sentiment Scores: {len(metrics.sentiment_scores)} tickers")
        print(f"  System Health: {len(metrics.system_health)} metrics")
        print(f"  Error Count: {metrics.error_count}")
        print(f"  Active Trades: {metrics.active_trades}")
        print(f"  Portfolio Value: ${metrics.portfolio_value:,.2f}")
        
        # Show sample metrics
        if metrics.model_performance:
            avg_model_perf = np.mean(list(metrics.model_performance.values()))
            print(f"  Average Model Performance: {avg_model_perf:.3f}")
        
        if metrics.system_health:
            cpu_usage = metrics.system_health.get('cpu_usage', 0)
            memory_usage = metrics.system_health.get('memory_usage', 0)
            print(f"  CPU Usage: {cpu_usage:.1f}%")
            print(f"  Memory Usage: {memory_usage:.1f}%")
    else:
        print("No system metrics collected")
    
    # Example 2: Trigger evaluation
    print("\nðŸ” Trigger Evaluation")
    
    decisions = controller.evaluate_triggers(metrics) if metrics else []
    
    print(f"Trigger Evaluation Results:")
    print(f"  Decisions Made: {len(decisions)}")
    
    for i, decision in enumerate(decisions):
        print(f"  Decision {i+1}:")
        print(f"    Action: {decision.action_type.value}")
        print(f"    Trigger: {decision.trigger_condition.value}")
        print(f"    Priority: {decision.priority}")
        print(f"    Reason: {decision.reason}")
        print(f"    Estimated Duration: {decision.estimated_duration} minutes")
        print(f"    Affected Components: {', '.join(decision.affected_components)}")
    
    # Example 3: Action execution
    print("\nâš¡ Action Execution")
    
    if decisions:
        print("Executing high-priority actions...")
        
        for decision in decisions:
            if decision.priority >= 3:  # Medium and high priority
                print(f"  Executing: {decision.action_type.value}")
                
                # Execute action
                result = asyncio.run(controller.execute_action(decision))
                
                print(f"    Action ID: {result.action_id}")
                print(f"    Success: {result.success}")
                print(f"    Duration: {result.duration_minutes:.1f} minutes")
                
                if result.errors:
                    print(f"    Errors: {', '.join(result.errors)}")
                
                if result.recommendations:
                    print(f"    Recommendations: {', '.join(result.recommendations)}")
    else:
        print("No actions to execute")
    
    # Example 4: Performance summary and recommendations
    print("\nðŸ“ˆ Performance Summary and Recommendations")
    
    summary = controller._get_performance_summary()
    
    if summary:
        print("Performance Summary:")
        if 'model_performance' in summary and summary['model_performance']:
            print(f"  Model Performance: {summary['model_performance'].get('average', 0):.3f}")
        
        if 'strategy_performance' in summary and summary['strategy_performance']:
            print(f"  Strategy Performance: {summary['strategy_performance'].get('average', 0):.3f}")
        
        print(f"  Action Success Rate: {summary.get('action_success_rate', 0):.1%}")
    
    # Generate recommendations
    recommendations = controller._generate_recommendations()
    
    print(f"\nSystem Recommendations ({len(recommendations)}):")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"  {i}. {recommendation}")
    
    # Example 5: System health monitoring
    print("\nðŸ¥ System Health Monitoring")
    
    # Simulate health check
    health_decision = controller._evaluate_system_health(metrics) if metrics else None
    
    if health_decision:
        print(f"Health Check Required:")
        print(f"  Action: {health_decision.action_type.value}")
        print(f"  Priority: {health_decision.priority}")
        print(f"  Reason: {health_decision.reason}")
        
        # Execute health check
        health_result = asyncio.run(controller.execute_action(health_decision))
        print(f"  Health Check Result: {'âœ… Healthy' if health_result.success else 'âŒ Issues Found'}")
        
        if health_result.results and 'components' in health_result.results:
            print("  Component Status:")
            for component, status in health_result.results['components'].items():
                status_icon = "âœ…" if status['status'] == 'healthy' else "âš ï¸" if status['status'] == 'warning' else "âŒ"
                print(f"    {status_icon} {component}: {status['status']}")
    else:
        print("System health is good - no health check required")
    
    # Example 6: Action history and audit trail
    print("\nðŸ“‹ Action History and Audit Trail")
    
    recent_actions = controller._get_recent_actions(hours=24)
    
    print(f"Recent Actions (Last 24 Hours): {len(recent_actions)}")
    
    if recent_actions:
        # Group by action type
        action_counts = {}
        success_counts = {}
        
        for action in recent_actions:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1
            if action.success:
                success_counts[action_type] = success_counts.get(action_type, 0) + 1
        
        print("Action Summary:")
        for action_type, count in action_counts.items():
            success_rate = success_counts.get(action_type, 0) / count
            print(f"  {action_type}: {count} actions ({success_rate:.1%} success rate)")
        
        # Show latest action
        latest_action = recent_actions[-1]
        print(f"\nLatest Action:")
        print(f"  Type: {latest_action.action_type.value}")
        print(f"  Time: {latest_action.start_time}")
        print(f"  Success: {latest_action.success}")
        print(f"  Duration: {latest_action.duration_minutes:.1f} minutes")
    else:
        print("No recent actions found")
    
    # Example 7: Threshold configuration
    print("\nâš™ï¸ Threshold Configuration")
    
    print("Current Trigger Thresholds:")
    for trigger_type, thresholds in controller.thresholds.items():
        print(f"  {trigger_type}:")
        for key, value in thresholds.items():
            if isinstance(value, (int, float)):
                print(f"    {key}: {value}")
            elif isinstance(value, dict):
                print(f"    {key}: {len(value)} sub-thresholds")
    
    # Example 8: Market state monitoring
    print("\nðŸ“ˆ Market State Monitoring")
    
    print(f"Current Market State:")
    print(f"  Volatility: {controller.market_state['volatility']:.3f}")
    print(f"  Trend: {controller.market_state['trend']}")
    print(f"  Sentiment: {controller.market_state['sentiment']:.3f}")
    print(f"  Volume: {controller.market_state['volume']:.3f}")
    
    # Example 9: Scheduled monitoring
    print("\nâ° Scheduled Monitoring Setup")
    
    # Start the scheduler in background
    print("Starting scheduled monitoring...")
    
    # Run the main controller
    results = controller.run()
    
    print(f"Controller Results:")
    print(f"  Status: {results['status']}")
    print(f"  Metrics Collected: {results['metrics_collected']}")
    print(f"  Decisions Made: {results['decisions_made']}")
    print(f"  Actions Executed: {results['actions_executed']}")
    print(f"  Scheduler Started: {results['scheduler_started']}")
    print(f"  System Status: {results['system_status']}")
    
    print("\nâœ… MetaControllerAgent example completed!")


def example_manual_trigger():
    """Example of manually triggering actions"""
    print("\nðŸŽ¯ Manual Action Trigger Example")
    
    controller = create_meta_controller()
    
    # Create manual action decision
    manual_decision = controller.ActionDecision(
        action_type=ActionType.REPORT_GENERATE,
        trigger_condition=TriggerCondition.MANUAL_REQUEST,
        timestamp=datetime.now().isoformat(),
        reason="Manual report generation request",
        priority=2,
        estimated_duration=30,
        affected_components=["reporting"],
        parameters={"manual_request": True}
    )
    
    print(f"Manual Action Created:")
    print(f"  Action: {manual_decision.action_type.value}")
    print(f"  Trigger: {manual_decision.trigger_condition.value}")
    print(f"  Priority: {manual_decision.priority}")
    print(f"  Reason: {manual_decision.reason}")
    
    # Execute manual action
    print("\nExecuting manual action...")
    result = asyncio.run(controller.execute_action(manual_decision))
    
    print(f"Manual Action Result:")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_minutes:.1f} minutes")
    
    if result.results:
        print(f"  Results: {result.results}")
    
    if result.recommendations:
        print(f"  Recommendations: {', '.join(result.recommendations)}")


def example_performance_monitoring():
    """Example of performance monitoring"""
    print("\nðŸ“Š Performance Monitoring Example")
    
    controller = create_meta_controller()
    
    # Simulate performance monitoring over time
    print("Simulating performance monitoring...")
    
    performance_data = []
    
    for i in range(10):
        # Simulate metrics collection
        metrics = controller.collect_system_metrics()
        
        if metrics:
            # Extract performance metrics
            avg_model_perf = np.mean(list(metrics.model_performance.values())) if metrics.model_performance else 0
            avg_strategy_perf = np.mean(list(metrics.strategy_performance.values())) if metrics.strategy_performance else 0
            
            performance_data.append({
                'timestamp': metrics.timestamp,
                'model_performance': avg_model_perf,
                'strategy_performance': avg_strategy_perf,
                'error_count': metrics.error_count,
                'active_trades': metrics.active_trades
            })
        
        # Simulate time passing
        time.sleep(0.1)
    
    # Analyze performance trends
    if performance_data:
        df = pd.DataFrame(performance_data)
        
        print(f"Performance Analysis ({len(df)} data points):")
        print(f"  Model Performance: {df['model_performance'].mean():.3f} Â± {df['model_performance'].std():.3f}")
        print(f"  Strategy Performance: {df['strategy_performance'].mean():.3f} Â± {df['strategy_performance'].std():.3f}")
        print(f"  Average Error Count: {df['error_count'].mean():.1f}")
        print(f"  Average Active Trades: {df['active_trades'].mean():.1f}")
        
        # Detect trends
        model_trend = "ðŸ“ˆ Improving" if df['model_performance'].iloc[-1] > df['model_performance'].iloc[0] else "ðŸ“‰ Declining"
        strategy_trend = "ðŸ“ˆ Improving" if df['strategy_performance'].iloc[-1] > df['strategy_performance'].iloc[0] else "ðŸ“‰ Declining"
        
        print(f"  Model Trend: {model_trend}")
        print(f"  Strategy Trend: {strategy_trend}")


def example_threshold_adjustment():
    """Example of adjusting thresholds dynamically"""
    print("\nâš™ï¸ Dynamic Threshold Adjustment Example")
    
    controller = create_meta_controller()
    
    print("Current Model Rebuild Thresholds:")
    model_thresholds = controller.thresholds['model_rebuild']
    for key, value in model_thresholds.items():
        print(f"  {key}: {value}")
    
    # Simulate market volatility increase
    print("\nSimulating high market volatility...")
    
    # Adjust thresholds for volatile market
    original_threshold = model_thresholds['performance_threshold']
    model_thresholds['performance_threshold'] = original_threshold * 0.8  # More sensitive
    
    print(f"Adjusted performance threshold: {original_threshold:.3f} â†’ {model_thresholds['performance_threshold']:.3f}")
    
    # Test trigger evaluation with adjusted thresholds
    metrics = controller.collect_system_metrics()
    if metrics:
        decisions = controller.evaluate_triggers(metrics)
        
        print(f"Trigger evaluation with adjusted thresholds: {len(decisions)} decisions")
        
        for decision in decisions:
            if decision.action_type == ActionType.MODEL_REBUILD:
                print(f"  Model rebuild triggered: {decision.reason}")
    
    # Restore original threshold
    model_thresholds['performance_threshold'] = original_threshold
    print(f"Restored performance threshold: {model_thresholds['performance_threshold']:.3f}")


def example_error_handling():
    """Example of error handling and recovery"""
    print("\nðŸ›¡ï¸ Error Handling and Recovery Example")
    
    controller = create_meta_controller()
    
    # Simulate system errors
    print("Simulating system errors...")
    
    # Create metrics with high error count
    error_metrics = controller.SystemMetrics(
        timestamp=datetime.now().isoformat(),
        model_performance={"model1": 0.3},  # Low performance
        strategy_performance={"strategy1": 0.2},  # Low performance
        sentiment_scores={},
        system_health={"cpu_usage": 95.0, "memory_usage": 98.0},  # High usage
        market_conditions={"volatility": 0.5},  # High volatility
        error_count=25,  # High error count
        active_trades=0,
        portfolio_value=0.0
    )
    
    # Evaluate triggers with error conditions
    decisions = controller.evaluate_triggers(error_metrics)
    
    print(f"Error Conditions Detected:")
    print(f"  Error Count: {error_metrics.error_count}")
    print(f"  CPU Usage: {error_metrics.system_health['cpu_usage']}%")
    print(f"  Memory Usage: {error_metrics.system_health['memory_usage']}%")
    print(f"  Market Volatility: {error_metrics.market_conditions['volatility']}")
    
    print(f"\nTriggered Actions: {len(decisions)}")
    
    for decision in decisions:
        print(f"  {decision.action_type.value} (Priority: {decision.priority})")
        print(f"    Reason: {decision.reason}")
        
        # Execute high-priority actions
        if decision.priority >= 4:
            print(f"    Executing high-priority action...")
            result = asyncio.run(controller.execute_action(decision))
            print(f"    Result: {'âœ… Success' if result.success else 'âŒ Failed'}")
            
            if result.errors:
                print(f"    Errors: {', '.join(result.errors)}")
            
            if result.recommendations:
                print(f"    Recovery Recommendations: {', '.join(result.recommendations)}")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Run specific examples
    example_manual_trigger()
    example_performance_monitoring()
    example_threshold_adjustment()
    example_error_handling()
    
    print("\nðŸŽ‰ All MetaControllerAgent examples completed!")
