#!/usr/bin/env python3
"""
Production Launch Script for Evolve Trading Platform

This script orchestrates the complete production launch:
- Validates all components
- Starts system monitoring
- Initializes all agents
- Launches the enhanced UI
- Performs health checks
- Provides production status
"""

import sys
import os
import time
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_environment():
    """Validate environment configuration."""
    print("🔍 Validating Environment Configuration...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Creating from template...")
        if os.path.exists('env.example'):
            import shutil
            shutil.copy('env.example', '.env')
            print("✅ Created .env from template. Please configure your API keys.")
            return False
        else:
            print("❌ env.example not found. Cannot create .env file.")
            return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment")
    
    # Check required environment variables
    required_vars = [
        'OPENAI_API_KEY',
        'FINNHUB_API_KEY',
        'ALPHA_VANTAGE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("Please set these variables in your .env file")
        return False
    
    print("✅ Environment validation passed")
    return True

def initialize_system_resilience():
    """Initialize system resilience and monitoring."""
    print("\n🔍 Initializing System Resilience...")
    
    try:
        from system_resilience import SystemResilience, start_system_monitoring
        
        # Initialize system resilience
        resilience = SystemResilience()
        
        # Start monitoring
        start_system_monitoring()
        print("✅ System monitoring started")
        
        # Perform initial health check
        health = resilience.get_system_health()
        print(f"✅ Initial health check: {health['overall_status']}")
        
        return resilience
        
    except Exception as e:
        print(f"❌ System resilience initialization failed: {e}")
        return None

def initialize_enhanced_interface():
    """Initialize enhanced interface."""
    print("\n🔍 Initializing Enhanced Interface...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        # Initialize interface
        interface = EnhancedUnifiedInterfaceV2()
        print("✅ Enhanced interface initialized")
        
        # Test component initialization
        components = [
            'agent_hub', 'data_feed', 'prompt_router', 'model_monitor',
            'strategy_logger', 'portfolio_manager', 'strategy_selector',
            'market_regime_agent', 'hybrid_engine', 'quant_gpt',
            'reporter', 'backtester'
        ]
        
        initialized_components = 0
        for component in components:
            if hasattr(interface, component):
                initialized_components += 1
        
        print(f"✅ {initialized_components}/{len(components)} components initialized")
        
        return interface
        
    except Exception as e:
        print(f"❌ Enhanced interface initialization failed: {e}")
        return None

def perform_comprehensive_health_check(resilience):
    """Perform comprehensive health check."""
    print("\n🔍 Performing Comprehensive Health Check...")
    
    try:
        health = resilience.get_system_health()
        
        # Check overall status
        if health['overall_status'] == 'healthy':
            print("✅ Overall system health: HEALTHY")
        elif health['overall_status'] == 'warning':
            print("⚠️  Overall system health: WARNING")
        else:
            print("❌ Overall system health: ERROR")
        
        # Check individual components
        for component, status in health['components'].items():
            if status['status'] == 'healthy':
                print(f"  ✅ {component}: {status['message']}")
            elif status['status'] == 'warning':
                print(f"  ⚠️  {component}: {status['message']}")
            else:
                print(f"  ❌ {component}: {status['message']}")
        
        # Check for issues
        if health['issues']:
            print(f"\n⚠️  Issues detected: {len(health['issues'])}")
            for issue in health['issues']:
                print(f"  - {issue}")
        
        return health['overall_status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def start_streamlit_interface():
    """Start Streamlit interface."""
    print("\n🚀 Starting Streamlit Interface...")
    
    try:
        import subprocess
        import sys
        
        # Start Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "unified_interface_v2.py", 
               "--server.port=8501", "--server.address=0.0.0.0"]
        
        print("Starting Streamlit server...")
        print("Access the application at: http://localhost:8501")
        
        # Run in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Streamlit server started successfully")
            return process
        else:
            print("❌ Streamlit server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return None

def display_production_status(interface, resilience):
    """Display production status."""
    print("\n" + "=" * 70)
    print("🏭 EVOLVE TRADING PLATFORM - PRODUCTION STATUS")
    print("=" * 70)
    
    # System status
    health = resilience.get_system_health()
    print(f"System Status: {health['overall_status'].upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Component status
    print("\n📊 Component Status:")
    for component, status in health['components'].items():
        icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'warning' else "❌"
        print(f"  {icon} {component}: {status['status']}")
    
    # Performance metrics
    performance = resilience.get_performance_report()
    if 'cpu_usage_avg' in performance:
        print(f"\n📈 Performance Metrics:")
        print(f"  CPU Usage: {performance['cpu_usage_avg']:.1f}%")
        print(f"  Memory Usage: {performance.get('memory_usage_avg', 'N/A')}%")
    
    # Access information
    print(f"\n🌐 Access Information:")
    print(f"  Web Interface: http://localhost:8501")
    print(f"  Health Check: http://localhost:8501/_stcore/health")
    
    # Features available
    print(f"\n🎯 Features Available:")
    features = [
        "Multi-tab UI (Forecast, Strategy, Backtest, Report, System)",
        "Natural language prompt processing",
        "Agentic intelligence and routing",
        "Advanced forecasting with multiple models",
        "Strategy development and optimization",
        "Comprehensive backtesting",
        "Unified reporting and export",
        "Real-time system monitoring",
        "Automatic fallback mechanisms",
        "Production-grade deployment"
    ]
    
    for feature in features:
        print(f"  ✅ {feature}")
    
    print("\n" + "=" * 70)
    print("🎉 EVOLVE TRADING PLATFORM IS NOW LIVE!")
    print("=" * 70)

def run_production_launch():
    """Run complete production launch."""
    print("🚀 EVOLVE TRADING PLATFORM - PRODUCTION LAUNCH")
    print("=" * 70)
    print(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Step 1: Validate environment
    if not validate_environment():
        print("❌ Environment validation failed. Please configure your environment.")
        return False
    
    # Step 2: Initialize system resilience
    resilience = initialize_system_resilience()
    if not resilience:
        print("❌ System resilience initialization failed.")
        return False
    
    # Step 3: Initialize enhanced interface
    interface = initialize_enhanced_interface()
    if not interface:
        print("❌ Enhanced interface initialization failed.")
        return False
    
    # Step 4: Perform health check
    if not perform_comprehensive_health_check(resilience):
        print("⚠️  Health check shows issues. Continuing with warnings...")
    
    # Step 5: Start Streamlit interface
    streamlit_process = start_streamlit_interface()
    if not streamlit_process:
        print("❌ Failed to start Streamlit interface.")
        return False
    
    # Step 6: Display production status
    display_production_status(interface, resilience)
    
    # Step 7: Keep running
    print("\n🔄 System is running. Press Ctrl+C to stop.")
    
    try:
        # Monitor system while running
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            # Perform periodic health check
            health = resilience.get_system_health()
            if health['overall_status'] == 'error':
                print("❌ System health degraded to ERROR status")
                break
            
            # Check if Streamlit is still running
            if streamlit_process.poll() is not None:
                print("❌ Streamlit process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        
        # Stop monitoring
        from system_resilience import stop_system_monitoring
        stop_system_monitoring()
        
        # Terminate Streamlit
        if streamlit_process:
            streamlit_process.terminate()
            streamlit_process.wait()
        
        print("✅ System shutdown complete")
    
    return True

if __name__ == "__main__":
    success = run_production_launch()
    sys.exit(0 if success else 1) 