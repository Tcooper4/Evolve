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
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_environment():
    """Validate environment configuration."""
    logger.info("🔍 Validating Environment Configuration...")
    
    # Check for .env file
    if not os.path.exists('.env'):
        logger.warning("⚠️  .env file not found. Creating from template...")
        if os.path.exists('env.example'):
            import shutil
            shutil.copy('env.example', '.env')
            logger.warning("✅ Created .env from template. Please configure your API keys.")
            return False
        else:
            logger.error("❌ env.example not found. Cannot create .env file.")
            return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("✅ Environment variables loaded")
    except ImportError:
        logger.warning("⚠️  python-dotenv not available, using system environment")
    
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
        logger.warning(f"⚠️  Missing environment variables: {missing_vars}")
        logger.warning("Please set these variables in your .env file")
        return False
    
    logger.info("✅ Environment validation passed")
    return True

def initialize_system_resilience():
    """Initialize system resilience and monitoring."""
    logger.info("🔍 Initializing System Resilience...")
    
    try:
        from system_resilience import SystemResilience, start_system_monitoring
        
        # Initialize system resilience
        resilience = SystemResilience()
        
        # Start monitoring
        start_system_monitoring()
        logger.info("✅ System monitoring started")
        
        # Perform initial health check
        health = resilience.get_system_health()
        logger.info(f"✅ Initial health check: {health['overall_status']}")
        
        return resilience
        
    except Exception as e:
        logger.error(f"❌ System resilience initialization failed: {e}")
        return None

def initialize_enhanced_interface():
    """Initialize enhanced interface."""
    logger.info("🔍 Initializing Enhanced Interface...")
    
    try:
        from unified_interface_v2 import EnhancedUnifiedInterfaceV2
        
        # Initialize interface
        interface = EnhancedUnifiedInterfaceV2()
        logger.info("✅ Enhanced interface initialized")
        
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
        
        logger.info(f"✅ {initialized_components}/{len(components)} components initialized")
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Enhanced interface initialization failed: {e}")
        return None

def perform_comprehensive_health_check(resilience):
    """Perform comprehensive health check."""
    logger.info("🔍 Performing Comprehensive Health Check...")
    
    try:
        health = resilience.get_system_health()
        
        # Check overall status
        if health['overall_status'] == 'healthy':
            logger.info("✅ Overall system health: HEALTHY")
        elif health['overall_status'] == 'warning':
            logger.warning("⚠️  Overall system health: WARNING")
        else:
            logger.error("❌ Overall system health: ERROR")
        
        # Check individual components
        for component, status in health['components'].items():
            if status['status'] == 'healthy':
                logger.info(f"  ✅ {component}: {status['message']}")
            elif status['status'] == 'warning':
                logger.warning(f"  ⚠️  {component}: {status['message']}")
            else:
                logger.error(f"  ❌ {component}: {status['message']}")
        
        # Check for issues
        if health['issues']:
            logger.warning(f"\n⚠️  Issues detected: {len(health['issues'])}")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
        
        return health['overall_status'] == 'healthy'
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

def start_streamlit_interface():
    """Start Streamlit interface."""
    logger.info("🚀 Starting Streamlit Interface...")
    
    try:
        import subprocess
        import sys
        
        # Start Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "unified_interface_v2.py", 
               "--server.port=8501", "--server.address=0.0.0.0"]
        
        logger.info("Starting Streamlit server...")
        logger.info("Access the application at: http://localhost:8501")
        
        # Run in background
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ Streamlit server started successfully")
            return process
        else:
            logger.error("❌ Streamlit server failed to start")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        return None

def display_production_status(interface, resilience):
    """Display production status."""
    logger.info("=" * 70)
    logger.info("🏭 EVOLVE TRADING PLATFORM - PRODUCTION STATUS")
    logger.info("=" * 70)
    
    # System status
    health = resilience.get_system_health()
    logger.info(f"System Status: {health['overall_status'].upper()}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Component status
    logger.info("\n📊 Component Status:")
    for component, status in health['components'].items():
        icon = "✅" if status['status'] == 'healthy' else "⚠️" if status['status'] == 'warning' else "❌"
        logger.info(f"  {icon} {component}: {status['status']}")
    
    # Performance metrics
    performance = resilience.get_performance_report()
    if 'cpu_usage_avg' in performance:
        logger.info(f"\n📈 Performance Metrics:")
        logger.info(f"  CPU Usage: {performance['cpu_usage_avg']:.1f}%")
        logger.info(f"  Memory Usage: {performance.get('memory_usage_avg', 'N/A')}%")
    
    # Access information
    logger.info(f"\n🌐 Access Information:")
    logger.info(f"  Web Interface: http://localhost:8501")
    logger.info(f"  Health Check: http://localhost:8501/_stcore/health")
    
    # Features available
    logger.info(f"\n🎯 Features Available:")
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
        logger.info(f"  ✅ {feature}")
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 EVOLVE TRADING PLATFORM IS NOW LIVE!")
    logger.info("=" * 70)

def run_production_launch():
    """Run complete production launch."""
    logger.info("🚀 EVOLVE TRADING PLATFORM - PRODUCTION LAUNCH")
    logger.info("=" * 70)
    logger.info(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # Step 1: Validate environment
    if not validate_environment():
        logger.error("❌ Environment validation failed. Please configure your environment.")
        return False
    
    # Step 2: Initialize system resilience
    resilience = initialize_system_resilience()
    if not resilience:
        logger.error("❌ System resilience initialization failed.")
        return False
    
    # Step 3: Initialize enhanced interface
    interface = initialize_enhanced_interface()
    if not interface:
        logger.error("❌ Enhanced interface initialization failed.")
        return False
    
    # Step 4: Perform health check
    if not perform_comprehensive_health_check(resilience):
        logger.warning("⚠️  Health check shows issues. Continuing with warnings...")
    
    # Step 5: Start Streamlit interface
    streamlit_process = start_streamlit_interface()
    if not streamlit_process:
        logger.error("❌ Failed to start Streamlit interface.")
        return False
    
    # Step 6: Display production status
    display_production_status(interface, resilience)
    
    # Step 7: Keep running
    logger.info("\n🔄 System is running. Press Ctrl+C to stop.")
    
    try:
        # Monitor system while running
        while True:
            time.sleep(30)  # Check every 30 seconds
            
            # Perform periodic health check
            health = resilience.get_system_health()
            if health['overall_status'] == 'error':
                logger.error("❌ System health degraded to ERROR status")
                break
            
            # Check if Streamlit is still running
            if streamlit_process.poll() is not None:
                logger.error("❌ Streamlit process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down system...")
        
        # Stop monitoring
        from system_resilience import stop_system_monitoring
        stop_system_monitoring()
        
        # Terminate Streamlit
        if streamlit_process:
            streamlit_process.terminate()
            streamlit_process.wait()
        
        logger.info("✅ System shutdown complete")
    
    return True

if __name__ == "__main__":
    success = run_production_launch()
    sys.exit(0 if success else 1) 