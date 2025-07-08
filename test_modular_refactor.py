#!/usr/bin/env python3
"""
Test Script for Modular Refactoring

This script tests the new modular architecture components including:
- Dependency injection container
- Event system
- Plugin system
- Interface definitions
- Logging system
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("🔧 Testing Core Module Imports...")
    
    try:
        from core import (
            # Interfaces
            IDataProvider, IModel, IStrategy, IAgent,
            DataRequest, DataResponse, ModelConfig, EventType,
            
            # Dependency injection
            ServiceContainer, get_container, register_service, resolve_service,
            
            # Event system
            EventBus, get_event_bus, publish_data_loaded,
            
            # Plugin system
            PluginManager, get_plugin_manager,
            
            # Logging
            SystemLogger, get_logger, log_info,
            
            # Core functions
            initialize_core_system, get_system_status
        )
        print("✅ All core imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dependency_injection():
    """Test the dependency injection container."""
    print("\n🔧 Testing Dependency Injection...")
    
    try:
        from core import ServiceContainer, ILogger, IEventBus
        
        # Create container
        container = ServiceContainer()
        print("✅ Service container created")
        
        # Test service registration
        container.register_singleton(ILogger, lambda: "test_logger")
        print("✅ Service registration successful")
        
        # Test service resolution
        logger = container.resolve(ILogger)
        assert logger == "test_logger"
        print("✅ Service resolution successful")
        
        # Test global container
        from core import get_container, register_service
        global_container = get_container()
        register_service(ILogger, lambda: "global_logger")
        global_logger = global_container.resolve(ILogger)
        print("✅ Global container working")
        
        return True
    except Exception as e:
        print(f"❌ Dependency injection error: {e}")
        return False

def test_event_system():
    """Test the event system."""
    print("\n🔧 Testing Event System...")
    
    try:
        from core import EventBus, EventType, SystemEvent, get_event_bus
        
        # Create event bus
        event_bus = EventBus()
        print("✅ Event bus created")
        
        # Test event subscription
        events_received = []
        
        def test_handler(event):
            events_received.append(event)
            print(f"✅ Event received: {event.event_type.value}")
        
        event_bus.subscribe(EventType.DATA_LOADED, test_handler)
        print("✅ Event subscription successful")
        
        # Test event publishing
        test_event = SystemEvent(
            event_type=EventType.DATA_LOADED,
            timestamp=datetime.now(),
            data={'symbol': 'AAPL', 'data': 'test_data'},
            source='test'
        )
        
        event_bus.publish(test_event)
        time.sleep(0.1)  # Allow async processing
        
        assert len(events_received) == 1
        print("✅ Event publishing successful")
        
        # Test global event bus
        global_bus = get_event_bus()
        from core import publish_data_loaded
        publish_data_loaded('TSLA', 'test_data', 'test_script')
        print("✅ Global event bus working")
        
        return True
    except Exception as e:
        print(f"❌ Event system error: {e}")
        return False

def test_plugin_system():
    """Test the plugin system."""
    print("\n🔧 Testing Plugin System...")
    
    try:
        from core import PluginManager, PluginStatus, get_plugin_manager
        
        # Create plugin manager
        plugin_manager = PluginManager(plugin_dirs=["./test_plugins"])
        print("✅ Plugin manager created")
        
        # Test plugin discovery
        plugins = plugin_manager.list_plugins()
        print(f"✅ Discovered {len(plugins)} plugins")
        
        # Test plugin info
        if plugins:
            plugin_info = plugins[0]
            print(f"✅ Plugin info: {plugin_info.name} v{plugin_info.version}")
        
        # Test global plugin manager
        global_manager = get_plugin_manager()
        stats = global_manager.get_plugin_stats()
        print(f"✅ Plugin stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Plugin system error: {e}")
        return False

def test_logging_system():
    """Test the logging system."""
    print("\n🔧 Testing Logging System...")
    
    try:
        from core import SystemLogger, get_logger, log_info, log_warning
        
        # Create logger
        logger = SystemLogger("test_logger")
        print("✅ Logger created")
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        print("✅ Logging methods working")
        
        # Test global logger
        global_logger = get_logger()
        log_info("Global info message")
        log_warning("Global warning message")
        print("✅ Global logging working")
        
        # Test stats
        stats = logger.get_stats()
        print(f"✅ Logger stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ Logging system error: {e}")
        return False

def test_core_system_initialization():
    """Test the core system initialization."""
    print("\n🔧 Testing Core System Initialization...")
    
    try:
        from core import initialize_core_system, get_system_status, shutdown_core_system
        
        # Initialize core system
        components = initialize_core_system()
        print("✅ Core system initialized")
        
        # Check components
        assert 'container' in components
        assert 'event_bus' in components
        assert 'plugin_manager' in components
        assert 'logger' in components
        print("✅ All components present")
        
        # Test system status
        status = get_system_status()
        assert status['status'] == 'healthy'
        print("✅ System status healthy")
        
        # Test shutdown
        shutdown_core_system()
        print("✅ Core system shutdown successful")
        
        return True
    except Exception as e:
        print(f"❌ Core system initialization error: {e}")
        return False

def test_interface_compliance():
    """Test that interfaces are properly defined."""
    print("\n🔧 Testing Interface Compliance...")
    
    try:
        from core import (
            IDataProvider, IModel, IStrategy, IAgent,
            DataRequest, DataResponse, ModelConfig
        )
        
        # Test that interfaces are abstract
        from abc import ABC
        assert issubclass(IDataProvider, ABC)
        assert issubclass(IModel, ABC)
        assert issubclass(IStrategy, ABC)
        assert issubclass(IAgent, ABC)
        print("✅ Interfaces are abstract")
        
        # Test data types
        request = DataRequest(symbol="AAPL", period="1d")
        assert request.symbol == "AAPL"
        assert request.period == "1d"
        print("✅ Data types working")
        
        return True
    except Exception as e:
        print(f"❌ Interface compliance error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Modular Refactoring Tests")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Dependency Injection", test_dependency_injection),
        ("Event System", test_event_system),
        ("Plugin System", test_plugin_system),
        ("Logging System", test_logging_system),
        ("Core System Initialization", test_core_system_initialization),
        ("Interface Compliance", test_interface_compliance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Modular refactoring is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 