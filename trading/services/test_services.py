#!/usr/bin/env python3
"""
Test Script for Trading Agent Services

Tests the service architecture and communication.
"""

import sys
import os
import time
import json
import logging
from pathlib import Path

# Add the trading directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from services.service_client import ServiceClient
from services.service_manager import ServiceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_redis_connection():
    """Test Redis connection."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379)
        client.ping()
        logger.info("✅ Redis connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False


def test_service_manager():
    """Test ServiceManager functionality."""
    try:
        manager = ServiceManager()
        
        # Test getting service status
        status = manager.get_service_status()
        logger.info(f"✅ ServiceManager initialized, found {len(status)} services")
        
        # Test manager stats
        stats = manager.get_manager_stats()
        logger.info(f"✅ ServiceManager stats: {stats['running_services']} running, {stats['stopped_services']} stopped")
        
        manager.shutdown()
        return True
    except Exception as e:
        logger.error(f"❌ ServiceManager test failed: {e}")
        return False


def test_service_client():
    """Test ServiceClient functionality."""
    try:
        client = ServiceClient()
        
        # Test ping service (should fail if no services running)
        result = client.ping_service('model_builder')
        if result:
            logger.info("✅ ServiceClient ping successful")
        else:
            logger.info("⚠️  ServiceClient ping failed (expected if no services running)")
        
        client.close()
        return True
    except Exception as e:
        logger.error(f"❌ ServiceClient test failed: {e}")
        return False


def test_service_communication():
    """Test service communication (requires services to be running)."""
    try:
        client = ServiceClient()
        
        # Test building a model
        logger.info("Testing model building...")
        result = client.build_model('lstm', 'BTCUSDT', '1h')
        if result:
            logger.info(f"✅ Model building test successful: {result.get('type')}")
        else:
            logger.info("⚠️  Model building test failed (expected if service not running)")
        
        # Test GitHub search
        logger.info("Testing GitHub search...")
        result = client.search_github('trading bot', max_results=3)
        if result:
            logger.info(f"✅ GitHub search test successful: {result.get('type')}")
        else:
            logger.info("⚠️  GitHub search test failed (expected if service not running)")
        
        # Test prompt routing
        logger.info("Testing prompt routing...")
        result = client.route_prompt('Build me an LSTM model for Bitcoin prediction')
        if result:
            logger.info(f"✅ Prompt routing test successful: {result.get('type')}")
        else:
            logger.info("⚠️  Prompt routing test failed (expected if service not running)")
        
        client.close()
        return True
    except Exception as e:
        logger.error(f"❌ Service communication test failed: {e}")
        return False


def test_individual_services():
    """Test individual service functionality."""
    services_to_test = [
        'model_builder',
        'performance_critic', 
        'updater',
        'research',
        'meta_tuner',
        'multimodal',
        'prompt_router'
    ]
    
    client = ServiceClient()
    
    for service_name in services_to_test:
        try:
            logger.info(f"Testing {service_name} service...")
            
            # Test service status
            result = client.get_service_status(service_name)
            if result:
                logger.info(f"✅ {service_name} status check successful")
            else:
                logger.info(f"⚠️  {service_name} status check failed (expected if not running)")
                
        except Exception as e:
            logger.error(f"❌ {service_name} test failed: {e}")
    
    client.close()


def run_full_test():
    """Run all tests."""
    logger.info("🚀 Starting Trading Agent Services Test Suite")
    logger.info("=" * 50)
    
    # Test Redis connection
    if not test_redis_connection():
        logger.error("❌ Redis connection failed. Please start Redis server.")
        return False
    
    # Test ServiceManager
    if not test_service_manager():
        logger.error("❌ ServiceManager test failed.")
        return False
    
    # Test ServiceClient
    if not test_service_client():
        logger.error("❌ ServiceClient test failed.")
        return False
    
    # Test individual services
    test_individual_services()
    
    # Test service communication (if services are running)
    test_service_communication()
    
    logger.info("=" * 50)
    logger.info("✅ Test suite completed!")
    logger.info("")
    logger.info("To run services:")
    logger.info("1. Start Redis: redis-server")
    logger.info("2. Start services: python service_manager.py --action start-all")
    logger.info("3. Run tests again to verify communication")
    
    return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Trading Agent Services')
    parser.add_argument('--test', choices=['redis', 'manager', 'client', 'communication', 'all'], 
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    if args.test == 'redis':
        test_redis_connection()
    elif args.test == 'manager':
        test_service_manager()
    elif args.test == 'client':
        test_service_client()
    elif args.test == 'communication':
        test_service_communication()
    elif args.test == 'all':
        run_full_test()


if __name__ == "__main__":
    main() 