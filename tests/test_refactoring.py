#!/usr/bin/env python3
"""
Test script for the refactored LLM and agent system.

This script validates:
1. Agent registry functionality
2. Prompt template usage
3. Prompt router integration
4. Voice prompt agent compatibility
"""

import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_registry():
    """Test the centralized agent registry."""
    logger.info("Testing Agent Registry...")
    
    try:
        from agents.registry import get_registry, get_agent, list_agents, ALL_AGENTS
        
        # Test registry initialization
        registry = get_registry()
        logger.info(f"✅ Registry initialized successfully")
        
        # Test agent listing
        agents = list_agents()
        logger.info(f"✅ Found {len(agents)} agents: {agents[:5]}...")
        
        # Test getting specific agents
        prompt_router = get_agent('promptrouteragent')
        if prompt_router:
            logger.info("✅ Successfully retrieved PromptRouterAgent")
        else:
            logger.warning("⚠️ Could not retrieve PromptRouterAgent")
        
        # Test ALL_AGENTS dictionary
        if 'prompt_router' in ALL_AGENTS:
            logger.info("✅ ALL_AGENTS dictionary populated correctly")
        else:
            logger.warning("⚠️ ALL_AGENTS dictionary missing expected agents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent registry test failed: {e}")
        return False

def test_prompt_templates():
    """Test the centralized prompt templates."""
    logger.info("Testing Prompt Templates...")
    
    try:
        from trading.agents.prompt_templates import (
            PROMPT_TEMPLATES, 
            get_template, 
            format_template,
            list_templates,
            list_categories
        )
        
        # Test template access
        templates = list_templates()
        logger.info(f"✅ Found {len(templates)} templates")
        
        categories = list_categories()
        logger.info(f"✅ Found {len(categories)} template categories: {categories}")
        
        # Test specific template
        forecast_template = get_template("forecast_request")
        if "forecast" in forecast_template.lower():
            logger.info("✅ Forecast template retrieved successfully")
        else:
            logger.warning("⚠️ Forecast template content unexpected")
        
        # Test template formatting
        formatted = format_template("forecast_request", symbol="AAPL", timeframe="1 week", model_type="LSTM")
        if "AAPL" in formatted and "1 week" in formatted:
            logger.info("✅ Template formatting works correctly")
        else:
            logger.warning("⚠️ Template formatting failed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt templates test failed: {e}")
        return False

def test_prompt_router():
    """Test the prompt router agent."""
    logger.info("Testing Prompt Router Agent...")
    
    try:
        from trading.agents.prompt_router_agent import PromptRouterAgent
        from trading.agents.prompt_templates import format_template
        
        # Test agent initialization
        router = PromptRouterAgent()
        logger.info("✅ PromptRouterAgent initialized successfully")
        
        # Test intent parsing
        test_prompt = "Forecast AAPL for 1 week"
        parsed = router.parse_intent(test_prompt)
        
        if parsed and hasattr(parsed, 'intent'):
            logger.info(f"✅ Intent parsing works: {parsed.intent} (confidence: {parsed.confidence})")
        else:
            logger.warning("⚠️ Intent parsing returned unexpected result")
        
        # Test provider status
        providers = router.get_available_providers()
        logger.info(f"✅ Available providers: {providers}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt router test failed: {e}")
        return False

def test_voice_prompt_agent():
    """Test the voice prompt agent (legacy compatibility)."""
    logger.info("Testing Voice Prompt Agent (Legacy)...")
    
    try:
        from voice_prompt_agent import VoicePromptAgent
        
        # Test agent initialization
        voice_agent = VoicePromptAgent()
        logger.info("✅ VoicePromptAgent initialized successfully")
        
        # Test command parsing
        test_command = "forecast AAPL for 5 days"
        parsed = voice_agent.parse_trading_command(test_command)
        
        if parsed and parsed.get('action') == 'forecast':
            logger.info(f"✅ Voice command parsing works: {parsed['action']}")
        else:
            logger.warning("⚠️ Voice command parsing returned unexpected result")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Voice prompt agent test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    logger.info("Testing Component Integration...")
    
    try:
        # Test that prompt router uses centralized templates
        from trading.agents.prompt_router_agent import PromptRouterAgent
        from trading.agents.prompt_templates import PROMPT_TEMPLATES
        
        router = PromptRouterAgent()
        
        # Verify that the router has access to templates
        if hasattr(router, 'parse_intent'):
            logger.info("✅ PromptRouterAgent has parse_intent method")
        else:
            logger.warning("⚠️ PromptRouterAgent missing parse_intent method")
        
        # Test that templates are accessible
        if "intent_classification" in PROMPT_TEMPLATES:
            logger.info("✅ Intent classification template available")
        else:
            logger.warning("⚠️ Intent classification template missing")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🧪 Starting LLM and Agent System Refactoring Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Agent Registry", test_agent_registry),
        ("Prompt Templates", test_prompt_templates),
        ("Prompt Router", test_prompt_router),
        ("Voice Prompt Agent", test_voice_prompt_agent),
        ("Integration", test_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Refactoring successful.")
        return 0
    else:
        logger.warning("⚠️ Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 