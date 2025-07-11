"""
Terminal-based NLP sandbox for PromptProcessor and LLMProcessor.
Usage: python sandbox_nlp.py
Enhanced with proper error handling and experimental logic isolation.
"""
import os
import sys
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MODEL = os.getenv("NLP_SANDBOX_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("NLP_SANDBOX_TEMP", 0.7))
MODERATION = os.getenv("NLP_SANDBOX_MODERATION", "1") == "1"
STREAMING = os.getenv("NLP_SANDBOX_STREAMING", "0") == "1"
MAX_RETRIES = int(os.getenv("NLP_SANDBOX_MAX_RETRIES", "3"))

class NLPSandbox:
    """NLP Sandbox for testing and experimentation."""
    
    def __init__(self):
        """Initialize the NLP sandbox with proper error handling."""
        self.processor = None
        self.llm = None
        self.initialized = False
        
        try:
            self._initialize_components()
        except Exception as e:
            logger.error(f"Failed to initialize NLP sandbox: {e}")
            self.initialized = False
    
    def _initialize_components(self):
        """Initialize NLP components with error handling."""
        try:
            # Import components with error handling
            from trading.nlp.prompt_processor import PromptProcessor
            from trading.nlp.llm_processor import LLMProcessor
            
            logger.info("Initializing PromptProcessor...")
            self.processor = PromptProcessor()
            
            logger.info("Initializing LLMProcessor...")
            llm_config = {
                "model": MODEL,
                "temperature": TEMPERATURE,
                "moderation": MODERATION,
                "max_tokens": 256
            }
            self.llm = LLMProcessor(llm_config)
            
            # Test model loading
            self._test_model_loading()
            
            self.initialized = True
            logger.info("NLP sandbox initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _test_model_loading(self):
        """Test if the model can be loaded successfully."""
        try:
            logger.info(f"Testing model loading for {MODEL}...")
            
            # Simple test prompt
            test_prompt = "Hello, this is a test."
            
            # Test prompt processing
            entities = self.processor.extract_entities(test_prompt)
            intent = self.processor.classify_intent(test_prompt)
            
            logger.info("Prompt processing test passed")
            
            # Test LLM processing (with timeout)
            try:
                response = self.llm.process(test_prompt)
                logger.info("LLM processing test passed")
            except Exception as e:
                logger.warning(f"LLM processing test failed: {e}")
                # Continue anyway, as this might be due to API issues
                
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            raise
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a single prompt with comprehensive error handling."""
        if not self.initialized:
            return {"error": "NLP sandbox not initialized"}
        
        try:
            result = {
                "prompt": prompt,
                "entities": {},
                "intent": None,
                "routed": {},
                "llm_response": None,
                "success": True,
                "errors": []
            }
            
            # Extract entities
            try:
                result["entities"] = self.processor.extract_entities(prompt)
            except Exception as e:
                error_msg = f"Entity extraction failed: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["success"] = False
            
            # Classify intent
            try:
                result["intent"] = self.processor.classify_intent(prompt)
            except Exception as e:
                error_msg = f"Intent classification failed: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["success"] = False
            
            # Route to agent
            try:
                result["routed"] = self.processor.route_to_agent(
                    result["entities"], 
                    result["intent"]
                )
            except Exception as e:
                error_msg = f"Agent routing failed: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["success"] = False
            
            # LLM response
            try:
                if STREAMING:
                    # Collect streaming response
                    response_chunks = []
                    for chunk in self.llm.process_stream(prompt):
                        response_chunks.append(chunk)
                    result["llm_response"] = "".join(response_chunks)
                else:
                    result["llm_response"] = self.llm.process(prompt)
            except Exception as e:
                error_msg = f"LLM processing failed: {e}"
                logger.error(error_msg)
                result["errors"].append(error_msg)
                result["success"] = False
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in prompt processing: {e}")
            return {
                "prompt": prompt,
                "error": f"Unexpected error: {e}",
                "success": False
            }
    
    def interactive_mode(self):
        """Run interactive mode with proper error handling."""
        if not self.initialized:
            logger.error("Cannot start interactive mode: sandbox not initialized")
            return
        
        logger.info("\n=== NLP Sandbox (Interactive Mode) ===")
        logger.info("Type your prompt and press Enter. Type 'exit' to quit.")
        logger.info(f"Model: {MODEL}, Temperature: {TEMPERATURE}")
        logger.info("---\n")
        
        while True:
            try:
                prompt = input("Prompt: ").strip()
                
                if prompt.lower() in ("exit", "quit", "q"):
                    logger.info("Exiting sandbox...")
                    break
                
                if not prompt:
                    continue
                
                # Process prompt
                result = self.process_prompt(prompt)
                
                # Display results
                self._display_results(result)
                
            except KeyboardInterrupt:
                logger.info("\nExiting sandbox...")
                break
            except EOFError:
                logger.info("\nExiting sandbox...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in interactive mode: {e}")
    
    def _display_results(self, result: Dict[str, Any]):
        """Display processing results in a formatted way."""
        try:
            if not result.get("success", True):
                logger.error("Processing failed:")
                for error in result.get("errors", []):
                    logger.error(f"  - {error}")
                return
            
            # Display entities
            if result.get("entities"):
                logger.info("\n[Entities]")
                logger.info(json.dumps(result["entities"], indent=2))
            
            # Display intent
            if result.get("intent"):
                logger.info(f"\n[Intent] {result['intent']}")
            
            # Display routing
            if result.get("routed"):
                logger.info("\n[Routed]")
                logger.info(json.dumps(result["routed"], indent=2))
            
            # Display LLM response
            if result.get("llm_response"):
                logger.info("\n[LLM Response]")
                logger.info(result["llm_response"])
            
            logger.info("\n---\n")
            
        except Exception as e:
            logger.error(f"Error displaying results: {e}")
    
    def batch_process(self, prompts: list) -> list:
        """Process multiple prompts in batch mode."""
        if not self.initialized:
            return [{"error": "NLP sandbox not initialized"} for _ in prompts]
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            result = self.process_prompt(prompt)
            results.append(result)
        
        return results
    
    def save_results(self, results: list, filename: str):
        """Save processing results to a file."""
        try:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

def main():
    """Main entry point for the NLP sandbox."""
    try:
        # Initialize sandbox
        sandbox = NLPSandbox()
        
        if not sandbox.initialized:
            logger.error("Failed to initialize NLP sandbox. Exiting.")
            sys.exit(1)
        
        # Run interactive mode
        sandbox.interactive_mode()
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)

# Experimental and testing functions - only run when script is executed directly
if __name__ == "__main__":
    # Configuration validation
    logger.info("Validating configuration...")
    
    # Check environment variables
    logger.info(f"Model: {MODEL}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Moderation: {MODERATION}")
    logger.info(f"Streaming: {STREAMING}")
    
    # Check API keys (if required)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
    
    # Run main function
    main()
else:
    # When imported as a module, provide access to the sandbox class
    logger.debug("NLP sandbox module imported") 