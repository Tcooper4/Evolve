"""
Terminal-based NLP sandbox for PromptProcessor and LLMProcessor.
Usage: python sandbox_nlp.py
"""
import os
import sys
import json
from trading.nlp.prompt_processor import PromptProcessor
from trading.nlp.llm_processor import LLMProcessor
import logging

logger = logging.getLogger(__name__)

MODEL = os.getenv("NLP_SANDBOX_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("NLP_SANDBOX_TEMP", 0.7))
MODERATION = os.getenv("NLP_SANDBOX_MODERATION", "1") == "1"
STREAMING = os.getenv("NLP_SANDBOX_STREAMING", "0") == "1"

def main():
    """NLP Sandbox main loop."""
    logger.info("\n=== NLP Sandbox ===")
    logger.info("Type your prompt and press Enter. Type 'exit' to quit.\n")

    processor = PromptProcessor()
    llm = LLMProcessor({
        "model": MODEL,
        "temperature": TEMPERATURE,
        "moderation": MODERATION,
        "max_tokens": 256
    })

    while True:
        prompt = input("Prompt: ").strip()
        if prompt.lower() in ("exit", "quit"): break
        # Process prompt
        logger.info("\n[Entities]")
        entities = processor.extract_entities(prompt)
        logger.info(json.dumps(entities, indent=2))
        intent = processor.classify_intent(prompt)
        logger.info(f"[Intent] {intent}")
        routed = processor.route_to_agent(entities, intent)
        logger.info(f"[Routed] {json.dumps(routed, indent=2)}")
        # LLM response
        logger.info("\n[LLM Response]")
        try:
            if STREAMING:
                logger.info("(streaming)", end=" ", flush=True)
                for chunk in llm.process_stream(prompt):
                    logger.info(chunk, end="", flush=True)
                logger.info()
            else:
                response = llm.process(prompt)
                logger.info(response)
        except Exception as e:
            logger.error(f"Error: {e}")
        logger.info("\n---\n")

if __name__ == "__main__":
    main() 