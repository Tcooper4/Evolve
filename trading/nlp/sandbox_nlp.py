"""
Terminal-based NLP sandbox for PromptProcessor and LLMProcessor.
Usage: python sandbox_nlp.py
"""
import os
import sys
import json
from trading.nlp.prompt_processor import PromptProcessor
from trading.nlp.llm_processor import LLMProcessor

MODEL = os.getenv("NLP_SANDBOX_MODEL", "gpt-3.5-turbo")
TEMPERATURE = float(os.getenv("NLP_SANDBOX_TEMP", 0.7))
MODERATION = os.getenv("NLP_SANDBOX_MODERATION", "1") == "1"
STREAMING = os.getenv("NLP_SANDBOX_STREAMING", "0") == "1"

print("\n=== NLP Sandbox ===")
print("Type your prompt and press Enter. Type 'exit' to quit.\n")

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
    print("\n[Entities]")
    entities = processor.extract_entities(prompt)
    print(json.dumps(entities, indent=2))
    intent = processor.classify_intent(prompt)
    print(f"[Intent] {intent}")
    routed = processor.route_to_agent(entities, intent)
    print(f"[Routed] {json.dumps(routed, indent=2)}")
    # LLM response
    print("\n[LLM Response]")
    try:
        if STREAMING:
            print("(streaming)", end=" ", flush=True)
            for chunk in llm.process_stream(prompt):
                print(chunk, end="", flush=True)
            print()
        else:
            response = llm.process(prompt)
            print(response)
    except Exception as e:
        print(f"Error: {e}")
    print("\n---\n") 