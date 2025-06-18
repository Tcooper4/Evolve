import os
import json
import unittest
from trading.nlp.prompt_processor import PromptProcessor, MEMORY_LOG_PATH

SAMPLE_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), '../../trading/nlp/sample_prompts.json')

class TestPromptProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PromptProcessor()
        # Clean up memory log before test
        if os.path.exists(MEMORY_LOG_PATH):
            os.remove(MEMORY_LOG_PATH)
        with open(SAMPLE_PROMPTS_PATH, 'r', encoding='utf-8') as f:
            self.prompts = json.load(f)

    def test_entity_extraction(self):
        for prompt in self.prompts:
            entities = self.processor.extract_entities(prompt)
            self.assertIsInstance(entities, dict)
            # At least one entity should be found for most prompts
            self.assertTrue(len(entities) > 0)

    def test_intent_classification(self):
        for prompt in self.prompts:
            intent = self.processor.classify_intent(prompt)
            self.assertIsInstance(intent, str)
            self.assertIn(intent, ["forecast", "backtest", "compare", "interpret", "summarize", "analyze", "unknown"])

    def test_routing_and_memory_log(self):
        for prompt in self.prompts:
            result = self.processor.process_and_route(prompt)
            self.assertIn("entities", result)
            self.assertIn("routed", result)
        # Check memory log file
        self.assertTrue(os.path.exists(MEMORY_LOG_PATH))
        with open(MEMORY_LOG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(self.prompts))
        for line in lines:
            entry = json.loads(line)
            self.assertIn("prompt", entry)
            self.assertIn("entities", entry)
            self.assertIn("routed", entry)

if __name__ == "__main__":
    unittest.main() 