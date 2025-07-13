import json
import os
import unittest

from trading.nlp.prompt_processor import MEMORY_LOG_PATH, PromptProcessor

SAMPLE_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "../../trading/nlp/sample_prompts.json")


class TestPromptProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PromptProcessor()
        # Clean up memory log before test
        if os.path.exists(MEMORY_LOG_PATH):
            os.remove(MEMORY_LOG_PATH)
        with open(SAMPLE_PROMPTS_PATH, "r", encoding="utf-8") as f:
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
        with open(MEMORY_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(self.prompts))
        for line in lines:
            entry = json.loads(line)
            self.assertIn("prompt", entry)
            self.assertIn("entities", entry)
            self.assertIn("routed", entry)

    def test_long_prompt_token_truncation(self):
        """Test long prompt token truncation edge cases and verify correct slicing."""
        print("\n📝 Testing long prompt token truncation edge cases")
        # Create a very long prompt (simulate >2048 tokens)
        long_prompt = "word " * 3000  # 3000 words, likely >2048 tokens
        max_tokens = 2048

        # Process with truncation
        truncated_prompt, was_truncated = self.processor.truncate_prompt(long_prompt, max_tokens=max_tokens)
        self.assertIsInstance(truncated_prompt, str)
        self.assertTrue(len(truncated_prompt) <= len(long_prompt))
        self.assertTrue(was_truncated, "Prompt should be truncated for long input")

        # Check that truncation is at token boundary (simulate tokenization)
        tokens = truncated_prompt.split()
        self.assertLessEqual(len(tokens), max_tokens)
        print(f"✅ Truncated to {len(tokens)} tokens (max {max_tokens})")

        # Test edge case: prompt exactly at max_tokens
        exact_prompt = "word " * max_tokens
        truncated_exact, was_truncated_exact = self.processor.truncate_prompt(exact_prompt, max_tokens=max_tokens)
        self.assertFalse(was_truncated_exact, "Prompt at max_tokens should not be truncated")
        self.assertEqual(len(truncated_exact.split()), max_tokens)
        print(f"✅ No truncation for prompt at exact token limit ({max_tokens} tokens)")

        # Test edge case: prompt just below max_tokens
        below_prompt = "word " * (max_tokens - 1)
        truncated_below, was_truncated_below = self.processor.truncate_prompt(below_prompt, max_tokens=max_tokens)
        self.assertFalse(was_truncated_below, "Prompt below max_tokens should not be truncated")
        self.assertEqual(len(truncated_below.split()), max_tokens - 1)
        print(f"✅ No truncation for prompt below token limit ({max_tokens - 1} tokens)")

        # Test edge case: prompt just above max_tokens
        above_prompt = "word " * (max_tokens + 1)
        truncated_above, was_truncated_above = self.processor.truncate_prompt(above_prompt, max_tokens=max_tokens)
        self.assertTrue(was_truncated_above, "Prompt above max_tokens should be truncated")
        self.assertEqual(len(truncated_above.split()), max_tokens)
        print(f"✅ Truncated to {max_tokens} tokens for prompt above token limit")

        print("✅ Long prompt token truncation edge cases passed")


if __name__ == "__main__":
    unittest.main()
