import os
import unittest
from trading.nlp.llm_processor import LLMProcessor

class TestLLMProcessor(unittest.TestCase):
    def setUp(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            self.skipTest("OPENAI_API_KEY not set; skipping LLMProcessor tests.")
        self.llm = LLMProcessor({
            "model": "gpt-3.5-turbo",
            "temperature": 0.1,
            "moderation": True,
            "max_tokens": 32
        })
        self.prompt = "Summarize the outlook for Apple stock."

    def test_standard_response(self):
        response = self.llm.process(self.prompt)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_streaming_response(self):
        chunks = list(self.llm.process_stream(self.prompt))
        self.assertTrue(any(isinstance(chunk, str) and chunk for chunk in chunks))

    def test_json_validation(self):
        valid_json = '{"response": "ok", "confidence": 0.95}'
        data = self.llm.validate_json_response(valid_json)
        self.assertEqual(data["response"], "ok")
        self.assertAlmostEqual(data["confidence"], 0.95)
        with self.assertRaises(ValueError):
            self.llm.validate_json_response('{"response": "ok"}')  # missing confidence
        with self.assertRaises(ValueError):
            self.llm.validate_json_response('not a json')

if __name__ == "__main__":
    unittest.main() 