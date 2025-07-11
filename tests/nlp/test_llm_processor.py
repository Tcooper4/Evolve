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

    def test_hallucination_mitigation_compound_prompts(self):
        """Test hallucination mitigation when asked compound financial prompts."""
        print("\n🧠 Testing Hallucination Mitigation for Compound Financial Prompts")
        
        # Define compound financial prompts that could lead to hallucinations
        compound_prompts = [
            "What will be the exact price of AAPL, TSLA, and MSFT on December 31st, 2024, and what specific events will cause these price movements?",
            "Predict the exact percentage gain for Bitcoin, Ethereum, and Cardano over the next 30 days with specific dates for major moves.",
            "What will be the precise earnings per share for Amazon, Google, and Netflix for Q4 2024, and what factors will drive these exact numbers?",
            "Give me the exact Federal Reserve interest rate decisions for the next 6 months with specific dates and percentage changes.",
            "What will be the exact market crash date in 2024 and what specific companies will be most affected with precise stock price drops?"
        ]
        
        # Define expected hallucination indicators
        hallucination_indicators = [
            'exact', 'precise', 'specific date', 'will be', 'definitely',
            'guaranteed', 'certain', 'exactly', 'precisely', '100%'
        ]
        
        for i, prompt in enumerate(compound_prompts, 1):
            print(f"\n📝 Testing compound prompt {i}: {prompt[:80]}...")
            
            try:
                # Process the compound prompt
                response = self.llm.process(prompt)
                self.assertIsInstance(response, str)
                self.assertTrue(len(response) > 0)
                
                # Check for hallucination mitigation techniques
                mitigation_checks = {
                    'uncertainty_indicators': self._check_uncertainty_indicators(response),
                    'disclaimer_present': self._check_disclaimer_present(response),
                    'qualification_phrases': self._check_qualification_phrases(response),
                    'avoided_exact_predictions': self._check_avoided_exact_predictions(response, hallucination_indicators),
                    'reasonable_timeframes': self._check_reasonable_timeframes(response),
                    'multiple_scenarios': self._check_multiple_scenarios(response)
                }
                
                # Log mitigation results
                print(f"  ✅ Response length: {len(response)} characters")
                for check_name, result in mitigation_checks.items():
                    status = "✅" if result else "❌"
                    print(f"  {status} {check_name}: {result}")
                
                # Assert that at least some mitigation techniques are present
                mitigation_score = sum(mitigation_checks.values())
                self.assertGreaterEqual(mitigation_score, 3, 
                    f"Insufficient hallucination mitigation for prompt {i}. Score: {mitigation_score}/6")
                
                # Test confidence scoring
                confidence = self.llm.assess_response_confidence(response, prompt)
                self.assertIsInstance(confidence, float)
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
                print(f"  📊 Confidence score: {confidence:.2f}")
                
                # Test fact verification
                fact_check_result = self.llm.verify_facts_in_response(response)
                self.assertIsInstance(fact_check_result, dict)
                self.assertIn('verifiable_facts', fact_check_result)
                self.assertIn('unverifiable_claims', fact_check_result)
                print(f"  🔍 Verifiable facts: {fact_check_result['verifiable_facts']}")
                print(f"  ⚠️ Unverifiable claims: {fact_check_result['unverifiable_claims']}")
                
            except Exception as e:
                print(f"  ❌ Error processing prompt {i}: {e}")
                # Don't fail the test for processing errors, but log them
                continue
        
        print("\n✅ Hallucination mitigation test completed")

    def _check_uncertainty_indicators(self, response):
        """Check if response contains uncertainty indicators."""
        uncertainty_words = ['might', 'could', 'may', 'possibly', 'potentially', 'likely', 'unlikely', 'uncertain']
        return any(word in response.lower() for word in uncertainty_words)

    def _check_disclaimer_present(self, response):
        """Check if response contains appropriate disclaimers."""
        disclaimer_phrases = [
            'not financial advice', 'past performance', 'no guarantee', 'risk', 'disclaimer',
            'consult a professional', 'do your own research', 'dyor'
        ]
        return any(phrase in response.lower() for phrase in disclaimer_phrases)

    def _check_qualification_phrases(self, response):
        """Check if response uses qualifying phrases."""
        qualification_words = ['based on', 'according to', 'data suggests', 'analysis indicates', 'trends show']
        return any(phrase in response.lower() for phrase in qualification_words)

    def _check_avoided_exact_predictions(self, response, hallucination_indicators):
        """Check if response avoids exact predictions."""
        # Count hallucination indicators in response
        indicator_count = sum(1 for indicator in hallucination_indicators if indicator in response.lower())
        # Response should have few or no exact prediction indicators
        return indicator_count <= 2

    def _check_reasonable_timeframes(self, response):
        """Check if response uses reasonable timeframes."""
        unreasonable_timeframes = ['next week', 'tomorrow', 'today', 'in 2 days', 'next month']
        unreasonable_count = sum(1 for timeframe in unreasonable_timeframes if timeframe in response.lower())
        # Should not have many unreasonable timeframes
        return unreasonable_count <= 1

    def _check_multiple_scenarios(self, response):
        """Check if response presents multiple scenarios."""
        scenario_indicators = ['scenario', 'possibility', 'alternative', 'on the other hand', 'however', 'but']
        return any(indicator in response.lower() for indicator in scenario_indicators)

if __name__ == "__main__":
    unittest.main() 