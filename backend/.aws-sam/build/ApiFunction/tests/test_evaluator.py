"""
Unit tests for the ResponseEvaluator class.
"""

import unittest
from unittest.mock import patch, MagicMock
from textblob import TextBlob

from backend.app.evaluator import ResponseEvaluator


class TestResponseEvaluator(unittest.TestCase):
    """Test cases for the ResponseEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ResponseEvaluator()
        
        # Sample texts for testing
        self.empty_text = ""
        self.short_text = "This is a short response."
        self.medium_text = (
            "This is a medium-length response with multiple sentences. "
            "It contains some longer words like 'comprehensive' and 'evaluation'. "
            "It also has questions? And some numbers like 42 and 7.5."
        )
        self.technical_text = (
            "AWS Lambda functions execute in containerized environments. "
            "When implementing serverless architectures, consider cold start latency. "
            "Provisioned concurrency can mitigate initialization overhead. "
            "CloudWatch provides metrics for monitoring invocation performance."
        )
        self.code_text = (
            "Here's a code example:\n```python\ndef hello_world():\n    print('Hello, world!')\n```\n"
            "And another example: <code>const x = 10;</code>"
        )
        
        # Sample model responses
        self.model_responses = {
            "model1": self.short_text,
            "model2": self.medium_text,
            "model3": self.technical_text
        }

    def test_evaluate_text_metrics(self):
        """Test basic text metrics evaluation."""
        # Test with medium text
        metrics = self.evaluator.evaluate_text_metrics(self.medium_text)
        
        # Verify all expected metrics are present
        self.assertIn("char_count", metrics)
        self.assertIn("word_count", metrics)
        self.assertIn("sentence_count", metrics)
        self.assertIn("avg_word_length", metrics)
        self.assertIn("avg_sentence_length", metrics)
        
        # Check reasonable values
        self.assertTrue(metrics["char_count"] > 0)
        self.assertTrue(metrics["word_count"] > 0)
        self.assertTrue(metrics["sentence_count"] > 0)
        self.assertTrue(0 < metrics["avg_word_length"] < 10)  # Reasonable word length
        self.assertTrue(0 < metrics["avg_sentence_length"] < 30)  # Reasonable sentence length
        
        # Test with empty text
        empty_metrics = self.evaluator.evaluate_text_metrics(self.empty_text)
        self.assertEqual(empty_metrics["char_count"], 0)
        self.assertEqual(empty_metrics["word_count"], 0)
        self.assertEqual(empty_metrics["sentence_count"], 0)

    def test_evaluate_readability(self):
        """Test readability metrics evaluation."""
        # Test with medium text
        readability = self.evaluator.evaluate_readability(self.medium_text)
        
        # Verify all expected metrics are present
        self.assertIn("flesch_reading_ease", readability)
        self.assertIn("flesch_kincaid_grade", readability)
        self.assertIn("gunning_fog", readability)
        self.assertIn("smog_index", readability)
        self.assertIn("automated_readability_index", readability)
        self.assertIn("coleman_liau_index", readability)
        self.assertIn("dale_chall_readability_score", readability)
        self.assertIn("difficult_words", readability)
        self.assertIn("linsear_write_formula", readability)
        self.assertIn("text_standard", readability)
        
        # Check reasonable values for a medium complexity text
        self.assertTrue(0 <= readability["flesch_reading_ease"] <= 100)
        self.assertTrue(0 <= readability["flesch_kincaid_grade"] <= 20)
        
        # Test with technical text (should be more complex)
        tech_readability = self.evaluator.evaluate_readability(self.technical_text)
        
        # Technical text should generally be more complex than medium text
        self.assertTrue(
            tech_readability["flesch_kincaid_grade"] >= readability["flesch_kincaid_grade"]
        )
        
        # Test with empty text
        empty_readability = self.evaluator.evaluate_readability(self.empty_text)
        self.assertEqual(empty_readability["flesch_reading_ease"], 0)

    def test_evaluate_content_metrics(self):
        """Test content metrics evaluation."""
        # Test with medium text
        content = self.evaluator.evaluate_content_metrics(self.medium_text)
        
        # Verify all expected metrics are present
        self.assertIn("sentiment", content)
        self.assertIn("subjectivity", content)
        self.assertIn("polarity", content)
        self.assertIn("question_count", content)
        self.assertIn("code_block_count", content)
        self.assertIn("list_count", content)
        self.assertIn("url_count", content)
        self.assertIn("has_numbers", content)
        self.assertIn("number_count", content)
        
        # Check sentiment structure
        self.assertIn("compound", content["sentiment"])
        self.assertIn("pos", content["sentiment"])
        self.assertIn("neu", content["sentiment"])
        self.assertIn("neg", content["sentiment"])
        
        # Check reasonable values
        self.assertTrue(-1 <= content["sentiment"]["compound"] <= 1)
        self.assertTrue(0 <= content["subjectivity"] <= 1)
        self.assertTrue(-1 <= content["polarity"] <= 1)
        
        # Specific checks for sample text content
        self.assertEqual(content["question_count"], 1)  # "It also has questions?"
        self.assertTrue(content["has_numbers"])
        self.assertTrue(content["number_count"] >= 2)  # At least "42 and 7.5"
        
        # Test with code text
        code_content = self.evaluator.evaluate_content_metrics(self.code_text)
        self.assertEqual(code_content["code_block_count"], 2)  # One markdown, one HTML
        
        # Test with empty text
        empty_content = self.evaluator.evaluate_content_metrics(self.empty_text)
        self.assertEqual(empty_content["sentiment"]["compound"], 0)
        self.assertEqual(empty_content["question_count"], 0)

    def test_evaluate_complexity(self):
        """Test complexity metrics evaluation."""
        # Test with medium text
        complexity = self.evaluator.evaluate_complexity(self.medium_text)
        
        # Verify all expected metrics are present
        self.assertIn("lexical_diversity", complexity)
        self.assertIn("long_word_ratio", complexity)
        self.assertIn("unique_word_ratio", complexity)
        self.assertIn("sentence_complexity", complexity)
        
        # Check reasonable values
        self.assertTrue(0 <= complexity["lexical_diversity"] <= 1)
        self.assertTrue(0 <= complexity["long_word_ratio"] <= 1)
        self.assertTrue(0 <= complexity["unique_word_ratio"] <= 1)
        self.assertTrue(complexity["sentence_complexity"] > 0)
        
        # Technical text should have more long words than short text
        tech_complexity = self.evaluator.evaluate_complexity(self.technical_text)
        short_complexity = self.evaluator.evaluate_complexity(self.short_text)
        
        self.assertTrue(
            tech_complexity["long_word_ratio"] > short_complexity["long_word_ratio"]
        )
        
        # Test with empty text
        empty_complexity = self.evaluator.evaluate_complexity(self.empty_text)
        self.assertEqual(empty_complexity["lexical_diversity"], 0)

    def test_evaluate_response(self):
        """Test comprehensive response evaluation."""
        # Full evaluation with all metrics
        evaluation = self.evaluator.evaluate_response(self.medium_text)
        
        # Verify all metric categories are present
        self.assertIn("char_count", evaluation)
        self.assertIn("word_count", evaluation)
        self.assertIn("readability", evaluation)
        self.assertIn("complexity", evaluation)
        self.assertIn("content", evaluation)
        
        # Basic evaluation without all metrics
        basic_evaluation = self.evaluator.evaluate_response(
            self.medium_text, include_all_metrics=False
        )
        
        # Basic should only have text metrics
        self.assertIn("char_count", basic_evaluation)
        self.assertIn("word_count", basic_evaluation)
        self.assertNotIn("readability", basic_evaluation)
        self.assertNotIn("complexity", basic_evaluation)
        self.assertNotIn("content", basic_evaluation)
        
        # Test with empty text
        empty_evaluation = self.evaluator.evaluate_response(self.empty_text)
        self.assertEqual(empty_evaluation["char_count"], 0)
        self.assertEqual(empty_evaluation["word_count"], 0)

    def test_compare_responses(self):
        """Test comparing multiple responses."""
        # Compare with all metrics
        comparison = self.evaluator.compare_responses(self.model_responses)
        
        # Verify structure
        self.assertEqual(comparison["model_count"], 3)
        self.assertEqual(set(comparison["models"]), {"model1", "model2", "model3"})
        self.assertIn("comparisons", comparison)
        
        # Check that comparisons exist
        self.assertIn("length", comparison["comparisons"])
        self.assertIn("word_count", comparison["comparisons"])
        self.assertIn("readability", comparison["comparisons"])
        self.assertIn("complexity", comparison["comparisons"])
        self.assertIn("sentiment", comparison["comparisons"])
        
        # Verify best model is identified
        self.assertIn(comparison["comparisons"]["length"]["best"], self.model_responses.keys())
        
        # Test with raw metrics included
        comparison_with_raw = self.evaluator.compare_responses(
            self.model_responses, include_raw_metrics=True
        )
        self.assertIn("raw_metrics", comparison_with_raw)
        
        # Test with empty responses
        empty_comparison = self.evaluator.compare_responses({})
        self.assertIn("error", empty_comparison)

    def test_rank_responses(self):
        """Test ranking multiple responses."""
        # Rank with default metrics
        ranking = self.evaluator.rank_responses(self.model_responses)
        
        # Verify structure
        self.assertIn("metrics_used", ranking)
        self.assertIn("weights_used", ranking)
        self.assertIn("ranking", ranking)
        
        # Check ranking format
        self.assertEqual(len(ranking["ranking"]), 3)  # Three models
        self.assertIn("rank", ranking["ranking"][0])
        self.assertIn("model_id", ranking["ranking"][0])
        self.assertIn("score", ranking["ranking"][0])
        
        # First rank should be 1
        self.assertEqual(ranking["ranking"][0]["rank"], 1)
        
        # Test with custom metrics and weights
        custom_metrics = ["word_count", "content.question_count"]
        custom_weights = {"word_count": 0.8, "content.question_count": 0.2}
        
        custom_ranking = self.evaluator.rank_responses(
            self.model_responses, metrics=custom_metrics, weights=custom_weights
        )
        
        # Verify custom settings were used
        self.assertEqual(custom_ranking["metrics_used"], custom_metrics)
        self.assertEqual(custom_ranking["weights_used"], custom_weights)
        
        # Test with empty responses
        empty_ranking = self.evaluator.rank_responses({})
        self.assertIn("error", empty_ranking)

    def test_detect_language(self):
        """Test language detection."""
        # Test direct implementation without mocking
        result = self.evaluator.detect_language(self.medium_text)
        
        # Ensure the result has the language key
        self.assertIn("language", result)
        
        # If there was an error, check for error key
        if "error" in result:
            self.assertEqual(result["language"], "unknown")
        # Otherwise expect confidence
        else:
            self.assertIn("confidence", result)
        
        # Test with a different text to ensure consistent behavior
        short_result = self.evaluator.detect_language(self.short_text)
        self.assertIn("language", short_result)

    def test_extract_topics(self):
        """Test topic extraction."""
        # Need to patch the TextBlob used within the method
        with patch('backend.app.evaluator.TextBlob') as MockTextBlob:
            mock_blob = MagicMock()
            mock_blob.noun_phrases = ["sample topic", "important concept", "key idea"]
            MockTextBlob.return_value = mock_blob
            
            # Test topic extraction
            topics = self.evaluator.extract_topics(self.medium_text)
            
            # Verify results
            self.assertTrue(len(topics) > 0)
            self.assertEqual(topics[0], "sample topic")
            
            # Test with fewer topics requested
            topics_limited = self.evaluator.extract_topics(self.medium_text, num_topics=2)
            self.assertEqual(len(topics_limited), 2)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with None input
        none_metrics = self.evaluator.evaluate_text_metrics(None)
        self.assertEqual(none_metrics["char_count"], 0)
        
        # Test with numeric input (should be handled gracefully)
        numeric_metrics = self.evaluator.evaluate_text_metrics(123)
        self.assertEqual(numeric_metrics["char_count"], 0)
        
        # Test with very long text
        long_text = "word " * 1000
        long_metrics = self.evaluator.evaluate_text_metrics(long_text)
        self.assertEqual(long_metrics["word_count"], 1000)
        
        # Test with special characters
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?\n\t"
        special_metrics = self.evaluator.evaluate_text_metrics(special_chars)
        self.assertEqual(special_metrics["char_count"], len(special_chars))
        self.assertTrue(special_metrics["word_count"] >= 0) # No actual words


if __name__ == '__main__':
    unittest.main()