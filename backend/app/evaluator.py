"""
Response Evaluator for the LLM Response Evaluator project.
This module provides metrics and comparison functionality for LLM responses.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from textblob import TextBlob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {str(e)}")


class ResponseEvaluator:
    """Evaluator for analyzing and comparing LLM responses."""

    def __init__(self):
        """Initialize the Response Evaluator."""
        self.sia = SentimentIntensityAnalyzer()
        logger.info("Initialized ResponseEvaluator")
        
    def evaluate_text_metrics(self, text: str) -> Dict[str, Any]:
        """
        Evaluate basic text metrics for a single response.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of text metrics
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for evaluation")
            return {
                "char_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "avg_sentence_length": 0
            }
            
        try:
            # Basic text metrics
            char_count = len(text)
            words = word_tokenize(text)
            word_count = len(words)
            sentences = sent_tokenize(text)
            sentence_count = len(sentences)
            
            # Average metrics
            avg_word_length = sum(len(word) for word in words) / max(1, word_count)
            avg_sentence_length = word_count / max(1, sentence_count)
            
            metrics = {
                "char_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_word_length": round(avg_word_length, 2),
                "avg_sentence_length": round(avg_sentence_length, 2)
            }
            
            logger.debug(f"Text metrics calculated: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating text metrics: {str(e)}")
            raise
    
    def evaluate_readability(self, text: str) -> Dict[str, Any]:
        """
        Evaluate readability metrics for a single response.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of readability metrics
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for readability evaluation")
            return {
                "flesch_reading_ease": 0,
                "flesch_kincaid_grade": 0,
                "gunning_fog": 0,
                "smog_index": 0,
                "automated_readability_index": 0,
                "coleman_liau_index": 0,
                "dale_chall_readability_score": 0,
                "difficult_words": 0,
                "linsear_write_formula": 0,
                "text_standard": "N/A"
            }
            
        try:
            # Calculate readability metrics using textstat
            readability = {
                "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
                "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
                "gunning_fog": round(textstat.gunning_fog(text), 2),
                "smog_index": round(textstat.smog_index(text), 2),
                "automated_readability_index": round(textstat.automated_readability_index(text), 2),
                "coleman_liau_index": round(textstat.coleman_liau_index(text), 2),
                "dale_chall_readability_score": round(textstat.dale_chall_readability_score(text), 2),
                "difficult_words": textstat.difficult_words(text),
                "linsear_write_formula": round(textstat.linsear_write_formula(text), 2),
                "text_standard": textstat.text_standard(text, float_output=False)
            }
            
            logger.debug(f"Readability metrics calculated: {readability}")
            return readability
            
        except Exception as e:
            logger.error(f"Error calculating readability metrics: {str(e)}")
            raise
    
    def evaluate_content_metrics(self, text: str) -> Dict[str, Any]:
        """
        Evaluate content-based metrics for a single response.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of content metrics
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for content evaluation")
            return {
                "sentiment": {"compound": 0, "pos": 0, "neu": 0, "neg": 0},
                "subjectivity": 0,
                "polarity": 0,
                "question_count": 0,
                "code_block_count": 0,
                "list_count": 0,
                "url_count": 0,
                "has_numbers": False,
                "number_count": 0
            }
            
        try:
            # Sentiment analysis using VADER
            sentiment = self.sia.polarity_scores(text)
            
            # Additional sentiment using TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Count questions using regex
            question_count = len(re.findall(r'\?(?:$|\s)', text))
            
            # Count code blocks (both Markdown and HTML style)
            code_block_count = text.count('```') // 2  # Markdown style
            code_block_count += len(re.findall(r'<code>.*?</code>', text, re.DOTALL))  # HTML style
            
            # Count numbered and bullet lists
            numbered_list_count = len(re.findall(r'^\d+\.\s', text, re.MULTILINE))
            bullet_list_count = len(re.findall(r'^[-*•]\s', text, re.MULTILINE))
            list_count = numbered_list_count + bullet_list_count
            
            # Count URLs
            url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
            url_count = len(re.findall(url_pattern, text))
            
            # Check for numbers and count them
            has_numbers = bool(re.search(r'\d', text))
            number_count = len(re.findall(r'\b\d+\b', text))
            
            content_metrics = {
                "sentiment": {
                    "compound": round(sentiment["compound"], 3),
                    "pos": round(sentiment["pos"], 3),
                    "neu": round(sentiment["neu"], 3),
                    "neg": round(sentiment["neg"], 3)
                },
                "subjectivity": round(subjectivity, 3),
                "polarity": round(polarity, 3),
                "question_count": question_count,
                "code_block_count": code_block_count,
                "list_count": list_count,
                "url_count": url_count,
                "has_numbers": has_numbers,
                "number_count": number_count
            }
            
            logger.debug(f"Content metrics calculated: {content_metrics}")
            return content_metrics
            
        except Exception as e:
            logger.error(f"Error calculating content metrics: {str(e)}")
            raise
    
    def evaluate_complexity(self, text: str) -> Dict[str, Any]:
        """
        Evaluate text complexity metrics.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary of complexity metrics
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text provided for complexity evaluation")
            return {
                "lexical_diversity": 0,
                "long_word_ratio": 0,
                "unique_word_ratio": 0,
                "sentence_complexity": 0
            }
            
        try:
            words = word_tokenize(text.lower())
            word_count = len(words)
            unique_words = set(words)
            unique_word_count = len(unique_words)
            
            # Lexical diversity (unique words / total words)
            lexical_diversity = unique_word_count / max(1, word_count)
            
            # Long word ratio (words with > 6 chars / total words)
            long_words = [word for word in words if len(word) > 6]
            long_word_ratio = len(long_words) / max(1, word_count)
            
            # Unique word ratio
            unique_word_ratio = unique_word_count / max(1, word_count)
            
            # Sentence complexity (avg words per sentence)
            sentences = sent_tokenize(text)
            words_per_sentence = [len(word_tokenize(sentence)) for sentence in sentences]
            sentence_complexity = sum(words_per_sentence) / max(1, len(sentences))
            
            complexity = {
                "lexical_diversity": round(lexical_diversity, 3),
                "long_word_ratio": round(long_word_ratio, 3),
                "unique_word_ratio": round(unique_word_ratio, 3),
                "sentence_complexity": round(sentence_complexity, 3)
            }
            
            logger.debug(f"Complexity metrics calculated: {complexity}")
            return complexity
            
        except Exception as e:
            logger.error(f"Error calculating complexity metrics: {str(e)}")
            raise
    
    def evaluate_response(self, text: str, include_all_metrics: bool = True) -> Dict[str, Any]:
        """
        Perform a comprehensive evaluation of a single response.
        
        Args:
            text: The text to analyze
            include_all_metrics: Whether to include all available metrics (default: True)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            # Start with basic text metrics
            evaluation = self.evaluate_text_metrics(text)
            
            # Add other metrics if requested
            if include_all_metrics:
                evaluation["readability"] = self.evaluate_readability(text)
                evaluation["complexity"] = self.evaluate_complexity(text)
                evaluation["content"] = self.evaluate_content_metrics(text)
            
            logger.info(f"Evaluated response with {len(evaluation)} metric categories")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}")
            return {"error": str(e)}
    
    def compare_responses(
        self, 
        responses: Dict[str, str],
        include_all_metrics: bool = True,
        include_raw_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Compare multiple model responses to the same prompt.
        
        Args:
            responses: Dictionary mapping model_ids to response texts
            include_all_metrics: Whether to include all available metrics (default: True)
            include_raw_metrics: Whether to include raw metrics for each model (default: False)
            
        Returns:
            Dictionary containing comparison results
        """
        if not responses:
            logger.warning("Empty responses provided for comparison")
            return {"error": "No responses to compare"}
        
        try:
            # Evaluate each response
            evaluations = {}
            for model_id, text in responses.items():
                evaluations[model_id] = self.evaluate_response(text, include_all_metrics)
            
            # Create comparison summary
            comparison = {
                "model_count": len(responses),
                "models": list(responses.keys()),
                "comparisons": {}
            }
            
            # Include raw metrics if requested
            if include_raw_metrics:
                comparison["raw_metrics"] = evaluations
            
            # Generate basic comparison metrics
            length_comparison = {
                model_id: metrics["char_count"] 
                for model_id, metrics in evaluations.items()
            }
            comparison["comparisons"]["length"] = {
                "metric": "char_count",
                "values": length_comparison,
                "best": max(length_comparison.items(), key=lambda x: x[1])[0] if length_comparison else None
            }
            
            word_count_comparison = {
                model_id: metrics["word_count"] 
                for model_id, metrics in evaluations.items()
            }
            comparison["comparisons"]["word_count"] = {
                "metric": "word_count",
                "values": word_count_comparison,
                "best": max(word_count_comparison.items(), key=lambda x: x[1])[0] if word_count_comparison else None
            }
            
            # Add readability comparison if available
            if include_all_metrics:
                readability_comparison = {
                    model_id: metrics["readability"]["flesch_reading_ease"] 
                    for model_id, metrics in evaluations.items() 
                    if "readability" in metrics
                }
                comparison["comparisons"]["readability"] = {
                    "metric": "flesch_reading_ease",
                    "values": readability_comparison,
                    "best": max(readability_comparison.items(), key=lambda x: x[1])[0] if readability_comparison else None
                }
                
                # Add complexity comparison
                complexity_comparison = {
                    model_id: metrics["complexity"]["lexical_diversity"] 
                    for model_id, metrics in evaluations.items() 
                    if "complexity" in metrics
                }
                comparison["comparisons"]["complexity"] = {
                    "metric": "lexical_diversity",
                    "values": complexity_comparison,
                    "best": max(complexity_comparison.items(), key=lambda x: x[1])[0] if complexity_comparison else None
                }
                
                # Add sentiment comparison
                sentiment_comparison = {
                    model_id: metrics["content"]["sentiment"]["compound"] 
                    for model_id, metrics in evaluations.items() 
                    if "content" in metrics
                }
                comparison["comparisons"]["sentiment"] = {
                    "metric": "sentiment_compound",
                    "values": sentiment_comparison,
                    "best": max(sentiment_comparison.items(), key=lambda x: x[1])[0] if sentiment_comparison else None
                }
            
            logger.info(f"Compared responses from {len(responses)} models")
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing responses: {str(e)}")
            return {"error": str(e)}
    
    def rank_responses(
        self, 
        responses: Dict[str, str],
        metrics: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Rank multiple model responses based on weighted metrics.
        
        Args:
            responses: Dictionary mapping model_ids to response texts
            metrics: List of metrics to consider for ranking (default: None, uses predefined set)
            weights: Dictionary mapping metrics to weights (default: None, equal weights)
            
        Returns:
            Dictionary containing ranking results
        """
        if not responses:
            logger.warning("Empty responses provided for ranking")
            return {"error": "No responses to rank"}
        
        # Default metrics if none provided
        if metrics is None:
            metrics = [
                "readability.flesch_reading_ease",
                "complexity.lexical_diversity",
                "content.sentiment.compound",
                "word_count"
            ]
        
        # Default weights if none provided (equal weights)
        if weights is None:
            weights = {metric: 1.0 / len(metrics) for metric in metrics}
        
        try:
            # Evaluate each response with all metrics
            evaluations = {}
            for model_id, text in responses.items():
                evaluations[model_id] = self.evaluate_response(text, include_all_metrics=True)
            
            # Extract metric values for each model
            model_scores = {model_id: 0.0 for model_id in responses.keys()}
            
            for metric in metrics:
                # Handle nested metrics with dot notation
                metric_parts = metric.split(".")
                
                # Get the values for this metric across all models
                metric_values = {}
                for model_id, evaluation in evaluations.items():
                    value = evaluation
                    try:
                        # Navigate through nested dictionaries
                        for part in metric_parts:
                            value = value[part]
                        metric_values[model_id] = value
                    except (KeyError, TypeError):
                        logger.warning(f"Metric {metric} not found for model {model_id}")
                        metric_values[model_id] = 0
                
                # Skip if no values found
                if not metric_values:
                    continue
                
                # Normalize values to 0-1 range
                min_val = min(metric_values.values())
                max_val = max(metric_values.values())
                range_val = max_val - min_val
                
                # Update scores (handle zero range case)
                if range_val > 0:
                    for model_id, value in metric_values.items():
                        normalized_value = (value - min_val) / range_val
                        model_scores[model_id] += normalized_value * weights.get(metric, 1.0)
            
            # Create ranking
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            ranking = {
                "metrics_used": metrics,
                "weights_used": weights,
                "ranking": [
                    {"rank": i+1, "model_id": model_id, "score": round(score, 4)}
                    for i, (model_id, score) in enumerate(ranked_models)
                ]
            }
            
            logger.info(f"Ranked {len(responses)} models based on {len(metrics)} metrics")
            return ranking
            
        except Exception as e:
            logger.error(f"Error ranking responses: {str(e)}")
            return {"error": str(e)}
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text response.
        Note: This is a basic implementation. For production,
        consider using a specialized library like langdetect or fastText.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with detected language information
        """
        try:
            # Using TextBlob for language detection
            blob = TextBlob(text)
            language = blob.detect_language()
            
            return {
                "language": language,
                "confidence": 0.8  # TextBlob doesn't provide confidence, so this is a placeholder
            }
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return {"language": "unknown", "error": str(e)}
    
    def extract_topics(self, text: str, num_topics: int = 5) -> List[str]:
        """
        Extract main topics/keywords from a response.
        This is a simplified implementation using TextBlob.
        
        Args:
            text: The text to analyze
            num_topics: Number of topics to extract (default: 5)
            
        Returns:
            List of extracted topics/keywords
        """
        try:
            blob = TextBlob(text)
            topics = []
            
            # Try to get noun phrases if available
            try:
                topics = list(blob.noun_phrases)
            except Exception as e:
                logger.warning(f"Could not extract noun phrases: {str(e)}")
            
            # If not enough topics, extract common words
            if len(topics) < num_topics:
                # Simple word extraction fallback
                words = [word.lower() for word in blob.words if len(word) > 3]
                word_freq = {}
                
                # Count word frequencies
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                    
                # Get most frequent words
                top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                for word, _ in top_words:
                    if word not in topics and len(topics) < num_topics:
                        topics.append(word)
            
            # Limit to requested number
            return topics[:num_topics]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []