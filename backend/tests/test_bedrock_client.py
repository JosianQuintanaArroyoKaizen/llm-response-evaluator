"""
Unit tests for the BedrockClient class.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import boto3
from botocore.exceptions import ClientError

from bedrock_client import BedrockClient


class TestBedrockClient(unittest.TestCase):
    """Test cases for the BedrockClient class."""

    @patch("boto3.client")
    def setUp(self, mock_boto_client):
        """Set up test fixtures."""
        # Create mock clients
        self.mock_bedrock = MagicMock()
        self.mock_bedrock_runtime = MagicMock()
        
        # Configure boto3.client to return the mock clients
        mock_boto_client.side_effect = lambda service_name, **kwargs: {
            "bedrock": self.mock_bedrock,
            "bedrock-runtime": self.mock_bedrock_runtime
        }[service_name]
        
        # Create the BedrockClient instance
        self.client = BedrockClient(region_name="us-east-1")

    def test_list_models(self):
        """Test listing models."""
        # Mock the response from list_foundation_models
        mock_models = [
            {"modelId": "model1", "modelName": "Model 1"},
            {"modelId": "model2", "modelName": "Model 2"}
        ]
        self.mock_bedrock.list_foundation_models.return_value = {
            "modelSummaries": mock_models
        }
        
        # Call the method
        result = self.client.list_models()
        
        # Assert the result
        self.assertEqual(result, mock_models)
        self.mock_bedrock.list_foundation_models.assert_called_once()

    def test_list_models_error(self):
        """Test error handling when listing models."""
        # Mock a ClientError
        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Access denied"
            }
        }
        self.mock_bedrock.list_foundation_models.side_effect = ClientError(
            error_response, "list_foundation_models"
        )
        
        # Call the method and assert the exception
        with self.assertRaises(ClientError):
            self.client.list_models()

    def test_get_model_details(self):
        """Test getting model details."""
        # Mock the response from get_foundation_model
        mock_details = {
            "providerName": "Anthropic",
            "inputModalities": ["TEXT"],
            "outputModalities": ["TEXT"],
            "responseStreamingSupported": True
        }
        self.mock_bedrock.get_foundation_model.return_value = {
            "modelDetails": mock_details
        }
        
        # Call the method
        result = self.client.get_model_details("model1")
        
        # Assert the result
        self.assertEqual(result, mock_details)
        self.mock_bedrock.get_foundation_model.assert_called_once_with(
            modelIdentifier="model1"
        )

    def test_invoke_model(self):
        """Test invoking a model."""
        # Mock the response from invoke_model
        mock_response = {
            "body": MagicMock()
        }
        mock_response["body"].read.return_value = json.dumps({
            "completion": "This is a generated response."
        })
        self.mock_bedrock_runtime.invoke_model.return_value = mock_response
        
        # Call the method
        result = self.client.invoke_model("anthropic.claude-v2", "Test prompt")
        
        # Assert the result
        self.assertEqual(result["model_id"], "anthropic.claude-v2")
        self.assertEqual(result["generated_text"], "This is a generated response.")
        self.mock_bedrock_runtime.invoke_model.assert_called_once()

    def test_invoke_model_streaming(self):
        """Test invoking a model with streaming."""
        # Mock the get_model_details response
        self.mock_bedrock.get_foundation_model.return_value = {
            "modelDetails": {"responseStreamingSupported": True}
        }
        
        # Mock the response from invoke_model_with_response_stream
        mock_chunk = MagicMock()
        mock_chunk.get.return_value = {"bytes": json.dumps({"completion": "This is a "}).encode()}
        
        mock_chunk2 = MagicMock()
        mock_chunk2.get.return_value = {"bytes": json.dumps({"completion": "streaming response."}).encode()}
        
        mock_stream = MagicMock()
        mock_stream.__iter__.return_value = [
            {"chunk": mock_chunk},
            {"chunk": mock_chunk2}
        ]
        
        self.mock_bedrock_runtime.invoke_model_with_response_stream.return_value = {
            "body": mock_stream
        }
        
        # Call the method
        result = self.client.invoke_model("anthropic.claude-v2", "Test prompt", response_streaming=True)
        
        # Assert the result
        self.assertEqual(result["model_id"], "anthropic.claude-v2")
        self.assertEqual(result["generated_text"], "This is a streaming response.")
        self.mock_bedrock_runtime.invoke_model_with_response_stream.assert_called_once()

    def test_batch_invoke_models(self):
        """Test batch invoking multiple models."""
        # Mock responses for different models
        def mock_invoke_side_effect(model_id, prompt, parameters=None, response_streaming=False):
            responses = {
                "model1": {"model_id": "model1", "generated_text": "Response from model 1"},
                "model2": {"model_id": "model2", "generated_text": "Response from model 2"}
            }
            return responses[model_id]
        
        # Patch the invoke_model method
        with patch.object(self.client, 'invoke_model', side_effect=mock_invoke_side_effect):
            # Call the method
            result = self.client.batch_invoke_models(["model1", "model2"], "Test prompt")
            
            # Assert the result
            self.assertEqual(len(result), 2)
            self.assertEqual(result["model1"]["generated_text"], "Response from model 1")
            self.assertEqual(result["model2"]["generated_text"], "Response from model 2")

    def test_create_batch_inference_job(self):
        """Test creating a batch inference job."""
        # Mock the response from create_model_invocation_job
        self.mock_bedrock.create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-id"
        }
        
        # Call the method
        result = self.client.create_batch_inference_job(
            "model1", 
            "s3://input-bucket/input/", 
            "s3://output-bucket/output/",
            job_name="test-job",
            role_arn="arn:aws:iam::123456789012:role/service-role"
        )
        
        # Assert the result
        self.assertEqual(result["jobArn"], "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-id")
        self.mock_bedrock.create_model_invocation_job.assert_called_once()

    def test_get_batch_job_status(self):
        """Test getting batch job status."""
        # Mock the response from get_model_invocation_job
        self.mock_bedrock.get_model_invocation_job.return_value = {
            "status": "COMPLETED",
            "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-id"
        }
        
        # Call the method
        result = self.client.get_batch_job_status("arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-id")
        
        # Assert the result
        self.assertEqual(result["status"], "COMPLETED")
        self.mock_bedrock.get_model_invocation_job.assert_called_once_with(
            jobIdentifier="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/job-id"
        )


if __name__ == '__main__':
    unittest.main()
