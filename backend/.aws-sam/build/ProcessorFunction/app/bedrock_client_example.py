"""
Example usage of the BedrockClient class.
This script demonstrates how to use the BedrockClient class to interact with AWS Bedrock models.
"""

import json
import logging
from bedrock_client import BedrockClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for the example.
    """
    # Initialize the BedrockClient
    client = BedrockClient(region_name="us-east-1")
    
    # Example 1: List available models
    logger.info("Listing available models...")
    models = client.list_models()
    print(f"Found {len(models)} models")
    
    # Print the first 5 models
    for model in models[:5]:
        print(f"Model: {model.get('modelId')} - {model.get('modelName')}")
    
    # Example 2: Invoke a single model
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Use a valid model ID
    prompt = "Explain the concept of large language models in one paragraph."
    
    logger.info(f"Invoking model {model_id}...")
    try:
        response = client.invoke_model(model_id, prompt)
        print("\nModel response:")
        print(response.get("generated_text", "No text generated"))
    except Exception as e:
        logger.error(f"Error invoking model: {str(e)}")
    
    # Example 3: Batch invoke multiple models
    model_ids = [
        "anthropic.claude-3-sonnet-20240229-v1:0",
        "amazon.titan-text-express-v1"
    ]
    
    logger.info("Batch invoking multiple models...")
    try:
        batch_responses = client.batch_invoke_models(model_ids, prompt)
        
        print("\nBatch model responses:")
        for model_id, response in batch_responses.items():
            print(f"\nModel: {model_id}")
            print(response.get("generated_text", "No text generated"))
    except Exception as e:
        logger.error(f"Error in batch invocation: {str(e)}")
    
    # Example 4: Get model details
    logger.info("Getting model details...")
    try:
        model_details = client.get_model_details(model_id)
        print("\nModel details:")
        print(f"Model: {model_id}")
        print(f"Provider: {model_details.get('providerName', 'Unknown')}")
        print(f"Input modalities: {model_details.get('inputModalities', [])}")
        print(f"Output modalities: {model_details.get('outputModalities', [])}")
        print(f"Streaming supported: {model_details.get('responseStreamingSupported', False)}")
    except Exception as e:
        logger.error(f"Error getting model details: {str(e)}")

if __name__ == "__main__":
    main()
