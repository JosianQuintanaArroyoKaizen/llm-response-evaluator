"""
AWS Bedrock Client for the LLM Response Evaluator project.
This module provides a client for interacting with AWS Bedrock models.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockClient:
    """Client for interacting with AWS Bedrock models."""

    def __init__(self, region_name: str = "us-east-1", max_retries: int = 3):
        """
        Initialize the Bedrock client.

        Args:
            region_name: AWS region name (default: "us-east-1")
            max_retries: Maximum number of retries for API calls (default: 3)
        """
        # Create a retry configuration
        self.retry_config = Config(
            retries={"max_attempts": max_retries, "mode": "standard"}
        )
        
        # Create Bedrock clients for model management and runtime inference
        self.bedrock_client = boto3.client(
            service_name="bedrock", 
            region_name=region_name,
            config=self.retry_config
        )
        
        self.bedrock_runtime = boto3.client(
            service_name="bedrock-runtime", 
            region_name=region_name,
            config=self.retry_config
        )
        
        logger.info(f"Initialized BedrockClient in region {region_name}")

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available Bedrock foundation models.

        Returns:
            List of model information dictionaries
        """
        try:
            response = self.bedrock_client.list_foundation_models()
            models = response.get("modelSummaries", [])
            logger.info(f"Found {len(models)} foundation models")
            return models
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error listing models: {error_code} - {error_message}")
            raise

    def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific Bedrock model.

        Args:
            model_id: The identifier of the model

        Returns:
            Dictionary containing model details
        """
        try:
            response = self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            model_details = response.get("modelDetails", {})
            logger.info(f"Retrieved details for model {model_id}")
            return model_details
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error getting model details for {model_id}: {error_code} - {error_message}")
            raise

    def _format_model_request(self, model_id: str, prompt: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Format the request payload based on the model provider.

        Args:
            model_id: The identifier of the model
            prompt: The text prompt to send to the model
            parameters: Additional parameters for the model (temperature, max_tokens, etc.)

        Returns:
            Formatted request payload ready for the model
        """
        # Default parameters if none provided
        if parameters is None:
            parameters = {}
        
        # Anthropic Claude models format
        if model_id.startswith("anthropic.claude"):
            default_params = {
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            if "anthropic_version" not in merged_params:
                merged_params["anthropic_version"] = "bedrock-2023-05-31"
                
            # Check if we need to use the newer messages format or older prompt format
            if "claude-3" in model_id:
                payload = {
                    **merged_params,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                }
            else:
                # Legacy prompt format for Claude 2 models
                payload = {
                    **merged_params,
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "stop_sequences": ["\n\nHuman:"]
                }
                
        # Amazon Titan models format
        elif model_id.startswith("amazon.titan"):
            default_params = {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            # Format depends on whether it's a text or embedding model
            if "embed" in model_id.lower():
                payload = {"inputText": prompt}
            else:
                payload = {
                    "inputText": prompt,
                    "textGenerationConfig": merged_params
                }
                
        # AI21 Jurassic models format
        elif model_id.startswith("ai21"):
            default_params = {
                "maxTokens": 512,
                "temperature": 0.7,
                "topP": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            payload = {
                "prompt": prompt,
                **merged_params
            }
            
        # Cohere models format
        elif model_id.startswith("cohere"):
            default_params = {
                "max_tokens": 512,
                "temperature": 0.7,
                "p": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            payload = {
                "prompt": prompt,
                **merged_params
            }
            
        # Meta Llama models format
        elif model_id.startswith("meta.llama"):
            default_params = {
                "max_gen_len": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            payload = {
                "prompt": prompt,
                **merged_params
            }
            
        # Mistral models format
        elif model_id.startswith("mistral"):
            default_params = {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            merged_params = {**default_params, **parameters}
            
            payload = {
                "prompt": prompt,
                **merged_params
            }
            
        # Stability AI models format
        elif model_id.startswith("stability"):
            default_params = {
                "cfg_scale": 10,
                "steps": 30,
                "seed": 0,
            }
            merged_params = {**default_params, **parameters}
            
            payload = {
                "text_prompts": [{"text": prompt}],
                **merged_params
            }
            
        # Default format if model provider is unknown
        else:
            payload = {
                "prompt": prompt,
                **parameters
            }
            logger.warning(f"Unknown model provider for {model_id}, using default payload format")
            
        return payload

    def invoke_model(
        self, 
        model_id: str, 
        prompt: str, 
        parameters: Optional[Dict[str, Any]] = None,
        response_streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke a single Bedrock model with a prompt.

        Args:
            model_id: The identifier of the model
            prompt: The text prompt to send to the model
            parameters: Additional parameters for the model (temperature, max_tokens, etc.)
            response_streaming: Whether to use streaming response (default: False)

        Returns:
            The model's response
        """
        # Format the request payload
        payload = self._format_model_request(model_id, prompt, parameters)
        body = json.dumps(payload).encode("utf-8")
        
        # Common parameters for both streaming and non-streaming invocations
        invoke_params = {
            "modelId": model_id,
            "body": body,
            "contentType": "application/json",
            "accept": "application/json",
        }
        
        try:
            # Determine whether to use streaming or standard invocation
            if response_streaming:
                # Check if the model supports streaming
                model_details = self.get_model_details(model_id)
                if not model_details.get("responseStreamingSupported", False):
                    logger.warning(f"Model {model_id} does not support streaming. Falling back to standard invocation.")
                    response = self.bedrock_runtime.invoke_model(**invoke_params)
                    return self._parse_model_response(model_id, response)
                
                # Use streaming response
                response = self.bedrock_runtime.invoke_model_with_response_stream(**invoke_params)
                
                # Process the streaming response
                stream = response.get("body", [])
                full_response = ""
                
                for event in stream:
                    chunk = event.get("chunk", {})
                    if chunk:
                        bytes_data = chunk.get("bytes")
                        if bytes_data:
                            chunk_data = json.loads(bytes_data.decode("utf-8"))
                            # Different models return different fields in streaming mode
                            if "completion" in chunk_data:
                                full_response += chunk_data["completion"]
                            elif "generated_text" in chunk_data:
                                full_response += chunk_data["generated_text"]
                            elif "generation" in chunk_data:
                                full_response += chunk_data["generation"]
                            elif "output" in chunk_data:
                                full_response += chunk_data["output"]
                
                return {"generated_text": full_response, "model_id": model_id}
            else:
                # Use standard invocation
                response = self.bedrock_runtime.invoke_model(**invoke_params)
                return self._parse_model_response(model_id, response)
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error invoking model {model_id}: {error_code} - {error_message}")
            
            # Handle specific error types
            if error_code == "ValidationException":
                logger.error(f"Invalid request for model {model_id}. Check model ID and parameters.")
            elif error_code == "AccessDeniedException":
                logger.error(f"Access denied for model {model_id}. Check IAM permissions.")
            elif error_code == "ThrottlingException":
                logger.error(f"Request throttled for model {model_id}. Consider implementing exponential backoff.")
            elif error_code == "ServiceUnavailableException":
                logger.error(f"Service unavailable for model {model_id}. Try again later.")
            elif error_code == "ModelTimeoutException":
                logger.error(f"Model timeout for {model_id}. Consider simplifying your prompt.")
            
            raise
        except Exception as e:
            logger.error(f"Unexpected error invoking model {model_id}: {str(e)}")
            raise

    def _parse_model_response(self, model_id: str, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the response from different model providers into a standard format.

        Args:
            model_id: The identifier of the model
            response: The raw response from the model

        Returns:
            Standardized response dictionary
        """
        try:
            # Parse the response body
            response_body = json.loads(response["body"].read())
            
            # Create standardized response with model ID
            standardized_response = {"model_id": model_id}
            
            # Anthropic Claude models format
            if model_id.startswith("anthropic.claude"):
                if "completion" in response_body:
                    # Claude 2 format
                    standardized_response["generated_text"] = response_body["completion"]
                elif "content" in response_body:
                    # Claude 3 format with content blocks
                    content_blocks = response_body.get("content", [])
                    text_blocks = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
                    standardized_response["generated_text"] = "".join(text_blocks)
                
            # Amazon Titan models format
            elif model_id.startswith("amazon.titan"):
                if "embedding" in response_body:
                    # Embedding model
                    standardized_response["embedding"] = response_body["embedding"]
                    standardized_response["input_token_count"] = response_body.get("inputTextTokenCount", 0)
                else:
                    # Text generation model
                    standardized_response["generated_text"] = response_body.get("results", [{}])[0].get("outputText", "")
                    
            # AI21 Jurassic models format
            elif model_id.startswith("ai21"):
                completions = response_body.get("completions", [{}])
                if completions:
                    standardized_response["generated_text"] = completions[0].get("data", {}).get("text", "")
                    
            # Cohere models format
            elif model_id.startswith("cohere"):
                generations = response_body.get("generations", [{}])
                if generations:
                    standardized_response["generated_text"] = generations[0].get("text", "")
                    
            # Meta Llama models format
            elif model_id.startswith("meta.llama"):
                standardized_response["generated_text"] = response_body.get("generation", "")
                
            # Mistral models format
            elif model_id.startswith("mistral"):
                standardized_response["generated_text"] = response_body.get("outputs", [{}])[0].get("text", "")
                
            # Stability AI models format (image generation)
            elif model_id.startswith("stability"):
                artifacts = response_body.get("artifacts", [{}])
                if artifacts:
                    standardized_response["image_base64"] = artifacts[0].get("base64", "")
                    standardized_response["seed"] = artifacts[0].get("seed", 0)
                    standardized_response["finish_reason"] = artifacts[0].get("finishReason", "")
                    
            # If unknown model format, return the raw response body
            else:
                logger.warning(f"Unknown model response format for {model_id}, returning raw response")
                standardized_response["raw_response"] = response_body
                
            return standardized_response
            
        except Exception as e:
            logger.error(f"Error parsing response from {model_id}: {str(e)}")
            # Return minimal information on failure
            return {
                "model_id": model_id,
                "error": str(e),
                "raw_response": response.get("body", "").read() if hasattr(response.get("body", ""), "read") else None
            }

    def batch_invoke_models(
        self, 
        model_ids: List[str], 
        prompt: str, 
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: int = 5
    ) -> Dict[str, Any]:
        """
        Invoke multiple models with the same prompt in parallel.

        Args:
            model_ids: List of model identifiers
            prompt: The text prompt to send to all models
            parameters: Dictionary mapping model_id to parameters (default: None)
            max_workers: Maximum number of parallel workers (default: 5)

        Returns:
            Dictionary mapping model_ids to their responses
        """
        if parameters is None:
            parameters = {}
            
        # Define the worker function for ThreadPoolExecutor
        def invoke_worker(model_id: str) -> Tuple[str, Dict[str, Any]]:
            model_params = parameters.get(model_id, {})
            try:
                response = self.invoke_model(model_id, prompt, model_params)
                return model_id, response
            except Exception as e:
                logger.error(f"Error in batch invoke for model {model_id}: {str(e)}")
                return model_id, {"error": str(e), "model_id": model_id}
        
        # Use ThreadPoolExecutor to run invocations in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(invoke_worker, model_id): model_id for model_id in model_ids}
            
            for future in futures:
                model_id, response = future.result()
                results[model_id] = response
                
        return results

    def create_batch_inference_job(
        self, 
        model_id: str, 
        input_s3_uri: str, 
        output_s3_uri: str, 
        job_name: Optional[str] = None,
        role_arn: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a batch inference job for processing multiple prompts asynchronously.
        Note: This requires proper S3 bucket setup and IAM roles.

        Args:
            model_id: The identifier of the model
            input_s3_uri: S3 URI for input data
            output_s3_uri: S3 URI for output data
            job_name: Optional name for the job (default: None)
            role_arn: ARN of the IAM role to use (default: None)

        Returns:
            Response from the CreateModelInvocationJob API
        """
        if job_name is None:
            timestamp = int(time.time())
            job_name = f"bedrock-batch-job-{timestamp}"
            
        try:
            request_params = {
                "modelId": model_id,
                "jobName": job_name,
                "inputDataConfig": {
                    "s3InputDataConfig": {
                        "s3Uri": input_s3_uri
                    }
                },
                "outputDataConfig": {
                    "s3OutputDataConfig": {
                        "s3Uri": output_s3_uri
                    }
                }
            }
            
            # Add role ARN if provided
            if role_arn:
                request_params["roleArn"] = role_arn
                
            response = self.bedrock_client.create_model_invocation_job(**request_params)
            logger.info(f"Created batch inference job {job_name} with job ARN: {response.get('jobArn')}")
            return response
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error creating batch inference job: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating batch inference job: {str(e)}")
            raise

    def get_batch_job_status(self, job_arn: str) -> Dict[str, Any]:
        """
        Get the status of a batch inference job.

        Args:
            job_arn: The ARN of the batch job

        Returns:
            Job status information
        """
        try:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = response.get("status")
            logger.info(f"Batch job {job_arn} status: {status}")
            return response
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Error getting batch job status: {error_code} - {error_message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting batch job status: {str(e)}")
            raise