"""
Processor for LLM Response Evaluator project.
This module handles async processing of evaluation jobs from the SQS queue.
"""

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

import boto3
import requests
from botocore.exceptions import ClientError

from .bedrock_client import BedrockClient
from .evaluator import ResponseEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS clients
sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')

# Get environment variables
QUEUE_URL = os.environ.get('EVALUATION_QUEUE_URL')
JOBS_TABLE_NAME = os.environ.get('JOBS_TABLE_NAME')
HISTORY_TABLE_NAME = os.environ.get('HISTORY_TABLE_NAME')
MAX_PROCESSING_TIME = int(os.environ.get('MAX_PROCESSING_TIME', '600'))  # In seconds

# Initialize DynamoDB tables
jobs_table = None
history_table = None

if JOBS_TABLE_NAME:
    jobs_table = dynamodb.Table(JOBS_TABLE_NAME)
    logger.info(f"Connected to DynamoDB jobs table: {JOBS_TABLE_NAME}")

if HISTORY_TABLE_NAME:
    history_table = dynamodb.Table(HISTORY_TABLE_NAME)
    logger.info(f"Connected to DynamoDB history table: {HISTORY_TABLE_NAME}")


class JobStatus:
    """Job status constants."""
    SUBMITTED = "SUBMITTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


def process_job(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single evaluation job.
    
    Args:
        job_data: The job data from SQS
        
    Returns:
        Processing results
    """
    job_id = job_data.get("job_id")
    prompt = job_data.get("prompt")
    model_ids = job_data.get("model_ids", [])
    parameters = job_data.get("parameters", {})
    metrics = job_data.get("metrics")
    save_history = job_data.get("save_history", True)
    callback_url = job_data.get("callback_url")
    
    logger.info(f"Processing job {job_id} with {len(model_ids)} models")
    
    try:
        # Update job status to PROCESSING
        update_job_status(job_id, JobStatus.PROCESSING)
        
        # Initialize clients
        region = os.environ.get('AWS_REGION', 'us-east-1')
        bedrock_client = BedrockClient(region_name=region)
        evaluator = ResponseEvaluator()
        
        # Invoke models and track progress
        model_responses = {}
        text_responses = {}
        
        for i, model_id in enumerate(model_ids):
            # Update progress
            progress = (i / len(model_ids)) * 100
            update_job_progress(job_id, progress)
            
            # Get model parameters
            model_params = parameters.get(model_id, {})
            
            try:
                # Invoke the model
                logger.info(f"Invoking model {model_id} for job {job_id}")
                response = bedrock_client.invoke_model(
                    model_id=model_id,
                    prompt=prompt,
                    parameters=model_params
                )
                
                model_responses[model_id] = response
                
                # Extract generated text
                if "error" in response:
                    text_responses[model_id] = f"Error: {response['error']}"
                elif "generated_text" in response:
                    text_responses[model_id] = response["generated_text"]
                else:
                    text_responses[model_id] = str(response)
                    
            except Exception as e:
                logger.error(f"Error invoking model {model_id}: {str(e)}")
                text_responses[model_id] = f"Error: {str(e)}"
        
        # Final progress update
        update_job_progress(job_id, 90)  # 90% complete after model invocations
        
        # Evaluate each response
        evaluations = {}
        for model_id, text in text_responses.items():
            if not text.startswith("Error:"):
                evaluations[model_id] = evaluator.evaluate_response(text)
            else:
                evaluations[model_id] = {"error": text}
        
        # Compare responses
        logger.info(f"Comparing responses for job {job_id}")
        comparisons = evaluator.compare_responses(text_responses)
        
        # Rank responses based on metrics (if specified)
        logger.info(f"Ranking responses for job {job_id}")
        ranking = evaluator.rank_responses(
            text_responses,
            metrics=metrics
        )
        
        # Prepare result data
        timestamp = datetime.utcnow().isoformat()
        result_data = {
            "job_id": job_id,
            "prompt": prompt,
            "timestamp": timestamp,
            "model_responses": text_responses,
            "evaluations": evaluations,
            "comparisons": comparisons,
            "ranking": ranking
        }
        
        # Save to history if requested
        if save_history and history_table:
            save_to_history(job_id, timestamp, result_data)
        
        # Update job with results
        logger.info(f"Updating job {job_id} with results")
        update_job_results(job_id, JobStatus.COMPLETED, result_data)
        
        # Send callback if provided
        if callback_url:
            send_callback(callback_url, {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "timestamp": timestamp
            })
        
        return result_data
        
    except Exception as e:
        error_message = f"Error processing job {job_id}: {str(e)}"
        logger.error(error_message)
        logger.error(traceback.format_exc())
        
        # Update job as failed
        update_job_status(job_id, JobStatus.FAILED, error=str(e))
        
        # Send callback if provided
        if callback_url:
            send_callback(callback_url, {
                "job_id": job_id,
                "status": JobStatus.FAILED,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "job_id": job_id,
            "error": str(e),
            "status": JobStatus.FAILED
        }


def update_job_status(job_id: str, status: str, error: str = None) -> None:
    """
    Update job status in DynamoDB.
    
    Args:
        job_id: The unique job identifier
        status: New job status
        error: Optional error message
    """
    if not jobs_table:
        logger.warning(f"Cannot update job {job_id}: Jobs table not configured")
        return
    
    try:
        update_expression = "SET #status = :status, #updated_at = :updated_at"
        expression_attr_names = {
            "#status": "status",
            "#updated_at": "updated_at"
        }
        expression_attr_values = {
            ":status": status,
            ":updated_at": datetime.utcnow().isoformat()
        }
        
        # Add error if provided
        if error:
            update_expression += ", #error = :error"
            expression_attr_names["#error"] = "error"
            expression_attr_values[":error"] = error
        
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attr_names,
            ExpressionAttributeValues=expression_attr_values
        )
        
        logger.info(f"Updated job {job_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Error updating job status: {str(e)}")


def update_job_progress(job_id: str, progress: float) -> None:
    """
    Update job progress in DynamoDB.
    
    Args:
        job_id: The unique job identifier
        progress: Progress percentage (0-100)
    """
    if not jobs_table:
        return
    
    try:
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #progress = :progress, #updated_at = :updated_at",
            ExpressionAttributeNames={
                "#progress": "progress",
                "#updated_at": "updated_at"
            },
            ExpressionAttributeValues={
                ":progress": progress,
                ":updated_at": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error updating job progress: {str(e)}")


def update_job_results(job_id: str, status: str, results: Dict[str, Any]) -> None:
    """
    Update job with results in DynamoDB.
    
    Args:
        job_id: The unique job identifier
        status: New job status
        results: Job results
    """
    if not jobs_table:
        logger.warning(f"Cannot update job {job_id} results: Jobs table not configured")
        return
    
    try:
        # Create an update expression that sets all result fields
        update_expression = "SET #status = :status, #updated_at = :updated_at"
        expression_attr_names = {
            "#status": "status",
            "#updated_at": "updated_at"
        }
        expression_attr_values = {
            ":status": status,
            ":updated_at": datetime.utcnow().isoformat()
        }
        
        # Add each result field to the update expression
        for key, value in results.items():
            if key not in ["job_id"]:  # Skip the primary key
                update_expression += f", #{key} = :{key}"
                expression_attr_names[f"#{key}"] = key
                
                # Convert complex objects to JSON strings if needed
                if isinstance(value, (dict, list)):
                    expression_attr_values[f":{key}"] = value
                else:
                    expression_attr_values[f":{key}"] = value
        
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attr_names,
            ExpressionAttributeValues=expression_attr_values
        )
        
        logger.info(f"Updated job {job_id} with results")
        
    except Exception as e:
        logger.error(f"Error updating job results: {str(e)}")
        logger.error(traceback.format_exc())


def save_to_history(job_id: str, timestamp: str, data: Dict[str, Any]) -> None:
    """
    Save evaluation data to DynamoDB history table.
    
    Args:
        job_id: The unique job identifier
        timestamp: ISO-formatted timestamp
        data: Evaluation data to save
    """
    if not history_table:
        logger.warning(f"Cannot save job {job_id} to history: History table not configured")
        return
    
    try:
        # Remove large data to keep history records small
        history_data = {
            "evaluation_id": job_id,
            "timestamp": timestamp,
            "prompt": data["prompt"],
            "model_ids": list(data["model_responses"].keys()),
            "ranking": data.get("ranking", {})
        }
        
        # Add the item to DynamoDB
        history_table.put_item(Item=history_data)
        logger.info(f"Saved job {job_id} to history")
        
    except Exception as e:
        logger.error(f"Error saving to history: {str(e)}")


def send_callback(callback_url: str, data: Dict[str, Any]) -> None:
    """
    Send a callback to notify about job completion.
    
    Args:
        callback_url: The URL to send the callback to
        data: Callback data
    """
    try:
        response = requests.post(
            url=callback_url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"Callback sent successfully to {callback_url}")
        else:
            logger.warning(f"Callback failed with status {response.status_code}: {response.text}")
            
    except Exception as e:
        logger.error(f"Error sending callback: {str(e)}")


def poll_queue():
    """
    Poll the SQS queue for evaluation jobs.
    """
    if not QUEUE_URL:
        logger.error("Cannot poll queue: EVALUATION_QUEUE_URL not configured")
        return
    
    try:
        # Receive message from SQS queue
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            AttributeNames=["All"],
            MessageAttributeNames=["All"],
            MaxNumberOfMessages=1,
            VisibilityTimeout=MAX_PROCESSING_TIME,
            WaitTimeSeconds=20
        )
        
        messages = response.get("Messages", [])
        if not messages:
            logger.info("No messages in queue")
            return
        
        for message in messages:
            receipt_handle = message["ReceiptHandle"]
            
            try:
                # Parse message body
                body = json.loads(message["Body"])
                logger.info(f"Processing message: {body.get('job_id')}")
                
                # Process the job
                process_job(body)
                
                # Delete message from queue after successful processing
                sqs.delete_message(
                    QueueUrl=QUEUE_URL,
                    ReceiptHandle=receipt_handle
                )
                
                logger.info(f"Deleted message {receipt_handle} from queue")
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Message will return to the queue after visibility timeout
                # or we could explicitly change visibility timeout
                
    except Exception as e:
        logger.error(f"Error polling queue: {str(e)}")


# Lambda handler
def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Processing result
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Check if this is an SQS event
    if "Records" in event:
        for record in event["Records"]:
            if record.get("eventSource") == "aws:sqs":
                try:
                    body = json.loads(record["body"])
                    result = process_job(body)
                    logger.info(f"Processed job {body.get('job_id')}")
                    return {
                        "statusCode": 200,
                        "body": json.dumps({"message": "Job processed successfully", "result": result})
                    }
                except Exception as e:
                    logger.error(f"Error processing SQS message: {str(e)}")
                    logger.error(traceback.format_exc())
                    return {
                        "statusCode": 500,
                        "body": json.dumps({"error": str(e)})
                    }
    
    # If this is a direct invocation, check for a job_id
    if "job_id" in event:
        try:
            result = process_job(event)
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Job processed successfully", "result": result})
            }
        except Exception as e:
            logger.error(f"Error processing direct invocation: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }
    
    # Otherwise, poll the queue
    try:
        poll_queue()
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Queue polling completed"})
        }
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# Direct invocation for local testing
if __name__ == "__main__":
    poll_queue()