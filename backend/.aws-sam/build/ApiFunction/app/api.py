"""
API endpoints for the LLM Response Evaluator project.
This module defines FastAPI routes for accessing Bedrock models and evaluating responses.
It implements an asynchronous architecture with SQS for handling LLM evaluation requests.
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Query, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

from .bedrock_client import BedrockClient
from .evaluator import ResponseEvaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="LLM Response Evaluator API",
    description="API for evaluating responses from AWS Bedrock models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AWS clients
sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')
jobs_table = None
history_table = None

# Get environment variables
QUEUE_URL = os.environ.get('EVALUATION_QUEUE_URL')
JOBS_TABLE_NAME = os.environ.get('JOBS_TABLE_NAME')
HISTORY_TABLE_NAME = os.environ.get('HISTORY_TABLE_NAME')

# Initialize DynamoDB tables if configured
if JOBS_TABLE_NAME:
    try:
        jobs_table = dynamodb.Table(JOBS_TABLE_NAME)
        logger.info(f"Connected to DynamoDB jobs table: {JOBS_TABLE_NAME}")
    except Exception as e:
        logger.error(f"Error connecting to DynamoDB jobs table: {str(e)}")

if HISTORY_TABLE_NAME:
    try:
        history_table = dynamodb.Table(HISTORY_TABLE_NAME)
        logger.info(f"Connected to DynamoDB history table: {HISTORY_TABLE_NAME}")
    except Exception as e:
        logger.error(f"Error connecting to DynamoDB history table: {str(e)}")


# Define job status constants
class JobStatus:
    SUBMITTED = "SUBMITTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Define Pydantic models for request/response validation

class ModelParameters(BaseModel):
    """Parameters for model invocation."""
    temperature: Optional[float] = Field(0.7, ge=0, le=1, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(1024, gt=0, description="Maximum number of tokens to generate")
    top_p: Optional[float] = Field(0.9, ge=0, le=1, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(None, gt=0, description="Top-k sampling parameter")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences that stop generation")


class EvaluateRequest(BaseModel):
    """Request model for the evaluate endpoint."""
    prompt: str = Field(..., description="The prompt to send to the models")
    model_ids: List[str] = Field(..., min_items=1, description="List of model IDs to evaluate")
    parameters: Optional[Dict[str, ModelParameters]] = Field(None, description="Model-specific parameters")
    metrics: Optional[List[str]] = Field(None, description="Specific metrics to evaluate")
    save_history: Optional[bool] = Field(True, description="Whether to save the evaluation to history")
    callback_url: Optional[str] = Field(None, description="URL to call when processing completes")


class MetricInfo(BaseModel):
    """Information about an evaluation metric."""
    name: str
    description: str
    category: str
    type: str = "numeric"


class ModelsResponse(BaseModel):
    """Response model for the models endpoint."""
    count: int
    models: List[Dict[str, Any]]


class MetricsResponse(BaseModel):
    """Response model for the metrics endpoint."""
    count: int
    metrics: List[MetricInfo]


class SubmitJobResponse(BaseModel):
    """Response model for job submission."""
    job_id: str
    status: str
    timestamp: str
    model_ids: List[str]
    prompt: str
    estimated_completion_time: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str
    created_at: str
    updated_at: str
    progress: Optional[float] = None
    error: Optional[str] = None
    model_ids: List[str]
    estimated_completion_time: Optional[str] = None


class JobResultResponse(BaseModel):
    """Response model for job results."""
    job_id: str
    status: str
    prompt: str
    timestamp: str
    model_responses: Optional[Dict[str, str]] = None
    evaluations: Optional[Dict[str, Any]] = None
    comparisons: Optional[Dict[str, Any]] = None
    ranking: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HistoryResponse(BaseModel):
    """Response model for the history endpoint."""
    count: int
    evaluations: List[Dict[str, Any]]


# Dependency for getting clients
def get_bedrock_client():
    """Dependency for getting a BedrockClient instance."""
    region = os.environ.get('AWS_REGION', 'us-east-1')
    return BedrockClient(region_name=region)


def get_evaluator():
    """Dependency for getting a ResponseEvaluator instance."""
    return ResponseEvaluator()


# Define API routes

@app.get("/models", response_model=ModelsResponse)
async def get_models(bedrock_client: BedrockClient = Depends(get_bedrock_client)):
    """
    Get a list of available AWS Bedrock models.
    
    Returns:
        List of model information
    """
    try:
        models = bedrock_client.list_models()
        return {
            "count": len(models),
            "models": models
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(evaluator: ResponseEvaluator = Depends(get_evaluator)):
    """
    Get a list of available evaluation metrics.
    
    Returns:
        List of metric information
    """
    # Define available metrics and their categories
    metrics = [
        # Basic text metrics
        MetricInfo(name="char_count", description="Number of characters", category="basic"),
        MetricInfo(name="word_count", description="Number of words", category="basic"),
        MetricInfo(name="sentence_count", description="Number of sentences", category="basic"),
        MetricInfo(name="avg_word_length", description="Average word length", category="basic"),
        MetricInfo(name="avg_sentence_length", description="Average sentence length", category="basic"),
        
        # Readability metrics
        MetricInfo(name="readability.flesch_reading_ease", description="Flesch Reading Ease score", category="readability"),
        MetricInfo(name="readability.flesch_kincaid_grade", description="Flesch-Kincaid Grade Level", category="readability"),
        MetricInfo(name="readability.gunning_fog", description="Gunning Fog Index", category="readability"),
        MetricInfo(name="readability.smog_index", description="SMOG Index", category="readability"),
        MetricInfo(name="readability.automated_readability_index", description="Automated Readability Index", category="readability"),
        MetricInfo(name="readability.coleman_liau_index", description="Coleman-Liau Index", category="readability"),
        MetricInfo(name="readability.dale_chall_readability_score", description="Dale-Chall Readability Score", category="readability"),
        MetricInfo(name="readability.difficult_words", description="Number of difficult words", category="readability"),
        MetricInfo(name="readability.linsear_write_formula", description="Linsear Write Formula", category="readability"),
        MetricInfo(name="readability.text_standard", description="Text Standard", category="readability", type="string"),
        
        # Complexity metrics
        MetricInfo(name="complexity.lexical_diversity", description="Lexical diversity (unique words / total words)", category="complexity"),
        MetricInfo(name="complexity.long_word_ratio", description="Ratio of long words (>6 chars)", category="complexity"),
        MetricInfo(name="complexity.unique_word_ratio", description="Ratio of unique words", category="complexity"),
        MetricInfo(name="complexity.sentence_complexity", description="Average words per sentence", category="complexity"),
        
        # Content metrics
        MetricInfo(name="content.sentiment.compound", description="Overall sentiment score (-1 to 1)", category="content"),
        MetricInfo(name="content.sentiment.pos", description="Positive sentiment score (0 to 1)", category="content"),
        MetricInfo(name="content.sentiment.neu", description="Neutral sentiment score (0 to 1)", category="content"),
        MetricInfo(name="content.sentiment.neg", description="Negative sentiment score (0 to 1)", category="content"),
        MetricInfo(name="content.subjectivity", description="Subjectivity score (0 to 1)", category="content"),
        MetricInfo(name="content.polarity", description="Polarity score (-1 to 1)", category="content"),
        MetricInfo(name="content.question_count", description="Number of questions", category="content"),
        MetricInfo(name="content.code_block_count", description="Number of code blocks", category="content"),
        MetricInfo(name="content.list_count", description="Number of lists", category="content"),
        MetricInfo(name="content.url_count", description="Number of URLs", category="content"),
        MetricInfo(name="content.has_numbers", description="Whether the text contains numbers", category="content", type="boolean"),
        MetricInfo(name="content.number_count", description="Number of numeric values", category="content"),
    ]
    
    return {
        "count": len(metrics),
        "metrics": metrics
    }


@app.post("/evaluate", response_model=SubmitJobResponse)
async def submit_evaluation_job(request: EvaluateRequest):
    """
    Submit a job to evaluate LLM responses asynchronously.
    
    Args:
        request: The evaluation request parameters
        
    Returns:
        Job submission information with job ID
    """
    # Validate SQS and DynamoDB are configured
    if not QUEUE_URL:
        raise HTTPException(
            status_code=501,
            detail="SQS queue not configured. Set EVALUATION_QUEUE_URL environment variable."
        )
    
    if not jobs_table:
        raise HTTPException(
            status_code=501,
            detail="Jobs table not configured. Set JOBS_TABLE_NAME environment variable."
        )
    
    try:
        # Generate a job ID
        job_id = f"job-{uuid.uuid4()}"
        timestamp = datetime.utcnow().isoformat()
        
        # Format model parameters
        formatted_parameters = {}
        if request.parameters:
            for model_id, params in request.parameters.items():
                formatted_parameters[model_id] = params.dict(exclude_none=True)
        
        # Create the job message
        job_message = {
            "job_id": job_id,
            "prompt": request.prompt,
            "model_ids": request.model_ids,
            "parameters": formatted_parameters,
            "metrics": request.metrics,
            "save_history": request.save_history,
            "callback_url": request.callback_url,
            "timestamp": timestamp
        }
        
        # Store job in DynamoDB
        job_item = {
            "job_id": job_id,
            "status": JobStatus.SUBMITTED,
            "created_at": timestamp,
            "updated_at": timestamp,
            "prompt": request.prompt,
            "model_ids": request.model_ids,
            "request": json.dumps(job_message)
        }
        
        # Estimate completion time (rough estimate: 5 seconds per model + 2 seconds overhead)
        estimated_seconds = len(request.model_ids) * 5 + 2
        estimated_completion = datetime.utcnow()
        estimated_completion = estimated_completion.timestamp() + estimated_seconds
        estimated_completion = datetime.fromtimestamp(estimated_completion).isoformat()
        job_item["estimated_completion_time"] = estimated_completion
        
        # Save to DynamoDB
        jobs_table.put_item(Item=job_item)
        
        # Send to SQS queue
        sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(job_message),
            MessageAttributes={
                'JobType': {
                    'DataType': 'String',
                    'StringValue': 'LLMEvaluation'
                }
            }
        )
        
        logger.info(f"Submitted evaluation job {job_id} to queue")
        
        # Return job information
        return {
            "job_id": job_id,
            "status": JobStatus.SUBMITTED,
            "timestamp": timestamp,
            "model_ids": request.model_ids,
            "prompt": request.prompt,
            "estimated_completion_time": estimated_completion
        }
        
    except ClientError as e:
        logger.error(f"AWS error submitting job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AWS error: {str(e)}")
    except Exception as e:
        logger.error(f"Error submitting evaluation job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of an evaluation job.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        Current job status information
    """
    if not jobs_table:
        raise HTTPException(
            status_code=501,
            detail="Jobs table not configured. Set JOBS_TABLE_NAME environment variable."
        )
    
    try:
        # Get job from DynamoDB
        response = jobs_table.get_item(
            Key={"job_id": job_id}
        )
        
        job = response.get("Item")
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Prepare and return the job status
        status_response = {
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "model_ids": job["model_ids"]
        }
        
        # Add optional fields if present
        if "progress" in job:
            status_response["progress"] = job["progress"]
        
        if "error" in job:
            status_response["error"] = job["error"]
            
        if "estimated_completion_time" in job:
            status_response["estimated_completion_time"] = job["estimated_completion_time"]
        
        return status_response
        
    except ClientError as e:
        logger.error(f"AWS error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AWS error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str):
    """
    Get the results of a completed evaluation job.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        Evaluation results
    """
    if not jobs_table:
        raise HTTPException(
            status_code=501,
            detail="Jobs table not configured. Set JOBS_TABLE_NAME environment variable."
        )
    
    try:
        # Get job from DynamoDB
        response = jobs_table.get_item(
            Key={"job_id": job_id}
        )
        
        job = response.get("Item")
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Check if job is completed
        if job["status"] != JobStatus.COMPLETED and job["status"] != JobStatus.FAILED:
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "prompt": job["prompt"],
                "timestamp": job["updated_at"]
            }
        
        # For failed jobs, return the error
        if job["status"] == JobStatus.FAILED:
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "prompt": job["prompt"],
                "timestamp": job["updated_at"],
                "error": job.get("error", "Unknown error")
            }
        
        # For completed jobs, return the results
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "prompt": job["prompt"],
            "timestamp": job["updated_at"],
            "model_responses": job.get("model_responses", {}),
            "evaluations": job.get("evaluations", {}),
            "comparisons": job.get("comparisons", {}),
            "ranking": job.get("ranking", {})
        }
        
    except ClientError as e:
        logger.error(f"AWS error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AWS error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    """
    Cancel a pending or in-progress job.
    
    Args:
        job_id: The unique job identifier
        
    Returns:
        No content on success
    """
    if not jobs_table:
        raise HTTPException(
            status_code=501,
            detail="Jobs table not configured. Set JOBS_TABLE_NAME environment variable."
        )
    
    try:
        # Get job from DynamoDB
        response = jobs_table.get_item(
            Key={"job_id": job_id}
        )
        
        job = response.get("Item")
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Check if job can be canceled
        if job["status"] == JobStatus.COMPLETED or job["status"] == JobStatus.FAILED:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel job with status {job['status']}"
            )
        
        # Update job status to FAILED with cancellation reason
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #status = :status, #error = :error, #updated_at = :updated_at",
            ExpressionAttributeNames={
                "#status": "status",
                "#error": "error",
                "#updated_at": "updated_at"
            },
            ExpressionAttributeValues={
                ":status": JobStatus.FAILED,
                ":error": "Job canceled by user",
                ":updated_at": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(f"Canceled job {job_id}")
        return None
        
    except ClientError as e:
        logger.error(f"AWS error canceling job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AWS error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error canceling job: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(10, gt=0, le=100),
    start_date: Optional[str] = Query(None, description="Start date filter (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date filter (ISO format)")
):
    """
    Get evaluation history records.
    
    Args:
        limit: Maximum number of records to return (default: 10)
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
        
    Returns:
        List of historical evaluation records
    """
    if not history_table:
        raise HTTPException(
            status_code=501, 
            detail="History table not configured. Set HISTORY_TABLE_NAME environment variable."
        )
    
    try:
        # Construct filter expression if dates are provided
        filter_expression = None
        expression_values = {}
        
        if start_date and end_date:
            filter_expression = "timestamp BETWEEN :start AND :end"
            expression_values = {
                ":start": start_date,
                ":end": end_date
            }
        elif start_date:
            filter_expression = "timestamp >= :start"
            expression_values = {":start": start_date}
        elif end_date:
            filter_expression = "timestamp <= :end"
            expression_values = {":end": end_date}
        
        # Query DynamoDB
        scan_params = {
            "Limit": limit
        }
        
        if filter_expression:
            scan_params["FilterExpression"] = filter_expression
            scan_params["ExpressionAttributeValues"] = expression_values
        
        response = history_table.scan(**scan_params)
        items = response.get("Items", [])
        
        # Sort by timestamp (newest first)
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return {
            "count": len(items),
            "evaluations": items
        }
    
    except Exception as e:
        logger.error(f"Error retrieving evaluation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        Status message
    """
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "sqs": "configured" if QUEUE_URL else "not configured",
            "jobs_table": "configured" if jobs_table else "not configured",
            "history_table": "configured" if history_table else "not configured"
        }
    }


# Webhook endpoint for processor service to update job status (optional)
@app.post("/webhook/job-update", status_code=204)
async def job_update_webhook(job_update: Dict[str, Any] = Body(...)):
    """
    Webhook endpoint for the processor service to update job status.
    This is useful when the processor is in a separate service/container.
    
    Args:
        job_update: Job update information
        
    Returns:
        No content on success
    """
    if not jobs_table:
        raise HTTPException(
            status_code=501,
            detail="Jobs table not configured. Set JOBS_TABLE_NAME environment variable."
        )
    
    try:
        # Validate update data
        job_id = job_update.get("job_id")
        if not job_id:
            raise HTTPException(status_code=400, detail="Missing job_id in update")
        
        status = job_update.get("status")
        if not status:
            raise HTTPException(status_code=400, detail="Missing status in update")
        
        # Get current job
        response = jobs_table.get_item(
            Key={"job_id": job_id}
        )
        
        job = response.get("Item")
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        # Update expressions for DynamoDB
        update_expression = "SET #status = :status, #updated_at = :updated_at"
        expression_attr_names = {
            "#status": "status",
            "#updated_at": "updated_at"
        }
        expression_attr_values = {
            ":status": status,
            ":updated_at": datetime.utcnow().isoformat()
        }
        
        # Add other fields to update
        for key, value in job_update.items():
            if key not in ["job_id", "status"]:
                update_expression += f", #{key} = :{key}"
                expression_attr_names[f"#{key}"] = key
                expression_attr_values[f":{key}"] = value
        
        # Update job in DynamoDB
        jobs_table.update_item(
            Key={"job_id": job_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attr_names,
            ExpressionAttributeValues=expression_attr_values
        )
        
        logger.info(f"Updated job {job_id} status to {status}")
        return None
        
    except ClientError as e:
        logger.error(f"AWS error updating job status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AWS error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating job status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mangum handler for AWS Lambda
handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)