import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set mock values directly in environment variables
os.environ['JOBS_TABLE_NAME'] = 'mock-jobs-table'
os.environ['HISTORY_TABLE_NAME'] = 'mock-history-table' 
os.environ['EVALUATION_QUEUE_URL'] = 'https://sqs.us-east-1.amazonaws.com/123456789012/mock-queue'
os.environ['AWS_REGION'] = 'us-east-1'

# Import the API with our mock environment variables
try:
    from app.api import app
    
    if __name__ == "__main__":
        print("\n=== Mock API Server ===")
        print("Starting API at http://localhost:8000")
        print("Note: AWS services are MOCKED - only read-only endpoints will work")
        print("Endpoints like /models and /metrics should work")
        print("Endpoints that write to DynamoDB or SQS will return errors\n")
        uvicorn.run(app, host="0.0.0.0", port=8000)
except Exception as e:
    print(f"Error initializing API: {e}")