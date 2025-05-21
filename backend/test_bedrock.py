from app.bedrock_client import BedrockClient

client = BedrockClient()
models = client.list_models()
print(f"Found {len(models)} Bedrock models")
for i, model in enumerate(models[:5]):  # Print first 5 models only
    print(f"{i+1}. {model.get('modelId', 'Unknown')}")