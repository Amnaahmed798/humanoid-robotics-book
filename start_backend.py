#!/usr/bin/env python3
"""
Script to start the backend server with proper environment configuration.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv('./backend/.env')

# Add backend to path
sys.path.insert(0, './backend')

# Now start the server
if __name__ == "__main__":
    import uvicorn

    # Set environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv('./backend/.env')

    print("Starting backend server...")
    print(f"QDRANT_URL: {os.environ.get('QDRANT_URL')}")
    print(f"COHERE_API_KEY is {'set' if os.environ.get('COHERE_API_KEY') else 'not set'}")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to avoid file locking issues
    )