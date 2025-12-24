#!/usr/bin/env python3
"""
Minimal backend startup script that delays Qdrant initialization to avoid locking issues.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables from backend/.env
load_dotenv('./backend/.env')

# Add backend to path
sys.path.insert(0, './backend')

def start_backend():
    """Start the backend server with delayed service initialization."""
    import uvicorn

    # Set environment variables from .env file
    load_dotenv('./backend/.env')

    print("Starting backend server...")
    print(f"QDRANT_URL: {os.environ.get('QDRANT_URL')}")
    print(f"COHERE_API_KEY is {'set' if os.environ.get('COHERE_API_KEY') else 'not set'}")

    # Import and start the app after environment is set
    from backend.main import app

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to avoid file locking issues
    )

if __name__ == "__main__":
    start_backend()