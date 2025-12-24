#!/usr/bin/env python3
"""
Simple test script to verify the API is working correctly.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

# Set a test API key to avoid errors during import
os.environ.setdefault('OPENAI_API_KEY', 'test-key')

print("Testing API import...")

try:
    from backend.agent.main import app
    print("✓ Successfully imported the FastAPI application")

    # Test that we can access the routes
    routes = [route.path for route in app.routes]
    print(f"✓ Application has {len(routes)} routes: {routes}")

    # Try to access the API router to ensure all modules can be loaded
    from backend.agent.api import router
    print("✓ Successfully imported the API router")

    print("\n🎉 API structure is working correctly!")
    print("\nTo run the server, use: uvicorn backend.agent.main:app --reload --port 8000")
    print("Then test with curl or Postman at http://localhost:8000/health")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()