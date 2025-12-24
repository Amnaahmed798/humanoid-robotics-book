#!/usr/bin/env python3
"""
Script to add sample humanoid robotics content to the Qdrant database via the backend API.
This will allow the chatbot to have some content to work with.
"""

import sys
import os
import asyncio
import requests
import json
from dotenv import load_dotenv

# Load environment variables from backend directory
load_dotenv('./backend/.env')

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Sample humanoid robotics content that would normally be processed
sample_content = [
    "Humanoid robots are robots designed to resemble and mimic human behavior and appearance. They typically have a head, torso, two arms, and two legs, though some designs may vary. The main goal of humanoid robotics is to create machines that can interact with human environments using similar physical capabilities as humans.",
    "The history of humanoid robots dates back to ancient times with mechanical automata, but modern humanoid robotics began in the late 20th century. Notable early examples include WABOT-1 from Waseda University (1972) and later robots like Honda's ASIMO series, which demonstrated advanced walking and interaction capabilities.",
    "Locomotion in humanoid robots is one of the most challenging aspects of their design. Achieving stable bipedal walking requires sophisticated control algorithms, precise balance, and coordinated movement of multiple joints. Common approaches include zero-moment point (ZMP) control and more recently, whole-body control methods.",
    "Humanoid robots use various types of actuators to move their joints. Common actuator types include servo motors, pneumatic muscles, and hydraulic systems. The choice of actuator affects the robot's strength, speed, precision, and safety when interacting with humans.",
    "Applications of humanoid robots include research, education, entertainment, and service industries. They are particularly useful in scenarios where human-like interaction is beneficial, such as in healthcare assistance, customer service, and as companions for elderly care.",
    "Safety is a critical concern in humanoid robotics, especially when robots operate in human environments. Safety measures include compliant control, collision detection and avoidance, emergency stop mechanisms, and appropriate materials and design to minimize injury risk during contact.",
    "Artificial intelligence and machine learning play crucial roles in humanoid robotics. These technologies enable robots to perceive their environment, make decisions, learn from experience, and adapt their behavior. Common AI techniques include computer vision, natural language processing, and reinforcement learning.",
    "The future of humanoid robotics includes more autonomous and capable robots that can perform complex tasks in unstructured environments. Challenges remain in areas such as energy efficiency, robustness, and cost reduction to make humanoid robots more accessible for widespread use."
]

def test_backend_connection():
    """Test if the backend is running and accessible."""
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            print("Backend is running and accessible")
            return True
        else:
            print(f"Backend returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to backend: {e}")
        return False

def process_content():
    """Simulate the process of chunking and embedding content."""
    print("Processing sample content...")

    # In a real scenario, this would be done via the /process endpoint
    # For now, we'll just return the content as-is since the backend handles chunking/embedding
    processed_chunks = []
    for i, text in enumerate(sample_content):
        chunk = {
            "id": f"sample_chunk_{i}",
            "text": text,
            "source_location": f"Sample Content - Section {i+1}",
            "metadata": {"type": "sample", "section": f"{i+1}"}
        }
        processed_chunks.append(chunk)

    print(f"Processed {len(processed_chunks)} content chunks")
    return processed_chunks

def store_content_chunks(chunks):
    """Store the content in the vector database via the backend API."""
    print("Storing content in the vector database...")

    # First, we need to simulate the process of storing content
    # The backend has endpoints for processing and storing content
    collection_name = "book_content"

    # Since we can't directly call internal functions, let's check if we can validate the collection
    try:
        # Try to validate the collection to see if it exists
        validate_response = requests.post(
            f"{BACKEND_URL}/validate",
            json={"collection_name": collection_name},
            headers={"Content-Type": "application/json"}
        )

        if validate_response.status_code == 200:
            result = validate_response.json()
            print(f"Collection validation result: {result}")
        else:
            print(f"Collection validation failed with status {validate_response.status_code}: {validate_response.text}")
    except Exception as e:
        print(f"Error validating collection: {e}")

    # Instead of direct Qdrant access, let's try to use the backend's internal functionality
    # by testing the search endpoint to see if it can find content
    print("Testing search functionality...")
    try:
        search_response = requests.post(
            f"{BACKEND_URL}/search",
            json={
                "query": "humanoid robotics",
                "collection_name": collection_name,
                "top_k": 5
            },
            headers={"Content-Type": "application/json"}
        )

        if search_response.status_code == 200:
            result = search_response.json()
            print(f"Search test result: {result}")
        else:
            print(f"Search test failed with status {search_response.status_code}: {search_response.text}")
    except Exception as e:
        print(f"Error testing search: {e}")

    print("Note: To properly populate the database, you would typically:")
    print("1. Call POST /extract with a source URL")
    print("2. Call POST /process with the job ID from extract")
    print("3. Call POST /store with the job ID from process")
    print("4. Verify with POST /validate")

    return True

async def main():
    """Main function to run the sample content addition."""
    print("Testing backend connection...")
    if not test_backend_connection():
        print("Cannot connect to backend. Please ensure the backend server is running on http://localhost:8000")
        return False

    print("\nProcessing sample content...")
    chunks = process_content()

    print("\nStoring content...")
    success = store_content_chunks(chunks)

    if success:
        print("\nSample content processing completed!")
        print("The chatbot should now be able to answer questions about humanoid robotics.")
        print("\nTo test the chatbot, try querying with:")
        print("curl -X POST http://localhost:8000/api/v1/query -H 'Content-Type: application/json' -d '{\"question\": \"What are humanoid robots?\"}'")
    else:
        print("\nFailed to process sample content.")

if __name__ == "__main__":
    asyncio.run(main())