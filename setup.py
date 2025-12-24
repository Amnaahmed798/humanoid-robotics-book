from setuptools import setup, find_packages

setup(
    name="humanoid-robotics-book-backend",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.103.1",
        "uvicorn==0.23.2",
        "qdrant-client==1.9.1",
        "cohere==5.5.3",
        "requests==2.31.0",
        "beautifulsoup4==4.12.2",
        "python-dotenv==1.0.0",
        "openai>=1.0.0",
        "pytest==7.4.2",
    ],
    python_requires=">=3.8",
)