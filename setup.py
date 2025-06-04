from setuptools import setup, find_packages

setup(
    name="openairealtimewebsocket",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "websockets>=12.0",
        "python-dotenv>=1.0.1",
        "pydantic>=2.6.1",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python SDK for OpenAI's Realtime API using WebSocket",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openairealtimewebsocket",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 