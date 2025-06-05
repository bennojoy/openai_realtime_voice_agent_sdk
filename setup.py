from setuptools import setup, find_packages

setup(
    name="openai_realtime_voice_agent_sdk",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "websockets",
        "pyaudio",
        "sounddevice",
        "numpy",
    ],
    author="Benno Joy",
    author_email="your.email@example.com",
    description="A Python SDK for interacting with OpenAI's Realtime API using WebSockets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/openai-realtime-websocket-python-sdk",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 