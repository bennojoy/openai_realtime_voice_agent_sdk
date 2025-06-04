# OpenAI Realtime WebSocket Python SDK

A Python SDK for interacting with OpenAI's Realtime WebSocket API, providing a simple interface for text and voice interactions.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Text Interaction

```python
from agents import Agent, Runner, Voice

# Create an agent with instructions
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant"
)

# Run a synchronous text interaction
result = Runner.run_sync(
    agent,
    "Write a haiku about programming",
    api_key="your-api-key"
)

print(result.text)
```

### Voice Interaction

```python
from agents import Agent, Runner, Voice
import soundfile as sf

# Create an agent with voice capabilities
agent = Agent(
    name="Voice Assistant",
    instructions="You are a helpful voice assistant",
    voice=Voice.SAGE  # Choose from available voices
)

# Read audio file
audio_data, _ = sf.read("input.wav", dtype='float32')
audio_bytes = (audio_data * 32767).astype('int16').tobytes()

# Run a synchronous voice interaction
result = Runner.run_sync(
    agent,
    audio_bytes,
    api_key="your-api-key"
)

# Save the response audio
if result.audio_chunks:
    with open("response.wav", "wb") as f:
        for chunk in result.audio_chunks:
            f.write(chunk)
```

### Streaming Response

```python
from agents import Agent, Runner, Voice

agent = Agent(
    name="Streaming Assistant",
    instructions="You are a helpful assistant"
)

# Get streaming response
response = Runner.run_stream(
    agent,
    "Tell me a story",
    api_key="your-api-key"
)

# Monitor the response as it comes in
while not response.is_done:
    if response.text:
        print(response.text, end="", flush=True)
    time.sleep(0.1)
```

### Using Tools

```python
from agents import Agent, Runner, Tool

# Define a tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
)

# Create an agent with the tool
agent = Agent(
    name="Weather Assistant",
    instructions="You are a helpful weather assistant",
    tools=[weather_tool]
)

# Run interaction
result = Runner.run_sync(
    agent,
    "What's the weather like in San Francisco?",
    api_key="your-api-key"
)

# Check for function calls
if result.function_calls:
    for call in result.function_calls:
        print(f"Function call: {call}")
```

## Available Voices

- ALLOY
- ASH
- BALLAD
- CORAL
- ECHO
- SAGE
- SHIMMER
- VERSE

## Features

- Simple interface for text and voice interactions
- Support for streaming responses
- Built-in voice activity detection (VAD)
- Function calling support
- Thread-safe response handling
- Automatic WebSocket connection management

## Configuration

### Session Configuration

The `SessionConfig` class allows you to configure various aspects of the session:

- `input_audio_format`: Audio format for input
- `output_audio_format`: Audio format for output
- `tools`: List of available functions
- `tool_choice`: Function calling behavior

### Response Configuration

The `ResponseConfig` class allows you to configure individual responses:

- `input_audio_format`: Audio format for this response
- `output_audio_format`: Audio format for this response
- `tools`: List of available functions for this response
- `tool_choice`: Function calling behavior for this response
- `conversation`: Conversation context
- `metadata`: Custom metadata
- `modalities`: List of modalities to use
- `instructions`: Custom instructions
- `input`: Custom input context

## Environment Variables

The SDK uses the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key

## License

MIT License 