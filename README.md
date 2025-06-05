# OpenAI Realtime WebSocket Python SDK

This SDK provides a Python interface for interacting with OpenAI's Realtime API using WebSockets. It supports both streaming and non-streaming modes for voice and text interactions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bennojoy/openai-realtime-websocket-python-sdk.git
   cd openai-realtime-websocket-python-sdk
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## System Requirement: FFmpeg

This SDK requires [FFmpeg](https://ffmpeg.org/) to be installed on your system for audio processing.

### Install FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
- Download the latest static build from [ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Extract the files and add the `bin` folder to your system `PATH`.

## Usage

### Non-Streaming Example

The `voice_non_streaming_example.py` demonstrates how to record audio, send it to OpenAI, and play the response:

```python
import pyaudio
import wave
import time
import logging
from agents import Agent, Voice, AudioConfig, AudioFormat
from runner import Runner
import sounddevice as sd
import numpy as np
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def record_audio(duration=5, output_file="input.wav"):
    """Record audio for specified duration and save to WAV file."""
    logger.info(f"Recording {duration} seconds of audio...")
    
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000  # Match OpenAI's sample rate
    
    p = pyaudio.PyAudio()
    
    # Open audio stream
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    logger.info(f"Recording started at {RATE}Hz...")
    frames = []
    
    # Record audio
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    logger.info("Recording finished")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    logger.info(f"Audio saved to {output_file} at {RATE}Hz")

def play_audio(audio_chunks):
    """Play audio chunks using sounddevice."""
    logger.info("Playing response audio...")
    
    # Combine all chunks
    audio_data = b''.join(audio_chunks)
    
    # Convert to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Play audio
    sd.play(audio_array, 24000)  # Using OpenAI's sample rate
    sd.wait()  # Wait until audio is finished playing
    
    logger.info("Audio playback finished")

def main():
    # Record audio
    record_audio(duration=5, output_file="input.wav")
    
    # Create voice + text agent
    voice_text_agent = Agent(
        name="Voice Assistant",
        instructions="You are a helpful assistant, respond to user in a friendly and helpful manner",
        voice=Voice.SAGE,
        input_audio=AudioConfig(
            format=AudioFormat.PCM16,
            sample_rate=24000,
            channels=1
        ),
        output_audio=AudioConfig(
            format=AudioFormat.PCM16,
            sample_rate=24000,
            channels=1
        ),
        enable_text=True  # Enable text modality
    )
    
    def response_done_handler(response):
        logger.info("Response done handler called")
        logger.info(f"Response text: {response.text}")
        if response.audio_chunks:
            play_audio(response.audio_chunks)
        else:
            logger.warning("No audio response received")
    
    # Create runner
    runner = Runner(voice_text_agent, is_streaming=False,
                    on_response_done=response_done_handler)
    
    # Run the conversation
    logger.info("Starting conversation...")
    runner.init()
    
    try:
        # Send the recorded audio file
        with open("input.wav", "rb") as f:
            audio_data = f.read()
        runner._send_audio_input(audio_data)
        
        # Wait for the response to complete
        time.sleep(10)  # Give enough time for the response to complete
                
    except KeyboardInterrupt:
        logger.info("Stopping conversation...")
    finally:
        runner._disconnect()
        # Clean up the temporary file
        try:
            os.remove("input.wav")
        except:
            pass

if __name__ == "__main__":
    main()
```

### Streaming Example

The `voice_streaming_example.py` demonstrates how to stream audio in real-time:

```python
import pyaudio
import wave
import time
import logging
from agents import Agent, Voice, AudioConfig, AudioFormat
from runner import Runner
import sounddevice as sd
import numpy as np
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def record_audio(duration=5, output_file="input.wav"):
    """Record audio for specified duration and save to WAV file."""
    logger.info(f"Recording {duration} seconds of audio...")
    
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000  # Match OpenAI's sample rate
    
    p = pyaudio.PyAudio()
    
    # Open audio stream
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    logger.info(f"Recording started at {RATE}Hz...")
    frames = []
    
    # Record audio
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    logger.info("Recording finished")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    logger.info(f"Audio saved to {output_file} at {RATE}Hz")

def play_audio(audio_chunks):
    """Play audio chunks using sounddevice."""
    logger.info("Playing response audio...")
    
    # Combine all chunks
    audio_data = b''.join(audio_chunks)
    
    # Convert to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Play audio
    sd.play(audio_array, 24000)  # Using OpenAI's sample rate
    sd.wait()  # Wait until audio is finished playing
    
    logger.info("Audio playback finished")

def main():
    # Record audio
    record_audio(duration=5, output_file="input.wav")
    
    # Create voice + text agent
    voice_text_agent = Agent(
        name="Voice Assistant",
        instructions="You are a helpful assistant, respond to user in a friendly and helpful manner",
        voice=Voice.SAGE,
        input_audio=AudioConfig(
            format=AudioFormat.PCM16,
            sample_rate=24000,
            channels=1
        ),
        output_audio=AudioConfig(
            format=AudioFormat.PCM16,
            sample_rate=24000,
            channels=1
        ),
        enable_text=True  # Enable text modality
    )
    
    def response_done_handler(response):
        logger.info("Response done handler called")
        logger.info(f"Response text: {response.text}")
        if response.audio_chunks:
            play_audio(response.audio_chunks)
        else:
            logger.warning("No audio response received")
    
    # Create runner
    runner = Runner(voice_text_agent, is_streaming=True,
                    on_response_done=response_done_handler)
    
    # Run the conversation
    logger.info("Starting conversation...")
    runner.init()
    
    try:
        # Send the recorded audio file
        with open("input.wav", "rb") as f:
            audio_data = f.read()
        runner._send_audio_input(audio_data)
        
        # Wait for the response to complete
        time.sleep(10)  # Give enough time for the response to complete
                
    except KeyboardInterrupt:
        logger.info("Stopping conversation...")
    finally:
        runner._disconnect()
        # Clean up the temporary file
        try:
            os.remove("input.wav")
        except:
            pass

if __name__ == "__main__":
    main()
```

## Key Features

- **Non-Streaming Mode**: Record audio, send it to OpenAI, and play the response.
- **Streaming Mode**: Stream audio in real-time for interactive conversations.
- **Audio Processing**: Convert audio to the required format (PCM16, 24000Hz, mono).
- **Response Handling**: Use callbacks to handle text and audio responses.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
