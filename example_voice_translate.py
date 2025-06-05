import pyaudio
import wave
import time
import logging
from agents import Agent, Voice
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
    """
    Record audio for specified duration and save to WAV file.
    
    Args:
        duration: Recording duration in seconds
        output_file: Path to save the WAV file
    """
    logger.info(f"Recording {duration} seconds of audio...")
    
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    
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
    """
    Play audio chunks using sounddevice.
    
    Args:
        audio_chunks: List of audio chunks to play
    """
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
        name="French Translator with Text",
        instructions="You are a helpful assistant, respond to user in a friendly and helpful manner",
        voice=Voice.SAGE,
        enable_text=True  # Enable text modality
    )
    
    # Run translation with voice+text agent
    logger.info("Running translation with voice+text agent...")
    result = Runner.run_sync(
        voice_text_agent,
        "input.wav",  # Pass the file path directly
        api_key=None  # Will use OPENAI_API_KEY environment variable
    )
    
    # Print the transcription and translation
    logger.info(f"Transcription: {result.text}")
    
    # Play the response audio
    if result.audio_chunks:
        play_audio(result.audio_chunks)
    else:
        logger.warning("No audio response received")

if __name__ == "__main__":
    main() 