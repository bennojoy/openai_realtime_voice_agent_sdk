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