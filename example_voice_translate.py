import pyaudio
import wave
import time
import logging
from agents import Agent, Voice
from runner import Runner
from audio_processor import AudioProcessor
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
    RATE = 24000  # Changed to 48kHz
    
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
    sd.play(audio_array, AudioProcessor.TARGET_SAMPLE_RATE)
    sd.wait()  # Wait until audio is finished playing
    
    logger.info("Audio playback finished")

def main():
    # Record audio at 48kHz
    record_audio(duration=5, output_file="input.wav")
    
    # Create voice + text agent
    voice_text_agent = Agent(
        name="French Translator with Text",
        instructions="You are a helpful assistant, respond to user in a friendly and helpful manner",
        voice=Voice.SAGE,
        enable_text=True  # Enable text modality
    )
    
    # Process the audio file and save the converted version if needed
    logger.info("Processing audio file for OpenAI...")
    input_size = os.path.getsize("input.wav")
    logger.info(f"Input file size: {input_size} bytes")
    
    processed_audio = AudioProcessor.process_audio_file("input.wav", voice_text_agent.input_audio)
    processed_size = len(processed_audio)
    logger.info(f"Processed audio size: {processed_size} bytes")
    
    # Only save if the audio was actually converted (accounting for small PCM16 conversion differences)
    size_diff = abs(processed_size - input_size)
    if size_diff > 100:  # Only consider it a conversion if size difference is significant
        logger.info(f"Audio was converted (size changed from {input_size} to {processed_size}), saving to converted_input.wav")
        try:
            with open("converted_input.wav", "wb") as f:
                logger.debug("Opening converted_input.wav for writing")
                f.write(processed_audio)
                logger.debug(f"Written {len(processed_audio)} bytes to converted_input.wav")
            logger.info("Successfully saved converted_input.wav")
        except Exception as e:
            logger.error(f"Failed to save converted_input.wav: {e}")
            raise
    else:
        logger.info(f"No significant conversion needed (size difference: {size_diff} bytes), using input.wav directly")
    
    # Run translation with voice+text agent using the processed audio
    logger.info("Running translation with voice+text agent...")
    result = Runner.run_sync(
        voice_text_agent,
        processed_audio,  # Use the processed audio directly instead of the file path
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