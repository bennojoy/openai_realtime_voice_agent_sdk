import pyaudio
import wave
import time
import logging
from agents import Agent, Voice, AudioConfig, AudioFormat
from runner import Runner
import sounddevice as sd
import numpy as np
import queue
import threading
import os
import base64
import struct

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to control audio streaming
is_streaming = True

class AudioPlayer:
    def __init__(self, sample_rate=24000, buffer_size_ms=1000):
        self.sample_rate = sample_rate
        self.buffer = []
        self.buffer_size = int(sample_rate * buffer_size_ms / 1000)  # Convert ms to samples
        self.is_playing = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        logger.info(f"Initialized AudioPlayer with {buffer_size_ms}ms buffer size")

    def play_chunk(self, audio_chunk: bytes):
        """Add an audio chunk to the buffer and start playback if needed"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            with self._lock:
                # Add to buffer
                self.buffer.append(audio_array)
                buffer_samples = sum(len(chunk) for chunk in self.buffer)
                
                # If not playing and buffer is full enough, start playback
                if not self.is_playing and buffer_samples >= self.buffer_size:
                    self.is_playing = True
                    # Start playback in a separate thread
                    threading.Thread(target=self._play_buffer, daemon=True).start()
                    logger.debug(f"Started playback thread with {buffer_samples} samples in buffer")
                    
        except Exception as e:
            logger.error(f"Error adding audio chunk to buffer: {e}")

    def _play_buffer(self):
        """Play audio from buffer in a separate thread"""
        try:
            while not self._stop_event.is_set():
                with self._lock:
                    if not self.buffer:
                        self.is_playing = False
                        logger.debug("Buffer empty, stopping playback")
                        break
                    
                    # Get next chunk to play
                    chunk = self.buffer.pop(0)
                
                # Play this chunk
                try:
                    sd.play(chunk, self.sample_rate)
                    sd.wait()
                except Exception as e:
                    logger.error(f"Error playing audio chunk: {e}")
                    
        except Exception as e:
            logger.error(f"Error in playback thread: {e}")
        finally:
            self.is_playing = False

    def stop(self):
        """Stop playback and clean up"""
        logger.info("Stopping audio player...")
        self._stop_event.set()
        
        with self._lock:
            self.buffer.clear()
            
        try:
            sd.stop()
        except Exception as e:
            logger.error(f"Error stopping audio playback: {e}")
            
        logger.info("Audio player stopped")

def stream_audio_chunks(runner, chunk_duration_ms=500):
    """Stream audio from microphone to OpenAI in fixed duration chunks, with responsive Ctrl+C handling."""
    global is_streaming
    TARGET_CHUNK_MS = chunk_duration_ms
    SUB_CHUNK_MS = 100  # Read in 100ms sub-chunks for responsiveness
    SUB_CHUNK = int(24000 * SUB_CHUNK_MS / 1000)
    NUM_SUB_CHUNKS = TARGET_CHUNK_MS // SUB_CHUNK_MS
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000

    p = pyaudio.PyAudio()
    stream = None

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=SUB_CHUNK)
        logger.info(f"Starting audio streaming in {chunk_duration_ms}ms chunks (read in {SUB_CHUNK_MS}ms sub-chunks)...")
        while True:
            try:
                if not is_streaming:
                    time.sleep(0.1)  # Sleep when not streaming to reduce CPU usage
                    continue
                    
                chunk_data = b''
                for _ in range(NUM_SUB_CHUNKS):
                    chunk_data += stream.read(SUB_CHUNK, exception_on_overflow=False)
                runner._send_audio_input(chunk_data, streaming=True)
                time.sleep(0.01)
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received in stream_audio_chunks")
                break
            except Exception as e:
                logger.error(f"Error reading audio: {e}")
                break
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during stream setup")
    finally:
        logger.info("Cleaning up audio stream...")
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
        try:
            p.terminate()
        except Exception as e:
            logger.error(f"Error terminating PyAudio: {e}")
        logger.info("Audio stream cleanup complete")

def main():
    global is_streaming
    # Initialize the voice agent
    agent = Agent(
        name="Voice Assistant",
        instructions="You are a helpful assistant. Respond to the user in a friendly and helpful manner.",
        model="gpt-4o-realtime-preview-2024-12-17",
        voice=Voice.ALLOY,
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
        tools=[],
        temperature=0.7,
        max_response_output_tokens=500,
        turn_detection={
            "type": "server_vad",
            "threshold": 0.9,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 2000,
            "create_response": True
        }
    )

    # Initialize audio player
    player = AudioPlayer()
    runner = None
    
    def handle_audio_delta(audio_delta: str):
        """Handle audio deltas"""
        logger.info("Audio delta handler called")
        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(audio_delta)
            # Play the audio chunk
            player.play_chunk(audio_data)
        except Exception as e:
            logger.error(f"Error playing audio chunk: {e}")
            
    def handle_user_transcript(transcript: str):
        """Handle user's speech transcription"""
        logger.info(f"User transcript handler called with: {transcript}")
        print(f"\nYou: {transcript}")
        
    def handle_ai_transcript(transcript: str):
        """Handle AI's speech transcription"""
        logger.info(f"AI transcript handler called with: {transcript}")
        print(f"\nAI (transcribed): {transcript}")

        
    def handle_speech_stopped():
        """Handle VAD detection of speech end"""
        global is_streaming
        logger.info("Speech stopped handler called")
        print("\nSpeech ended. Waiting for AI response...")
        # Stop audio streaming when speech ends
        is_streaming = False
        logger.info("Stopped audio streaming")
        
    def handle_response_done(response_data):
        """Handle completion of AI response"""
        logger.info(f"Response done handler called with data: {response_data}")
        print("\nAI response complete. You can speak again or press Ctrl+C to exit.")
        
    def handle_audio_transcript_done():
        """Handle completion of AI audio transcript"""

        logger.info("Audio transcript done handler called")
        print("\nAudio transcript complete. You can speak again or press Ctrl+C to exit.")
        # Resume audio streaming when AI's response is complete
        is_streaming = True
    
    try:
        # Create runner and start streaming
        logger.info("Initializing runner...")
        runner = Runner(api_key=os.getenv("OPENAI_API_KEY"))
        runner.agent = agent
        runner._is_streaming = True
        runner.on_audio_delta = handle_audio_delta
        runner.on_transcript_user = handle_user_transcript
        runner.on_transcript_ai = handle_ai_transcript
        runner.on_speech_stopped = handle_speech_stopped
        runner.on_response_done = handle_response_done
        runner.on_audio_transcript_done = handle_audio_transcript_done
        logger.info("All handlers registered")
        
        # Connect and wait for session creation
        logger.info("Connecting to websocket...")
        runner._connect()
        logger.info("Updating session configuration...")
        runner._update_session(agent, is_streaming=True)
        try:
            runner._session_updated.wait(timeout=10)
            logger.info("Session configuration updated successfully")
        except Exception as e:
            logger.error(f"Failed to update session configuration: {e}")
            raise

        print("Starting voice interaction... (Press Ctrl+C to stop)")
        print("Speak into your microphone...")
        
        # Start streaming audio in 500ms chunks
        stream_audio_chunks(runner, chunk_duration_ms=500)
                
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main try block")
        print("\nStopping voice interaction...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        # Clean up
        logger.info("Starting cleanup...")
        try:
            # First stop the audio player
            player.stop()
        except Exception as e:
            logger.error(f"Error stopping audio player: {e}")
            
        # Then disconnect the websocket
        if runner is not None:
            try:
                logger.info("Disconnecting websocket...")
                runner._disconnect()  # Ensure websocket is closed
            except Exception as e:
                logger.error(f"Error disconnecting websocket: {e}")
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main() 