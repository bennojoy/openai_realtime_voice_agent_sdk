import logging
import threading
import queue
import time
from typing import Optional, Callable, Union, List, Dict, Any
from dataclasses import dataclass, field
from agents import Agent
from agents import AudioFormat
import json
import websocket
import base64
from audio_processor import AudioProcessor
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Response:
    """Represents a response from the agent."""
    text: str = ""
    audio_chunks: List[bytes] = field(default_factory=list)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    is_done: bool = False

class Runner:
    """
    Handles interactions with the agent through WebSocket connection.
    
    This class manages the WebSocket connection, audio processing, and response handling
    for voice interactions with the agent.
    """
    
    def __init__(self, 
                 agent: Agent,
                 api_key: Optional[str] = None, 
                 is_streaming: bool = True,
                 on_audio_delta: Optional[Callable[[str], None]] = None,
                 on_transcript_user: Optional[Callable[[str], None]] = None,
                 on_transcript_ai: Optional[Callable[[str], None]] = None,
                 on_speech_stopped: Optional[Callable[[], None]] = None):
        """
        Initialize the runner.
        
        Args:
            agent: The agent to use for the interaction
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
            is_streaming: Whether to use streaming mode
            on_audio_delta: Callback for audio delta events
            on_transcript_user: Callback for user transcript events
            on_transcript_ai: Callback for AI transcript events
            on_speech_stopped: Callback for speech stopped events
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or through OPENAI_API_KEY environment variable")
            
        self.agent = agent
        self._is_streaming = is_streaming
        self.on_audio_delta = on_audio_delta
        self.on_transcript_user = on_transcript_user
        self.on_transcript_ai = on_transcript_ai
        self.on_speech_stopped = on_speech_stopped
        
        self.ws = None
        self.response_queue = queue.Queue()
        self.current_response = Response()
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._session_updated = threading.Event()
        self._audio_buffer = []
        self.on_response_done = None  # Initialize response done callback
        self.on_audio_transcript_done = None  # Initialize audio transcript done callback

    def init(self):
        """Initialize the connection and session configuration"""
        logger.info("Initializing runner...")
        self._connect()
        logger.info("Updating session configuration...")
        self._update_session(self.agent, is_streaming=self._is_streaming)
        try:
            self._session_updated.wait(timeout=10)
            logger.info("Session configuration updated successfully")
        except Exception as e:
            logger.error(f"Failed to update session configuration: {e}")
            raise

    def _on_message(self, ws, message):
        event = json.loads(message)
        event_type = event.get("type")
        
        # Log all events with their data
        if event_type.endswith('_error'):
            # Handle error events
            logger.error(f"Error event received: {event_type}")
            logger.error(f"Error details: {json.dumps(event, indent=2)}")
            # Set error flag if it's a session update error
            if event_type == "session.update_error":
                self._session_updated.clear()
        else:
            # Log regular events with full data, but hide delta content
            logger.debug(f"Event received: {event_type}")
            if "delta" in event:
                # Create a copy of the event to modify
                log_event = event.copy()
                log_event["delta"] = f"[{len(event['delta'])} bytes of data]"
                logger.debug(f"Event data: {json.dumps(log_event, indent=2)}")
            else:
                logger.debug(f"Event data: {json.dumps(event, indent=2)}")

        # Handle specific events
        if event_type == "session.created":
            self._connected.set()
        elif event_type == "session.updated":
            # Don't clear the event here, let _update_session handle it
            self._session_updated.set()
            logger.debug(f"Session updated: {json.dumps(event, indent=2)}")
        elif event_type == "input_audio_buffer.speech_stopped":
            # Handle VAD detection of speech end
            logger.info("VAD detected end of speech")
            if hasattr(self, 'on_speech_stopped') and self.on_speech_stopped:
                self.on_speech_stopped()
        elif event_type == "response.text.delta":
            with self._lock:
                self.current_response.text += event.get("delta", "")
        elif event_type == "response.audio.delta":
            # Get the delta data
            delta = event.get("delta", "")
            # If we have an audio delta callback, call it immediately
            if hasattr(self, 'on_audio_delta') and self.on_audio_delta:
                self.on_audio_delta(delta)
            # Also buffer the audio data for complete response
            audio_data = base64.b64decode(delta)
            with self._lock:
                self._audio_buffer.append(audio_data)
        elif event_type == "response.audio.done":
            # Process all buffered audio chunks at once
            with self._lock:
                if self._audio_buffer:
                    combined_audio = AudioProcessor.buffer_audio_chunks(self._audio_buffer)
                    processed_audio = AudioProcessor.output_audio_codec(
                        combined_audio,
                        self.agent.output_audio.format,
                        self.agent.output_audio.codec_params
                    )
                    self.current_response.audio_chunks.append(processed_audio)
                    self._audio_buffer = []  # Clear the buffer
        elif event_type == "conversation.item.input_audio_transcription.completed":
            # Handle user's speech transcription
            if hasattr(self, 'on_transcript_user') and self.on_transcript_user:
                transcript = event.get("transcript", "")
                self.on_transcript_user(transcript)
        elif event_type == "response.audio_transcript.done":
            # Handle AI's speech transcription
            if hasattr(self, 'on_transcript_ai') and self.on_transcript_ai:
                transcript = event.get("transcript", "")
                self.on_transcript_ai(transcript)
            # Call audio transcript done callback
            if hasattr(self, 'on_audio_transcript_done') and self.on_audio_transcript_done:
                self.on_audio_transcript_done()
        elif event_type == "response.function_call_arguments.delta":
            with self._lock:
                self.current_response.function_calls.append(event.get("delta", {}))
        elif event_type == "response.done":
            with self._lock:
                self.current_response.is_done = True
                if not self._is_streaming:
                    self.response_queue.put(self.current_response)
                # Call response done callback
                if hasattr(self, 'on_response_done') and self.on_response_done:
                    self.on_response_done(event)
                self.current_response = Response()

    def _on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
        self._connected.clear()
        self._session_updated.clear()

    def _on_open(self, ws):
        """
        Called when WebSocket connection is established.
        """
        logger.info("WebSocket connection established")

    def _disconnect(self):
        """
        Disconnect the WebSocket connection.
        """
        logger.info("Disconnecting WebSocket...")
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self.ws = None
        logger.info("WebSocket disconnected")

    def _connect(self):
        """
        Establish WebSocket connection and wait for session creation.
        """
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        headers = [
            f"Authorization: Bearer {self.api_key}",
            "OpenAI-Beta: realtime=v1"
        ]
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=self.ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for session creation
        logger.info("Waiting for session creation...")
        self._connected.wait(timeout=10)
        if not self._connected.is_set():
            raise Exception("Failed to establish WebSocket connection - no session created")

    def _update_session(self, agent: Agent, is_streaming: bool = False):
        """
        Update the session configuration after session is created.
        """
        self._session_updated.clear()
        logger.info(f"Updating session configuration... (streaming={is_streaming})")
        session_config = agent.to_session_config()
        
        
        # Disable VAD for sync mode
        if not is_streaming:
            logger.debug("Disabling VAD for sync mode")
            session_config["turn_detection"] = None

        logger.debug(f"Session config: {json.dumps(session_config, indent=2)}")
            
        event = {
            "type": "session.update",
            "session": session_config
        }
        logger.debug(f"Sending session update event: {json.dumps(event, indent=2)}")
        self.ws.send(json.dumps(event))
        # Session updates will be handled asynchronously by _on_message

    def _send_text_input(self, text: str):
        logger.info("Sending text input")
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text
                    }
                ]
            }
        }
        logger.debug(f"Sending text event: {json.dumps(event, indent=2)}")
        self.ws.send(json.dumps(event))

    def _send_audio_input(self, input_data: Union[str, bytes], streaming: bool = False):
        """Send audio input to the WebSocket."""
        logger.info(f"Processing input data (streaming={streaming})")
        
        if isinstance(input_data, str):
            if input_data.endswith(('.wav', '.mp3', '.flac', '.opus', '.aac')):
                # It's an audio file path
                logger.info(f"Processing audio file: {input_data}")
                # Read the file and determine its format
                with open(input_data, 'rb') as f:
                    audio_data = f.read()
                input_format = AudioFormat.WAV if input_data.endswith('.wav') else \
                             AudioFormat.MP3 if input_data.endswith('.mp3') else \
                             AudioFormat.FLAC if input_data.endswith('.flac') else \
                             AudioFormat.OPUS if input_data.endswith('.opus') else \
                             AudioFormat.AAC
                # Convert to OpenAI format
                audio_bytes = AudioProcessor.input_audio_codec(audio_data, input_format)
                self._send_audio_bytes(audio_bytes, streaming)
            else:
                # It's a text input
                logger.info("Sending text input")
                self._send_text_input(input_data)
        else:
            # It's audio bytes, check if it matches agent's audio config
            if (self.agent.input_audio.format == AudioFormat.PCM16 and 
                self.agent.input_audio.sample_rate == 24000 and 
                self.agent.input_audio.channels == 1):
                # Already in correct format, sample rate, and channels, send directly
                logger.info("Input matches agent's audio config (PCM16, 24kHz, mono), sending directly")
                self._send_audio_bytes(input_data, streaming)
            else:
                # Try to detect format from magic bytes
                logger.info(f"Processing audio bytes: {len(input_data)} bytes")
                if input_data.startswith(b'RIFF') and input_data[8:12] == b'WAVE':
                    input_format = AudioFormat.WAV
                    logger.debug("Detected WAV format from magic bytes")
                elif input_data.startswith(b'ID3') or input_data.startswith(b'\xFF\xFB'):
                    input_format = AudioFormat.MP3
                    logger.debug("Detected MP3 format from magic bytes")
                elif input_data.startswith(b'fLaC'):
                    input_format = AudioFormat.FLAC
                    logger.debug("Detected FLAC format from magic bytes")
                elif input_data.startswith(b'OggS'):
                    # Check if it's Opus by looking for OpusHead in the Ogg container
                    if b'OpusHead' in input_data[:100]:
                        input_format = AudioFormat.OPUS
                        logger.debug("Detected OPUS format from magic bytes")
                    else:
                        input_format = AudioFormat.OGG
                        logger.debug("Detected OGG format from magic bytes")
                elif input_data.startswith(b'\xFF\xF1') or input_data.startswith(b'\xFF\xF9'):
                    input_format = AudioFormat.AAC
                    logger.debug("Detected AAC format from magic bytes")
                else:
                    # If we can't detect the format, assume PCM16 WAV
                    logger.warning("Could not detect audio format from magic bytes, assuming PCM16 WAV")
                    input_format = AudioFormat.PCM16
                
                audio_bytes = AudioProcessor.input_audio_codec(input_data, input_format)
                self._send_audio_bytes(audio_bytes, streaming)

    def _send_audio_bytes(self, audio_bytes: bytes, streaming: bool):
        """Send processed audio bytes to the WebSocket."""
        logger.info(f"Sending audio bytes (streaming={streaming}, size={len(audio_bytes)} bytes)")
        
        if streaming:
            logger.debug("Sending streaming audio chunk")
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_bytes).decode()
            }))
        else:
            logger.debug("Sending complete audio message")
            # For sync mode, we need to append the audio and then commit it
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_bytes).decode()
            }))
            logger.debug("Appending audio data")
            
            # Commit the audio buffer
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.commit"
            }))
            logger.debug("Committed audio buffer")

    def _create_response(self):
        logger.info("Creating response")
        event = {
            "type": "response.create"
        }
        logger.debug("Sending response create event")
        self.ws.send(json.dumps(event)) 