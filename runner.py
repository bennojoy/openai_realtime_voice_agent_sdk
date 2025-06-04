import logging
import threading
import queue
import time
from typing import Optional, Callable, Union, List, Dict, Any
from dataclasses import dataclass, field
from agents import Agent
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
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the runner.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to get from OPENAI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as a parameter or through OPENAI_API_KEY environment variable")
            
        self.ws = None
        self.response_queue = queue.Queue()
        self.current_response = Response()
        self._lock = threading.Lock()
        self._connected = threading.Event()
        self._session_updated = threading.Event()
        self._audio_buffer = []
        self._is_streaming = False
        self.agent = None  # Will be set during run_sync or run_stream

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
            
        elif event_type == "response.text.delta":
            with self._lock:
                self.current_response.text += event.get("delta", "")
        elif event_type == "response.audio.delta":
            audio_data = base64.b64decode(event.get("delta", ""))
            # Process audio data to match output format
            processed_audio = AudioProcessor.process_response_audio(audio_data, self.agent.output_audio)
            with self._lock:
                self.current_response.audio_chunks.append(processed_audio)
        elif event_type == "response.function_call_arguments.delta":
            with self._lock:
                self.current_response.function_calls.append(event.get("delta", {}))
        elif event_type == "response.done":
            with self._lock:
                self.current_response.is_done = True
                if not self._is_streaming:
                    self.response_queue.put(self.current_response)
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

    def _send_audio_input(self, audio_data: bytes, streaming: bool = False):
        """Send audio input to the WebSocket."""
        logger.info(f"Sending audio input (streaming={streaming}, size={len(audio_data)} bytes)")
        
        if streaming:
            logger.debug("Sending streaming audio chunk")
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode()
            }))
        else:
            logger.debug("Sending complete audio message")
            # For sync mode, we need to append the audio and then commit it
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode()
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

    @staticmethod
    def run_sync(agent: Agent, input_data: Union[str, bytes], api_key: str) -> Response:
        """
        Run a synchronous interaction with the agent.
        """
        logger.info("Starting synchronous interaction")
        runner = Runner(api_key)
        runner.agent = agent  # Set the agent for audio processing
        
        # Connect and wait for session creation
        logger.info("Connecting to WebSocket...")
        runner._connect()
        
        # Update session with VAD disabled
        logger.info("Updating session configuration...")
        runner._update_session(agent, is_streaming=False)
        try:
            runner._session_updated.wait(timeout=10)
        except Exception as e:
            logger.error(f"Failed to update session configuration: {e}")
            raise
        
        if isinstance(input_data, str):
            if input_data.endswith(('.wav', '.mp3', '.flac', '.opus', '.aac')):
                # It's an audio file path
                logger.info(f"Processing audio file: {input_data}")
                audio_bytes = AudioProcessor.process_audio_file(input_data, agent.input_audio)
                runner._send_audio_input(audio_bytes, streaming=False)
            else:
                # It's a text input
                logger.info("Sending text input")
                runner._send_text_input(input_data)
        else:
            # It's audio bytes
            logger.info(f"Processing audio bytes: {len(input_data)} bytes")
            audio_bytes = AudioProcessor.process_audio_stream(input_data, agent.input_audio)
            runner._send_audio_input(audio_bytes, streaming=False)

        # For sync mode, we need to explicitly create response
        logger.info("Creating response...")
        runner._create_response()
        logger.info("Waiting for response...")
        return runner.response_queue.get()

    @staticmethod
    def run_stream(agent: Agent, input_data: Union[str, bytes], api_key: str, 
                  on_text: Optional[Callable[[str], None]] = None,
                  on_audio: Optional[Callable[[bytes], None]] = None) -> Response:
        """
        Run a streaming interaction with the agent.
        """
        logger.info("Starting streaming interaction")
        runner = Runner(api_key)
        runner.agent = agent  # Set the agent for audio processing
        runner._is_streaming = True
        
        # Connect and wait for session creation
        logger.info("Connecting to WebSocket...")
        runner._connect()
        
        # Update session with VAD enabled
        logger.info("Updating session configuration...")
        runner._update_session(agent, is_streaming=True)

        if isinstance(input_data, str):
            if input_data.endswith(('.wav', '.mp3', '.flac', '.opus', '.aac')):
                # It's an audio file path
                logger.info(f"Processing audio file: {input_data}")
                audio_bytes = AudioProcessor.process_audio_file(input_data, agent.input_audio)
                runner._send_audio_input(audio_bytes, streaming=True)
            else:
                # It's a text input
                logger.info("Sending text input")
                runner._send_text_input(input_data)
        else:
            # It's audio bytes
            logger.info(f"Processing audio bytes: {len(input_data)} bytes")
            audio_bytes = AudioProcessor.process_audio_stream(input_data, agent.input_audio)
            runner._send_audio_input(audio_bytes, streaming=True)

        # For streaming mode, response is created automatically by VAD
        # Set up callbacks
        if on_text:
            logger.info("Setting up text callback")
            def text_callback():
                while not runner.current_response.is_done:
                    if runner.current_response.text:
                        on_text(runner.current_response.text)
                    time.sleep(0.1)
            threading.Thread(target=text_callback, daemon=True).start()
            
        if on_audio:
            logger.info("Setting up audio callback")
            def audio_callback():
                while not runner.current_response.is_done:
                    if runner.current_response.audio_chunks:
                        for chunk in runner.current_response.audio_chunks:
                            on_audio(chunk)
                    time.sleep(0.1)
            threading.Thread(target=audio_callback, daemon=True).start()
            
        logger.info("Streaming interaction started")
        return runner.current_response 