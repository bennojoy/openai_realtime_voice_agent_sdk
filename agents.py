from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Voice(Enum):
    """Available voices for the agent."""
    ALLOY = "alloy"
    ASH = "ash"
    BALLAD = "ballad"
    CORAL = "coral"
    ECHO = "echo"
    SAGE = "sage"
    SHIMMER = "shimmer"
    VERSE = "verse"

class AudioFormat(Enum):
    """Supported audio formats and codecs."""
    # Raw formats
    PCM16 = "pcm16"
    PCM24 = "pcm24"
    PCM32 = "pcm32"
    FLOAT32 = "float32"
    
    # Codecs
    OPUS = "opus"
    MP3 = "mp3"
    AAC = "aac"
    FLAC = "flac"
    WAV = "wav"

@dataclass
class AudioConfig:
    """Audio configuration for input and output."""
    format: AudioFormat = AudioFormat.PCM16
    sample_rate: int = 24000  # OpenAI's supported sample rate
    channels: int = 1  # Mono audio
    codec_params: Dict[str, Any] = field(default_factory=dict)  # Codec-specific parameters

    def __post_init__(self):
        """Set default codec parameters based on format."""
        if self.format == AudioFormat.OPUS:
            self.codec_params.setdefault("bitrate", 64000)  # 64 kbps
            self.codec_params.setdefault("application", "voip")  # voip, audio, or restricted_lowdelay
        elif self.format == AudioFormat.MP3:
            self.codec_params.setdefault("bitrate", 128000)  # 128 kbps
        elif self.format == AudioFormat.AAC:
            self.codec_params.setdefault("bitrate", 128000)  # 128 kbps
            self.codec_params.setdefault("profile", "aac_he")  # aac_low, aac_he, aac_he_v2

@dataclass
class Tool:
    """Represents a tool that the agent can use."""
    type: str = "function"
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Agent:
    """
    Represents an AI agent with specific capabilities and configuration.
    
    Attributes:
        name: Name of the agent
        instructions: System instructions for the agent
        tools: List of tools available to the agent
        voice: Voice to use for audio output (defaults to SAGE)
        enable_text: Whether to enable text modality (defaults to False)
        temperature: Temperature for response generation
        max_response_output_tokens: Maximum tokens in response
        input_audio: Audio configuration for input
        output_audio: Audio configuration for output
        model: Model to use for the agent (defaults to 'gpt-4o-realtime-preview-2024-12-17')
        speed: Speech speed for audio output (defaults to 1.0)
        tool_choice: How to handle tool calls (defaults to 'auto')
        tracing: Optional tracing configuration for debugging (defaults to None)
        input_audio_transcription: Configuration for audio transcription
        output_audio_synthesis: Configuration for audio synthesis (defaults to None)
        turn_detection: Configuration for Voice Activity Detection
    """
    name: str
    instructions: str
    tools: List[Tool] = field(default_factory=list)
    voice: Voice = Voice.SAGE  # Default voice
    enable_text: bool = False  # Text is optional
    temperature: float = 0.8
    max_response_output_tokens: Union[int, str] = "inf"
    input_audio: AudioConfig = field(default_factory=AudioConfig)
    output_audio: AudioConfig = field(default_factory=AudioConfig)
    model: str = "gpt-4o-realtime-preview-2024-12-17"
    speed: float = 1.0
    tool_choice: str = "auto"
    tracing: Optional[Dict[str, Any]] = None
    input_audio_transcription: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4o-transcribe",
        "prompt": "",
        "language": "en"
    })
    output_audio_synthesis: Optional[Dict[str, Any]] = None
    turn_detection: Dict[str, Any] = field(default_factory=lambda: {
        "type": "server_vad",
        "threshold": 0.9,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "create_response": True
    })

    def to_session_config(self) -> dict:
        logger.debug(f"Converting agent {self.name} to session config")
        config = {
            "model": self.model,
            "instructions": self.instructions,
            "voice": self.voice.value if self.voice else None,
            "tools": [tool.to_dict() for tool in self.tools],
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            "speed": self.speed,
            "tool_choice": self.tool_choice,
            "tracing": self.tracing,
            "input_audio_format": self.input_audio.format.value if self.input_audio else None,
            "output_audio_format": self.output_audio.format.value if self.output_audio else None,
            "input_audio_transcription": self.input_audio_transcription,
            "output_audio_synthesis": self.output_audio_synthesis,
            "turn_detection": self.turn_detection,
        }
        # Set modalities
        if self.voice or self.input_audio:
            config["modalities"] = ["audio", "text"]
        else:
            config["modalities"] = ["text"]
        # Remove None values
        return {k: v for k, v in config.items() if v is not None} 