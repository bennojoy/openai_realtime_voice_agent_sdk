import soundfile as sf
import numpy as np
from scipy import signal
import logging
from typing import Tuple, Union, Dict, Any, Optional
import wave
import io
import subprocess
import tempfile
import os
from agents import AudioFormat, AudioConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Handles all audio processing operations."""
    
    TARGET_SAMPLE_RATE = 24000  # OpenAI's supported sample rate
    
    @staticmethod
    def convert_audio_codec(audio_data: bytes, input_format: AudioFormat, output_format: AudioFormat,
                          input_params: Dict[str, Any], output_params: Dict[str, Any]) -> bytes:
        """
        Convert audio between different codecs using ffmpeg.
        
        Args:
            audio_data: Raw audio data
            input_format: Input audio format
            output_format: Output audio format
            input_params: Input codec parameters
            output_params: Output codec parameters
            
        Returns:
            bytes: Converted audio data
        """
        logger.info(f"Starting audio conversion from {input_format.value} to {output_format.value}")
        logger.debug(f"Input params: {input_params}")
        logger.debug(f"Output params: {output_params}")
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=f".{input_format.value}", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=f".{output_format.value if output_format != AudioFormat.PCM16 else 'wav'}", delete=False) as output_file:
            
            logger.debug(f"Created temporary files: {input_file.name} -> {output_file.name}")
            
            # Write input data
            input_file.write(audio_data)
            input_file.flush()
            logger.debug(f"Written {len(audio_data)} bytes to input file")
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Input options
            cmd.extend(["-f", "s16le"])  # Use s16le format for PCM16
            if input_format.value == AudioFormat.OPUS:
                cmd.extend(["-application", input_params.get("application", "voip")])
            cmd.extend(["-i", input_file.name])
            
            # Add audio filters for all required conversions
            filters = []
            
            # Convert to mono if needed
            filters.append("channelsplit=channel_layout=stereo[left][right];[left][right]amerge=inputs=2")
            
            # Resample to target rate with high quality
            filters.append("aresample=24000:filter_size=256:cutoff=0.8:phase_shift=10")
            
            # Combine all filters
            cmd.extend(["-filter_complex", ",".join(filters)])
            
            # Output options
            if output_format == AudioFormat.OPUS:
                cmd.extend([
                    "-c:a", "libopus",
                    "-b:a", str(output_params.get("bitrate", 64000)),
                    "-application", output_params.get("application", "voip")
                ])
            elif output_format == AudioFormat.MP3:
                cmd.extend([
                    "-c:a", "libmp3lame",
                    "-b:a", str(output_params.get("bitrate", 128000))
                ])
            elif output_format == AudioFormat.AAC:
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", str(output_params.get("bitrate", 128000)),
                    "-profile:a", output_params.get("profile", "aac_he")
                ])
            elif output_format == AudioFormat.FLAC:
                cmd.extend(["-c:a", "flac"])
            elif output_format == AudioFormat.PCM16:
                cmd.extend(["-c:a", "pcm_s16le", "-f", "wav"])  # Output as WAV
            elif output_format == AudioFormat.WAV:
                cmd.extend(["-c:a", "pcm_s16le"])
            
            cmd.append(output_file.name)
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            try:
                result = subprocess.run(cmd, check=True, capture_output=True)
                logger.debug("FFmpeg conversion completed successfully")
                
                # Read output file
                with open(output_file.name, 'rb') as f:
                    output_data = f.read()
                    logger.info(f"Read {len(output_data)} bytes from output file")
                    return output_data
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
                raise
            finally:
                # Clean up temporary files
                os.unlink(input_file.name)
                os.unlink(output_file.name)
                logger.debug("Cleaned up temporary files")

    @staticmethod
    def process_audio_file(file_path: str, audio_config: AudioConfig) -> bytes:
        """
        Process audio file and convert to the required format.
        
        Args:
            file_path: Path to the audio file
            audio_config: Audio configuration for processing
            
        Returns:
            bytes: Processed audio data in the required format
        """
        logger.info(f"Starting audio file processing: {file_path}")
        logger.debug(f"Audio config: format={audio_config.format}, sample_rate={audio_config.sample_rate}, channels={audio_config.channels}")
        
        # First check if we need conversion by reading WAV header
        with wave.open(file_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            
            logger.info(f"Input WAV: {framerate}Hz, {channels} channels, {sample_width} bytes per sample")
            
            # Check if conversion is needed
            needs_conversion = (
                framerate != audio_config.sample_rate or  # Sample rate mismatch
                channels != audio_config.channels or  # Channel count mismatch
                sample_width != 2 or  # Not 16-bit PCM
                audio_config.format != AudioFormat.PCM16  # Format mismatch
            )
            
            logger.info(f"Conversion check: sample_rate_match={framerate == audio_config.sample_rate}, "
                       f"channels_match={channels == audio_config.channels}, "
                       f"format_match={sample_width == 2 and audio_config.format == AudioFormat.PCM16}")
            
            if not needs_conversion:
                logger.info("Input audio already matches required format, using raw PCM data")
                return wf.readframes(n_frames)
        
        logger.info("Conversion needed, processing with FFmpeg...")
        # If conversion is needed, use FFmpeg
        with open(file_path, 'rb') as f:
            audio_data = f.read()
            
        return AudioProcessor.convert_audio_codec(
            audio_data,
            AudioFormat.WAV,
            audio_config.format,
            {"sample_rate": framerate},
            audio_config.codec_params
        )

    @staticmethod
    def process_audio_stream(audio_data: bytes, audio_config: AudioConfig) -> bytes:
        """
        Process audio stream data and convert to the required format.
        
        Args:
            audio_data: Raw audio data
            audio_config: Audio configuration for processing
            
        Returns:
            bytes: Processed audio data in the required format
        """
        logger.info(f"Starting audio stream processing: {len(audio_data)} bytes")
        logger.debug(f"Audio config: format={audio_config.format}, sample_rate={audio_config.sample_rate}, channels={audio_config.channels}")
        
        # If input format is not PCM16, convert it
        if audio_config.format != AudioFormat.PCM16:
            logger.debug(f"Converting from {audio_config.format} to PCM16")
            audio_data = AudioProcessor.convert_audio_codec(
                audio_data,
                audio_config.format,
                AudioFormat.PCM16,
                audio_config.codec_params,
                {}  # No parameters for PCM16
            )
            logger.debug(f"Conversion completed: {len(audio_data)} bytes")
        
        return audio_data

    @staticmethod
    def process_response_audio(audio_data: bytes, audio_config: AudioConfig) -> bytes:
        """
        Process response audio data to match the output format.
        
        Args:
            audio_data: Raw audio data from the response
            audio_config: Audio configuration for processing
            
        Returns:
            bytes: Processed audio data in the required output format
        """
        logger.info(f"Starting response audio processing: {len(audio_data)} bytes")
        logger.debug(f"Audio config: format={audio_config.format}, sample_rate={audio_config.sample_rate}, channels={audio_config.channels}")
        
        # If output format is not PCM16, convert it
        if audio_config.format != AudioFormat.PCM16:
            logger.debug(f"Converting from PCM16 to {audio_config.format}")
            audio_data = AudioProcessor.convert_audio_codec(
                audio_data,
                AudioFormat.PCM16,
                audio_config.format,
                {},  # No parameters for PCM16
                audio_config.codec_params
            )
            logger.debug(f"Conversion completed: {len(audio_data)} bytes")
        
        return audio_data

    @staticmethod
    def save_audio_chunks(chunks: list, output_path: str):
        """
        Save audio chunks to a WAV file.
        
        Args:
            chunks: List of audio chunks
            output_path: Path to save the WAV file
            
        Raises:
            Exception: If saving fails
        """
        try:
            logger.info(f"Saving audio chunks to: {output_path}")
            # Combine all chunks
            audio_data = b''.join(chunks)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Save as WAV file
            sf.write(
                output_path,
                audio_array,
                AudioProcessor.TARGET_SAMPLE_RATE,
                subtype='PCM_16'
            )
            logger.info("Audio file saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save audio chunks: {e}")
            raise 