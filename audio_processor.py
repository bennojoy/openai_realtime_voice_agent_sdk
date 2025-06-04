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
        logger.debug(f"Converting audio from {input_format.value} to {output_format.value}")
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=f".{input_format.value}", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=f".{output_format.value}", delete=False) as output_file:
            
            # Write input data
            input_file.write(audio_data)
            input_file.flush()
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Input options
            cmd.extend(["-f", input_format.value])
            if input_format == AudioFormat.OPUS:
                cmd.extend(["-application", input_params.get("application", "voip")])
            cmd.extend(["-i", input_file.name])
            
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
            elif output_format == AudioFormat.WAV:
                cmd.extend(["-c:a", "pcm_s16le"])
            
            cmd.append(output_file.name)
            
            # Run ffmpeg
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Read output file
                with open(output_file.name, 'rb') as f:
                    return f.read()
                    
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
                raise
            finally:
                # Clean up temporary files
                os.unlink(input_file.name)
                os.unlink(output_file.name)

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
        logger.info(f"Processing audio file: {file_path}")
        
        # Read the file
        data, samplerate = sf.read(file_path, dtype='float32')
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            logger.debug("Converting stereo to mono")
            data = data.mean(axis=1)
        
        # Resample to target sample rate if needed
        if samplerate != audio_config.sample_rate:
            logger.debug(f"Resampling from {samplerate}Hz to {audio_config.sample_rate}Hz")
            samples = len(data)
            new_samples = int(samples * audio_config.sample_rate / samplerate)
            data = signal.resample(data, new_samples)
        
        # Convert to PCM16 for OpenAI
        pcm_data = (data * 32767).astype(np.int16)
        pcm_bytes = pcm_data.tobytes()
        
        # If input format is not PCM16, convert it
        if audio_config.format != AudioFormat.PCM16:
            pcm_bytes = AudioProcessor.convert_audio_codec(
                pcm_bytes,
                AudioFormat.PCM16,
                audio_config.format,
                {},  # No parameters for PCM16
                audio_config.codec_params
            )
        
        return pcm_bytes

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
        logger.debug("Processing audio stream")
        
        # If input format is not PCM16, convert it
        if audio_config.format != AudioFormat.PCM16:
            audio_data = AudioProcessor.convert_audio_codec(
                audio_data,
                audio_config.format,
                AudioFormat.PCM16,
                audio_config.codec_params,
                {}  # No parameters for PCM16
            )
        
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
        logger.debug("Processing response audio")
        
        # If output format is not PCM16, convert it
        if audio_config.format != AudioFormat.PCM16:
            audio_data = AudioProcessor.convert_audio_codec(
                audio_data,
                AudioFormat.PCM16,
                audio_config.format,
                {},  # No parameters for PCM16
                audio_config.codec_params
            )
        
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