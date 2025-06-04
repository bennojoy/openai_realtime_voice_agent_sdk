import logging
from typing import Dict, Any
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
    def input_audio_codec(audio_data: bytes, input_format: AudioFormat) -> bytes:
        """
        Convert any input audio to OpenAI's required format (PCM16, 24000Hz, mono).
        
        Args:
            audio_data: Raw audio data
            input_format: Input audio format (WAV, MP3, etc.)
            
        Returns:
            bytes: Converted audio data in PCM16 WAV format at 24000Hz mono
        """
        logger.info(f"Converting input audio from {input_format.value} to OpenAI format (PCM16, 24000Hz, mono)")
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=f".{input_format.value}", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            
            logger.debug(f"Created temporary files: {input_file.name} -> {output_file.name}")
            
            # Write input data
            input_file.write(audio_data)
            input_file.flush()
            logger.debug(f"Written {len(audio_data)} bytes to input file")
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Input options
            cmd.extend(["-i", input_file.name])
            
            # Add audio filters for required conversions
            filters = []
            
            # Convert to mono and resample using the working filter chain
            filters.append("[0:a]pan=stereo|c0=0.5*c0+0.5*c1,aresample=24000:filter_size=256:cutoff=0.8:phase_shift=10[a]")
            
            # Combine all filters
            cmd.extend(["-filter_complex", ",".join(filters)])
            
            # Map the filtered output
            cmd.extend(["-map", "[a]"])
            
            # Output options - always PCM16 WAV
            cmd.extend([
                "-ac", "1",           # Force mono output
                "-c:a", "pcm_s16le",  # 16-bit PCM
                "-f", "wav",          # WAV container
                "-ar", "24000"        # 24000Hz sample rate
            ])
            
            cmd.append(output_file.name)
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            try:
                result = subprocess.run(cmd, check=True, capture_output=True)
                logger.debug("FFmpeg conversion completed successfully")
                
                # Read output file
                with open(output_file.name, 'rb') as f:
                    output_data = f.read()
                    logger.info(f"Converted audio: {len(output_data)} bytes, PCM16 WAV at 24000Hz mono")
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
    def output_audio_codec(audio_data: bytes, output_format: AudioFormat, output_params: Dict[str, Any] = None) -> bytes:
        """
        Convert OpenAI's response (PCM16, 24000Hz, mono) to user's desired format.
        
        Args:
            audio_data: Raw audio data from OpenAI (PCM16 WAV at 24000Hz mono)
            output_format: Desired output format (WAV, MP3, etc.)
            output_params: Optional parameters for output format (e.g., bitrate)
            
        Returns:
            bytes: Converted audio data in the desired format
        """
        logger.info(f"Converting OpenAI response to {output_format.value}")
        
        # If output format is already PCM16 WAV, return as is
        if output_format == AudioFormat.PCM16:
            logger.info("Output format is already PCM16 WAV, skipping conversion")
            return audio_data
            
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=f".{output_format.value}", delete=False) as output_file:
            
            logger.debug(f"Created temporary files: {input_file.name} -> {output_file.name}")
            
            # Write input data
            input_file.write(audio_data)
            input_file.flush()
            logger.debug(f"Written {len(audio_data)} bytes to input file")
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Input options - we know it's PCM16 WAV at 24000Hz
            cmd.extend([
                "-f", "wav",
                "-ar", "24000",
                "-i", input_file.name
            ])
            
            # Output options based on desired format
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
            
            logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            try:
                result = subprocess.run(cmd, check=True, capture_output=True)
                logger.debug("FFmpeg conversion completed successfully")
                
                # Read output file
                with open(output_file.name, 'rb') as f:
                    output_data = f.read()
                    logger.info(f"Converted audio: {len(output_data)} bytes, format: {output_format.value}")
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
    def process_response_audio(audio_data: bytes, output_config: AudioConfig) -> bytes:
        """
        Process audio response data from OpenAI.
        
        Args:
            audio_data: Raw audio data from OpenAI
            output_config: Output audio configuration
            
        Returns:
            bytes: Processed audio data in the desired format
        """
        logger.info(f"Processing response audio to {output_config.format.value}")
        
        # If output format is already PCM16 WAV, return as is
        if output_config.format == AudioFormat.PCM16:
            logger.info("Output format is already PCM16 WAV, skipping conversion")
            return audio_data
            
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix=f".{output_config.format.value}", delete=False) as output_file:
            
            logger.debug(f"Created temporary files: {input_file.name} -> {output_file.name}")
            
            # Write input data
            input_file.write(audio_data)
            input_file.flush()
            logger.debug(f"Written {len(audio_data)} bytes to input file")
            
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]  # -y to overwrite output file
            
            # Input options - we know it's PCM16 WAV at 24000Hz
            cmd.extend([
                "-f", "wav",
                "-ar", "24000",
                "-i", input_file.name
            ])
            
            # Output options based on desired format
            if output_config.format == AudioFormat.OPUS:
                cmd.extend([
                    "-c:a", "libopus",
                    "-b:a", str(output_config.codec_params.get("bitrate", 64000)),
                    "-application", output_config.codec_params.get("application", "voip")
                ])
            elif output_config.format == AudioFormat.MP3:
                cmd.extend([
                    "-c:a", "libmp3lame",
                    "-b:a", str(output_config.codec_params.get("bitrate", 128000))
                ])
            elif output_config.format == AudioFormat.AAC:
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", str(output_config.codec_params.get("bitrate", 128000)),
                    "-profile:a", output_config.codec_params.get("profile", "aac_he")
                ])
            elif output_config.format == AudioFormat.FLAC:
                cmd.extend(["-c:a", "flac"])
            elif output_config.format == AudioFormat.WAV:
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
                    logger.info(f"Converted audio: {len(output_data)} bytes, format: {output_config.format.value}")
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
    def buffer_audio_chunks(chunks: list[bytes]) -> bytes:
        """
        Combine multiple audio chunks into a single audio buffer.
        
        Args:
            chunks: List of audio data chunks
            
        Returns:
            bytes: Combined audio data
        """
        logger.debug(f"Combining {len(chunks)} audio chunks")
        return b''.join(chunks) 