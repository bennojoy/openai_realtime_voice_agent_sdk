import logging
from typing import Dict, Any, List, Optional
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
    def buffer_audio_chunks(chunks: List[bytes]) -> bytes:
        """Combine multiple audio chunks into a single buffer."""
        return b''.join(chunks)

    @staticmethod
    def output_audio_codec(audio_data: bytes, output_format: AudioFormat, codec_params: Optional[Dict[str, Any]] = None) -> bytes:
        """Convert audio data to the specified output format."""
        if output_format == AudioFormat.PCM16:
            return audio_data  # Already in PCM16 format
            
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as input_file, \
             tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
            try:
                # Write input data to temporary file
                input_file.write(audio_data)
                input_file.flush()
                
                # Build ffmpeg command based on output format
                if output_format == AudioFormat.MP3:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', input_file.name,
                        '-c:a', 'libmp3lame',
                        '-q:a', str(codec_params.get('quality', 2) if codec_params else 2),
                        output_file.name
                    ]
                elif output_format == AudioFormat.OPUS:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', input_file.name,
                        '-c:a', 'libopus',
                        '-b:a', f"{codec_params.get('bitrate', 64)}k" if codec_params else '64k',
                        output_file.name
                    ]
                elif output_format == AudioFormat.AAC:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', input_file.name,
                        '-c:a', 'aac',
                        '-b:a', f"{codec_params.get('bitrate', 128)}k" if codec_params else '128k',
                        output_file.name
                    ]
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
                
                # Run ffmpeg command
                logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {result.stderr}")
                    raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                
                # Read converted audio data
                with open(output_file.name, 'rb') as f:
                    return f.read()
                    
            finally:
                # Clean up temporary files
                try:
                    os.unlink(input_file.name)
                    os.unlink(output_file.name)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary files: {e}") 