"""
Podcast Generator Module.
Handles the complete pipeline from content to podcast audio.
"""

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .config import Config, get_config
from .tts_server import EmbeddedTTSServer, is_port_in_use

logger = logging.getLogger(__name__)


@dataclass
class PodcastSegment:
    """A single segment of the podcast."""
    speaker: str
    text: str
    voice: str


@dataclass
class PodcastScript:
    """Generated podcast script."""
    segments: List[PodcastSegment]
    raw_text: str
    source_file: Optional[str] = None


class LLMClient:
    """Client for interacting with LM Studio (OpenAI-compatible API)."""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Create custom HTTP client with very long timeout for large context processing
        # Large contexts (100k+ tokens) can take 15+ minutes to process
        import httpx
        http_client = httpx.Client(
            timeout=httpx.Timeout(
                connect=30.0,      # Connection timeout
                read=1800.0,       # Read timeout: 30 minutes
                write=1800.0,      # Write timeout: 30 minutes
                pool=30.0          # Pool timeout
            )
        )
        
        self.client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            http_client=http_client  # Use custom client with extended timeout
        )
        self.model = config.llm.model
    
    def check_connection(self) -> bool:
        """Check if LM Studio is running and accessible."""
        try:
            # Try a minimal request
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Cannot connect to LM Studio at {self.config.llm.base_url}: {e}")
            return False
    
    def _calculate_word_count(self, content: str, podcast_mode: str) -> int:
        """
        Calculate word count based on podcast mode.
        
        Modes:
        - "summary": Quick overview (~1500 words = ~10 min max) - key points only
        - "analysis": Detailed discussion (~800 words) - insights and analysis
        - "full": Complete coverage - covers EVERYTHING, dynamic based on content
        
        Args:
            content: The source content
            podcast_mode: The podcast generation mode
        
        Returns:
            Calculated target word count for the podcast
        """
        content_words = len(content.split())
        
        if podcast_mode == "summary":
            # Quick overview - ~10 minutes max (~1500 words at 150 wpm speaking rate)
            return 1500
        elif podcast_mode == "analysis":
            # Detailed discussion with insights
            return 800
        else:  # "full" mode
            # Complete coverage - covers EVERYTHING, scales with content
            if content_words < 500:
                return max(600, int(content_words * 1.5))
            elif content_words < 2000:
                return max(1000, int(content_words * 1.2))
            elif content_words < 5000:
                return max(1500, int(content_words * 1.0))
            else:
                # For very long content, cover everything but cap reasonably
                return min(5000, int(content_words * 0.8))
    
    def _get_mode_instructions(self, podcast_mode: str) -> str:
        """
        Get specific instructions for each podcast mode.
        
        Args:
            podcast_mode: The podcast generation mode
        
        Returns:
            Mode-specific instructions for the LLM
        """
        if podcast_mode == "summary":
            return """
SUMMARY MODE - Quick Overview (~10 minutes max):
- Focus on the most important key points (top 5-10 points)
- Keep it concise but informative - like a news summary
- Each speaker should say 2-3 sentences per turn
- Skip minor details, but cover the main ideas clearly
- Total output: ~1500 words maximum (~10 min at 150 wpm)
- Think: "What would a 10-minute podcast highlight reel cover?"
"""
        elif podcast_mode == "analysis":
            return """
ANALYSIS MODE - Detailed Discussion:
- Cover the main topics with insights and analysis
- Discuss implications, context, and significance
- Include some examples to illustrate points
- Each speaker can say 2-3 sentences per turn
- Balance breadth and depth
- Total output: ~1500 words
- Think: "What would a thoughtful 10-minute analysis cover?"
"""
        else:  # "full"
            return """
FULL MODE - Complete Coverage (covers EVERYTHING):
- Cover ALL content comprehensively - do NOT skip anything
- Include ALL details, examples, explanations, and nuances
- Natural conversation flow with transitions between topics
- Each speaker can have longer turns when needed
- Every section, point, and detail from the source must be discussed
- Total output: dynamic length - as long as needed to cover everything
- Think: "What would a thorough educational podcast episode that covers the entire document look like?"
"""
    
    def generate_script(self, content: str, conversation_config: dict) -> str:
        """
        Generate a podcast script from content.
        
        Args:
            content: The source content to transform
            conversation_config: Conversation style configuration
        
        Returns:
            Generated script as text
        """
        style = ", ".join(conversation_config.get("conversation_style", ["casual", "informative"]))
        podcast_mode = conversation_config.get("podcast_mode", "summary")
        podcast_name = conversation_config.get("podcast_name", "Local Podcast")
        creativity = conversation_config.get("creativity", 0.7)
        user_instructions = conversation_config.get("user_instructions", "")
        
        # Calculate word count based on mode
        word_count = self._calculate_word_count(content, podcast_mode)
        content_words = len(content.split())
        mode_instructions = self._get_mode_instructions(podcast_mode)
        logger.info(f"📊 Mode: {podcast_mode} | Content: {content_words} words → Target: {word_count} words")
        
        system_prompt = f"""You are a world-class podcast script writer. Transform the provided content into an engaging 2-person conversation between a 'Host' and an 'Expert'.

{mode_instructions}

Guidelines:
- Style: {style}
- Target length: approximately {word_count} words (STRICT LIMIT - do not exceed)
- Podcast name: {podcast_name}
- Make it conversational with natural flow
- The Host should ask questions and show interest
- The Expert should provide accurate information from the source
- Format each line as: Host: [dialogue] or Expert: [dialogue]

{f'Additional instructions: {user_instructions}' if user_instructions else ''}"""
        
        logger.info(f"🧠 Generating script with {self.model}...")
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Transform the following content into a podcast dialogue:\n\n{content}"}
                ],
                temperature=creativity,
                max_tokens=self.config.llm.max_tokens
            )
            
            elapsed = time.time() - start_time
            script = response.choices[0].message.content
            logger.info(f"✅ Script generated in {elapsed:.1f}s ({len(script)} characters)")
            
            return script
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to generate script: {e}")
            
            # Provide helpful error messages
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                raise RuntimeError(
                    "Lost connection to LM Studio. Please ensure LM Studio is still running "
                    "with the model loaded. For very long content, the model may take time to respond."
                )
            elif "timeout" in error_msg.lower():
                raise RuntimeError(
                    "Request timed out. The content may be too long. "
                    "Try with shorter content or check if LM Studio is responding."
                )
            else:
                raise RuntimeError(f"Script generation failed: {e}")


class TTSClient:
    """Client for text-to-speech synthesis using Kokoro."""
    
    def __init__(self, config: Config):
        self.config = config
        self.server = None
        self.client = None
    
    def start_server(self):
        """Start the embedded TTS server if needed."""
        host = self.config.tts.server.host
        port = self.config.tts.server.port
        
        if is_port_in_use(host, port):
            logger.info(f"🔊 TTS server already running at http://{host}:{port}")
        else:
            logger.info("🔊 Starting embedded TTS server...")
            self.server = EmbeddedTTSServer(
                host=host,
                port=port,
                model_name=self.config.tts.model
            )
            self.server.start(blocking=False)
        
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="not-needed"
        )
    
    def synthesize(self, text: str, voice: str) -> bytes:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            voice: Voice name (e.g., "af_bella", "am_adam")
        
        Returns:
            Audio bytes (WAV format)
        """
        if not self.client:
            raise RuntimeError("TTS client not initialized. Call start_server() first.")
        
        try:
            response = self.client.audio.speech.create(
                model="kokoro",
                voice=voice,
                input=text
            )
            return response.content
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def synthesize_batch(self, segments: List[PodcastSegment], max_workers: int = 4) -> List[bytes]:
        """
        Synthesize multiple segments in parallel.
        
        Args:
            segments: List of podcast segments
            max_workers: Maximum parallel workers
        
        Returns:
            List of audio bytes in order
        """
        logger.info(f"🎙️ Synthesizing {len(segments)} audio segments...")
        start_time = time.time()
        
        results = [None] * len(segments)
        
        # Use ThreadPoolExecutor for parallel synthesis
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.synthesize, seg.text, seg.voice): i
                for i, seg in enumerate(segments)
            }
            
            completed = 0
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                    completed += 1
                    if completed % 5 == 0:
                        logger.info(f"   Progress: {completed}/{len(segments)} segments")
                except Exception as e:
                    logger.error(f"Failed to synthesize segment {idx}: {e}")
                    raise
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Audio synthesis complete in {elapsed:.1f}s")
        
        return results


def parse_script(script: str, voices: dict) -> List[PodcastSegment]:
    """
    Parse a script into segments.
    
    Args:
        script: Raw script text
        voices: Voice mapping with speaker_1 and speaker_2
    
    Returns:
        List of PodcastSegment objects
    """
    segments = []
    lines = script.strip().split('\n')
    
    speaker_1_voice = voices.get("speaker_1", "af_bella")
    speaker_2_voice = voices.get("speaker_2", "am_adam")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Match "Host: text" or "Expert: text" patterns
        match = re.match(r'^(Host|Expert|Speaker\s*1|Speaker\s*2)\s*:\s*(.+)$', line, re.IGNORECASE)
        if match:
            speaker = match.group(1).lower()
            text = match.group(2).strip()
            
            # Map speaker to voice
            if speaker in ('host', 'speaker 1', 'speaker1'):
                voice = speaker_1_voice
                speaker_name = "Host"
            else:
                voice = speaker_2_voice
                speaker_name = "Expert"
            
            if text:  # Only add non-empty segments
                segments.append(PodcastSegment(
                    speaker=speaker_name,
                    text=text,
                    voice=voice
                ))
    
    return segments


def load_content(input_path: str) -> str:
    """
    Load content from a file.
    
    Args:
        input_path: Path to input file (markdown, text, etc.)
    
    Returns:
        Content as string
    """
    path = Path(input_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def save_script(script: str, output_path: str):
    """Save script to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(script)


def concatenate_audio(audio_chunks: List[bytes], output_path: str):
    """
    Concatenate audio chunks and save to file.
    
    Args:
        audio_chunks: List of WAV audio bytes
        output_path: Output file path
    """
    import scipy.io.wavfile as wavfile
    import numpy as np
    import io
    
    # Parse all WAV files and concatenate
    samples_list = []
    sample_rate = 24000  # Kokoro default
    
    for chunk in audio_chunks:
        if chunk:
            buffer = io.BytesIO(chunk)
            sr, samples = wavfile.read(buffer)
            if samples_list:
                samples_list.append(samples)
            else:
                sample_rate = sr
                samples_list.append(samples)
    
    if not samples_list:
        raise ValueError("No audio data to save")
    
    # Concatenate all samples
    combined = np.concatenate(samples_list)
    
    # Save to file
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    wavfile.write(path, sample_rate, combined)
    logger.info(f"💾 Audio saved to: {output_path}")


class PodcastGenerator:
    """
    Main podcast generator that orchestrates the entire pipeline.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.llm_client = LLMClient(self.config)
        self.tts_client = TTSClient(self.config)
    
    def generate(self, input_path: str, output_path: str, 
                 script_only: bool = False) -> str:
        """
        Generate a podcast from input content.
        
        Args:
            input_path: Path to input content file
            output_path: Path for output audio file
            script_only: If True, only generate script without audio
        
        Returns:
            Path to generated file (audio or script)
        """
        # Load content
        logger.info(f"📂 Loading content from: {input_path}")
        content = load_content(input_path)
        logger.info(f"   Content length: {len(content)} characters")
        
        # Check LLM connection
        if not self.llm_client.check_connection():
            raise RuntimeError(
                f"Cannot connect to LM Studio at {self.config.llm.base_url}. "
                "Please ensure LM Studio is running with the model loaded."
            )
        
        # Generate script
        script_text = self.llm_client.generate_script(
            content, 
            vars(self.config.conversation)
        )
        
        # Parse script into segments
        segments = parse_script(script_text, self.config.tts.voices)
        logger.info(f"📝 Parsed {len(segments)} dialogue segments")
        
        # Save script if requested
        script_path = Path(output_path).with_suffix('.txt')
        if self.config.output.keep_temp_script or script_only:
            save_script(script_text, str(script_path))
            logger.info(f"📄 Script saved to: {script_path}")
        
        if script_only:
            return str(script_path)
        
        # Start TTS server
        self.tts_client.start_server()
        
        # Synthesize audio
        audio_chunks = self.tts_client.synthesize_batch(segments)
        
        # Concatenate and save
        concatenate_audio(audio_chunks, output_path)
        
        return output_path
    
    def generate_from_text(self, text: str, output_path: str,
                          script_only: bool = False) -> str:
        """
        Generate a podcast from raw text.
        
        Args:
            text: Raw text content
            output_path: Path for output audio file
            script_only: If True, only generate script without audio
        
        Returns:
            Path to generated file
        """
        # Check LLM connection
        if not self.llm_client.check_connection():
            raise RuntimeError(
                f"Cannot connect to LM Studio at {self.config.llm.base_url}. "
                "Please ensure LM Studio is running with the model loaded."
            )
        
        # Generate script
        script_text = self.llm_client.generate_script(
            text,
            vars(self.config.conversation)
        )
        
        # Parse script into segments
        segments = parse_script(script_text, self.config.tts.voices)
        logger.info(f"📝 Parsed {len(segments)} dialogue segments")
        
        # Save script if requested
        script_path = Path(output_path).with_suffix('.txt')
        if self.config.output.keep_temp_script or script_only:
            save_script(script_text, str(script_path))
            logger.info(f"📄 Script saved to: {script_path}")
        
        if script_only:
            return str(script_path)
        
        # Start TTS server
        self.tts_client.start_server()
        
        # Synthesize audio
        audio_chunks = self.tts_client.synthesize_batch(segments)
        
        # Concatenate and save
        concatenate_audio(audio_chunks, output_path)
        
        return output_path