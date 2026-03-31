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
        self.client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key
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
    
    def _calculate_dynamic_word_count(self, content: str) -> int:
        """
        Calculate dynamic word count based on input content length.
        
        The podcast will be as long as it needs to be to cover all content:
        - Short content (< 500 words): 2x expansion for engaging discussion
        - Medium content (500-2000 words): 1.5x for thorough coverage
        - Long content (2000-5000 words): 1.2x for detailed discussion
        - Very long content (> 5000 words): 1.1x - full coverage, no summarization
        
        Args:
            content: The source content
        
        Returns:
            Calculated target word count for the podcast
        """
        # Count words in content
        content_words = len(content.split())
        
        if content_words < 500:
            # Short content - expand for engaging discussion
            target = max(800, int(content_words * 2))
        elif content_words < 2000:
            # Medium content - thorough coverage
            target = int(content_words * 1.5)
        elif content_words < 5000:
            # Long content - detailed discussion
            target = int(content_words * 1.2)
        else:
            # Very long content - full coverage, no summarization
            target = int(content_words * 1.1)
        
        return target
    
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
        base_word_count = conversation_config.get("word_count", 2000)
        podcast_name = conversation_config.get("podcast_name", "Local Podcast")
        creativity = conversation_config.get("creativity", 0.7)
        user_instructions = conversation_config.get("user_instructions", "")
        
        # Calculate dynamic word count based on content length
        word_count = self._calculate_dynamic_word_count(content)
        content_words = len(content.split())
        logger.info(f"📊 Content: {content_words} words → Target podcast: {word_count} words")
        
        system_prompt = f"""You are a world-class podcast script writer. Transform the provided content into an engaging, natural-sounding 2-person conversation between a 'Host' and an 'Expert'.

Guidelines:
- Style: {style}
- Target length: approximately {word_count} words
- Podcast name: {podcast_name}
- Cover ALL the content comprehensively - do not skip or summarize important details
- Make it conversational with natural flow, occasional verbal fillers, and genuine curiosity
- The Host should ask clarifying questions and show interest
- The Expert should provide detailed, accurate information from the source content
- Include brief transitions between topics
- Avoid overly formal language; make it feel like a real conversation
- The podcast should be as long as needed to cover all the content properly

Format each line as:
Host: [dialogue]
Expert: [dialogue]

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
            logger.error(f"Failed to generate script: {e}")
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