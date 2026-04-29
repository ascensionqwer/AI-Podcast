"""
Podcast Generator Module.
Handles the complete pipeline from content to podcast audio.
"""

import logging
import re
import time
import numpy as np
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
        - "summary": Quick overview (~2500 words = ~17 min) - key points with context
        - "analysis": Detailed discussion (~4000 words) - deep insights and analysis
        - "full": Complete coverage - covers EVERYTHING, dynamic based on content
        
        Args:
            content: The source content
            podcast_mode: The podcast generation mode
        
        Returns:
            Calculated target word count for the podcast
        """
        content_words = len(content.split())
        
        if podcast_mode == "summary":
            # Quick overview with substance - ~17 minutes (~2500 words at 150 wpm speaking rate)
            return 2500
        elif podcast_mode == "analysis":
            # Deep discussion with thorough insights and examples
            return 4000
        else:  # "full" mode
            # Complete coverage - covers EVERYTHING, scales with content
            if content_words < 500:
                return max(1000, int(content_words * 2.0))
            elif content_words < 2000:
                return max(2000, int(content_words * 1.5))
            elif content_words < 5000:
                return max(3000, int(content_words * 1.2))
            else:
                # For very long content, cover everything comprehensively
                return min(10000, int(content_words * 1.0))
    
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
SUMMARY MODE - Comprehensive Overview (~17 minutes):
- Cover the key points thoroughly, not just surface-level
- Explain the "why" behind each point, not just the "what"
- Each speaker should say 3-5 sentences per turn minimum
- Include relevant context and background for each major point
- The Host should ask follow-up questions to dig deeper into important concepts
- The Expert should provide concrete examples and real-world applications
- Total output: ~2500 words minimum
- Think: "What would a thorough 17-minute podcast that actually teaches the listener something cover?"
- DO NOT rush through topics - take time to explain concepts properly
- When the Expert explains something, the Host should often ask "Can you give an example?" or "Why does that matter?"
"""
        elif podcast_mode == "analysis":
            return """
ANALYSIS MODE - Deep Dive Discussion:
- Go DEEP into every topic - this is an in-depth analysis, not a surface overview
- For each major point, explore: the context, the implications, the evidence, the counterarguments
- The Host should actively challenge ideas, ask "why" repeatedly, and push for deeper explanations
- The Expert should provide thorough explanations with multiple examples, case studies, or data points
- Include historical context, industry trends, and future predictions where relevant
- Each speaker should say 4-6+ sentences per turn - this is a detailed conversation
- When a concept is introduced, explore it fully before moving on
- Discuss what the source DOESN'T say - gaps, limitations, alternative viewpoints
- Total output: ~4000 words minimum - this should feel like a thorough, meaty discussion
- Think: "What would a 25+ minute podcast that leaves the listener truly understanding the topic look like?"
- IMPORTANT: Do NOT summarize - ANALYZE. Break down concepts, examine assumptions, explore consequences
"""
        else:  # "full"
            return """
FULL MODE - Exhaustive Coverage (covers EVERYTHING in depth):
- Cover ALL content comprehensively with DEEP exploration of every point
- For EVERY detail in the source, provide: explanation, context, examples, and significance
- The Host should be an active learner - ask clarifying questions, request examples, challenge assumptions
- The Expert should teach thoroughly - explain concepts from first principles, provide multiple examples, address common misconceptions
- Include ALL details, examples, explanations, nuances, AND expand on them meaningfully
- When the source mentions a concept, explore what it means, why it matters, and how it connects to other ideas
- Each speaker should have substantial turns - 5-8+ sentences when explaining complex ideas
- Create natural back-and-forth: Host asks, Expert explains, Host probes deeper, Expert elaborates with examples
- Discuss implications, applications, limitations, and connections to broader topics
- Total output: as long as needed to cover everything thoroughly - typically 5000-10000+ words
- Think: "What would a comprehensive educational podcast that leaves no stone unturned look like?"
- CRITICAL: This is NOT a summary. This is a deep educational discussion. Every point deserves thorough exploration.
- When in doubt, go deeper rather than moving on to the next topic
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
        
        system_prompt = f"""You are a world-class podcast script writer. Transform the provided content into an engaging, in-depth 2-person conversation between a 'Host' and an 'Expert'.

{mode_instructions}

Guidelines:
- Style: {style}
- Target length: approximately {word_count} words (aim for this length to ensure thorough coverage)
- Podcast name: {podcast_name}
- Make it conversational with natural flow
- The Host should ask probing questions, challenge ideas, and request examples
- The Expert should provide thorough, well-explained information with concrete examples
- Format each line as: Host: [dialogue] or Expert: [dialogue]

IMPORTANT - Make the dialogue sound NATURAL and CONVERSATIONAL (not like reading):
- Include natural speech patterns: "Well...", "So...", "You know...", "I think...", "Hmm..."
- Add reactions: "That's interesting!", "Right.", "Exactly!", "Oh, I see."
- Use contractions: "don't" instead of "do not", "can't" instead of "cannot"
- Include brief pauses with "... " for emphasis or thinking
- Vary sentence length - mix short responses with longer explanations
- Add filler words occasionally: "like", "kind of", "sort of", "basically"
- Show personality: Host should be curious and engaging, Expert should be knowledgeable but approachable
- Avoid overly formal language - make it sound like friends discussing a topic

CRITICAL FOR DEPTH:
- Never rush through a topic - if something is important, spend time on it
- The Host should frequently ask follow-up questions like "Can you explain that more?" or "What does that mean in practice?"
- The Expert should always provide concrete examples, analogies, or real-world applications when explaining concepts
- Explore the "why" and "how" behind every point, not just the "what"
- When a concept has implications, discuss them thoroughly

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


class KokoroTTSClient:
    """Client for text-to-speech synthesis using Kokoro."""
    
    def __init__(self, config: Config):
        self.config = config
        self.server = None
        self.client = None
    
    def start(self):
        """Start the embedded Kokoro TTS server if needed."""
        kokoro_config = self.config.tts.kokoro
        host = kokoro_config.server.host
        port = kokoro_config.server.port
        
        if is_port_in_use(host, port):
            logger.info(f"🔊 Kokoro TTS server already running at http://{host}:{port}")
        else:
            logger.info("🔊 Starting embedded Kokoro TTS server...")
            self.server = EmbeddedTTSServer(
                host=host,
                port=port,
                model_name=kokoro_config.model
            )
            self.server.start(blocking=False)
        
        self.client = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="not-needed"
        )
    
    def synthesize(self, text: str, voice: str) -> bytes:
        """
        Synthesize text to audio using Kokoro.
        
        Args:
            text: Text to synthesize
            voice: Voice name (e.g., "af_bella", "am_adam")
        
        Returns:
            Audio bytes (WAV format)
        """
        if not self.client:
            raise RuntimeError("Kokoro TTS client not initialized. Call start() first.")
        
        try:
            response = self.client.audio.speech.create(
                model="kokoro",
                voice=voice,
                input=text
            )
            audio_content = response.content
            logger.debug(f"KokoroTTS synthesize: Got {len(audio_content) if audio_content else 0} bytes for text: '{text[:50]}...'")
            return audio_content
        except Exception as e:
            logger.error(f"Kokoro TTS synthesis failed: {e}")
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
        logger.info(f"🎙️ Synthesizing {len(segments)} audio segments with Kokoro...")
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
                    audio_data = future.result()
                    results[idx] = audio_data
                    # Log audio data size for debugging
                    if audio_data:
                        logger.debug(f"   Segment {idx}: Got {len(audio_data)} bytes of audio")
                    else:
                        logger.warning(f"   Segment {idx}: Got EMPTY audio data!")
                    completed += 1
                    if completed % 5 == 0:
                        logger.info(f"   Progress: {completed}/{len(segments)} segments")
                except Exception as e:
                    logger.error(f"Failed to synthesize segment {idx}: {e}")
                    raise
            
            # Log final results summary
            non_none_count = sum(1 for r in results if r is not None)
            empty_count = sum(1 for r in results if r is not None and len(r) == 0)
            logger.info(f"   Batch synthesis complete: {non_none_count}/{len(segments)} segments have audio, {empty_count} are empty")
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Kokoro audio synthesis complete in {elapsed:.1f}s")
        
        return results


class VoiceCloneTTSClient:
    """Client for text-to-speech synthesis using Qwen3-TTS voice cloning."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.voice_prompts = {}  # Cache for voice prompts per speaker
    
    def start(self):
        """Load the Qwen3-TTS model and create voice prompts."""
        vc_config = self.config.tts.voice_clone
        
        logger.info(f"🔊 Loading Qwen3-TTS model: {vc_config.model}")
        
        try:
            from qwen_tts import Qwen3TTSModel
            import torch
            
            # Determine device
            device = vc_config.device
            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            
            # Determine dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(vc_config.dtype, torch.bfloat16)
            
            # Check attention implementation
            attn = vc_config.attention
            if attn == "flash_attention_2" and not torch.cuda.is_available():
                logger.info("Flash attention 2 requires CUDA. Falling back to sdpa.")
                attn = "sdpa"
            
            logger.info(f"   Device: {device}, dtype: {vc_config.dtype}, attention: {attn}")
            
            self.model = Qwen3TTSModel.from_pretrained(
                vc_config.model,
                device_map=device,
                dtype=torch_dtype,
                attn_implementation=attn,
            )
            
            logger.info("✅ Qwen3-TTS model loaded successfully")
            
            # Create voice prompts for each speaker
            self._create_voice_prompts()
            
        except ImportError:
            raise ImportError(
                "qwen-tts is not installed. Install it with: pip install qwen-tts"
            )
        except Exception as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            raise
    
    def _create_voice_prompts(self):
        """Create voice clone prompts for each speaker."""
        vc_config = self.config.tts.voice_clone
        
        # Create voice prompt for speaker_1
        speaker_1 = vc_config.voices.speaker_1
        if speaker_1.ref_audio:
            logger.info(f"🎤 Creating voice prompt for speaker_1 ({speaker_1.profile})...")
            ref_audio_path = self._resolve_path(speaker_1.ref_audio)
            self.voice_prompts["speaker_1"] = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=speaker_1.ref_text,
                x_vector_only_mode=False,
            )
            logger.info(f"   ✅ Voice prompt created for {speaker_1.profile}")
        
        # Create voice prompt for speaker_2
        speaker_2 = vc_config.voices.speaker_2
        if speaker_2.ref_audio:
            logger.info(f"🎤 Creating voice prompt for speaker_2 ({speaker_2.profile})...")
            ref_audio_path = self._resolve_path(speaker_2.ref_audio)
            self.voice_prompts["speaker_2"] = self.model.create_voice_clone_prompt(
                ref_audio=ref_audio_path,
                ref_text=speaker_2.ref_text,
                x_vector_only_mode=False,
            )
            logger.info(f"   ✅ Voice prompt created for {speaker_2.profile}")
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to config file location."""
        config_path = Path(self.config.output.directory).parent
        resolved = config_path / path
        if resolved.exists():
            return str(resolved)
        # Try absolute path
        if Path(path).exists():
            return path
        raise FileNotFoundError(f"Reference audio not found: {path}")
    
    def _preprocess_text_for_conversation(self, text: str) -> str:
        """
        Preprocess text to make it sound more conversational for voice cloning.
        
        This adds natural speech patterns that help the TTS model produce
        more natural-sounding output instead of flat "reading" style.
        
        Args:
            text: Original text to preprocess
        
        Returns:
            Preprocessed text with conversational markers
        """
        import re
        
        # Add brief pauses after punctuation for natural rhythm
        # Replace "." with ". " (slight pause)
        # Replace "?" with "? " (question pause)
        # Replace "!" with "! " (emphasis pause)
        text = re.sub(r'\.(\s|$)', '. ', text)
        text = re.sub(r'\?(\s|$)', '? ', text)
        text = re.sub(r'\!(\s|$)', '! ', text)
        
        # Add comma pauses for natural breathing
        text = re.sub(r',(\s)', ', ', text)
        
        # Handle ellipsis for thinking pauses - ensure they're recognized
        text = re.sub(r'\.\.\.', '... ', text)
        
        # Add slight pause before certain words for emphasis
        emphasis_words = ['however', 'but', 'actually', 'in fact', 'you know', 'so', 'well']
        for word in emphasis_words:
            text = re.sub(rf'\b{word}\b', f'... {word}', text, count=1)
        
        # Clean up any double spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def synthesize(self, text: str, speaker: str) -> bytes:
        """
        Synthesize text to audio using voice cloning.
        
        Args:
            text: Text to synthesize
            speaker: Speaker identifier ("speaker_1" or "speaker_2")
        
        Returns:
            Audio bytes (WAV format)
        """
        if not self.model:
            raise RuntimeError("Voice clone TTS client not initialized. Call start() first.")
        
        voice_prompt = self.voice_prompts.get(speaker)
        if not voice_prompt:
            raise RuntimeError(f"No voice prompt for speaker: {speaker}")
        
        vc_config = self.config.tts.voice_clone
        speaker_config = vc_config.voices.__dict__.get(speaker)
        language = speaker_config.language if speaker_config else "English"
        
        # Preprocess text for more conversational output
        processed_text = self._preprocess_text_for_conversation(text)
        
        try:
            import soundfile as sf
            import io
            
            wavs, sr = self.model.generate_voice_clone(
                text=processed_text,
                language=language,
                voice_clone_prompt=voice_prompt,
                max_new_tokens=vc_config.max_tokens,
            )
            
            if wavs:
                # Convert to WAV bytes
                buffer = io.BytesIO()
                sf.write(buffer, wavs[0], sr, format='WAV')
                buffer.seek(0)
                return buffer.read()
            else:
                raise RuntimeError("No audio generated")
                
        except Exception as e:
            logger.error(f"Voice clone TTS synthesis failed: {e}")
            raise
    
    def synthesize_batch(self, segments: List[PodcastSegment], max_workers: int = 1) -> List[bytes]:
        """
        Synthesize multiple segments sequentially (voice cloning doesn't support parallel well).
        
        Args:
            segments: List of podcast segments
            max_workers: Ignored for voice cloning (sequential processing)
        
        Returns:
            List of audio bytes in order
        """
        logger.info(f"🎙️ Synthesizing {len(segments)} audio segments with Voice Cloning...")
        start_time = time.time()
        
        results = []
        
        # Voice cloning is memory-intensive, process sequentially
        for i, seg in enumerate(segments):
            try:
                # Map speaker name to speaker_id
                speaker_id = "speaker_1" if seg.speaker.lower() == "host" else "speaker_2"
                audio = self.synthesize(seg.text, speaker_id)
                results.append(audio)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"   Progress: {i + 1}/{len(segments)} segments")
                    
            except Exception as e:
                logger.error(f"Failed to synthesize segment {i}: {e}")
                raise
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Voice clone audio synthesis complete in {elapsed:.1f}s")
        
        return results


class TTSClient:
    """Factory for TTS clients based on provider configuration."""
    
    def __init__(self, config: Config):
        self.config = config
        self._client = None
    
    def _get_client(self):
        """Get the appropriate TTS client based on provider."""
        if self._client is None:
            provider = self.config.tts.provider
            if provider == "kokoro":
                self._client = KokoroTTSClient(self.config)
            elif provider == "voice_clone":
                self._client = VoiceCloneTTSClient(self.config)
            else:
                raise ValueError(f"Unknown TTS provider: {provider}")
        return self._client
    
    def start_server(self):
        """Start the TTS server/client (legacy method name)."""
        client = self._get_client()
        client.start()
    
    def synthesize(self, text: str, voice: str) -> bytes:
        """Synthesize text to audio."""
        client = self._get_client()
        return client.synthesize(text, voice)
    
    def synthesize_batch(self, segments: List[PodcastSegment], max_workers: int = 4) -> List[bytes]:
        """Synthesize multiple segments."""
        client = self._get_client()
        return client.synthesize_batch(segments, max_workers)


def parse_script(script: str, config: Config) -> List[PodcastSegment]:
    """
    Parse a script into segments.
    
    Args:
        script: Raw script text
        config: Configuration object (used to determine TTS provider and voice settings)
    
    Returns:
        List of PodcastSegment objects
    """
    segments = []
    lines = script.strip().split('\n')
    
    # Get voice info based on provider
    provider = config.tts.provider
    
    if provider == "voice_clone":
        # For voice cloning, voice field stores the speaker_id for lookup
        speaker_1_voice = "speaker_1"
        speaker_2_voice = "speaker_2"
    else:  # kokoro
        # For Kokoro, voice field stores the voice name
        speaker_1_voice = config.tts.kokoro.voices.speaker_1
        speaker_2_voice = config.tts.kokoro.voices.speaker_2
    
    # Track parsing stats for debugging
    total_lines = len(lines)
    matched_lines = 0
    skipped_lines = 0
    
    # More flexible regex patterns to handle various LLM output formats
    # Pattern 1: "Host: text" or "Expert: text" (standard)
    # Pattern 2: "**Host:** text" (markdown bold)
    # Pattern 3: "Speaker 1: text" or "Speaker 2: text"
    # Pattern 4: "HOST:" or "EXPERT:" (uppercase)
    patterns = [
        r'^\*?\*?(Host|Expert|Speaker\s*1|Speaker\s*2)\*?\*?\s*:\s*(.+)$',  # With optional markdown bold
        r'^(HOST|EXPERT)\s*:\s*(.+)$',  # Uppercase
        r'^(Speaker\s*[12])\s*:\s*(.+)$',  # Speaker 1/2
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            skipped_lines += 1
            continue
        
        # Skip markdown headers, titles, and non-dialogue lines
        if line.startswith('#') or line.startswith('---') or line.startswith('==='):
            skipped_lines += 1
            continue
        
        matched = False
        for pattern in patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                matched = True
                matched_lines += 1
                speaker = match.group(1).lower().replace('*', '').strip()
                text = match.group(2).strip()
                
                # Remove markdown formatting from text
                text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold
                text = re.sub(r'\*(.+?)\*', r'\1', text)      # Remove italic
                
                # Map speaker to voice
                if speaker in ('host', 'speaker 1', 'speaker1', 'speaker1'):
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
                break
        
        if not matched:
            skipped_lines += 1
            # Log first few skipped lines for debugging
            if skipped_lines <= 5:
                logger.debug(f"   Skipped line: '{line[:80]}...'")
    
    # Log parsing stats
    logger.info(f"   Script parsing: {total_lines} lines, {matched_lines} matched, {skipped_lines} skipped")
    
    if not segments:
        logger.warning("⚠️ No dialogue segments parsed from script!")
        logger.warning("   Script preview (first 500 chars):")
        logger.warning(f"   {script[:500]}")
        logger.warning("   Expected format: 'Host: text' or 'Expert: text'")
    
    return segments


def load_content(input_path: str) -> str:
    """
    Load content from a file.
    
    Supports:
    - Native text formats: .md, .txt, .markdown (read directly)
    - Document formats: .pdf, .docx, .pptx, .doc, .ppt, .xlsx, .xls, .html (converted to markdown)
    
    Args:
        input_path: Path to input file
    
    Returns:
        Content as string (markdown format)
    """
    from .doc_converter import load_document_content
    
    return load_document_content(input_path)


def save_script(script: str, output_path: str):
    """Save script to file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(script)


def generate_silence(duration_seconds: float, sample_rate: int = 24000) -> np.ndarray:
    """
    Generate silence audio samples.
    
    Args:
        duration_seconds: Duration of silence in seconds
        sample_rate: Sample rate for the audio
    
    Returns:
        numpy array of silence samples
    """
    import numpy as np
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.int16)


def concatenate_audio(audio_chunks: List[bytes], output_path: str,
                      segments: List[PodcastSegment] = None,
                      pause_between: float = 0.5,
                      pause_on_change: float = 1.0):
    """
    Concatenate audio chunks with natural pauses between dialogue segments.
    
    Args:
        audio_chunks: List of WAV audio bytes
        output_path: Output file path
        segments: List of podcast segments (to detect speaker changes)
        pause_between: Seconds of silence between each segment
        pause_on_change: Extra pause when speaker changes (turn-taking)
    """
    import scipy.io.wavfile as wavfile
    import io
    
    # Parse all WAV files and concatenate with pauses
    samples_list = []
    sample_rate = 24000  # Default
    
    prev_speaker = None
    
    logger.info(f"🔍 Debug: Received {len(audio_chunks)} audio chunks")
    for i, chunk in enumerate(audio_chunks):
        if chunk is None:
            logger.warning(f"   Chunk {i}: is None")
        elif len(chunk) == 0:
            logger.warning(f"   Chunk {i}: is empty bytes (0 bytes)")
        else:
            logger.debug(f"   Chunk {i}: {len(chunk)} bytes")
        
    for i, chunk in enumerate(audio_chunks):
        if chunk:
            buffer = io.BytesIO(chunk)
            sr, samples = wavfile.read(buffer)
            
            if not samples_list:
                sample_rate = sr
                samples_list.append(samples)
            else:
                # Add pause before this segment
                if segments and i < len(segments):
                    current_speaker = segments[i].speaker
                    
                    # Determine pause duration based on speaker change
                    if prev_speaker and current_speaker != prev_speaker:
                        # Speaker changed - add longer pause (turn-taking)
                        pause_duration = pause_between + pause_on_change
                    else:
                        # Same speaker continuing - shorter pause
                        pause_duration = pause_between
                    
                    silence = generate_silence(pause_duration, sample_rate)
                    samples_list.append(silence)
                    prev_speaker = current_speaker
                else:
                    # No segment info - use default pause
                    silence = generate_silence(pause_between, sample_rate)
                    samples_list.append(silence)
                
                samples_list.append(samples)
    
    if not samples_list:
        # Provide more detailed error message
        none_count = sum(1 for c in audio_chunks if c is None)
        empty_count = sum(1 for c in audio_chunks if c is not None and len(c) == 0)
        valid_count = sum(1 for c in audio_chunks if c is not None and len(c) > 0)
        error_msg = (
            f"No audio data to save. "
            f"Total chunks: {len(audio_chunks)}, "
            f"Valid: {valid_count}, "
            f"Empty: {empty_count}, "
            f"None: {none_count}"
        )
        logger.error(f"❌ {error_msg}")
        raise ValueError(error_msg)
    
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
        segments = parse_script(script_text, self.config)
        logger.info(f"📝 Parsed {len(segments)} dialogue segments")
        
        # Check if we have any segments to synthesize
        if not segments:
            raise RuntimeError(
                "No dialogue segments found in the generated script. "
                "The LLM may have returned a non-dialogue format. "
                "Check the script output and ensure it uses 'Host:' and 'Expert:' format."
            )
        
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
        
        # Concatenate and save with natural pauses
        concatenate_audio(
            audio_chunks,
            output_path,
            segments=segments,
            pause_between=self.config.tts.pause_between_segments,
            pause_on_change=self.config.tts.pause_on_speaker_change
        )
        
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
        segments = parse_script(script_text, self.config)
        logger.info(f"📝 Parsed {len(segments)} dialogue segments")
        
        # Check if we have any segments to synthesize
        if not segments:
            raise RuntimeError(
                "No dialogue segments found in the generated script. "
                "The LLM may have returned a non-dialogue format. "
                "Check the script output and ensure it uses 'Host:' and 'Expert:' format."
            )
        
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
        
        # Concatenate and save with natural pauses
        concatenate_audio(
            audio_chunks,
            output_path,
            segments=segments,
            pause_between=self.config.tts.pause_between_segments,
            pause_on_change=self.config.tts.pause_on_speaker_change
        )
        
        return output_path