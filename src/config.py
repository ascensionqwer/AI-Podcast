"""
Configuration management for Podcastfy Local.
Loads and validates configuration from YAML file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    model: str = "qwen3.5-122b-a10b"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"
    temperature: float = 0.7
    max_tokens: int = 8192


@dataclass
class TTSServerConfig:
    """TTS server configuration for Kokoro."""
    host: str = "127.0.0.1"
    port: int = 8880
    auto_start: bool = True


@dataclass
class KokoroVoiceConfig:
    """Kokoro voice configuration."""
    speaker_1: str = "af_bella"  # Female voice (Host)
    speaker_2: str = "am_adam"    # Male voice (Expert)


@dataclass
class KokoroConfig:
    """Kokoro TTS configuration."""
    model: str = "mlx-community/Kokoro-82M-bf16"
    server: TTSServerConfig = field(default_factory=TTSServerConfig)
    voices: KokoroVoiceConfig = field(default_factory=KokoroVoiceConfig)


@dataclass
class VoiceCloneVoiceProfile:
    """Voice profile for voice cloning."""
    profile: str = ""
    ref_audio: str = ""
    ref_text: str = ""
    language: str = "English"


@dataclass
class VoiceCloneVoicesConfig:
    """Voice cloning voices configuration."""
    speaker_1: VoiceCloneVoiceProfile = field(default_factory=VoiceCloneVoiceProfile)
    speaker_2: VoiceCloneVoiceProfile = field(default_factory=VoiceCloneVoiceProfile)


@dataclass
class VoiceCloneConfig:
    """Voice cloning (Qwen3-TTS) configuration."""
    model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device: str = "auto"
    dtype: str = "bfloat16"
    attention: str = "flash_attention_2"
    max_tokens: int = 2048
    voices: VoiceCloneVoicesConfig = field(default_factory=VoiceCloneVoicesConfig)


@dataclass
class TTSConfig:
    """TTS configuration settings."""
    provider: str = "kokoro"  # Options: "kokoro" or "voice_clone"
    kokoro: KokoroConfig = field(default_factory=KokoroConfig)
    voice_clone: VoiceCloneConfig = field(default_factory=VoiceCloneConfig)
    sample_rate: int = 24000
    output_format: str = "wav"
    # Conversation flow settings
    pause_between_segments: float = 0.5  # Seconds of silence between dialogue segments
    pause_on_speaker_change: float = 1.0  # Extra pause when speaker changes (turn-taking)
    
    def get_voices(self) -> dict:
        """Get voice configuration based on current provider."""
        if self.provider == "voice_clone":
            return {
                "speaker_1": self.voice_clone.voices.speaker_1,
                "speaker_2": self.voice_clone.voices.speaker_2
            }
        else:  # kokoro
            return {
                "speaker_1": self.kokoro.voices.speaker_1,
                "speaker_2": self.kokoro.voices.speaker_2
            }


@dataclass
class ConversationConfig:
    """Conversation style configuration."""
    word_count: int = 2000
    conversation_style: List[str] = field(default_factory=lambda: ["casual", "informative"])
    podcast_name: str = "Local Podcast"
    creativity: float = 0.7
    podcast_mode: str = "summary"
    user_instructions: str = ""


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str = "./output"
    default_filename: str = "podcast.wav"
    keep_temp_script: bool = False


@dataclass
class Config:
    """Main configuration container."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {path}, using defaults")
            return cls()
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        llm_data = data.get("llm", {})
        tts_data = data.get("tts", {})
        conversation_data = data.get("conversation", {})
        output_data = data.get("output", {})
        
        # Parse Kokoro config
        kokoro_data = tts_data.get("kokoro", {})
        kokoro_server_data = kokoro_data.get("server", {})
        kokoro_server_config = TTSServerConfig(
            host=kokoro_server_data.get("host", "127.0.0.1"),
            port=kokoro_server_data.get("port", 8880),
            auto_start=kokoro_server_data.get("auto_start", True)
        )
        kokoro_voices_data = kokoro_data.get("voices", {})
        kokoro_voices_config = KokoroVoiceConfig(
            speaker_1=kokoro_voices_data.get("speaker_1", "af_bella"),
            speaker_2=kokoro_voices_data.get("speaker_2", "am_adam")
        )
        kokoro_config = KokoroConfig(
            model=kokoro_data.get("model", "mlx-community/Kokoro-82M-bf16"),
            server=kokoro_server_config,
            voices=kokoro_voices_config
        )
        
        # Parse Voice Clone config
        voice_clone_data = tts_data.get("voice_clone", {})
        vc_voices_data = voice_clone_data.get("voices", {})
        
        # Parse speaker_1 voice profile
        speaker_1_data = vc_voices_data.get("speaker_1", {})
        speaker_1_profile = VoiceCloneVoiceProfile(
            profile=speaker_1_data.get("profile", ""),
            ref_audio=speaker_1_data.get("ref_audio", ""),
            ref_text=speaker_1_data.get("ref_text", ""),
            language=speaker_1_data.get("language", "English")
        )
        
        # Parse speaker_2 voice profile
        speaker_2_data = vc_voices_data.get("speaker_2", {})
        speaker_2_profile = VoiceCloneVoiceProfile(
            profile=speaker_2_data.get("profile", ""),
            ref_audio=speaker_2_data.get("ref_audio", ""),
            ref_text=speaker_2_data.get("ref_text", ""),
            language=speaker_2_data.get("language", "English")
        )
        
        vc_voices_config = VoiceCloneVoicesConfig(
            speaker_1=speaker_1_profile,
            speaker_2=speaker_2_profile
        )
        
        voice_clone_config = VoiceCloneConfig(
            model=voice_clone_data.get("model", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
            device=voice_clone_data.get("device", "auto"),
            dtype=voice_clone_data.get("dtype", "bfloat16"),
            attention=voice_clone_data.get("attention", "flash_attention_2"),
            max_tokens=voice_clone_data.get("max_tokens", 2048),
            voices=vc_voices_config
        )
        
        return cls(
            llm=LLMConfig(
                model=llm_data.get("model", "qwen3.5-122b-a10b"),
                base_url=llm_data.get("base_url", "http://localhost:1234/v1"),
                api_key=llm_data.get("api_key", "lm-studio"),
                temperature=llm_data.get("temperature", 0.7),
                max_tokens=llm_data.get("max_tokens", 8192)
            ),
            tts=TTSConfig(
                provider=tts_data.get("provider", "kokoro"),
                kokoro=kokoro_config,
                voice_clone=voice_clone_config,
                sample_rate=tts_data.get("sample_rate", 24000),
                output_format=tts_data.get("output_format", "wav"),
                pause_between_segments=tts_data.get("pause_between_segments", 0.5),
                pause_on_speaker_change=tts_data.get("pause_on_speaker_change", 1.0)
            ),
            conversation=ConversationConfig(
                word_count=conversation_data.get("word_count", 2000),
                conversation_style=conversation_data.get("conversation_style", ["casual", "informative"]),
                podcast_name=conversation_data.get("podcast_name", "Local Podcast"),
                creativity=conversation_data.get("creativity", 0.7),
                podcast_mode=conversation_data.get("podcast_mode", "summary"),
                user_instructions=conversation_data.get("user_instructions", "")
            ),
            output=OutputConfig(
                directory=output_data.get("directory", "./output"),
                default_filename=output_data.get("default_filename", "podcast.wav"),
                keep_temp_script=output_data.get("keep_temp_script", False)
            )
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check LLM settings
        if not self.llm.base_url:
            issues.append("LLM base_url is not set")
        
        if not self.llm.model:
            issues.append("LLM model is not set")
        
        # Check TTS settings based on provider
        if self.tts.provider == "kokoro":
            if self.tts.kokoro.server.port < 1 or self.tts.kokoro.server.port > 65535:
                issues.append(f"Invalid Kokoro TTS server port: {self.tts.kokoro.server.port}")
            if not self.tts.kokoro.voices.speaker_1 or not self.tts.kokoro.voices.speaker_2:
                issues.append("Kokoro TTS voices must have both speaker_1 and speaker_2 configured")
        elif self.tts.provider == "voice_clone":
            vc_voices = self.tts.voice_clone.voices
            if not vc_voices.speaker_1.ref_audio or not vc_voices.speaker_2.ref_audio:
                issues.append("Voice clone TTS voices must have ref_audio configured for both speakers")
            if not vc_voices.speaker_1.ref_text or not vc_voices.speaker_2.ref_text:
                issues.append("Voice clone TTS voices must have ref_text configured for both speakers")
        else:
            issues.append(f"Unknown TTS provider: {self.tts.provider}. Use 'kokoro' or 'voice_clone'")
        
        # Check output directory
        output_path = Path(self.output.directory)
        if not output_path.exists():
            logger.info(f"Output directory does not exist, will create: {output_path}")
        
        return issues


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults."""
    if config_path is None:
        # Look for config.yaml in current directory or parent directories
        config_path = "config.yaml"
    
    config = Config.from_yaml(config_path)
    issues = config.validate()
    
    if issues:
        for issue in issues:
            logger.warning(f"Configuration issue: {issue}")
    
    return config


# Global config instance
_config: Optional[Config] = None


def get_config(reload: bool = False, config_path: Optional[str] = None) -> Config:
    """Get the global configuration instance."""
    global _config
    
    if _config is None or reload:
        _config = load_config(config_path)
    
    return _config