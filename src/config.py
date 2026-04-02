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
    """TTS server configuration."""
    host: str = "127.0.0.1"
    port: int = 8880
    auto_start: bool = True


@dataclass
class TTSConfig:
    """TTS configuration settings."""
    provider: str = "kokoro"
    model: str = "mlx-community/Kokoro-82M-bf16"
    server: TTSServerConfig = field(default_factory=TTSServerConfig)
    voices: dict = field(default_factory=lambda: {
        "speaker_1": "af_bella",
        "speaker_2": "am_adam"
    })
    sample_rate: int = 24000
    output_format: str = "wav"


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
        
        # Parse nested TTS server config
        server_data = tts_data.get("server", {})
        server_config = TTSServerConfig(
            host=server_data.get("host", "127.0.0.1"),
            port=server_data.get("port", 8880),
            auto_start=server_data.get("auto_start", True)
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
                model=tts_data.get("model", "mlx-community/Kokoro-82M-bf16"),
                server=server_config,
                voices=tts_data.get("voices", {"speaker_1": "af_bella", "speaker_2": "am_adam"}),
                sample_rate=tts_data.get("sample_rate", 24000),
                output_format=tts_data.get("output_format", "wav")
            ),
            conversation=ConversationConfig(
                word_count=conversation_data.get("word_count", 2000),
                conversation_style=conversation_data.get("conversation_style", ["casual", "informative"]),
                podcast_name=conversation_data.get("podcast_name", "Local Podcast"),
                creativity=conversation_data.get("creativity", 0.7),
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
        
        # Check TTS settings
        if self.tts.server.port < 1 or self.tts.server.port > 65535:
            issues.append(f"Invalid TTS server port: {self.tts.server.port}")
        
        if not self.tts.voices.get("speaker_1") or not self.tts.voices.get("speaker_2"):
            issues.append("TTS voices must have both speaker_1 and speaker_2 configured")
        
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