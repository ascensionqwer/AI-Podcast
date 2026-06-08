#!/usr/bin/env python3
"""
Podcastfy Local - Unified podcast generation CLI
Uses LM Studio (any LLM) + Kokoro TTS for 100% local podcast generation.

Usage:
    python podcast.py --input ./assets/article.md --output ./output/podcast.wav
    python podcast.py --input ./assets/article.md --output ./output/podcast.wav --script-only
    python podcast.py --input ./assets/article.md --output ./output/podcast.wav --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from src.config import Config, load_config
from src.generator import PodcastGenerator
from src.tts_server import run_standalone_server

console = Console()


def setup_logging(verbose: bool = False):
    """Configure logging with rich output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=verbose,
                rich_tracebacks=True
            )
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate podcasts from content using LM Studio + Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input ./assets/article.md --output ./output/podcast.wav
  %(prog)s -i ./assets/article.md -o ./output/podcast.wav --script-only
  %(prog)s -i ./assets/article.md -o ./output/podcast.wav -v
  %(prog)s --serve-tts  # Run standalone TTS server
"""
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input file path (markdown, text, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output/podcast.wav",
        help="Output audio file path (default: ./output/podcast.wav)"
    )
    
    # Generation options
    parser.add_argument(
        "--script-only",
        action="store_true",
        help="Generate script only, skip audio synthesis"
    )
    parser.add_argument(
        "--keep-script",
        action="store_true",
        help="Keep intermediate script file after audio generation"
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--conversation-config",
        type=str,
        help="Custom conversation style config (YAML file)"
    )
    
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    
    # TTS server mode
    parser.add_argument(
        "--serve-tts",
        action="store_true",
        help="Run standalone TTS server (no podcast generation)"
    )
    parser.add_argument(
        "--tts-port",
        type=int,
        default=8880,
        help="Port for standalone TTS server (default: 8880)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle TTS server mode
    if args.serve_tts:
        logger.info("Running standalone TTS server mode")
        run_standalone_server(port=args.tts_port)
        return
    
    # Validate input
    if not args.input:
        console.print("[red]Error: --input is required for podcast generation[/red]")
        parser.print_help()
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {args.input}[/red]")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override keep_script if specified
    if args.keep_script:
        config.output.keep_temp_script = True
    
    # Load custom conversation config if provided
    if args.conversation_config:
        import yaml
        conv_path = Path(args.conversation_config)
        if conv_path.exists():
            with open(conv_path, 'r') as f:
                conv_data = yaml.safe_load(f)
                if conv_data:
                    for key, value in conv_data.items():
                        if hasattr(config.conversation, key):
                            setattr(config.conversation, key, value)
            logger.info(f"Loaded custom conversation config: {args.conversation_config}")
        else:
            logger.warning(f"Conversation config not found: {args.conversation_config}")
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Display configuration summary
    console.print("\n[bold cyan]Podcastfy Local[/bold cyan]")
    console.print(f"  LLM: {config.llm.model} @ {config.llm.base_url}")
    console.print(f"  TTS: {config.tts.provider} @ http://{config.tts.kokoro.server.host}:{config.tts.kokoro.server.port}")
    console.print(f"  Input: {args.input}")
    console.print(f"  Output: {args.output}")
    if args.script_only:
        console.print("  Mode: [yellow]Script only[/yellow]")
    console.print()
    
    # Generate podcast
    try:
        generator = PodcastGenerator(config)
        result = generator.generate(
            input_path=args.input,
            output_path=args.output,
            script_only=args.script_only
        )
        
        console.print(f"\n[bold green]✅ Success![/bold green]")
        console.print(f"   Generated: {result}")
        
        # Show file size
        result_path = Path(result)
        if result_path.exists():
            size_kb = result_path.stat().st_size / 1024
            console.print(f"   File size: {size_kb:.1f} KB")
        
    except FileNotFoundError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
    except RuntimeError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("\n[yellow]Tips:[/yellow]")
        console.print("  1. Ensure LM Studio is running with any LLM loaded")
        console.print("  2. Check that LM Studio's server is enabled (port 1234)")
        console.print("  3. Verify Kokoro model is available")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()