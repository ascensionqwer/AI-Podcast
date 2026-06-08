"""
Embedded Kokoro TTS Server for Podcastfy Local.
Provides a FastAPI server that wraps MLX-Audio's Kokoro model.
Can run standalone or be embedded in the main application.
"""

import io
import logging
import socket
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, Generator
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)

# Global model reference
_model = None
_model_lock = threading.Lock()


def find_available_port(start_port: int = 8880, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found between {start_port} and {start_port + max_attempts - 1}")


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception:
        return False


def load_kokoro_model(model_name: str = "mlx-community/Kokoro-82M-bf16"):
    """Load the Kokoro model into memory."""
    global _model
    
    with _model_lock:
        if _model is not None:
            return _model
        
        try:
            from mlx_audio.tts.utils import load_model
            logger.info(f"🚀 Loading Kokoro model: {model_name}")
            _model = load_model(model_name)
            logger.info("✅ Kokoro model loaded successfully")
            return _model
        except ImportError:
            raise ImportError(
                "mlx-audio is not installed. Install it with: pip install mlx-audio"
            )
        except Exception as e:
            logger.error(f"Failed to load Kokoro model: {e}")
            raise


def generate_audio(text: str, voice: str = "af_bella", model_name: str = "mlx-community/Kokoro-82M-bf16") -> np.ndarray:
    """
    Generate audio from text using Kokoro.
    
    Args:
        text: Text to synthesize
        voice: Voice name (e.g., "af_bella", "am_adam")
        model_name: Model identifier
    
    Returns:
        numpy array of audio samples
    """
    model = load_kokoro_model(model_name)
    
    # Generate audio chunks
    audio_chunks = []
    for result in model.generate(text, voice=voice):
        audio_chunks.append(result.audio)
    
    if not audio_chunks:
        raise RuntimeError("No audio generated")
    
    # Concatenate all chunks
    import mlx.core as mx
    combined = mx.concatenate(audio_chunks, axis=0)
    return np.array(combined)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio)
    return buffer.getvalue()


# FastAPI application
def create_tts_app(model_name: str = "mlx-community/Kokoro-82M-bf16"):
    """Create a FastAPI TTS application."""
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import Response
    
    @asynccontextmanager
    async def lifespan(app):
        # Load model on startup
        load_kokoro_model(model_name)
        yield
        # Cleanup if needed
        global _model
        _model = None
    
    app = FastAPI(
        title="Kokoro TTS Server",
        description="Local TTS server using Kokoro via MLX-Audio",
        version="1.0.0",
        lifespan=lifespan
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "model": model_name}
    
    @app.post("/v1/audio/speech")
    async def text_to_speech(request: Request):
        """
        OpenAI-compatible TTS endpoint.
        
        Request body:
        {
            "model": "kokoro",
            "voice": "af_bella",
            "input": "Text to synthesize"
        }
        """
        try:
            data = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        
        text = data.get("input", "")
        voice = data.get("voice", "af_bella")
        
        if not text:
            raise HTTPException(status_code=400, detail="Missing 'input' field")
        
        try:
            audio = generate_audio(text, voice, model_name)
            wav_bytes = audio_to_wav_bytes(audio)
            
            # Log audio size for debugging
            logger.info(f"TTS: Generated {len(wav_bytes)} bytes for text: '{text[:50]}...'")
            
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "attachment; filename=speech.wav"
                }
            )
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


class EmbeddedTTSServer:
    """
    Embedded TTS server that can run in a background thread.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8880, 
                 model_name: str = "mlx-community/Kokoro-82M-bf16"):
        self.host = host
        self.port = port
        self.model_name = model_name
        self._server = None
        self._thread = None
        self._running = False
    
    def start(self, blocking: bool = False):
        """
        Start the TTS server.
        
        Args:
            blocking: If True, block until server stops. If False, run in background.
        """
        if is_port_in_use(self.host, self.port):
            logger.info(f"TTS server already running at http://{self.host}:{self.port}")
            return
        
        # Find available port if needed
        if is_port_in_use(self.host, self.port):
            self.port = find_available_port(self.port + 1)
            logger.info(f"Using alternative port: {self.port}")
        
        import uvicorn
        
        app = create_tts_app(self.model_name)
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False
        )
        self._server = uvicorn.Server(config)
        
        if blocking:
            self._running = True
            self._server.run()
        else:
            self._thread = threading.Thread(target=self._server.run, daemon=True)
            self._thread.start()
            self._running = True
            
            # Wait for server to be ready
            max_wait = 30
            for _ in range(max_wait):
                if is_port_in_use(self.host, self.port):
                    logger.info(f"🔊 TTS server started at http://{self.host}:{self.port}")
                    return
                time.sleep(0.5)
            
            logger.warning("TTS server may not have started properly")
    
    def stop(self):
        """Stop the TTS server."""
        if self._server:
            self._server.should_exit = True
            self._running = False
            logger.info("TTS server stopped")
    
    def is_running(self) -> bool:
        """Check if the server is running."""
        return is_port_in_use(self.host, self.port)
    
    def get_client(self):
        """Get an OpenAI-compatible client for this server."""
        from openai import OpenAI
        return OpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key="not-needed"
        )


def run_standalone_server(host: str = "127.0.0.1", port: int = 8880,
                          model_name: str = "mlx-community/Kokoro-82M-bf16"):
    """Run the TTS server standalone (blocking)."""
    import uvicorn
    
    app = create_tts_app(model_name)
    print(f"🔊 Starting Kokoro TTS server at http://{host}:{port}")
    print(f"📝 Model: {model_name}")
    print("Press Ctrl+C to stop")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kokoro TTS Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8880, help="Port to bind to")
    parser.add_argument("--model", default="mlx-community/Kokoro-82M-bf16", help="Model name")
    
    args = parser.parse_args()
    run_standalone_server(args.host, args.port, args.model)