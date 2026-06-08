# Podcastfy Local - AI Podcast Generator

A unified podcast generation system that runs **100% locally** on Apple Silicon using:
- A local LLM server for script generation
- Kokoro TTS via MLX-Audio for high-quality voice synthesis
- Docling for document conversion (PDF, DOCX, PPTX, etc.)

---

## Supported Input Formats

Podcastfy Local now supports a wide range of document formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| Markdown | `.md`, `.markdown` | Native format, read directly |
| Text | `.txt` | Plain text files |
| PDF | `.pdf` | PDF documents (converted to markdown) |
| Word | `.docx`, `.doc` | Microsoft Word documents |
| PowerPoint | `.pptx`, `.ppt` | Microsoft PowerPoint presentations |
| Excel | `.xlsx`, `.xls` | Microsoft Excel spreadsheets |
| HTML | `.html`, `.htm` | HTML web pages |

Documents are automatically converted to markdown using Docling before being processed by the LLM. No intermediate files are saved - the converted content is fed directly to the script generator.

---

## Complete Setup Guide for New MacBook

Follow these steps to set up Podcastfy Local on a brand new MacBook (Apple Silicon M1/M2/M3/M4).

### Step 1: Install Homebrew (Package Manager)

Open Terminal (press `Cmd + Space`, type "Terminal", press Enter) and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

After installation, add Homebrew to your PATH:

```bash
# For Apple Silicon Macs
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verify installation:

```bash
brew --version
```

---

### Step 2: Install Python 3.11+

```bash
brew install python@3.11
```

Verify installation:

```bash
python3 --version
# Should show Python 3.11.x or higher
```

---

### Step 3: Install Local LLM Server

1. Download a local LLM server application from the official website
2. Open the downloaded `.dmg` file
3. Drag to Applications folder
4. Open the application from Applications

**Download the Required Model:**

1. In the app, click the **Search** icon (magnifying glass)
2. Search for a suitable open-source LLM model
3. Download the model (this may take time depending on size)
4. Once downloaded, click the **Chat** icon
5. Select your downloaded model from the dropdown
6. Click **Load Model**

**Enable the Local Server:**

1. Click the **Server** icon (left sidebar, looks like a network icon)
2. Click **Start Server**
3. Ensure it's running on port `1234`
4. You should see: `Server running at http://localhost:1234`

---

### Step 4: Clone the Project

```bash
# Navigate to your preferred location (example)
cd ~/Documents/Projects

# Clone the repository
git clone <YOUR_REPOSITORY_URL> Podcastfy

# Enter the project directory
cd Podcastfy
```

---

### Step 5: Create Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

---

### Step 6: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

This will install:
- `openai` - For local LLM API communication
- `mlx-audio` - For Kokoro TTS on Apple Silicon
- `customtkinter` - For the GUI
- `docling` - For document conversion (PDF, DOCX, PPTX, etc.)
- And other dependencies

---

### Step 7: Run the Application

**Option A: GUI Application (Recommended)**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the GUI
python gui.py
```

**Option B: Command Line**

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Generate a podcast
python podcast.py --input ./assets/example_content.md --output ./output/podcast.wav
```

---

### Step 8: Use the Desktop Launcher

The project includes a ready-to-use desktop launcher: `Podcastfy.command`

**Option A: Use from Project Folder**

1. Navigate to the project folder in Finder
2. Double-click `Podcastfy.command`
3. The GUI will launch automatically

**Option B: Copy to Desktop**

```bash
# Copy launcher to Desktop (adjust path to your project location)
cp ./Podcastfy.command ~/Desktop/

# Now you can double-click it from your Desktop
```

**Option C: Create Alias (Alternative)**

If you prefer a simpler approach, create a launcher that navigates to your project:

```bash
# Create the launcher script (adjust path to your project location)
cat > ~/Desktop/Podcastfy.command << 'EOF'
#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source venv/bin/activate
python gui.py
EOF

# Make it executable
chmod +x ~/Desktop/Podcastfy.command
```

Now you can double-click `Podcastfy.command` on your Desktop to launch the app!

**Note:** On first run, macOS may ask for permission to run the script. Click "Open" to allow it.

---

## Quick Start Guide

### Using the GUI

1. **Start your LLM server** - Ensure the server is running (port 1234)
2. **Launch Podcastfy** - Run `python gui.py` or double-click the desktop launcher
3. **Select Input File** - Click on a file in the Assets list or upload a new one
4. **Choose Podcast Mode** - Select from Summary, Analysis, or Full mode
5. **Custom Instructions (Analysis Mode)** - When using Analysis mode, a text box appears where you can optionally enter specific focus areas or instructions for the podcast discussion
6. **Set Output Name** - Enter a filename (e.g., `my_podcast.wav`)
7. **Click Generate** - Press the "Generate Podcast" button
8. **Wait for Completion** - The progress bar shows generation status
9. **Play Output** - Click on the generated file in the Output list

### Using Command Line

```bash
# Basic usage
python podcast.py -i ./assets/article.md -o ./output/podcast.wav

# Script only (no audio)
python podcast.py -i ./assets/article.md -o ./output/podcast.wav --script-only

# Verbose mode
python podcast.py -i ./assets/article.md -o ./output/podcast.wav -v
```

---

## Project Structure

```
Podcastfy/
├── gui.py                  # Desktop GUI application
├── podcast.py              # CLI entry point
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── generator.py        # Podcast generation logic
│   ├── doc_converter.py    # Document to markdown converter
│   └── tts_server.py       # Embedded Kokoro TTS server
├── assets/                 # Input content files (.md, .txt, .pdf, .docx, etc.)
│   └── example_content.md  # Place your input files here
└── output/                 # Generated podcasts (.wav, .mp3)
```

---

## Configuration

Edit `config.yaml` to customize settings:

### TTS Provider Selection

You can choose between two TTS providers by changing `tts.provider`:

- **`kokoro`** - Built-in Kokoro TTS voices (fast, no setup required)
- **`voice_clone`** - Custom cloned voices using a voice cloning model (requires reference audio)

```yaml
# TTS Configuration
tts:
  provider: "voice_clone"  # Options: "kokoro" or "voice_clone"
```

### Kokoro TTS Configuration (provider="kokoro")

```yaml
tts:
  provider: "kokoro"
  kokoro:
    model: "mlx-community/Kokoro-82M-bf16"
    server:
      host: "127.0.0.1"
      port: 8880
      auto_start: true
    voices:
      speaker_1: "af_bella"  # Female voice (Host)
      speaker_2: "am_adam"    # Male voice (Expert)
```

### Voice Cloning Configuration (provider="voice_clone")

Voice cloning uses a TTS model to clone voices from reference audio samples. Place your voice profiles in the `Profile/` directory.

```yaml
tts:
  provider: "voice_clone"
  voice_clone:
    model: "<YOUR_TTS_MODEL_ID>"
    device: "auto"  # auto, cuda:0, mps, cpu
    dtype: "bfloat16"
    attention: "flash_attention_2"
    max_tokens: 2048
    voices:
      speaker_1:  # Female voice (Host)
        profile: "ProfileA"
        ref_audio: "Profile/ProfileA/profile_a.wav"
        ref_text: "The transcript of the reference audio..."
        language: "English"
      speaker_2:  # Male voice (Expert)
        profile: "ProfileB"
        ref_audio: "Profile/ProfileB/profile_b.wav"
        ref_text: "The transcript of the reference audio..."
        language: "English"
```

### Voice Profile Setup

To use voice cloning, you need reference audio files and their transcripts:

1. **Create a profile directory**: `Profile/<ProfileName>/`
2. **Add reference audio**: `<ProfileName>.wav` (5-30 seconds of clear speech)
3. **Add transcript**: `<ProfileName>.txt` (exact text spoken in the audio)

Example structure:
```
Profile/
├── ProfileA/
│   ├── profile_a.wav    # Reference audio
│   └── profile_a.txt    # Transcript
├── ProfileB/
│   ├── profile_b.wav
│   └── profile_b.txt
```

### Full Configuration Example

```yaml
# LLM Configuration
llm:
  model: "<YOUR_LLM_MODEL_ID>"
  base_url: "http://localhost:1234/v1"
  temperature: 0.7
  max_tokens: 262144

# TTS Configuration (change provider to switch)
tts:
  provider: "voice_clone"  # or "kokoro"
  # ... provider-specific settings ...

# Conversation Style
conversation:
  conversation_style:
    - casual
    - informative
  podcast_name: "Local Podcast"
  creativity: 0.7
  podcast_mode: "summary"  # Options: summary, analysis, full
  user_instructions: ""    # Custom instructions for analysis mode
```

---

## Podcast Modes

| Mode | Description | Length | Use Case |
|------|-------------|--------|----------|
| **Summary** | Quick overview of key points | ~10 min | Get a fast summary of the main ideas |
| **Analysis** | Detailed discussion with insights | ~5 min | Deep dive with custom focus areas |
| **Full** | Complete coverage of all content | Dynamic | Comprehensive coverage of entire document |

### Custom Instructions (Analysis Mode)

When using **Analysis mode** in the GUI, a text box appears where you can optionally enter custom instructions to guide the podcast discussion. This allows you to:

- **Focus on specific topics** - e.g., "Focus on the financial implications"
- **Set discussion tone** - e.g., "Make it more critical and analytical"
- **Highlight specific sections** - e.g., "Emphasize the methodology section"
- **Add context** - e.g., "Discuss this in context of recent market trends"

**Example custom instructions:**
```
Focus on the legal implications and discuss how this affects small businesses
```

The instructions are passed directly to the LLM and influence how the podcast script is generated.

---

**Note:** Podcast length is now **dynamic** based on input content:
- Short content (< 500 words): ~2x expansion for engaging discussion
- Medium content (500-2000 words): ~1.5x for thorough coverage
- Long content (2000-5000 words): ~1.2x for detailed discussion
- Very long content (> 5000 words): ~1.1x - full coverage, no summarization

The podcast will be as long as needed to cover all content properly!

---

## Available Voices

### Kokoro TTS Voices (provider="kokoro")

| Voice ID | Description |
|----------|-------------|
| `af_bella` | Female, natural and warm |
| `af_sarah` | Female, professional |
| `am_adam` | Male, clear and articulate |
| `am_michael` | Male, deeper tone |

### Voice Cloning Profiles (provider="voice_clone")

Voice cloning uses custom voice profiles from the `Profile/` directory. Each profile contains:
- Reference audio file (5-30 seconds of clear speech)
- Transcript of the reference audio

**Included Profiles:**
| Profile | Description | Use Case |
|---------|-------------|----------|
| `ProfileA` | Female voice | Host/Speaker 1 |
| `ProfileB` | Male voice, authoritative | Expert/Speaker 2 |
| `ProfileC` | Custom voice | User-defined |

**Creating Custom Profiles:**
1. Record 5-30 seconds of clear speech
2. Create a transcript of exactly what was spoken
3. Place files in `Profile/<YourName>/` directory

---

## Troubleshooting

### Local LLM Server Connection Error

```
Cannot connect to local LLM server at http://localhost:1234/v1
```

**Solution:**
1. Open your LLM server application
2. Go to Server tab (left sidebar)
3. Click "Start Server"
4. Ensure port is 1234

### Kokoro Model Not Found

```
Error: Failed to load Kokoro model
```

**Solution:**
1. Ensure internet connection for first download
2. Model is cached locally after first download (~82MB)
3. Check MLX-Audio installation: `pip install mlx-audio --upgrade`

### Voice Cloning Model Not Found

```
Error: Failed to load voice cloning model
```

**Solution:**
1. Ensure internet connection for first download
2. Model is cached locally after first download (~1.7GB)
3. Check the TTS package installation: `pip install <tts-package> --upgrade`

### Voice Cloning Reference Audio Not Found

```
Error: Reference audio not found: Profile/Name/name.wav
```

**Solution:**
1. Check that the profile directory exists in `Profile/`
2. Verify the audio file has the correct name
3. Ensure the path in `config.yaml` matches the actual file location

### Memory Issues with Voice Cloning

Voice cloning requires more memory than Kokoro. If you encounter memory errors:

1. Use `device: "cpu"` in config.yaml (slower but more stable)
2. Use `dtype: "float16"` instead of `bfloat16`
3. Close other applications to free memory

### Python Virtual Environment Issues

If you see errors about missing packages:

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Port Already in Use

```
Error: Port 8880 already in use
```

**Solution:**
1. The system auto-finds available ports
2. Or kill existing process: `lsof -ti:8880 | xargs kill -9`

---

## Tips for Best Results

1. **Content Quality** - Well-structured documents with clear sections produce better podcasts
2. **Document Formats** - PDF, DOCX, and PPTX files are automatically converted to markdown
3. **Temperature** - Lower (0.5-0.7) for factual, higher (0.8-1.0) for creative content
4. **Word Count** - 2000 words ≈ 10-15 minutes of audio
5. **Input Format** - Use markdown headers and bullet points for better structure
6. **PDF Documents** - Text-based PDFs work best; scanned PDFs may need OCR preprocessing

---

## Updating the Project

```bash
# Navigate to your project directory (adjust path as needed)
cd Podcastfy
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## License

This project uses open-source components:
- Kokoro TTS model from mlx-community
- MLX-Audio framework from Apple
- CustomTkinter for GUI

---

## Acknowledgments

- Original Podcastfy project - Inspiration
- Local LLM server tools - Local inference runtime
- MLX-Audio - Apple Silicon TTS

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Open an issue on GitHub at your repository's issue tracker
