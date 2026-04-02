# Podcastfy Local - AI Podcast Generator

A unified podcast generation system that runs **100% locally** on Apple Silicon using:
- **LM Studio** with qwen3.5-122b-a10b for script generation
- **Kokoro TTS** via MLX-Audio for high-quality voice synthesis
- **Docling** for document conversion (PDF, DOCX, PPTX, etc.)

![Podcastfy Local GUI](https://via.placeholder.com/800x400?text=Podcastfy+Local+GUI)

---

## 📄 Supported Input Formats

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

Documents are automatically converted to markdown using [Docling](https://github.com/docling-project/docling) before being processed by the LLM. No intermediate files are saved - the converted content is fed directly to the script generator.

---

## 📋 Complete Setup Guide for New MacBook

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

### Step 3: Install LM Studio

1. Download LM Studio from: https://lmstudio.ai/download
2. Open the downloaded `.dmg` file
3. Drag LM Studio to Applications folder
4. Open LM Studio from Applications

**Download the Required Model:**

1. In LM Studio, click the **Search** icon (magnifying glass)
2. Search for: `qwen3.5-122b-a10b`
3. Download the model (this is a large model, ~70GB, may take time)
4. Once downloaded, click the **Chat** icon
5. Select `qwen3.5-122b-a10b` from the model dropdown
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
cd ~/Documents/Work/Codes

# Clone the repository
git clone https://github.com/ascensionqwer/AI_Local_Podcast.git Podcastfy

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
- `openai` - For LM Studio API communication
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
python podcast.py --input ./assets/FSO_Law_Indonesia.md --output ./output/podcast.wav
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

## 🎯 Quick Start Guide

### Using the GUI

1. **Start LM Studio** - Ensure the server is running (port 1234)
2. **Launch Podcastfy** - Run `python gui.py` or double-click the desktop launcher
3. **Select Input File** - Click on a file in the Assets list or upload a new one
4. **Set Output Name** - Enter a filename (e.g., `my_podcast.wav`)
5. **Click Generate** - Press the "🚀 Generate Podcast" button
6. **Wait for Completion** - The progress bar shows generation status
7. **Play Output** - Click on the generated file in the Output list

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

## 📁 Project Structure

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
│   └── FSO_Law_Indonesia.md
└── output/                 # Generated podcasts (.wav, .mp3)
```

---

## ⚙️ Configuration

Edit `config.yaml` to customize settings:

```yaml
# LLM Configuration
llm:
  model: "qwen3.5-122b-a10b"
  base_url: "http://localhost:1234/v1"
  temperature: 0.7
  max_tokens: 262144

# TTS Configuration
tts:
  provider: "kokoro"
  model: "mlx-community/Kokoro-82M-bf16"
  server:
    host: "127.0.0.1"
    port: 8880
    auto_start: true
  voices:
    speaker_1: "af_bella"  # Female voice (Host)
    speaker_2: "am_adam"    # Male voice (Expert)

# Conversation Style
conversation:
  conversation_style:
    - casual
    - informative
  podcast_name: "Local Podcast"
  creativity: 0.7
```

**Note:** Podcast length is now **dynamic** based on input content:
- Short content (< 500 words): ~2x expansion for engaging discussion
- Medium content (500-2000 words): ~1.5x for thorough coverage
- Long content (2000-5000 words): ~1.2x for detailed discussion
- Very long content (> 5000 words): ~1.1x - full coverage, no summarization

The podcast will be as long as needed to cover all content properly!

---

## 🎙️ Available Voices

| Voice ID | Description |
|----------|-------------|
| `af_bella` | Female, natural and warm |
| `af_sarah` | Female, professional |
| `am_adam` | Male, clear and articulate |
| `am_michael` | Male, deeper tone |

---

## 🔧 Troubleshooting

### LM Studio Connection Error

```
❌ Cannot connect to LM Studio at http://localhost:1234/v1
```

**Solution:**
1. Open LM Studio
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

## 💡 Tips for Best Results

1. **Content Quality** - Well-structured documents with clear sections produce better podcasts
2. **Document Formats** - PDF, DOCX, and PPTX files are automatically converted to markdown
3. **Temperature** - Lower (0.5-0.7) for factual, higher (0.8-1.0) for creative content
4. **Word Count** - 2000 words ≈ 10-15 minutes of audio
5. **Input Format** - Use markdown headers and bullet points for better structure
6. **PDF Documents** - Text-based PDFs work best; scanned PDFs may need OCR preprocessing

---

## 🔄 Updating the Project

```bash
# Navigate to your project directory (adjust path as needed)
cd Podcastfy
git pull origin main
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## 📝 License

This project uses open-source components:
- Kokoro TTS model from mlx-community
- MLX-Audio framework from Apple
- CustomTkinter for GUI

---

## 🙏 Acknowledgments

- [Podcastfy](https://github.com/souzatharsis/podcastfy) - Inspiration
- [LM Studio](https://lmstudio.ai/) - Local LLM runtime
- [MLX-Audio](https://github.com/Blaizzy/mlx-audio) - Apple Silicon TTS

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section above
2. Open an issue on GitHub: https://github.com/ascensionqwer/AI_Local_Podcast/issues