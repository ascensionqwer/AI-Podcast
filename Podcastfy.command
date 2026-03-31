#!/bin/bash
# Podcastfy Local - Desktop Launcher
# Double-click this file to launch the GUI application

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Launch the GUI
echo "🎙️ Launching Podcastfy Local..."
python gui.py

# Deactivate when done
deactivate