#!/bin/bash
# Podcastfy Local - Desktop Launcher
# Double-click this file to launch the GUI application

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Navigate to the project directory
cd "$SCRIPT_DIR"

echo "🎙️ Podcastfy Local - AI Podcast Generator"
echo "=========================================="
echo ""

# Check if virtual environment exists and is valid
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "✅ Found existing virtual environment"
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found or incomplete"
    echo "📦 Setting up for first-time use..."
    echo ""
    
    # Create virtual environment
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        echo "Please ensure Python 3 is installed"
        read -p "Press Enter to exit..."
        exit 1
    fi
    
    source venv/bin/activate
    echo "✅ Virtual environment created"
    
    # Install dependencies
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Launching Podcastfy GUI..."
echo ""

# Launch the GUI
python gui.py

# Keep terminal open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with an error"
    read -p "Press Enter to close..."
fi

# Deactivate when done
deactivate 2>/dev/null