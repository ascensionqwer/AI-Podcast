#!/bin/bash
# Podcastfy Local - Desktop Launcher
# Double-click this file to launch the GUI application
# Uses global Python (no virtual environment)

# Project directory - UPDATE THIS PATH if you moved the project
PROJECT_DIR="/Users/wesleygwn/Documents/Work/Codes/Podcastfy"

echo "🎙️ Podcastfy Local - AI Podcast Generator"
echo "=========================================="
echo ""

# Navigate to the project directory
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "📂 Project directory: $PROJECT_DIR"
else
    echo "❌ Project not found at: $PROJECT_DIR"
    echo ""
    echo "Please update the PROJECT_DIR path in this script"
    echo "or clone the project to the expected location."
    read -p "Press Enter to close..."
    exit 1
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python3 -c "import customtkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Dependencies not installed"
    echo "📦 Installing dependencies globally..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        echo "Try running: pip3 install -r requirements.txt"
        read -p "Press Enter to exit..."
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies found"
fi

echo ""
echo "🚀 Launching Podcastfy GUI..."
echo ""

# Launch the GUI using global Python
python3 gui.py

# Keep terminal open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with an error"
    read -p "Press Enter to close..."
fi