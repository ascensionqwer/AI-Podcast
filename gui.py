#!/usr/bin/env python3
"""
Podcastfy Local - Desktop GUI Application
A modern GUI for generating podcasts using LM Studio + Kokoro TTS.

Run with: python gui.py
"""

import os
import sys
import threading
import shutil
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
from PIL import Image

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class PodcastfyApp(ctk.CTk):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Podcastfy Local - AI Podcast Generator")
        self.geometry("900x650")
        self.minsize(800, 550)
        
        # Get base directory
        self.base_dir = Path(__file__).parent.absolute()
        self.assets_dir = self.base_dir / "assets"
        self.output_dir = self.base_dir / "output"
        
        # Ensure directories exist
        self.assets_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # State variables
        self.selected_file = None
        self.output_filename = ctk.StringVar(value="podcast.wav")
        self.podcast_mode = ctk.StringVar(value="summary")
        self.is_generating = False
        self.status_text = ctk.StringVar(value="Ready - Select a file to begin")
        
        # Configure grid layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Create UI components
        self._create_sidebar()
        self._create_main_area()
        self._create_status_bar()
        
        # Load existing files
        self._refresh_file_lists()
        
        # Bind focus event for auto-refresh
        self.bind("<FocusIn>", self._on_focus_in)
    
    def _on_focus_in(self, event):
        """Refresh file lists when window gains focus."""
        self._refresh_file_lists()
    
    def _create_sidebar(self):
        """Create the left sidebar with controls."""
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(5, weight=1)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🎙️ Podcastfy Local",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar,
            text="AI Podcast Generator",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # File Selection Section
        self.file_section = ctk.CTkLabel(
            self.sidebar,
            text="─── Input File ───",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.file_section.grid(row=2, column=0, padx=20, pady=(10, 5))
        
        # Upload button
        self.upload_btn = ctk.CTkButton(
            self.sidebar,
            text="📁 Upload File (.md/.txt)",
            command=self._upload_file,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.upload_btn.grid(row=3, column=0, padx=20, pady=5)
        
        # Selected file display
        self.selected_file_label = ctk.CTkLabel(
            self.sidebar,
            text="No file selected",
            font=ctk.CTkFont(size=11),
            wraplength=240
        )
        self.selected_file_label.grid(row=4, column=0, padx=20, pady=5)
        
        # Assets folder buttons
        self.assets_btn = ctk.CTkButton(
            self.sidebar,
            text="📂 Open Assets Folder",
            command=self._open_assets_folder,
            height=32,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            font=ctk.CTkFont(size=12)
        )
        self.assets_btn.grid(row=5, column=0, padx=20, pady=5, sticky="n")
        
        # Output Section
        self.output_section = ctk.CTkLabel(
            self.sidebar,
            text="─── Output ───",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.output_section.grid(row=6, column=0, padx=20, pady=(15, 5))
        
        # Output filename entry
        self.output_entry = ctk.CTkEntry(
            self.sidebar,
            placeholder_text="podcast.wav",
            textvariable=self.output_filename,
            height=32,
            font=ctk.CTkFont(size=12)
        )
        self.output_entry.grid(row=7, column=0, padx=20, pady=5)
        
        # Output folder button
        self.output_btn = ctk.CTkButton(
            self.sidebar,
            text="📂 Open Output Folder",
            command=self._open_output_folder,
            height=32,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "#DCE4EE"),
            font=ctk.CTkFont(size=12)
        )
        self.output_btn.grid(row=8, column=0, padx=20, pady=5)
        
        # Podcast Mode Section
        self.mode_section = ctk.CTkLabel(
            self.sidebar,
            text="─── Podcast Mode ───",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.mode_section.grid(row=9, column=0, padx=20, pady=(15, 5))
        
        # Podcast mode dropdown
        self.mode_dropdown = ctk.CTkOptionMenu(
            self.sidebar,
            values=["summary", "analysis", "full"],
            variable=self.podcast_mode,
            command=self._on_mode_change,
            height=32,
            font=ctk.CTkFont(size=12)
        )
        self.mode_dropdown.grid(row=10, column=0, padx=20, pady=5)
        
        # Mode description
        self.mode_desc = ctk.CTkLabel(
            self.sidebar,
            text="summary: Quick overview (~400 words)",
            font=ctk.CTkFont(size=10),
            text_color="gray",
            wraplength=240
        )
        self.mode_desc.grid(row=11, column=0, padx=20, pady=(0, 5))
        
        # Generate Section
        self.gen_section = ctk.CTkLabel(
            self.sidebar,
            text="─── Generate ───",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.gen_section.grid(row=12, column=0, padx=20, pady=(10, 5))
        
        # Script only checkbox
        self.script_only_var = ctk.StringVar(value="off")
        self.script_only_cb = ctk.CTkCheckBox(
            self.sidebar,
            text="Script only (no audio)",
            variable=self.script_only_var,
            onvalue="on",
            offvalue="off",
            font=ctk.CTkFont(size=12)
        )
        self.script_only_cb.grid(row=13, column=0, padx=20, pady=5)
        
        # Generate button
        self.generate_btn = ctk.CTkButton(
            self.sidebar,
            text="🚀 Generate Podcast",
            command=self._generate_podcast,
            height=50,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color="#1f6aa5",
            hover_color="#144870"
        )
        self.generate_btn.grid(row=14, column=0, padx=20, pady=10)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            self.sidebar,
            width=220,
            mode="indeterminate"
        )
        self.progress_bar.grid(row=15, column=0, padx=20, pady=5)
        self.progress_bar.set(0)
        
        # Version label
        self.version_label = ctk.CTkLabel(
            self.sidebar,
            text="v1.0.0 | LM Studio + Kokoro",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.version_label.grid(row=16, column=0, padx=20, pady=(10, 10))
    
    def _create_main_area(self):
        """Create the main content area."""
        # Main frame
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Header with refresh button
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.header_frame.grid_columnconfigure(0, weight=1)
        
        self.header_label = ctk.CTkLabel(
            self.header_frame,
            text="Files in Assets Folder",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        self.header_label.grid(row=0, column=0, sticky="w")
        
        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            self.header_frame,
            text="🔄 Refresh",
            command=self._refresh_file_lists,
            width=100,
            height=32,
            font=ctk.CTkFont(size=12)
        )
        self.refresh_btn.grid(row=0, column=1, padx=10)
        
        # File list frame
        self.file_list_frame = ctk.CTkScrollableFrame(
            self.main_frame,
            label_text="Available Input Files (click to select)"
        )
        self.file_list_frame.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Output files frame
        self.output_list_frame = ctk.CTkScrollableFrame(
            self.main_frame,
            label_text="Generated Podcasts (click to play)"
        )
        self.output_list_frame.grid(row=2, column=0, sticky="nsew", pady=10)
    
    def _create_status_bar(self):
        """Create the bottom status bar."""
        self.status_frame = ctk.CTkFrame(self, height=40, corner_radius=0)
        self.status_frame.grid(row=3, column=1, sticky="ew", padx=20, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            textvariable=self.status_text,
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=10)
        
        # LM Studio status indicator
        self.llm_status = ctk.CTkLabel(
            self.status_frame,
            text="🟡 LM Studio",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.llm_status.pack(side="right", padx=10)
        
        # Check LM Studio status
        self._check_llm_status()
    
    def _refresh_file_lists(self):
        """Refresh the file lists in the main area."""
        # Clear existing widgets
        for widget in self.file_list_frame.winfo_children():
            widget.destroy()
        
        for widget in self.output_list_frame.winfo_children():
            widget.destroy()
        
        # List assets files
        md_files = list(self.assets_dir.glob("*.md"))
        txt_files = list(self.assets_dir.glob("*.txt"))
        all_input_files = sorted(md_files + txt_files, key=lambda x: x.name.lower())
        
        if all_input_files:
            for file_path in all_input_files:
                # Highlight if this is the selected file
                is_selected = self.selected_file and self.selected_file == file_path
                btn_text = f"{'✅ ' if is_selected else '📄 '}{file_path.name}"
                
                btn = ctk.CTkButton(
                    self.file_list_frame,
                    text=btn_text,
                    command=lambda p=file_path: self._select_file(p),
                    height=36,
                    fg_color="#2d4a6f" if is_selected else "transparent",
                    hover_color="#3d5a7f" if is_selected else "#1a1a2e",
                    border_width=1,
                    text_color="white",
                    anchor="w",
                    font=ctk.CTkFont(size=13)
                )
                btn.pack(fill="x", pady=2)
        else:
            empty_label = ctk.CTkLabel(
                self.file_list_frame,
                text="No .md or .txt files in assets folder\n\nClick 'Upload File' or add files to the assets folder",
                text_color="gray",
                font=ctk.CTkFont(size=12)
            )
            empty_label.pack(pady=20)
        
        # List output files
        wav_files = sorted(self.output_dir.glob("*.wav"), key=lambda x: x.name.lower())
        mp3_files = sorted(self.output_dir.glob("*.mp3"), key=lambda x: x.name.lower())
        txt_outputs = sorted(self.output_dir.glob("*.txt"), key=lambda x: x.name.lower())
        all_outputs = wav_files + mp3_files + txt_outputs
        
        if all_outputs:
            for file_path in all_outputs:
                btn = ctk.CTkButton(
                    self.output_list_frame,
                    text=f"🎵 {file_path.name}",
                    command=lambda p=file_path: self._play_output(p),
                    height=36,
                    fg_color="transparent",
                    hover_color="#1a1a2e",
                    border_width=1,
                    text_color="white",
                    anchor="w",
                    font=ctk.CTkFont(size=13)
                )
                btn.pack(fill="x", pady=2)
        else:
            empty_label = ctk.CTkLabel(
                self.output_list_frame,
                text="No generated podcasts yet\n\nSelect a file and click 'Generate Podcast'",
                text_color="gray",
                font=ctk.CTkFont(size=12)
            )
            empty_label.pack(pady=20)
    
    def _upload_file(self):
        """Open file dialog to upload a file."""
        file_path = filedialog.askopenfilename(
            title="Select Input File",
            filetypes=[
                ("Markdown files", "*.md"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialdir=str(self.assets_dir)
        )
        
        if file_path:
            source_path = Path(file_path)
            
            # Copy to assets folder if not already there
            if source_path.parent != self.assets_dir:
                dest_path = self.assets_dir / source_path.name
                
                # Check if file already exists
                if dest_path.exists():
                    # Ask to overwrite
                    from tkinter import messagebox
                    if not messagebox.askyesno("File Exists", f"'{source_path.name}' already exists in assets. Overwrite?"):
                        return
                
                shutil.copy2(file_path, dest_path)
                self._select_file(dest_path)
                self._refresh_file_lists()
                self.status_text.set(f"✅ File copied: {source_path.name}")
            else:
                self._select_file(source_path)
    
    def _select_file(self, file_path: Path):
        """Select a file for podcast generation."""
        self.selected_file = file_path
        self.selected_file_label.configure(text=f"✅ {file_path.name}")
        self.status_text.set(f"Selected: {file_path.name}")
        
        # Update output filename based on input
        base_name = file_path.stem
        self.output_filename.set(f"{base_name}_podcast.wav")
        
        # Refresh to show selection highlight
        self._refresh_file_lists()
    
    def _open_assets_folder(self):
        """Open the assets folder in Finder."""
        if sys.platform == "darwin":  # macOS
            os.system(f'open "{self.assets_dir}"')
        elif sys.platform == "win32":  # Windows
            os.system(f'explorer "{self.assets_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{self.assets_dir}"')
    
    def _open_output_folder(self):
        """Open the output folder in Finder."""
        if sys.platform == "darwin":  # macOS
            os.system(f'open "{self.output_dir}"')
        elif sys.platform == "win32":  # Windows
            os.system(f'explorer "{self.output_dir}"')
        else:  # Linux
            os.system(f'xdg-open "{self.output_dir}"')
    
    def _play_output(self, file_path: Path):
        """Open/play the output file."""
        if sys.platform == "darwin":  # macOS
            os.system(f'open "{file_path}"')
        elif sys.platform == "win32":  # Windows
            os.system(f'explorer "{file_path}"')
        else:  # Linux
            os.system(f'xdg-open "{file_path}"')
    
    def _check_llm_status(self):
        """Check if LM Studio is running."""
        try:
            import httpx
            response = httpx.get("http://localhost:1234/v1/models", timeout=2.0)
            if response.status_code == 200:
                self.llm_status.configure(text="🟢 LM Studio Connected", text_color="green")
            else:
                self.llm_status.configure(text="🔴 LM Studio Error", text_color="red")
        except Exception:
            self.llm_status.configure(text="🔴 LM Studio Offline", text_color="red")
    
    def _generate_podcast(self):
        """Start podcast generation."""
        if self.is_generating:
            return
        
        if not self.selected_file:
            self.status_text.set("❌ Please select a file first!")
            return
        
        # Check LM Studio
        try:
            import httpx
            response = httpx.get("http://localhost:1234/v1/models", timeout=2.0)
            if response.status_code != 200:
                self.status_text.set("❌ LM Studio not responding. Please start LM Studio.")
                return
        except Exception:
            self.status_text.set("❌ Cannot connect to LM Studio. Please start LM Studio first.")
            return
        
        # Start generation in background thread
        self.is_generating = True
        self.generate_btn.configure(state="disabled", text="⏳ Generating...")
        self.progress_bar.start()
        self.status_text.set("🚀 Starting podcast generation...")
        
        thread = threading.Thread(
            target=self._generation_thread,
            daemon=True
        )
        thread.start()
    
    def _on_mode_change(self, choice: str):
        """Handle podcast mode selection change."""
        descriptions = {
            "summary": "summary: Quick overview (~10 min max)",
            "analysis": "analysis: Detailed discussion (~5 min)",
            "full": "full: Complete coverage (covers everything)"
        }
        self.mode_desc.configure(text=descriptions.get(choice, ""))
    
    def _generation_thread(self):
        """Background thread for podcast generation."""
        try:
            from src.config import load_config
            from src.generator import PodcastGenerator
            
            # Load config
            config = load_config(str(self.base_dir / "config.yaml"))
            
            # Override podcast mode based on GUI selection
            config.conversation.podcast_mode = self.podcast_mode.get()
            
            # Get output path with mode suffix
            output_name = self.output_filename.get()
            # Remove extension if present
            base_name = output_name.rsplit('.', 1)[0] if '.' in output_name else output_name
            # Add mode suffix
            mode_suffix = f"_podcast_{self.podcast_mode.get()}"
            output_name = f"{base_name}{mode_suffix}.wav"
            output_path = self.output_dir / output_name
            
            # Script only mode
            script_only = self.script_only_var.get() == "on"
            
            # Update status with mode info
            mode_name = self.podcast_mode.get().capitalize()
            self.after(0, lambda: self.status_text.set(f"🧠 Generating {mode_name} podcast script..."))
            
            # Generate
            generator = PodcastGenerator(config)
            result = generator.generate(
                input_path=str(self.selected_file),
                output_path=str(output_path),
                script_only=script_only
            )
            
            # Success
            self.after(0, lambda: self._generation_complete(True, str(output_path)))
            
        except Exception as e:
            self.after(0, lambda exc=str(e): self._generation_complete(False, exc))
    
    def _generation_complete(self, success: bool, message: str):
        """Handle generation completion."""
        self.is_generating = False
        self.generate_btn.configure(state="normal", text="🚀 Generate Podcast")
        self.progress_bar.stop()
        self.progress_bar.set(0)
        
        if success:
            self.status_text.set(f"✅ Success! Generated: {Path(message).name}")
            self._refresh_file_lists()
        else:
            self.status_text.set(f"❌ Error: {message}")
    
    def run(self):
        """Start the application."""
        self.mainloop()


def main():
    """Main entry point."""
    app = PodcastfyApp()
    app.run()


if __name__ == "__main__":
    main()