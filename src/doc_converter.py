"""
Document to Markdown converter using Docling.
Converts PDF, DOCX, PPTX and other supported formats to markdown.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Supported document extensions (besides .md and .txt which are handled natively)
SUPPORTED_DOC_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.doc', '.ppt', '.xlsx', '.xls', '.html', '.htm'}

# Cache for the converter instance
_converter = None


def get_converter():
    """
    Get or create the DocumentConverter instance (lazy loading).
    
    Returns:
        DocumentConverter instance
    """
    global _converter
    if _converter is None:
        try:
            from docling.document_converter import DocumentConverter
            _converter = DocumentConverter()
            logger.info("📄 Docling converter initialized")
        except ImportError:
            logger.warning("Docling not installed. Document conversion will be limited.")
            return None
    return _converter


def is_document_file(file_path: str) -> bool:
    """
    Check if a file is a supported document format (not .md or .txt).
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a supported document format
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_DOC_EXTENSIONS


def is_supported_file(file_path: str) -> bool:
    """
    Check if a file is supported for content loading.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file format is supported
    """
    ext = Path(file_path).suffix.lower()
    return ext in SUPPORTED_DOC_EXTENSIONS or ext in {'.md', '.txt', '.markdown'}


def convert_document_to_markdown(source_path: str) -> str:
    """
    Convert a document to markdown content.
    
    Args:
        source_path: Path to the source document (PDF, DOCX, PPTX, etc.)
    
    Returns:
        The markdown content as a string.
    
    Raises:
        FileNotFoundError: If the source file doesn't exist
        ImportError: If docling is not installed
        RuntimeError: If conversion fails
    """
    source = Path(source_path)
    
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    converter = get_converter()
    if converter is None:
        raise ImportError(
            "Docling is not installed. Install it with: pip install docling\n"
            "Document conversion requires the 'docling' package."
        )
    
    try:
        logger.info(f"📄 Converting document: {source_path}")
        
        # Convert the document
        result = converter.convert(str(source))
        
        # Export to markdown (no file saving, just return the content)
        markdown_content = result.document.export_to_markdown()
        
        logger.info(f"✅ Document converted successfully ({len(markdown_content)} characters)")
        
        return markdown_content
        
    except Exception as e:
        logger.error(f"Document conversion failed: {e}")
        raise RuntimeError(f"Failed to convert document {source_path}: {e}")


def load_document_content(file_path: str) -> str:
    """
    Load content from a file, converting to markdown if needed.
    
    This function handles both native text files (.md, .txt) and
    document files (.pdf, .docx, .pptx, etc.) that need conversion.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Content as markdown string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
        RuntimeError: If conversion fails
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    ext = path.suffix.lower()
    
    # Native text formats - read directly
    if ext in {'.md', '.txt', '.markdown'}:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"📄 Loaded text file: {file_path} ({len(content)} characters)")
        return content
    
    # Document formats - convert to markdown
    if ext in SUPPORTED_DOC_EXTENSIONS:
        return convert_document_to_markdown(file_path)
    
    # Unsupported format
    supported = ', '.join(SUPPORTED_DOC_EXTENSIONS | {'.md', '.txt', '.markdown'})
    raise ValueError(
        f"Unsupported file format: {ext}\n"
        f"Supported formats: {supported}"
    )