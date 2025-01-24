"""Shared utilities for audio processing scripts"""
from pathlib import Path
import logging
from typing import List, Dict, Optional, Union
import argparse

def setup_logging(log_file: str) -> None:
    """Configure consistent logging across all scripts"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_directory(path: Union[str, Path]) -> Path:
    """Validate and convert directory path"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    return path

def get_audio_files(directory: Path, extensions: List[str] = ['.wav', '.mp3']) -> List[Path]:
    """Recursively find audio files with given extensions"""
    return [f for f in directory.rglob('*') if f.suffix.lower() in extensions]

class AudioScriptArgumentParser(argparse.ArgumentParser):
    """Standardized argument parser for audio scripts"""
    def __init__(self, description: str):
        super().__init__(description=description)
        self.add_standard_arguments()
        
    def add_standard_arguments(self):
        """Add standard arguments common to all audio scripts"""
        self.add_argument('directory', type=str, help='Directory to process')
        self.add_argument('--log', default='utils.log', help='Log file path')
        
    def parse_args(self):
        args = super().parse_args()
        args.directory = validate_directory(args.directory)
        return args
