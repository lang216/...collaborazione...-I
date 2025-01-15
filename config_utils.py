"""Configuration utilities for audio processing system."""

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    
    # Audio parameters
    sr: int
    mono: bool
    hop_length: int
    n_fft: int
    
    # Feature parameters
    n_mfcc: int
    
    # Freesound parameters
    duration_multiplier: float
    max_parallel_searches: int
    min_results_per_file: int
    output_subfolder_format: str
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.max_parallel_searches <= 0:
            raise ValueError("max_parallel_searches must be positive")

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to config.json. If None, looks in script directory.
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.json"
    
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")

def create_audio_config(config: Dict[str, Any]) -> AudioConfig:
    """Create AudioConfig from configuration dictionary.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        AudioConfig instance with parameters from config
    """
    return AudioConfig(
        sr=config["audio"]["sr"],
        mono=config["audio"]["mono"],
        hop_length=config["audio"]["hop_length"],
        n_fft=config["audio"]["n_fft"],
        n_mfcc=config["features"]["n_mfcc"],
        duration_multiplier=config["freesound"]["duration_multiplier"],
        max_parallel_searches=config["freesound"]["max_parallel_searches"],
        min_results_per_file=config["freesound"]["min_results_per_file"],
        output_subfolder_format=config["freesound"]["output_subfolder_format"]
    ) 