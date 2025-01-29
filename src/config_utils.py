"""Configuration utilities for audio processing system."""

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SprayNotesConfig:
    """Configuration for spray notes generation."""
    density_notes_per_second: int
    total_duration: int
    lower_freq: float
    upper_freq: float
    min_note_duration: float
    max_note_duration: float
    min_velocity: int
    max_velocity: int
    output_filename: str
    
    def validate(self) -> None:
        """Validate spray notes configuration parameters."""
        if self.density_notes_per_second <= 0:
            raise ValueError("density_notes_per_second must be positive")
        if self.total_duration <= 0:
            raise ValueError("total_duration must be positive")
        if self.lower_freq <= 0 or self.upper_freq <= 0:
            raise ValueError("frequencies must be positive")
        if self.lower_freq >= self.upper_freq:
            raise ValueError("upper_freq must be greater than lower_freq")
        if self.min_note_duration <= 0 or self.max_note_duration <= 0:
            raise ValueError("note durations must be positive")
        if self.min_note_duration >= self.max_note_duration:
            raise ValueError("max_note_duration must be greater than min_note_duration")
        if not (0 <= self.min_velocity <= 127) or not (0 <= self.max_velocity <= 127):
            raise ValueError("velocity must be between 0 and 127")
        if self.min_velocity >= self.max_velocity:
            raise ValueError("max_velocity must be greater than min_velocity")

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
        config_path = Path(__file__).parent.parent / "config.json"
    
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

def create_spray_notes_config(config: Dict[str, Any]) -> SprayNotesConfig:
    """Create SprayNotesConfig from configuration dictionary.
    
    Args:
        config: Configuration dictionary from load_config()
        
    Returns:
        SprayNotesConfig instance with parameters from config
    """
    return SprayNotesConfig(
        density_notes_per_second=config["spray_notes"]["density_notes_per_second"],
        total_duration=config["spray_notes"]["total_duration"],
        lower_freq=config["spray_notes"]["lower_freq"],
        upper_freq=config["spray_notes"]["upper_freq"],
        min_note_duration=config["spray_notes"]["min_note_duration"],
        max_note_duration=config["spray_notes"]["max_note_duration"],
        min_velocity=config["spray_notes"]["min_velocity"],
        max_velocity=config["spray_notes"]["max_velocity"],
        output_filename=config["spray_notes"]["output_filename"]
    )
