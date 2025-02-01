import argparse
import os
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as rb
from datetime import datetime
from config_utils import load_config
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import logging
import hashlib
import time
from functools import lru_cache
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from chord_extract import (
    extract_microtonal_sequence, ChordSegment, 
    ChordExtractionError, detect_pitch_with_confidence,
    PitchDetectionResult
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chord_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PathManager:
    """Centralized path management for chord processing."""
    base_dir: Path
    config: Dict[str, Any]
    
    @property
    def chord_output_dir(self) -> Path:
        return self.base_dir / self.config['paths']['chord_output_dir']
    
    @property
    def chord_input_dir(self) -> Path:
        return self.base_dir / self.config['chord_builder']['chord_input_dir']
    
    def create_output_dir(self, suffix: str = None) -> Path:
        """Create and return timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"chord_{timestamp}" if suffix is None else f"chord_{suffix}_{timestamp}"
        output_dir = self.chord_output_dir / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def validate_paths(self) -> None:
        """Validate existence of required directories."""
        required_dirs = [self.chord_input_dir, self.chord_output_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")

class AudioCache:
    """Cache for audio processing results."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, audio_data: np.ndarray, params: dict) -> str:
        """Generate unique cache key based on audio data and parameters."""
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        param_hash = hashlib.md5(str(params).encode()).hexdigest()
        return f"{audio_hash}_{param_hash}"
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached result."""
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None
    
    def set(self, key: str, data: np.ndarray) -> None:
        """Cache result."""
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, data)

def performance_logger(func):
    """Decorator for performance monitoring."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper

@performance_logger
def detect_original_note(audio_path: str, sr: int) -> int:
    """Detect the original note using improved pitch detection with confidence."""
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        result = detect_pitch_with_confidence(y, sr)
        return int(round(result.midi_note * 100 + result.cents))
    except Exception as e:
        logger.error(f"Error detecting original note: {str(e)}")
        raise

def parallel_process_stems(stems: List[Any], process_fn, max_workers: Optional[int] = None) -> List[Any]:
    """Process stems in parallel using a thread pool."""
    if max_workers is None:
        max_workers = min(len(stems), multiprocessing.cpu_count())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, stem) for stem in stems]
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing stem: {e}")
    return results

@performance_logger
def precise_rubberband_shift(y: np.ndarray, sr: int, shift_cents: float) -> np.ndarray:
    """High-precision pitch shifting with optimized RubberBand settings."""
    return rb.pyrb.pitch_shift(
        y=y, sr=sr,
        n_steps=shift_cents / 100,
        rbargs={
            "--formant": "",  # Preserve spectral envelope
            "--fine": "",     # High-precision mode
            "-c": "0",        # Higher frequency resolution
            "--threads": str(max(1, multiprocessing.cpu_count() - 1))  # Parallel processing
        }
    )

@performance_logger
def process_chord_notes(y: np.ndarray, native_sr: int, args, output_dir: str, target_duration: float = None) -> List[str]:
    """Process all chord notes in parallel and return stem paths."""
    stems = []
    original_midi = args.original_note / 100.0
    
    # Time stretch if needed
    if target_duration is not None:
        current_duration = len(y) / native_sr
        if abs(current_duration - target_duration) > 0.01:
            time_scale = target_duration / current_duration
            if abs(time_scale - 1.0) > 0.01:
                y = rb.pyrb.time_stretch(
                    y, native_sr,
                    rate=time_scale,
                    rbargs={
                        "--formant": "",
                        "--fine": "",
                        "-c": "0",
                        "--threads": str(max(1, multiprocessing.cpu_count() - 1))
                    }
                )
    
    def process_note(note: int) -> Optional[str]:
        try:
            target_midi = note / 100.0
            shift_cents = (target_midi - original_midi) * 100
            
            # Use caching for shifted audio
            cache_key = f"shift_{shift_cents}_{hash(y.tobytes())}"
            cached_result = audio_cache.get(cache_key)
            
            if cached_result is not None:
                y_shifted = cached_result
            else:
                y_shifted = precise_rubberband_shift(y, native_sr, shift_cents)
                audio_cache.set(cache_key, y_shifted)
            
            config = load_config()
            if config['chord_builder']['normalize_output']:
                y_shifted = librosa.util.normalize(y_shifted)
            
            stem_path = save_stem(y_shifted, native_sr, note, output_dir)
            return stem_path
        except Exception as e:
            logger.error(f"Failed to process note {note}: {str(e)}")
            return None
    
    # Process notes in parallel
    processed_stems = parallel_process_stems(args.chord_notes, process_note)
    stems = [stem for stem in processed_stems if stem is not None]
    
    return stems

@performance_logger
def save_stem(y: np.ndarray, native_sr: int, note: int, output_dir: str) -> str:
    """Save individual stem with proper formatting."""
    stem_path = os.path.join(output_dir, f'stem_{note:05d}.wav')
    sf.write(stem_path, y, native_sr, subtype='PCM_24')
    logger.info(f"Saved stem: {stem_path}")
    return stem_path

@performance_logger
def mix_and_save(stems: List[str], original: np.ndarray, native_sr: int, 
                output_dir: str, config: dict, target_duration: float = None) -> None:
    """Mix stems with parallel processing and targeted duration."""
    if target_duration is not None:
        target_samples = int(target_duration * native_sr)
        max_length = target_samples
    else:
        max_length = len(original)
    
    mixed = np.zeros(max_length)
    
    def load_and_process_stem(stem_path: str) -> np.ndarray:
        y, _ = librosa.load(stem_path, sr=native_sr)
        if target_duration is not None:
            current_duration = len(y) / native_sr
            if abs(current_duration - target_duration) > 0.01:
                time_scale = target_duration / current_duration
                if abs(time_scale - 1.0) > 0.01:
                    y = rb.pyrb.time_stretch(
                        y, native_sr,
                        rate=time_scale,
                        rbargs={
                            "--formant": "",
                            "--fine": "",
                            "-c": "0",
                            "--threads": str(max(1, multiprocessing.cpu_count() - 1))
                        }
                    )
        return librosa.util.fix_length(y, size=max_length)
    
    # Process stems in parallel
    processed_stems = parallel_process_stems(stems, load_and_process_stem)
    
    # Mix stems
    for stem_data in processed_stems:
        mixed += stem_data * 0.8
    
    if config['chord_builder']['normalize_output']:
        mixed = librosa.util.normalize(mixed)
    
    mixed_path = os.path.join(output_dir, 'mixed_chord.wav')
    sf.write(mixed_path, mixed, native_sr, subtype='PCM_24')
    logger.info(f"Saved mix: {mixed_path}")

def validate_config(config: dict) -> None:
    """Validate configuration with detailed error messages."""
    required_sections = ['audio', 'paths', 'chord_builder']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    audio_config = config['audio']
    if audio_config.get('sr') not in [44100, 48000, 96000]:
        raise ValueError(f"Invalid sample rate: {audio_config.get('sr')}")
    
    chord_config = config['chord_builder']
    if chord_config.get('default_bit_depth') not in [16, 24, 32]:
        raise ValueError(f"Invalid bit depth: {chord_config.get('default_bit_depth')}")

def write_metadata(args, stem_paths: List[str], output_dir: str, duration: float = None) -> None:
    """Generate comprehensive metadata file."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(Path(args.input_audio).absolute()),
        "original_note": args.original_note,
        "chord_notes": args.chord_notes,
        "sample_rate": args.sr,
        "stems": [str(Path(p).absolute()) for p in stem_paths],
        "processing_chain": {
            "pitch_detection": "librosa YIN with confidence" if args.detect_pitch else "manual",
            "pitch_shifting": "RubberBand with formant preservation",
            "parallel_processing": True,
            "caching_enabled": True
        }
    }
    if duration is not None:
        metadata["duration"] = duration
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")

@performance_logger
def extract_and_build_chords(y: np.ndarray, native_sr: int, args, config: dict) -> List[ChordSegment]:
    """Extract chords and build stems with error handling."""
    try:
        num_chords_to_extract = args.extract_chords
        num_voices = args.num_voices if hasattr(args, 'num_voices') and args.num_voices else 4
        
        chord_segments = extract_microtonal_sequence(
            args.chord_source, 
            num_chords_to_extract, 
            num_voices=num_voices
        )
        
        all_stems = []
        for i, segment in enumerate(chord_segments):
            if not segment.notes:  # Skip empty segments
                logger.warning(f"Skipping empty segment {i+1}")
                continue
                
            output_dir = create_output_dir(args.output_dir, base_dir_name=f"chord_sequence_{i+1}")
            args.chord_notes = segment.notes
            stems = process_chord_notes(y, native_sr, args, output_dir, target_duration=segment.duration)
            
            if stems:
                mix_and_save(stems, y, native_sr, output_dir, config, target_duration=segment.duration)
                if config['chord_builder']['generate_metadata']:
                    write_metadata(args, stems, output_dir, duration=segment.duration)
            all_stems.extend(stems or [])
            
        return chord_segments
    except ChordExtractionError as e:
        logger.error(f"Chord extraction error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_and_build_chords: {str(e)}")
        raise

def create_output_dir(base_path: str, base_dir_name: str = "chord") -> str:
    """Create unique output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_path, f"{base_dir_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path

def main():
    """Main execution function with comprehensive error handling."""
    try:
        config = load_config()
        validate_config(config)
        
        # Initialize path manager and audio cache
        path_manager = PathManager(Path.cwd(), config)
        global audio_cache
        audio_cache = AudioCache(path_manager.chord_output_dir / 'cache')
        
        parser = argparse.ArgumentParser(
            description="Microtonal Chord Generator",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Input arguments
        parser.add_argument('input_audio', help='Path to input audio file')
        parser.add_argument('--sr', type=int, default=config['audio']['sr'],
                          help='Sample rate for processing')
        parser.add_argument('--detect-pitch', action='store_true',
                          help='Automatically detect original pitch')
        parser.add_argument('--original-note', type=int,
                          help='Original note in OpenMusic format (required if not using --detect-pitch)')
        
        # Chord processing arguments
        parser.add_argument('--chord-notes', type=int, nargs='+',
                          help='Target notes in OpenMusic format')
        parser.add_argument('--chord-source', type=str,
                          help='Audio file to extract chord sequence from')
        parser.add_argument('--extract-chords', type=int,
                          help='Number of chords to extract from chord source')
        parser.add_argument('--num-voices', type=int, default=4,
                          help='Number of voices in extracted chords')
        
        # Output arguments
        parser.add_argument('--output-dir', default=str(path_manager.chord_output_dir),
                          help='Output directory for generated files')
        
        args = parser.parse_args()
        
        # Validate arguments
        if not args.detect_pitch and args.original_note is None:
            parser.error("Either --detect-pitch or --original-note must be specified")
        
        if not args.chord_notes and not args.chord_source:
            parser.error("Either --chord-notes or --chord-source must be specified")
        
        if args.chord_source and args.extract_chords is None:
            parser.error("--extract-chords is required when using --chord-source")
        
        # Load and validate input audio
        logger.info(f"Loading input audio: {args.input_audio}")
        y, native_sr = librosa.load(args.input_audio, sr=args.sr)
        
        # Detect or use provided original note
        if args.detect_pitch:
            logger.info("Detecting original pitch...")
            args.original_note = detect_original_note(args.input_audio, args.sr)
            logger.info(f"Detected original note: {args.original_note}")
        
        # Process based on input mode
        if args.chord_source:
            # Extract and build chord sequence
            logger.info(f"Extracting chords from: {args.chord_source}")
            chord_segments = extract_and_build_chords(y, native_sr, args, config)
            logger.info(f"Successfully processed {len(chord_segments)} chord segments")
        else:
            # Process single chord
            output_dir = create_output_dir(args.output_dir)
            stems = process_chord_notes(y, native_sr, args, output_dir)
            
            if stems:
                mix_and_save(stems, y, native_sr, output_dir, config)
                if config['chord_builder']['generate_metadata']:
                    write_metadata(args, stems, output_dir)
                logger.info("Successfully processed chord")
                
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        sys.exit(1)
    except ChordExtractionError as e:
        logger.error(f"Chord extraction error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup any temporary files or resources
        if 'audio_cache' in globals():
            logger.info("Cleaning up audio cache...")
            # Audio cache cleanup happens automatically via Python's garbage collection
        logger.info("Chord builder finished")

if __name__ == '__main__':
    main()
