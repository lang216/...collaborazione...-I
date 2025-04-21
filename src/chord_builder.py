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
from typing import List, Optional, Dict, Any, Callable, TypeVar
from dataclasses import dataclass

from chord_extract import (
    extract_microtonal_sequence,
    ChordSegment,
    ChordExtractionError,
    detect_pitch_with_confidence,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chord_processing.log"), logging.StreamHandler()],
)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class PathManager:
    """Centralized path management for chord processing."""

    base_dir: Path
    config: Dict[str, Any]

    @property
    def chord_output_dir(self) -> Path:
        return self.base_dir / self.config["paths"]["chord_output_dir"]

    @property
    def chord_input_dir(self) -> Path:
        return self.base_dir / self.config["chord_builder"]["chord_input_dir"]

    def create_output_dir(self, suffix: str = None) -> Path:
        """Create and return timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = (
            f"chord_{timestamp}" if suffix is None else f"chord_{suffix}_{timestamp}"
        )
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
        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, audio_data: np.ndarray, params: Dict[str, Any]) -> str:
        """Generate unique cache key based on audio data and parameters."""
        audio_hash: str = hashlib.md5(audio_data.tobytes()).hexdigest()
        param_hash: str = hashlib.md5(str(params).encode()).hexdigest()
        return f"{audio_hash}_{param_hash}"

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached result."""
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None

    def set(self, key: str, data: np.ndarray) -> None:
        """Cache result."""
        cache_file: Path = self.cache_dir / f"{key}.npy"
        np.save(cache_file, data)


F = TypeVar("F", bound=Callable[..., Any])


def performance_logger(func: F) -> F:
    """Decorator for performance monitoring."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time: float = time.time()
        result: Any = func(*args, **kwargs)
        duration: float = time.time() - start_time
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result

    return wrapper


@performance_logger
def detect_original_note(audio_path: str, sr: int) -> int:
    """Detect the original note using improved pitch detection with confidence.

    Args:
        audio_path: Path to the input audio file.
        sr: Target sample rate for loading the audio.

    Returns:
        The detected original note in OpenMusic format (MIDI * 100 + cents).

    Raises:
        Exception: If pitch detection fails or an error occurs during loading.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        result = detect_pitch_with_confidence(y, sr)
        return int(round(result.midi_note * 100 + result.cents))
    except Exception as e:
        logger.error(f"Error detecting original note: {str(e)}")
        raise


def parallel_process_stems(
    items: List[Any],
    process_fn: Callable[[Any], Optional[Any]],
    max_workers: Optional[int] = None,
) -> List[Any]:
    """Process a list of items in parallel using a thread pool.

    Args:
        items: A list of items to process.
        process_fn: The function to apply to each item. Should accept one item
                    and return a result or None on failure.
        max_workers: The maximum number of worker threads. Defaults to CPU count.

    Returns:
        A list containing the results of applying process_fn to each item.
        Results from failed tasks are omitted.
    """
    if max_workers is None:
        max_workers = min(len(items), multiprocessing.cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, stem) for stem in items]
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
    """High-precision pitch shifting with optimized RubberBand settings.

    Args:
        y: Input audio time series.
        sr: Sample rate of the audio.
        shift_cents: The amount to shift the pitch in cents.

    Returns:
        The pitch-shifted audio time series.
    """
    return rb.pyrb.pitch_shift(
        y=y,
        sr=sr,
        n_steps=shift_cents / 100,
        rbargs={
            "--formant": "",  # Preserve spectral envelope
            "--fine": "",  # High-precision mode
            "-c": "0",  # Higher frequency resolution
            "--threads": str(
                max(1, multiprocessing.cpu_count() - 1)
            ),  # Parallel processing
        },
    )


@performance_logger
def process_chord_notes(
    y: np.ndarray,
    native_sr: int,
    args: argparse.Namespace,
    output_dir: str,
    target_duration: Optional[float] = None,
) -> List[str]:
    """Process all chord notes in parallel, generating individual stems.

    Performs pitch shifting from the original note to each target chord note,
    optionally time-stretches, normalizes, and saves each resulting stem.

    Args:
        y: Input audio time series of the original note.
        native_sr: Sample rate of the input audio.
        args: Command-line arguments containing original_note and chord_notes.
        output_dir: Directory to save the generated stems.
        target_duration: Optional target duration in seconds for time stretching.

    Returns:
        A list of file paths to the generated stems. Returns an empty list if
        processing fails for all notes.
    """
    stems: List[str] = []
    original_midi: float = args.original_note / 100.0
    y_processed: np.ndarray = y.copy()  # Work on a copy

    # Time stretch original audio if target duration is specified
    # and differs significantly from the current duration.
    if target_duration is not None:
        current_duration: float = len(y_processed) / native_sr
        if abs(current_duration - target_duration) > 0.01:
            time_scale: float = target_duration / current_duration
            if abs(time_scale - 1.0) > 0.01:
                y_processed = rb.pyrb.time_stretch(
                    y_processed,
                    native_sr,
                    rate=time_scale,
                    rbargs={
                        "--formant": "",
                        "--fine": "",
                        "-c": "0",
                        "--threads": str(max(1, multiprocessing.cpu_count() - 1)),
                    },
                )

    def process_note(note: int) -> Optional[str]:
        try:
            target_midi: float = note / 100.0
            shift_cents: float = (target_midi - original_midi) * 100

            # Use caching for shifted audio
            # Note: Hashing the entire audio array might be slow for large files.
            # Consider alternative caching strategies if performance is an issue.
            cache_key: str = f"shift_{shift_cents}_{hash(y_processed.tobytes())}"
            cached_result: Optional[np.ndarray] = audio_cache.get(cache_key)
            y_shifted: np.ndarray

            if cached_result is not None:
                y_shifted = cached_result
            else:
                y_shifted = precise_rubberband_shift(
                    y_processed, native_sr, shift_cents
                )
                audio_cache.set(cache_key, y_shifted)

            config: Dict[str, Any] = load_config()
            if config["chord_builder"]["normalize_output"]:
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
    """Save an individual audio stem to a WAV file.

    Args:
        y: Audio time series data for the stem.
        native_sr: Sample rate of the audio data.
        note: The note identifier (OpenMusic format) for naming the file.
        output_dir: The directory where the stem file will be saved.

    Returns:
        The full path to the saved stem file.
    """
    stem_path = os.path.join(output_dir, f"stem_{note:05d}.wav")
    sf.write(stem_path, y, native_sr, subtype="PCM_24")
    logger.info(f"Saved stem: {stem_path}")
    return stem_path


@performance_logger
def mix_and_save(
    stems: List[str],
    original: np.ndarray,
    native_sr: int,
    output_dir: str,
    config: Dict[str, Any],
    target_duration: Optional[float] = None,
) -> None:
    """Mix individual stems into a single chord audio file.

    Loads stems in parallel, optionally time-stretches them to a target duration,
    mixes them, normalizes the result (optional), and saves the final mix.

    Args:
        stems: A list of file paths to the audio stems.
        original: The original audio time series (used for length reference if
                  target_duration is None).
        native_sr: The sample rate of the audio stems.
        output_dir: The directory where the mixed file will be saved.
        config: The application configuration dictionary.
        target_duration: Optional target duration in seconds for the final mix.
                         If provided, stems will be time-stretched.
    """
    if target_duration is not None:
        target_samples = int(target_duration * native_sr)
        max_length = target_samples
    else:
        max_length = len(original)

    mixed: np.ndarray = np.zeros(max_length)

    def load_and_process_stem(stem_path: str) -> np.ndarray:
        y_stem: np.ndarray
        sr_stem: int
        y_stem, sr_stem = librosa.load(
            stem_path, sr=native_sr
        )  # Ensure native_sr is used
        if target_duration is not None:
            current_duration: float = len(y_stem) / sr_stem
            if abs(current_duration - target_duration) > 0.01:
                time_scale: float = target_duration / current_duration
                if abs(time_scale - 1.0) > 0.01:
                    y_stem = rb.pyrb.time_stretch(
                        y_stem,
                        sr_stem,  # Use the actual sample rate from loading
                        rate=time_scale,
                        rbargs={
                            "--formant": "",
                            "--fine": "",
                            "-c": "0",
                            "--threads": str(max(1, multiprocessing.cpu_count() - 1)),
                        },
                    )
        return librosa.util.fix_length(y_stem, size=max_length)

    # Process stems in parallel
    processed_stems: List[np.ndarray] = parallel_process_stems(
        stems, load_and_process_stem
    )

    # Mix stems
    for stem_data in processed_stems:
        mixed += stem_data * 0.8  # Consider making mix level configurable

    if config["chord_builder"]["normalize_output"]:
        mixed = librosa.util.normalize(mixed)

    mixed_path = os.path.join(output_dir, "mixed_chord.wav")
    sf.write(mixed_path, mixed, native_sr, subtype="PCM_24")
    logger.info(f"Saved mix: {mixed_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate essential sections and parameters in the configuration dictionary.

    Checks for required sections ('audio', 'paths', 'chord_builder')
    and validates specific values like sample rate and bit depth.

    Args:
        config: The configuration dictionary loaded from the JSON file.

    Raises:
        ValueError: If a required section is missing or a parameter has an invalid value.
    """
    required_sections = ["audio", "paths", "chord_builder"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    audio_config = config["audio"]
    if audio_config.get("sr") not in [44100, 48000, 96000]:
        raise ValueError(f"Invalid sample rate: {audio_config.get('sr')}")

    chord_config = config["chord_builder"]
    if chord_config.get("default_bit_depth") not in [16, 24, 32]:
        raise ValueError(f"Invalid bit depth: {chord_config.get('default_bit_depth')}")


def write_metadata(
    args: argparse.Namespace,
    stem_paths: List[str],
    output_dir: str,
    duration: Optional[float] = None,
) -> None:
    """Generate and save a JSON metadata file summarizing the processing run.

    Includes information about input file, notes, sample rate, generated stems,
    processing parameters, and optional duration.

    Args:
        args: Command-line arguments containing processing parameters.
        stem_paths: List of absolute paths to the generated stem files.
        output_dir: Directory where the metadata.json file will be saved.
        duration: Optional duration of the processed segment in seconds.
    """
    metadata: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "input_file": str(Path(args.input_audio).resolve()), # Use resolve
        "original_note": args.original_note,
        "chord_notes": args.chord_notes,
        "sample_rate": args.sr,
        "stems": [str(Path(p).absolute()) for p in stem_paths],
        "processing_chain": {
            "pitch_detection": (
                "librosa YIN with confidence" if args.detect_pitch else "manual"
            ),
            "pitch_shifting": "RubberBand with formant preservation",
            "parallel_processing": True,
            "caching_enabled": True,
        },
    }
    if duration is not None:
        metadata["duration"] = duration

    metadata_path: str = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
    except IOError as e:
        logger.error(f"Failed to write metadata file {metadata_path}: {e}")


@performance_logger
def extract_and_build_chords(
    y: np.ndarray, native_sr: int, args: argparse.Namespace, config: Dict[str, Any]
) -> List[ChordSegment]:
    """Extract a sequence of chords from a source audio and build stems for each.

    Uses `extract_microtonal_sequence` to get chord segments (notes and durations)
    from the source specified in `args.chord_source`. Then, for each segment,
    it processes the input audio `y` to generate stems corresponding to the
    segment's notes and duration.

    Args:
        y: The input audio time series to be processed (pitch-shifted).
        native_sr: The sample rate of the input audio `y`.
        args: Command-line arguments, including `chord_source`, `extract_chords`,
              `num_voices`, `output_dir`, and `original_note`.
        config: The application configuration dictionary.

    Returns:
        A list of ChordSegment objects representing the extracted chords.

    Raises:
        ChordExtractionError: If chord extraction from the source fails.
        Exception: For other unexpected errors during processing.
    """
    try:
        num_chords_to_extract: int = args.extract_chords
        num_voices: int = (
            args.num_voices if hasattr(args, "num_voices") and args.num_voices else 4
        )

        chord_segments: List[ChordSegment] = extract_microtonal_sequence(
            args.chord_source, num_chords_to_extract, num_voices=num_voices
        )

        all_stems: List[str] = []
        for i, segment in enumerate(chord_segments):
            if not segment.notes:  # Skip empty segments
                logger.warning(f"Skipping empty segment {i + 1}")
                continue

            segment_output_dir: str = create_output_dir(
                args.output_dir, base_dir_name=f"chord_sequence_{i + 1}"
            )
            # Create a temporary args object or modify a copy to avoid side effects
            segment_args = argparse.Namespace(**vars(args))
            segment_args.chord_notes = segment.notes

            segment_stems: List[str] = process_chord_notes(
                y,
                native_sr,
                segment_args,
                segment_output_dir,
                target_duration=segment.duration,
            )

            if segment_stems:
                mix_and_save(
                    segment_stems,
                    y,  # Pass original y for length reference if needed
                    native_sr,
                    segment_output_dir,
                    config,
                    target_duration=segment.duration,
                )
                if config["chord_builder"]["generate_metadata"]:
                    write_metadata(
                        segment_args,
                        segment_stems,
                        segment_output_dir,
                        duration=segment.duration,
                    )
            all_stems.extend(segment_stems)  # Extend with generated stems

        return chord_segments
    except ChordExtractionError as e:
        logger.error(f"Chord extraction error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in extract_and_build_chords: {str(e)}")
        raise


def create_output_dir(base_path: str, base_dir_name: str = "chord") -> str:
    """Create a unique, timestamped output directory.

    Args:
        base_path: The parent directory where the new directory will be created.
        base_dir_name: A prefix for the new directory name (default: "chord").

    Returns:
        The path to the newly created output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_path, f"{base_dir_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path


def _process_chord_sequence(
    y: np.ndarray, native_sr: int, args: argparse.Namespace, config: Dict[str, Any]
) -> None:
    """Helper function to process a chord sequence extracted from a source file."""
    logger.info(f"Extracting chords from: {args.chord_source}")
    chord_segments = extract_and_build_chords(y, native_sr, args, config)
    logger.info(f"Successfully processed {len(chord_segments)} chord segments")


def _process_single_chord(
    y: np.ndarray, native_sr: int, args: argparse.Namespace, config: Dict[str, Any]
) -> None:
    """Helper function to process a single, explicitly defined chord."""
    output_dir = create_output_dir(args.output_dir)
    stems = process_chord_notes(y, native_sr, args, output_dir)

    if stems:
        mix_and_save(stems, y, native_sr, output_dir, config)
        if config["chord_builder"]["generate_metadata"]:
            write_metadata(args, stems, output_dir)
        logger.info("Successfully processed chord")
    else:
        logger.warning("No stems were generated for the single chord.")


def _parse_and_validate_args(
    config: Dict[str, Any], path_manager: PathManager
) -> argparse.Namespace:
    """Parse command-line arguments and validate them."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Microtonal Chord Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Argument Parsing ---
    parser.add_argument("input_audio", help="Path to input audio file")
    parser.add_argument(
        "--sr",
        type=int,
        default=config["audio"]["sr"],
        help="Sample rate for processing",
    )
    parser.add_argument(
        "--detect-pitch",
        action="store_true",
        help="Automatically detect original pitch",
    )
    parser.add_argument(
        "--original-note",
        type=int,
        help=(
            "Original note in OpenMusic format "
            "(required if not using --detect-pitch)"
        ),
    )
    parser.add_argument(
        "--chord-notes",
        type=int,
        nargs="+",
        help="Target notes in OpenMusic format",
    )
    parser.add_argument(
        "--chord-source", type=str, help="Audio file to extract chord sequence from"
    )
    parser.add_argument(
        "--extract-chords",
        type=int,
        help="Number of chords to extract from chord source",
    )
    parser.add_argument(
        "--num-voices",
        type=int,
        default=4,
        help="Number of voices in extracted chords",
    )
    parser.add_argument(
        "--output-dir",
        default=str(path_manager.chord_output_dir),
        help="Output directory for generated files",
    )

    args: argparse.Namespace = parser.parse_args()

    # --- Argument Validation ---
    if not args.detect_pitch and args.original_note is None:
        parser.error("Either --detect-pitch or --original-note must be specified")

    if not args.chord_notes and not args.chord_source:
        parser.error("Either --chord-notes or --chord-source must be specified")

    if args.chord_source and args.extract_chords is None:
        parser.error("--extract-chords is required when using --chord-source")

    return args


def main() -> None:
    """Main execution function with comprehensive error handling."""
    try:
        config: Optional[Dict[str, Any]] = load_config()
        if config is None:
            logger.error("Failed to load configuration. Exiting.")
            sys.exit(1)
        validate_config(config)

        # Initialize path manager and audio cache
        path_manager = PathManager(Path.cwd(), config)
        path_manager.validate_paths()  # Validate paths early
        global audio_cache  # Declaring global is necessary here
        audio_cache = AudioCache(path_manager.chord_output_dir / "cache")

        # Parse and validate arguments
        args: argparse.Namespace = _parse_and_validate_args(config, path_manager)

        # --- Load Audio ---
        logger.info(f"Loading input audio: {args.input_audio}")
        y: np.ndarray
        native_sr: int
        y, native_sr = librosa.load(args.input_audio, sr=args.sr)  # Use configured SR

        # --- Determine Original Note ---
        if args.detect_pitch:
            logger.info("Detecting original pitch...")
            args.original_note = detect_original_note(args.input_audio, args.sr)
            logger.info(f"Detected original note: {args.original_note}")

        # --- Process Chord(s) ---
        if args.chord_source:
            _process_chord_sequence(y, native_sr, args, config)
        else:
            _process_single_chord(y, native_sr, args, config)

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
        if "audio_cache" in globals():
            logger.info("Cleaning up audio cache...")
            # Audio cache cleanup happens automatically via Python's garbage collection
        logger.info("Chord builder finished")


if __name__ == "__main__":
    main()
