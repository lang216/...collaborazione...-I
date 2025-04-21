import librosa
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable, TypeVar, Any
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import hashlib
import json  # use JSON sorted keys for consistent param hashing
import argparse # Import argparse
import sys # Import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chord_processing.log"), logging.StreamHandler()],
)
logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ChordSegment:
    """Class to store chord notes and duration information."""

    notes: List[int]  # OpenMusic format notes
    duration: float  # Duration in seconds


@dataclass
class PitchDetectionResult:
    """Stores pitch detection results with confidence."""

    frequency: float
    confidence: float
    midi_note: int
    cents: int


class ChordExtractionError(Exception):
    """Base exception for chord extraction errors."""

    print('ChordExtractionError: Base exception for chord extraction errors.')


class InsufficientPeaksError(ChordExtractionError):
    """Raised when not enough peaks are detected."""

    print('InsufficientPeaksError: Raised when not enough peaks are detected.')


class LowConfidencePitchError(ChordExtractionError):
    """Raised when pitch detection confidence is below threshold."""

    print('LowConfidencePitchError: Raised when pitch detection confidence is below threshold.')


class AudioCache:
    """Cache for audio processing results."""

    def __init__(self, cache_dir: Path):
        self.cache_dir: Path = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, audio_data: np.ndarray, params: Dict[str, Any]) -> str:
        """Generate unique cache key based on audio data and parameters."""
        audio_hash: str = hashlib.md5(audio_data.tobytes()).hexdigest()
        param_hash: str = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
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


def memory_efficient_processing(
    audio_path: str, chunk_size: int = 1048576
) -> Tuple[np.ndarray, int]:
    """Load an audio file efficiently by processing it in chunks.

    This is useful for large audio files that might not fit entirely into memory.

    Args:
        audio_path: Path to the input audio file.
        chunk_size: The number of frames to read per chunk (default: 1MB).

    Returns:
        A tuple containing:
            - The entire audio data as a NumPy array.
            - The sample rate of the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: For other errors during file reading.
    """
    logger.info(f"Loading audio file: {audio_path}")
    chunks: List[np.ndarray] = []
    sr: int = 0  # Initialize sr
    with sf.SoundFile(audio_path) as audio_file:
        sr = audio_file.samplerate
        while audio_file.tell() < audio_file.frames:
            chunk: np.ndarray = audio_file.read(chunk_size)
            chunks.append(chunk)
    logger.info(f"Loaded audio file with sample rate: {sr}Hz")
    return np.concatenate(chunks), sr


def detect_pitch_with_confidence(y: np.ndarray, sr: int) -> PitchDetectionResult:
    """Detect the fundamental frequency (pitch) of an audio signal with confidence.

    Uses the librosa PYIN algorithm and calculates a confidence score based on
    voiced probabilities.

    Args:
        y: Input audio time series.
        sr: Sample rate of the audio time series.

    Returns:
        A PitchDetectionResult object containing the detected frequency,
        confidence score, MIDI note number, and cents deviation.

    Raises:
        LowConfidencePitchError: If the calculated confidence score is below 0.8.
    """
    logger.debug("Performing pitch detection with confidence scoring")
    f0: np.ndarray
    voiced_flag: np.ndarray
    voiced_probs: np.ndarray
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=27.5, fmax=4186, sr=sr, frame_length=4096
    )

    # Calculate confidence from voiced probabilities
    voiced_conf: float = (
        np.mean(voiced_probs[voiced_flag]) if np.any(voiced_flag) else 0.0
    )

    if voiced_conf < 0.8:
        raise LowConfidencePitchError(
            f"Low confidence pitch detection: {voiced_conf:.2f}"
        )

    # Ensure f0[voiced_flag] is not empty before calculating median
    valid_f0: np.ndarray = f0[voiced_flag]
    if len(valid_f0) == 0:
        raise LowConfidencePitchError(
            "No voiced frames detected for pitch calculation."
        )

    freq: float = float(np.median(valid_f0))  # Cast median result to float
    midi_note: float = librosa.hz_to_midi(freq)
    cents: int = int(round((midi_note - int(midi_note)) * 100))  # Round cents

    logger.debug(f"Detected pitch: {freq}Hz, confidence: {voiced_conf:.2f}")
    return PitchDetectionResult(freq, voiced_conf, int(midi_note), cents)


@performance_logger
def optimized_spectral_analysis(
    seg: np.ndarray, sr: int, cache: AudioCache
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform spectral analysis (FFT) on an audio segment, optimized for speed and caching.

    Uses NumPy's RFFT, applies a Blackman window, and leverages caching to avoid
    recomputing results for identical segments and parameters.
    Uses parallel processing for very large segments.

    Args:
        seg: The audio segment (time series) to analyze.
        sr: The sample rate of the audio segment.
        cache: An AudioCache instance for storing and retrieving FFT results.

    Returns:
        A tuple containing:
            - Magnitudes of the frequency components.
            - Corresponding frequencies.
    """
    # Check cache first
    cache_key: str = cache._get_cache_key(seg, {"sr": sr})
    cached_result: Optional[np.ndarray] = cache.get(cache_key)
    if cached_result is not None:
        logger.debug("Using cached FFT result")
        # Ensure cached result is unpacked correctly
        # Assuming cached result stores magnitudes and freqs as rows
        if cached_result.ndim == 2 and cached_result.shape[0] == 2:
            return cached_result[0, :], cached_result[1, :]
        else:
            logger.warning(
                "Invalid cached FFT result format or dimensions, recomputing."
            )

    # Use power of 2 length for optimal FFT performance
    n: int = 2 ** int(np.ceil(np.log2(len(seg))))
    window: np.ndarray = np.blackman(len(seg))
    fft: np.ndarray

    if len(seg) > 1048576:  # 1MB threshold
        max_workers_count = max(1, multiprocessing.cpu_count() - 1)
        with ThreadPoolExecutor(max_workers=max_workers_count) as executor:
            chunk_size: int = len(seg) // max_workers_count
            chunks: List[np.ndarray] = [seg[i: i + chunk_size] for i in range(0, len(seg), chunk_size)]
            futures = [executor.submit(np.fft.rfft, chunk * window[:len(chunk)], n) for chunk in chunks]
            fft_parts: List[np.ndarray] = [f.result() for f in futures if isinstance(f.result(), np.ndarray)]
            if not fft_parts:
                raise ChordExtractionError("FFT computation failed for all chunks.")
            fft = np.concatenate(fft_parts)
    else:
        fft = np.fft.rfft(seg * window, n=n)

    freqs: np.ndarray = np.fft.rfftfreq(n, d=1 / sr)
    magnitudes: np.ndarray = np.abs(fft)
    # result: Tuple[np.ndarray, np.ndarray] = (magnitudes, freqs) # Removed unused assignment

    # Cache the result - stack magnitudes and freqs for consistent storage
    cache.set(cache_key, np.vstack((magnitudes, freqs)))

    return magnitudes, freqs


@performance_logger
def optimized_peak_detection(
    magnitudes: np.ndarray,
    freqs: np.ndarray,
    min_freq: float,
    max_freq: float,
    num_voices: int,
) -> List[Tuple[float, float]]:
    """Detect prominent spectral peaks within a specified frequency range.

    Filters the frequency spectrum, calculates an adaptive noise floor, and uses
    `scipy.signal.find_peaks` with optimized parameters to find the most
    prominent peaks corresponding to the desired number of voices.

    Args:
        magnitudes: Array of spectral magnitudes from FFT.
        freqs: Array of corresponding frequencies from FFT.
        min_freq: Minimum frequency threshold for peak detection.
        max_freq: Maximum frequency threshold for peak detection.
        num_voices: The desired number of peaks (voices) to detect.

    Returns:
        A list of tuples, where each tuple contains (frequency, magnitude)
        for one of the detected peaks, sorted by magnitude in descending order.

    Raises:
        ChordExtractionError: If no valid frequencies are found in the specified range.
        InsufficientPeaksError: If fewer peaks than `num_voices` are detected
                                above the threshold and prominence criteria.
    """
    logger.debug(f"Detecting peaks in range {min_freq}-{max_freq}Hz")

    # Pre-filter frequency range
    mask: np.ndarray = (freqs >= min_freq) & (freqs <= max_freq)
    valid_magnitudes: np.ndarray = magnitudes[mask]
    valid_freqs: np.ndarray = freqs[mask]

    if len(valid_magnitudes) == 0:
        raise ChordExtractionError(
            f"No valid frequencies found in range {min_freq}-{max_freq}Hz"
        )

    # Adaptive noise floor
    noise_floor: float = float(np.median(valid_magnitudes))  # Cast median result

    # Find peaks with optimized parameters
    peaks: np.ndarray
    peak_properties: Dict[str, Any]
    peaks, peak_properties = find_peaks(
        valid_magnitudes,
        height=noise_floor * 2,
        distance=int(len(valid_magnitudes) * 0.01),  # Adaptive distance
        prominence=noise_floor * 1.5,
    )

    if len(peaks) < num_voices:
        raise InsufficientPeaksError(
            f"Insufficient peaks detected: found {len(peaks)}, needed {num_voices}"
        )

    # Sort peaks by magnitude and take top num_voices
    peak_data: List[Tuple[float, float]] = [
        (valid_freqs[p], valid_magnitudes[p]) for p in peaks
    ]
    peaks_found = sorted(peak_data, key=lambda x: x[1], reverse=True)[:num_voices]
    logger.debug(f"Detected {len(peaks_found)} peaks")
    return peaks_found


def _process_audio_segment(
    seg: np.ndarray,
    sr: int,
    duration: float,
    audio_cache: AudioCache,
    min_freq: float,
    max_freq: float,
    num_voices: int,
) -> ChordSegment:
    """Processes a single audio segment to extract chord notes.

    Performs spectral analysis, peak detection, and converts frequencies
    to OpenMusic format notes. Handles insufficient peaks gracefully.

    Args:
        seg: The audio time series for the segment.
        sr: The sample rate of the audio.
        duration: The duration of the segment in seconds.
        audio_cache: The cache object for spectral analysis results.
        min_freq: Minimum frequency for peak detection.
        max_freq: Maximum frequency for peak detection.
        num_voices: The number of voices (peaks) to detect.

    Returns:
        A ChordSegment object containing the detected notes and duration.
        Notes list will be empty if peak detection fails.
    """
    try:
        # Optimized spectral analysis with caching
        magnitudes: np.ndarray
        freqs: np.ndarray
        magnitudes, freqs = optimized_spectral_analysis(seg, sr, audio_cache)

        # Optimized peak detection with validation
        detected_peaks: List[Tuple[float, float]] = optimized_peak_detection(
            magnitudes, freqs, min_freq, max_freq, num_voices
        )

        # Convert to OpenMusic MIDI with cent precision
        om_notes: List[int] = []
        for freq, _ in detected_peaks:
            if freq <= 0:
                continue
            midi: float = 69 + 12 * np.log2(freq / 440)
            om: int = int(round(midi * 100))
            om_notes.append(om)

        return ChordSegment(notes=om_notes, duration=duration)

    except InsufficientPeaksError as e:
        logger.warning(f"Warning: {str(e)} in segment at {duration:.2f}s")
        return ChordSegment(notes=[], duration=duration)


@performance_logger
def extract_microtonal_sequence(
    audio_path: str,
    num_chords: int,
    num_voices: int = 4,
    min_freq: float = 40,
    max_freq: float = 12000,
    cache_dir: Optional[Path] = None,
) -> List[ChordSegment]:
    """Extract a sequence of microtonal chords from an audio file.

    Segments the audio using MFCCs and agglomerative clustering, then analyzes
    each segment to find prominent spectral peaks (notes) and their duration.

    Args:
        audio_path: Path to the input audio file.
        num_chords: The target number of chords to extract (segmentation target).
        num_voices: The number of notes (peaks) to detect within each chord segment.
        min_freq: Minimum frequency for peak detection.
        max_freq: Maximum frequency for peak detection.
        cache_dir: Optional path to a directory for caching intermediate results.
                   Defaults to "chord_cache" in the current directory.

    Returns:
        A list of ChordSegment objects, each containing the detected notes
        (in OpenMusic format) and the duration of the segment. Segments where
        peak detection failed may have an empty list of notes.

    Raises:
        ChordExtractionError: If a fatal error occurs during the extraction process.
    """
    logger.info(f"Extracting {num_chords} chords from {audio_path}")

    try:
        # Initialize audio cache
        audio_cache: AudioCache = AudioCache(cache_dir or Path("chord_cache"))

        # Memory efficient audio loading
        y: np.ndarray
        sr: int
        y, sr = memory_efficient_processing(audio_path)

        # --- Feature Extraction (MFCC) with parallel processing ---
        hop_length: int = 512
        n_mfcc: int = 20
        chunk_size_samples = hop_length * 128
        # Prepare audio chunks
        audio_chunks: List[np.ndarray] = [y[i: i + chunk_size_samples] for i in range(0, len(y), chunk_size_samples) if len(y[i: i + chunk_size_samples]) > 0]
        # Parallel MFCC computation
        max_workers_count = max(1, multiprocessing.cpu_count() - 1)
        def compute_mfcc(chunk: np.ndarray) -> np.ndarray:
            return librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        with ThreadPoolExecutor(max_workers=max_workers_count) as executor:
            mfcc_frames: List[np.ndarray] = list(executor.map(compute_mfcc, audio_chunks))

        if not mfcc_frames:
            raise ChordExtractionError("Could not extract MFCC features.")

        mfcc_full: np.ndarray = np.hstack(mfcc_frames)
        mfcc_norm: np.ndarray = librosa.util.normalize(mfcc_full, axis=1)

        # --- Segmentation ---
        boundaries: np.ndarray = librosa.segment.agglomerative(mfcc_norm, k=num_chords)

        # Convert frame boundaries to sample indices
        boundary_samples: List[int] = [
            librosa.frames_to_samples(b, hop_length=hop_length) for b in boundaries
        ]
        boundary_samples.append(len(y))  # Add end of audio as final boundary

        # Prepare tasks for parallel processing
        tasks: List[Tuple[np.ndarray, float]] = []
        for start, end in zip(boundary_samples[:-1], boundary_samples[1:]):
            seg_len = end - start
            duration = seg_len / sr
            if seg_len < sr * 0.1:  # skip <100ms
                logger.debug(f"Skipping segment (too short: {duration:.3f}s)")
                continue
            tasks.append((y[start:end], duration))

        sequence: List[ChordSegment] = []
        with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
            futures = [executor.submit(_process_audio_segment, seg, sr, duration, audio_cache, min_freq, max_freq, num_voices) for seg, duration in tasks]
            for future in futures:
                sequence.append(future.result())

    except Exception as e:
        logger.error(f"Error during chord extraction: {str(e)}")
        raise ChordExtractionError(f"Error during chord extraction: {str(e)}")

    logger.info(f"Successfully extracted {len(sequence)} chord segments")
    return sequence


def main():
    """Main function to run chord extraction from the command line."""
    parser = argparse.ArgumentParser(
        description="Extract a sequence of microtonal chords from an audio file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_path", help="Path to the input audio file.")
    parser.add_argument(
        "num_chords",
        type=int,
        help="Target number of chords to extract (segmentation target).",
    )
    parser.add_argument(
        "--num-voices",
        type=int,
        default=4,
        help="Number of notes (peaks) to detect within each chord segment.",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=40,
        help="Minimum frequency (Hz) for peak detection.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=12000,
        help="Maximum frequency (Hz) for peak detection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("chord_cache"),
        help="Directory for caching intermediate results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the extracted chord sequence as a JSON file.",
    )

    args = parser.parse_args()

    try:
        extracted_sequence = extract_microtonal_sequence(
            audio_path=args.audio_path,
            num_chords=args.num_chords,
            num_voices=args.num_voices,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            cache_dir=args.cache_dir,
        )

        print("\n--- Chord Extraction Results ---")
        if not extracted_sequence:
            print("No chord segments were extracted.")
        else:
            for i, segment in enumerate(extracted_sequence):
                notes_str = ", ".join(map(str, segment.notes)) if segment.notes else "None"
                print(f"Segment {i+1}: Duration={segment.duration:.3f}s, Notes=[{notes_str}]")

            if args.output_json:
                # Convert list of dataclasses to list of dicts for JSON serialization
                results_list = [
                    {"notes": seg.notes, "duration": seg.duration}
                    for seg in extracted_sequence
                ]
                try:
                    args.output_json.parent.mkdir(parents=True, exist_ok=True)
                    with open(args.output_json, "w") as f:
                        json.dump(results_list, f, indent=2)
                    print(f"\nResults saved to: {args.output_json}")
                except IOError as e:
                    logger.error(f"Failed to write output JSON file: {e}")
                    print(f"Error: Could not write results to {args.output_json}")

    except ChordExtractionError as e:
        logger.error(f"Chord extraction failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
