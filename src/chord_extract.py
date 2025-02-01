import librosa
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
import soundfile as sf
from pathlib import Path
from typing import List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import hashlib

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
class ChordSegment:
    """Class to store chord notes and duration information."""
    notes: list[int]  # OpenMusic format notes
    duration: float   # Duration in seconds

@dataclass
class PitchDetectionResult:
    """Stores pitch detection results with confidence."""
    frequency: float
    confidence: float
    midi_note: int
    cents: int

class ChordExtractionError(Exception):
    """Base exception for chord extraction errors."""
    pass

class InsufficientPeaksError(ChordExtractionError):
    """Raised when not enough peaks are detected."""
    pass

class LowConfidencePitchError(ChordExtractionError):
    """Raised when pitch detection confidence is below threshold."""
    pass

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

def memory_efficient_processing(audio_path: str, chunk_size: int = 1048576) -> tuple[np.ndarray, int]:
    """Process audio in chunks to reduce memory usage."""
    logger.info(f"Loading audio file: {audio_path}")
    chunks = []
    with sf.SoundFile(audio_path) as audio_file:
        sr = audio_file.samplerate
        while audio_file.tell() < audio_file.frames:
            chunk = audio_file.read(chunk_size)
            chunks.append(chunk)
    logger.info(f"Loaded audio file with sample rate: {sr}Hz")
    return np.concatenate(chunks), sr

def detect_pitch_with_confidence(y: np.ndarray, sr: int) -> PitchDetectionResult:
    """Enhanced pitch detection with confidence scoring."""
    logger.debug("Performing pitch detection with confidence scoring")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=27.5, fmax=4186, sr=sr, frame_length=4096
    )
    
    # Calculate confidence from voiced probabilities
    voiced_conf = np.mean(voiced_probs[voiced_flag]) if any(voiced_flag) else 0
    
    if voiced_conf < 0.8:
        raise LowConfidencePitchError(
            f"Low confidence pitch detection: {voiced_conf:.2f}"
        )
    
    freq = np.median(f0[voiced_flag])
    midi_note = librosa.hz_to_midi(freq)
    cents = int((midi_note - int(midi_note)) * 100)
    
    logger.debug(f"Detected pitch: {freq}Hz, confidence: {voiced_conf:.2f}")
    return PitchDetectionResult(freq, voiced_conf, int(midi_note), cents)

@performance_logger
def optimized_spectral_analysis(seg: np.ndarray, sr: int, cache: AudioCache) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized spectral analysis using scipy's FFT with caching."""
    # Check cache first
    cache_key = cache._get_cache_key(seg, {'sr': sr})
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        logger.debug("Using cached FFT result")
        return cached_result[0], cached_result[1]

    # Use power of 2 length for optimal FFT performance
    n = 2 ** int(np.ceil(np.log2(len(seg))))
    window = np.blackman(len(seg))
    
    # Parallel FFT processing for large segments
    if len(seg) > 1048576:  # 1MB threshold
        with ThreadPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
            chunk_size = len(seg) // executor._max_workers
            chunks = [seg[i:i+chunk_size] for i in range(0, len(seg), chunk_size)]
            futures = [executor.submit(np.fft.rfft, chunk * window[:len(chunk)], n) for chunk in chunks]
            fft_parts = [f.result() for f in futures]
            fft = np.concatenate(fft_parts)
    else:
        fft = np.fft.rfft(seg * window, n=n)
    
    freqs = np.fft.rfftfreq(n, d=1/sr)
    result = (np.abs(fft), freqs)
    
    # Cache the result
    cache.set(cache_key, np.array(result))
    
    return result

@performance_logger
def optimized_peak_detection(magnitudes: np.ndarray, freqs: np.ndarray, 
                           min_freq: float, max_freq: float, num_voices: int) -> List[Tuple[float, float]]:
    """Optimized peak detection with frequency filtering."""
    logger.debug(f"Detecting peaks in range {min_freq}-{max_freq}Hz")
    
    # Pre-filter frequency range
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    valid_magnitudes = magnitudes[mask]
    valid_freqs = freqs[mask]
    
    if len(valid_magnitudes) == 0:
        raise ChordExtractionError(f"No valid frequencies found in range {min_freq}-{max_freq}Hz")
    
    # Adaptive noise floor
    noise_floor = np.median(valid_magnitudes)
    
    # Find peaks with optimized parameters
    peaks, _ = find_peaks(
        valid_magnitudes,
        height=noise_floor * 2,
        distance=int(len(valid_magnitudes) * 0.01),  # Adaptive distance
        prominence=noise_floor * 1.5
    )
    
    if len(peaks) < num_voices:
        raise InsufficientPeaksError(
            f"Insufficient peaks detected: found {len(peaks)}, needed {num_voices}"
        )
    
    # Sort peaks by magnitude and take top num_voices
    peak_data = [(valid_freqs[p], valid_magnitudes[p]) for p in peaks]
    peaks_found = sorted(peak_data, key=lambda x: x[1], reverse=True)[:num_voices]
    logger.debug(f"Detected {len(peaks_found)} peaks")
    return peaks_found

@performance_logger
def extract_microtonal_sequence(audio_path: str, num_chords: int, num_voices: int = 4, 
                              min_freq: float = 40, max_freq: float = 12000,
                              cache_dir: Optional[Path] = None) -> List[ChordSegment]:
    """
    Extract a sequence of chords from audio with duration information.
    Returns a list of ChordSegment objects containing notes and durations.
    """
    logger.info(f"Extracting {num_chords} chords from {audio_path}")
    
    try:
        # Initialize audio cache
        audio_cache = AudioCache(cache_dir or Path('chord_cache'))
        
        # Memory efficient audio loading
        y, sr = memory_efficient_processing(audio_path)
        
        # Extract MFCC features for agglomerative clustering
        # Process in chunks for large files
        hop_length = 512
        n_mfcc = 20
        mfcc_frames = []
        
        for i in range(0, len(y), hop_length * 128):
            chunk = y[i:i + hop_length * 128]
            if len(chunk) == 0:
                break
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            mfcc_frames.append(mfcc)
        
        mfcc = np.hstack(mfcc_frames)
        mfcc = librosa.util.normalize(mfcc, axis=1)
        
        # Perform agglomerative clustering
        boundaries = librosa.segment.agglomerative(mfcc, k=num_chords)
        
        # Convert boundaries to sample indices
        boundary_samples = [librosa.frames_to_samples(b, hop_length=hop_length) for b in boundaries]
        boundary_samples.append(len(y))
        
        sequence = []
        
        for i, (start, end) in enumerate(zip(boundary_samples[:-1], boundary_samples[1:])):
            logger.debug(f"Processing segment {i+1}/{len(boundary_samples)-1}")
            duration = (end - start) / sr
            
            if end - start < sr * 0.1:  # Skip segments shorter than 100ms
                logger.debug(f"Skipping segment {i+1} (too short: {duration:.3f}s)")
                continue
                
            seg = y[start:end]
            
            try:
                # Optimized spectral analysis with caching
                magnitudes, freqs = optimized_spectral_analysis(seg, sr, audio_cache)
                
                # Optimized peak detection with validation
                peaks = optimized_peak_detection(magnitudes, freqs, min_freq, max_freq, num_voices)
                
                # Convert to OpenMusic MIDI with cent precision
                om_notes = []
                for freq, _ in peaks:
                    if freq <= 0:
                        continue
                    midi = 69 + 12 * np.log2(freq/440)
                    om = int(round(midi * 100))
                    om_notes.append(om)
                
                sequence.append(ChordSegment(notes=om_notes, duration=duration))
                logger.debug(f"Processed segment {i+1}: {len(om_notes)} notes, {duration:.2f}s")
                
            except InsufficientPeaksError as e:
                logger.warning(f"Warning: {str(e)} in segment at {duration:.2f}s")
                sequence.append(ChordSegment(notes=[], duration=duration))
                
    except Exception as e:
        logger.error(f"Error during chord extraction: {str(e)}")
        raise ChordExtractionError(f"Error during chord extraction: {str(e)}")
    
    logger.info(f"Successfully extracted {len(sequence)} chord segments")
    return sequence
