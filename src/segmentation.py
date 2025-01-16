import numpy as np
import librosa
import os
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed, Memory
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup memory caching
cachedir = '.cache'
memory = Memory(cachedir, verbose=0)

@dataclass
class AudioConfig:
    """Configuration class for audio processing parameters."""
    sr: Optional[int] = None
    mono: bool = True
    hop_length: int = 512
    n_fft: int = 2048  # Default FFT window size
    n_mfcc: int = 20
    onset_params: Dict[str, Union[int, float]] = None
    
    def __post_init__(self):
        self.onset_params = self.onset_params or {
            'pre_max': 30,
            'post_max': 30,
            'pre_avg': 100,
            'post_avg': 100,
            'delta': 0.2,
            'wait': 30
        }

    def validate(self):
        """Validate configuration parameters."""
        assert self.n_mfcc > 0, "n_mfcc must be positive"
        assert self.hop_length > 0, "hop_length must be positive"

CONFIG = AudioConfig()

@memory.cache
def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load audio file with caching."""
    try:
        audio, sr = librosa.load(file_path, mono=CONFIG.mono, sr=CONFIG.sr)
        return audio, sr
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        raise

@memory.cache
def detect_onsets(audio: np.ndarray, sr: int) -> np.ndarray:
    """Optimized onset detection with caching."""
    onset_env = librosa.onset.onset_strength(
        y=audio, 
        sr=sr,
        hop_length=CONFIG.hop_length,
        # aggregate=np.mean
    )
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=CONFIG.hop_length,
        **CONFIG.onset_params
    )
    
    onset_times = librosa.frames_to_samples(onset_frames, hop_length=CONFIG.hop_length)
    return np.append(onset_times, len(audio))

class AudioFeatureExtractor:
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize feature extractor with optional custom config"""
        self.config = config or CONFIG
        self.config.validate()
        # Configure parallel processing based on available cores
        self.n_jobs = max(1, os.cpu_count() - 1)
    
    @staticmethod
    def _extract_single_feature(segment: np.ndarray, sr: int, feature_type: str, config: AudioConfig) -> np.ndarray:
        """Extract a single feature type from an audio segment."""
        if len(segment) < config.n_fft:
            # Pad short segments with zeros to reach n_fft length
            padding = config.n_fft - len(segment)
            segment = np.pad(segment, (0, padding), mode='constant')
            logger.info(f"Padded short segment from {len(segment)-padding} to {len(segment)} samples")
            
        if feature_type == 'mfcc':
            return np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=config.n_mfcc, n_fft=config.n_fft), axis=1)
        elif feature_type == 'spectral_centroid':
            return np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=config.n_fft))
        elif feature_type == 'spectral_flatness':
            return np.mean(librosa.feature.spectral_flatness(y=segment, n_fft=config.n_fft))
        elif feature_type == 'rms':
            return np.mean(librosa.feature.rms(y=segment, frame_length=config.n_fft))
    
    def extract_segment_features(self, segment: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract multiple audio features in parallel from a segment.
        
        Args:
            segment: Audio samples to extract features from
            sr: Sample rate of the audio
            
        Returns:
            Dictionary mapping feature names to their extracted values
            
        Raises:
            RuntimeError: If feature extraction fails
        """
        feature_types = ['mfcc', 
                         'spectral_centroid', 
                         'spectral_flatness', 
                         'rms', 
                        #  'chroma'
                         ]
        try:
            features = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._extract_single_feature)(segment, sr, feat_type, self.config)
                for feat_type in feature_types
            )
            
            # Filter out empty feature arrays
            valid_features = {
                feat_type: feat 
                for feat_type, feat in zip(feature_types, features)
                if feat.size > 0
            }
            
            return valid_features
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}") from e

    def filter_chunk_durations(self, input_dir, min_duration=0.2, max_duration=2.0, min_files=3):
        """Filters audio files in directory based on duration constraints.
        
        Args:
            input_dir: Input directory containing audio files
            min_duration: Minimum duration in seconds (default: 0.2)
            max_duration: Maximum duration in seconds (default: 2.0)
            min_files: Minimum number of files per cluster (default: 3)
            
        Creates new directory with '_filtered' suffix and maintains original structure.
        Removes clusters containing fewer than min_files after filtering.
        """
        input_path = Path(input_dir)
        output_path = input_path.parent / f"{input_path.name}_Filtered"

        if not input_path.exists():
            logger.error(f"Input directory does not exist: {input_path}")
            return

        # Create output directory structure
        output_path.mkdir(exist_ok=True)
        
        total_files = 0
        copied_files = 0
        skipped_files = 0

        # Get list of all .wav files first
        wav_files = [item for item in input_path.rglob('*') 
                    if item.is_file() and item.suffix == '.wav']
        total_files = len(wav_files)
        
        # Process files with progress bar
        for item in tqdm(wav_files, desc="Filtering audio chunks"):
            try:
                # Get audio duration without loading the entire file
                duration = librosa.get_duration(path=item)
                if min_duration <= duration <= max_duration:
                    # Create corresponding directory in output
                    relative_path = item.relative_to(input_path)
                    output_file_path = output_path / relative_path
                    output_file_path.parent.mkdir(parents=True, exist_ok=True)
                    # Copy file to corresponding location in output directory
                    shutil.copy(item, output_file_path)
                    copied_files += 1
                else:
                    skipped_files += 1
            except Exception as e:
                skipped_files += 1
                logger.error(f"Error processing {item}: {e}")

        # Remove cluster folders with fewer than min_files
        removed_clusters = 0
        for feature_type in os.listdir(output_path):
            feature_path = os.path.join(output_path, feature_type)
            if not os.path.isdir(feature_path):
                continue
                
            for piece_name in os.listdir(feature_path):
                piece_path = os.path.join(feature_path, piece_name)
                if not os.path.isdir(piece_path):
                    continue
                    
                for cluster in os.listdir(piece_path):
                    cluster_path = os.path.join(piece_path, cluster)
                    if not os.path.isdir(cluster_path):
                        continue
                        
                    # Count .wav files in cluster folder
                    wav_files = [f for f in os.listdir(cluster_path) 
                               if f.endswith('.wav')]
                    
                    if len(wav_files) < min_files:
                        try:
                            shutil.rmtree(cluster_path)
                            removed_clusters += 1
                            logger.info(f"Removed cluster {cluster_path} with {len(wav_files)} files")
                        except Exception as e:
                            logger.error(f"Error removing cluster {cluster_path}: {e}")

        logger.info(f"Filtering complete. Processed {total_files} files: "
                   f"{copied_files} copied, {skipped_files} skipped. "
                   f"Removed {removed_clusters} clusters with fewer than {min_files} files.")

def process_audio_files(
    directory: str, 
    k: int,
    fade_duration: float = 0.1  # Add fade_duration parameter
) -> Dict[str, Dict]:
    """
    Process audio files in directory, extracting features and clustering segments.
    
    Args:
        directory: Path to directory containing .wav files
        k: Number of clusters to create
        
    Returns:
        Dictionary containing processed audio data with features and cluster labels
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If processing fails
    """
    if not Path(directory).exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if k < 1:
        raise ValueError("Number of clusters must be positive")
    
    extractor = AudioFeatureExtractor()
    results = {}
    
    try:
        audio_files = list(Path(directory).glob('*.wav'))
        if not audio_files:
            logger.warning(f"No .wav files found in {directory}")
            return results
            
        for i, file_path in enumerate(tqdm(audio_files, desc="Processing audio files")):
            try:
                # Load audio with memory management
                audio, sr = load_audio(str(file_path))
                if len(audio) == 0:
                    logger.warning(f"Empty audio file: {file_path}")
                    continue
                
                # Split audio by onsets with progress
                onset_points = detect_onsets(audio, sr)
                segments = [audio[start:end] for start, end in 
                          tqdm(zip(onset_points[:-1], onset_points[1:]),
                              desc="Splitting audio", leave=False)]
                
                # Extract features with memory cleanup
                segment_features = []
                with Parallel(n_jobs=-1, max_nbytes='100M') as parallel:
                    segment_features = parallel(
                        delayed(extractor.extract_segment_features)(segment, sr)
                        for segment in tqdm(segments, 
                                          desc=f"Extracting features for {file_path.name}", 
                                          leave=False)
                    )
                
                # Cluster features with progress
                clustered_features = {}
                for feature_name in tqdm(['mfcc', 
                                          'spectral_centroid', 
                                          'spectral_flatness', 
                                          'rms', 
                                        #   'chroma'
                                          ],
                                       desc="Clustering features", leave=False):
                    feature_data = np.array([feat[feature_name] for feat in segment_features])
                    if feature_data.ndim == 1:
                        feature_data = feature_data.reshape(-1, 1)
                    
                    clusterer = AgglomerativeClustering(n_clusters=min(k, len(segments)))
                    labels = clusterer.fit_predict(feature_data)
                    clustered_features[feature_name] = labels
                
                # Store results with memory cleanup
                results[f"piano_piece_{i}"] = {
                    'segments': segments,
                    'sr': sr,
                    'cluster_labels': clustered_features
                }
                
                # Explicit cleanup
                del audio, segments, segment_features
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        return results
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise RuntimeError(f"Processing failed: {str(e)}") from e

def apply_fade(audio: np.ndarray, fade_duration: float = 0.02, sr: int = 48000) -> np.ndarray:
    """Apply fade in and fade out to audio segment.
    
    Args:
        audio: Audio segment to apply fades to
        fade_duration: Duration of fade in seconds
        sr: Sample rate
        
    Returns:
        Audio with fades applied
    """
    fade_length = int(fade_duration * sr)
    
    # Ensure fade length isn't longer than half the audio
    if len(audio) < 2 * fade_length:
        fade_length = len(audio) // 4
    
    # Create fade curves
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    
    # Apply fades
    audio = audio.copy()  # Create a copy to avoid modifying original
    audio[:fade_length] *= fade_in
    audio[-fade_length:] *= fade_out
    
    return audio

def save_audio_chunks(
    chunks: List[np.ndarray], 
    sr: int, 
    output_dir: str, 
    piece_name: str, 
    segment_labels: np.ndarray,
    fade_duration: float = 0.02  # 20ms fade by default
) -> None:
    """Save audio chunks to disk with fades, organized into subfolders by piece and cluster."""
    # Create output directory for the piece if it doesn't exist
    piece_dir = os.path.join(output_dir, piece_name)
    os.makedirs(piece_dir, exist_ok=True)
    
    # Save each chunk in its corresponding cluster subfolder
    for i, chunk in enumerate(chunks):
        # Get the cluster label for this chunk
        cluster_label = segment_labels[i]
        
        # Create a subfolder for this cluster
        cluster_dir = os.path.join(piece_dir, f"cluster_{cluster_label}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        # Create the output path with cluster subfolder
        output_path = os.path.join(cluster_dir, f"chunk_{i}.wav")
        
        # Apply fades to the chunk
        faded_chunk = apply_fade(chunk, fade_duration=fade_duration, sr=sr)
        
        # Save the chunk
        sf.write(output_path, faded_chunk, sr)
