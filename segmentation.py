import numpy as np
import librosa
import os
from typing import Tuple, List, Dict, Any, Optional
from sklearn.cluster import AgglomerativeClustering
from joblib import Parallel, delayed, Memory
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging
from dataclasses import dataclass
from functools import lru_cache

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
    n_mfcc: int = 20
    n_chroma: int = 12
    onset_params: Dict = None
    
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
        assert self.n_chroma > 0, "n_chroma must be positive"
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
        aggregate=np.median
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
        if feature_type == 'mfcc':
            return np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=config.n_mfcc), axis=1)
        elif feature_type == 'spectral_centroid':
            return np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
        elif feature_type == 'spectral_flatness':
            return np.mean(librosa.feature.spectral_flatness(y=segment))
        elif feature_type == 'rms':
            return np.mean(librosa.feature.rms(y=segment))
        elif feature_type == 'chroma':
            return np.mean(librosa.feature.chroma_stft(y=segment, sr=sr, n_chroma=config.n_chroma), axis=1)
    
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
        feature_types = ['mfcc', 'spectral_centroid', 'spectral_flatness', 'rms', 'chroma']
        try:
            features = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._extract_single_feature)(segment, sr, feat_type, self.config)
                for feat_type in feature_types
            )
            return dict(zip(feature_types, features))
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise RuntimeError(f"Feature extraction failed: {str(e)}") from e

def process_audio_files(directory: str, k: int) -> Dict[str, Dict]:
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
                for feature_name in tqdm(['mfcc', 'spectral_centroid', 'spectral_flatness', 'rms', 'chroma'],
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

def save_audio_chunks(chunks: List[np.ndarray], sr: int, output_dir: str, piece_name: str, segment_labels: np.ndarray) -> None:
    """
    Save audio chunks to disk, organized into subfolders by piece and cluster.
    
    Args:
        chunks: List of audio segments to save
        sr: Sample rate of the audio
        output_dir: Base output directory
        piece_name: Name of the musical piece
        segment_labels: Cluster labels for each segment
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If saving fails
    """
    if not chunks:
        logger.warning("No chunks to save")
        return
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
    if not segment_labels or len(segment_labels) != len(chunks):
        raise ValueError("Segment labels must match chunks")
    
    try:
        # Create output directory for the piece
        piece_dir = os.path.join(output_dir, piece_name)
        os.makedirs(piece_dir, exist_ok=True)
        
        # Save chunks with progress tracking
        for i, chunk in enumerate(tqdm(chunks, desc=f"Saving {piece_name} chunks", leave=False)):
            try:
                # Get cluster label and create subfolder
                cluster_label = segment_labels[i]
                cluster_dir = os.path.join(piece_dir, f"cluster_{cluster_label}")
                os.makedirs(cluster_dir, exist_ok=True)
                
                # Create output path with format based on chunk size
                format = 'wav' if len(chunk) < 1000000 else 'flac'  # Use FLAC for larger files
                output_path = os.path.join(cluster_dir, f"chunk_{i}.{format}")
                
                # Save with appropriate format and compression
                sf.write(output_path, chunk, sr, format=format)
                
                # Explicit cleanup
                del chunk
                
            except Exception as e:
                logger.error(f"Error saving chunk {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error saving audio chunks: {str(e)}")
        raise RuntimeError(f"Error saving audio chunks: {str(e)}") from e

if __name__ == "__main__":
    AUDIO_DIR = Path("Piano Piece/example_audio")
    OUTPUT_DIR = Path("Piano Piece/Segmented_Audio")
    
    try:
        CONFIG.validate()
        results = process_audio_files(str(AUDIO_DIR), k=8)
        
        for piece_name, data in results.items():
            segments = data['segments']
            sr = data['sr']
            
            for feature_name, labels in tqdm(data['cluster_labels'].items(), 
                                          desc=f"Saving {piece_name}"):
                feature_output_dir = OUTPUT_DIR / feature_name
                save_audio_chunks(segments, sr, str(feature_output_dir), piece_name, labels)
                
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
