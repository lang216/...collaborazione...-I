from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Optional, Dict, List, Tuple

import librosa
import numpy as np
from hmmlearn import hmm
from joblib import Memory, Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import logging
import os
import time
import json

class AudioError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message):
        super().__init__(message)

@dataclass
class StitchingConfig:
    """Audio stitching configuration."""
    n_components: int = 2
    crossfade_duration: float = 0.2
    overlap_ratio: float = 0.2
    sampling_rate: int = 48000
    num_segments: int = 10
    cache_dir: str = '.cache'
    n_jobs: int = 4
    data_dir: str = 'Segmented_Audio_filtered'
    n_iterations: int = 1000
    min_covar: float = 1e-2
    means_prior: float = 1e-2
    means_weight: float = 1e-2
    version: str = "1.0"
    
    def get_worker_count(self) -> int:
        """Get number of workers based on n_jobs setting."""
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return self.n_jobs
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
        if self.crossfade_duration <= 0:
            raise ValueError("crossfade_duration must be positive")
        if not 0 <= self.overlap_ratio <= 1:
            raise ValueError("overlap_ratio must be between 0 and 1")
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.num_segments < 1:
            raise ValueError("num_segments must be positive")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup memory caching
memory = Memory(StitchingConfig().cache_dir, verbose=0)

@memory.cache
def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio file with caching."""
    try:
        abs_path = os.path.abspath(file_path)
        logger.debug(f"Loading audio from {abs_path}")
        return librosa.load(abs_path, sr=None)
    except Exception as e:
        logger.error(f"Error loading {abs_path}: {str(e)}")
        raise AudioError(f"Failed to load audio: {abs_path}") from e

class StitchingFeatureExtractor:
    """Extracts audio features specifically for audio stitching using parallel processing."""
    
    def __init__(self, config: StitchingConfig):
        self.config = config
        self.scaler = StandardScaler()

    def extract_segment_features(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        """Extract audio features from segment using parallel processing.
        
        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        try:
            parallel = Parallel(n_jobs=self.config.n_jobs, backend='threading')
            results = parallel([
                delayed(librosa.feature.mfcc)(y=audio, sr=sr),
                delayed(librosa.feature.spectral_centroid)(y=audio, sr=sr),
                delayed(librosa.feature.spectral_flatness)(y=audio)
            ])
            
            mfcc, spectral_centroid, spectral_flatness = [
                np.mean(r.T, axis=0) for r in results
            ]
            
            return {
                'mfcc': mfcc,
                'spectral_centroid': spectral_centroid,
                'spectral_flatness': spectral_flatness
            }
        except Exception as e:
            logger.error(f"Error extracting segment features: {str(e)}")
            raise AudioError(f"Failed to extract segment features: {str(e)}") from e

def extract_features(file_path: str, config: StitchingConfig) -> np.ndarray | None:
    """Extract and normalize audio features from file path."""
    try:
        audio, sr = load_audio(file_path)
        extractor = StitchingFeatureExtractor(config)
        segment_features = extractor.extract_segment_features(audio, sr)
        
        combined_features = np.hstack([
            segment_features['mfcc'].reshape(1, -1),
            segment_features['spectral_centroid'].reshape(1, -1),
            segment_features['spectral_flatness'].reshape(1, -1)
        ])
        
        return extractor.scaler.fit_transform(combined_features)
    except Exception as e:
        logger.error(f"Error extracting features from {file_path}: {str(e)}")
        return None

def save_cluster_dataset(cluster_path: str, features: np.ndarray) -> None:
    """Save cluster dataset to file directly in cluster folder with descriptive name."""
    try:
        # Extract piano piece, feature type and cluster name from path
        parts = Path(cluster_path).parts
        piano_piece = parts[-2]  # e.g. 'piano_piece_0'
        feature_type = parts[-3]  # e.g. 'mfcc'
        cluster_name = parts[-1]  # e.g. 'cluster_0'
        
        # Create descriptive filename
        dataset_name = f"{feature_type}_{piano_piece}_{cluster_name}_dataset.npy"
        dataset_path = os.path.join(cluster_path, dataset_name)
        logger.info(f"Attempting to save dataset to {dataset_path}")
        
        # Save directly without temporary file
        np.save(dataset_path, features)
        
        logger.info(f"Successfully saved dataset to {dataset_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset to {cluster_path}: {str(e)}")
        raise AudioError(f"Dataset save failed: {str(e)}") from e

def load_cluster_dataset(cluster_path: str) -> Optional[np.ndarray]:
    """Load cluster dataset from file if it exists using descriptive naming."""
    # Extract piano piece, feature type and cluster name from path
    parts = Path(cluster_path).parts
    piano_piece = parts[-2]  # e.g. 'piano_piece_0'
    feature_type = parts[-3]  # e.g. 'mfcc'
    cluster_name = parts[-1]  # e.g. 'cluster_0'
    
    # Look for dataset with descriptive name
    dataset_name = f"{feature_type}_{piano_piece}_{cluster_name}_dataset.npy"
    dataset_path = os.path.join(cluster_path, dataset_name)
    
    if os.path.exists(dataset_path):
        return np.load(dataset_path)
    return None

def prepare_dataset(config: StitchingConfig) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Prepare dataset for HMM training from audio files in cluster directories."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Segmented_Audio_filtered'))
    
    # Add debug logging for the base directory
    logger.info(f"Loading dataset from: {base_dir}")
    
    # Get all cluster directories
    cluster_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if 'cluster_' in root:
            cluster_dirs.append(root)
            # Add debug logging for each cluster directory
            logger.info(f"Found cluster directory: {root}")
    
    dataset = {}
    
    # Process each cluster directory
    with tqdm(total=len(cluster_dirs), desc="Processing clusters") as pbar:
        def process_cluster(cluster_dir: str) -> Optional[tuple[str, str, str, np.ndarray]]:
            # Parse path to get piece, feature and cluster info
            parts = Path(cluster_dir).relative_to(base_dir).parts
            if len(parts) < 3:
                logger.warning(f"Invalid cluster directory structure: {cluster_dir}")
                return None
                
            feature_type = parts[0]
            piece_name = parts[1]
            cluster_name = parts[2]
            
            # Debug logging for processing
            logger.info(f"Processing cluster: {piece_name}/{feature_type}/{cluster_name}")
            
            # List and log wav files
            wav_files = list(Path(cluster_dir).glob('*.wav'))
            logger.info(f"Found {len(wav_files)} wav files in cluster")
            for wav_file in wav_files:
                logger.info(f"Processing file: {wav_file}")
            
            # Check for existing dataset
            cached_data = load_cluster_dataset(cluster_dir)
            if cached_data is not None:
                pbar.update(1)
                return piece_name, feature_type, cluster_name, cached_data
                
            # Process files if no cached dataset
            features = []
            
            for file_path in wav_files:
                result = extract_features(str(file_path), config)
                if result is not None:
                    features.append(result)
            
            if features:
                # Save dataset for this cluster
                cluster_data = np.vstack(features)
                save_cluster_dataset(cluster_dir, cluster_data)
                pbar.update(1)
                return piece_name, feature_type, cluster_name, cluster_data
            return None
            
        # Use file lock for concurrent access
        from filelock import FileLock
        lock_path = os.path.join(base_dir, '.dataset_lock')
        
        # Reduce parallel workers for file operations
        file_workers = max(1, min(4, config.n_jobs // 2))
        
        def safe_process_cluster(cluster_dir: str) -> Optional[tuple[str, str, str, np.ndarray]]:
            # Retry logic for file operations
            max_retries = 3
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                try:
                    with FileLock(lock_path, timeout=10):
                        return process_cluster(cluster_dir)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to process cluster {cluster_dir} after {max_retries} attempts: {str(e)}")
                        return None
                    time.sleep(retry_delay * (attempt + 1))
                    retry_delay *= 2
            
            return None

        # Process clusters with reduced parallelism for file operations
        with Parallel(n_jobs=file_workers, backend='threading', verbose=10) as parallel:
            results = parallel(
                delayed(safe_process_cluster)(cd) for cd in cluster_dirs
            )
        
        # Build nested dataset dictionary
        for result in results:
            if result is not None:
                piece_name, feature_type, cluster_name, features = result
                
                # Initialize nested dictionaries if needed
                if piece_name not in dataset:
                    dataset[piece_name] = {}
                if feature_type not in dataset[piece_name]:
                    dataset[piece_name][feature_type] = {}
                
                # Add features to nested structure
                dataset[piece_name][feature_type][cluster_name] = features
                
    return dataset

def validate_model(model: hmm.GaussianHMM, data: np.ndarray) -> bool:
    try:
        score = model.score(data)
        if np.isfinite(score) and score > config.min_model_score:
            return True
        logger.warning(f"Model score {score} below threshold")
        return False
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def train_hmm(
    dataset: dict[str, dict[str, dict[str, np.ndarray]]], 
    config: StitchingConfig,
    save_model: bool = True
) -> tuple[
    dict[tuple[str, str, str], hmm.GaussianHMM],
    dict[int, list[dict[str, np.ndarray]]],  # Changed back to global state mapping
    Optional[str]
]:
    """Train separate Gaussian HMMs for each cluster and create state-to-chunks mapping."""
    models = {}
    state_to_chunks = {}  # Global state mapping
    state_counter = 0
    
    # Process clusters sequentially to avoid multiprocessing issues
    with tqdm(total=sum(
        len(clusters) 
        for piece in dataset.values() 
        for clusters in piece.values()
    ), desc="Training HMMs") as pbar:
        
        for piece_name, piece_data in dataset.items():
            for feature_type, clusters in piece_data.items():
                for cluster_name, features in clusters.items():
                    # Get cluster directory from name
                    cluster_dir = Path(
                        config.data_dir
                    ) / feature_type / piece_name / cluster_name
                    
                    # Get audio files for this cluster
                    audio_files = list(Path(cluster_dir).glob('*.wav'))
                    
                    # Create state mapping with additional logging
                    state_to_chunks[state_counter] = []
                    for i, fp in enumerate(audio_files):
                        chunk_info = {
                            'audio_path': str(fp),
                            'features': features[i],
                            'model_key': (piece_name, feature_type, cluster_name),
                            'state': state_counter  # Add state information
                        }
                        state_to_chunks[state_counter].append(chunk_info)
                        logger.info(f"Mapping state {state_counter} to audio: {fp.name} from {piece_name}/{feature_type}/{cluster_name}")
                    
                    # Train separate HMM for this cluster
                    cluster_data = features
                    
                    # Add small random noise and normalize data
                    cluster_data += np.random.normal(0, 1e-6, cluster_data.shape)
                    scaler = StandardScaler()
                    cluster_data = scaler.fit_transform(cluster_data)
                    cluster_data = np.clip(cluster_data, -3, 3)
                    
                    # Validate data quality
                    if np.any(np.isnan(cluster_data)) or np.any(np.isinf(cluster_data)):
                        logger.warning(f"Cluster {piece_name}/{feature_type}/{cluster_name} contains invalid data")
                        continue
                        
                    # Create and train HMM model
                    model = hmm.GaussianHMM(
                        n_components=config.n_components,
                        covariance_type="diag",
                        n_iter=config.n_iterations,
                        init_params='stmc',
                        params='stmc',
                        random_state=42,
                        min_covar=config.min_covar,
                        transmat_prior=1.0 + 1e-6,
                        means_prior=config.means_prior,
                        means_weight=config.means_weight,
                        startprob_prior=1.0 + 1e-6,
                        algorithm='map'
                    )
                    
                    try:
                        model.fit(cluster_data)
                        models[(piece_name, feature_type, cluster_name)] = model
                    except ValueError as e:
                        logger.error(f"Error fitting HMM for {piece_name}/{feature_type}/{cluster_name}: {str(e)}")
                    
                    state_counter += 1
                    pbar.update(1)
                    
                    # Add more detailed progress information
                    logger.info(f"Training model for {piece_name}/{feature_type}/{cluster_name}")
                    logger.info(f"Feature shape: {cluster_data.shape}")
                    logger.info(f"Model convergence score: {model.score(cluster_data)}")
        
        # If no valid models created, raise error
        if not models:
            raise AudioError("No valid clusters found for training")
    
    model_path = None
    if save_model:
        from joblib import dump
        import os
        
        # Ensure cache directory exists
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # Add versioning and metadata to saved models
        model_metadata = {
            'timestamp': time.time(),
            'config': asdict(config),
            'dataset_stats': {
                'n_clusters': len(models),
                'feature_types': list(set(k[1] for k in models.keys())),
                'pieces': list(set(k[0] for k in models.keys()))
            }
        }
        
        save_path = os.path.join(
            config.cache_dir, 
            f"hmm_models_v{config.version}_{int(time.time())}"
        )
        
        # Save the model and update model_path
        model_path = f"{save_path}.joblib"  # Store the full path
        dump({'models': models, 'metadata': model_metadata}, model_path)
        
        metadata_path = f"{save_path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info(f"Saved trained HMM models to {save_path}")
    
    return models, state_to_chunks, model_path

def select_chunk(state: int, state_to_chunks: dict[int, list[dict[str, np.ndarray]]], config: StitchingConfig) -> np.ndarray:
    """Select and load an audio chunk for a given state.
    
    Args:
        state: HMM state number
        state_to_chunks: Mapping of states to available audio chunks
        config: Configuration object with sampling rate
        
    Returns:
        Audio data as numpy array
        
    Raises:
        AudioError: If chunk selection or loading fails
    """
    try:
        chunks = state_to_chunks[state]
        selected = np.random.choice(chunks)
        # Load audio from path
        audio, _ = librosa.load(selected['audio_path'], sr=config.sampling_rate)
        return audio
    except Exception as e:
        logger.error(f"Error selecting/loading chunk for state {state}: {str(e)}")
        raise AudioError(f"Failed to select/load chunk: {str(e)}") from e

def layer_audio_with_hmm(
    audio_chunks: list[np.ndarray],
    model: hmm.GaussianHMM,
    features: np.ndarray,  # Features of the chunks
    sr: int = 48000,
    min_overlap: float = 0.1,
    max_overlap: float = 0.3
) -> np.ndarray:
    """Layer audio chunks with HMM-determined overlaps.
    
    Args:
        audio_chunks: List of audio segments to layer
        model: Trained HMM model
        features: Features of the audio chunks
        sr: Sample rate
        min_overlap: Minimum overlap ratio
        max_overlap: Maximum overlap ratio
        
    Returns:
        Layered audio array with HMM-determined overlaps
    """
    # Get state sequence from HMM
    state_sequence = model.predict(features)
    
    # Use state probabilities to determine overlap amounts
    state_probs = model.predict_proba(features)
    
    # Initialize output array
    max_length = max(len(c) for c in audio_chunks)
    total_length = max_length * len(audio_chunks)  # Initial estimate
    layered = np.zeros(total_length)
    
    current_position = 0
    
    for i, chunk in enumerate(audio_chunks[:-1]):
        # Place current chunk
        end_pos = current_position + len(chunk)
        layered[current_position:end_pos] += chunk
        
        # Determine overlap with next chunk based on HMM state probabilities
        current_state = state_sequence[i]
        next_state = state_sequence[i + 1]
        transition_prob = model.transmat_[current_state, next_state]
        
        # Scale transition probability to overlap range
        overlap_ratio = min_overlap + (max_overlap - min_overlap) * transition_prob
        overlap_samples = int(overlap_ratio * sr)
        
        # Update position for next chunk
        current_position = end_pos - overlap_samples
    
    # Add final chunk
    if audio_chunks:
        final_chunk = audio_chunks[-1]
        layered[current_position:current_position + len(final_chunk)] += final_chunk
    
    # Trim any excess zeros
    non_zero = np.nonzero(layered)[0]
    if len(non_zero) > 0:
        layered = layered[:non_zero[-1] + 1]
    
    # Normalize to prevent clipping
    max_amp = np.max(np.abs(layered))
    if max_amp > 0:
        layered = layered / max_amp
        
    return layered

def generate_sequences(
    models: dict[tuple[str, str, str], hmm.GaussianHMM],
    state_to_chunks: dict[int, list[dict[str, np.ndarray]]],
    length: int = 10,
    config: StitchingConfig = StitchingConfig()
) -> dict[tuple[str, str, str], list[list[np.ndarray]]]:
    """Generate sequences of original audio chunks using all HMM models."""
    sequences = {}
    
    try:
        for model_key, model in tqdm(models.items(), desc="Generating sequences from models"):
            piece_name, feature_type, cluster_name = model_key
            logger.info(f"Generating sequence for {piece_name}/{feature_type}/{cluster_name}")
            
            # Find states associated with this model
            model_states = []
            for state, chunks in state_to_chunks.items():
                if any(chunk['model_key'] == model_key for chunk in chunks):
                    model_states.append(state)
            
            logger.info(f"Found {len(model_states)} states for model {model_key}")
            
            model_sequences = []
            num_sequences = 3  # Number of sequences to generate per model
            
            for seq_idx in range(num_sequences):
                # Generate state sequence for this model
                state_sequence = model.sample(length)[1]
                
                # Get chunks and their features
                selected_chunks = []
                chunk_features = []
                
                for state in state_sequence:
                    # Only select chunks from the correct model/cluster
                    valid_chunks = [
                        chunk for chunk in state_to_chunks[state]
                        if chunk['model_key'] == model_key
                    ]
                    
                    if not valid_chunks:
                        logger.warning(f"No valid chunks found for state {state} in model {model_key}")
                        continue
                        
                    selected = np.random.choice(valid_chunks)
                    logger.info(f"Selected chunk from {selected['model_key']} for state {state}")
                    
                    audio, _ = librosa.load(selected['audio_path'], sr=config.sampling_rate)
                    selected_chunks.append(audio)
                    chunk_features.append(selected['features'])
                
                if selected_chunks:  # Only add if we found chunks
                    chunk_features = np.vstack(chunk_features)
                    model_sequences.append({
                        'chunks': selected_chunks,
                        'features': chunk_features
                    })
                else:
                    logger.warning(f"No chunks selected for sequence {seq_idx} of model {model_key}")
            
            if model_sequences:
                sequences[model_key] = model_sequences
            
    except Exception as e:
        logger.error(f"Error generating sequences: {str(e)}")
        raise
        
    return sequences

def crossfade(audio1: np.ndarray, audio2: np.ndarray, 
             fade_duration: float = 0.08, sr: int = 48000) -> np.ndarray:
    """Crossfade between two audio segments using linear interpolation.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment 
        fade_duration: Crossfade duration in seconds
        sr: Sample rate
        
    Returns:
        Crossfaded audio array
        
    Raises:
        ValueError: If audio segments are too short or sample rate is invalid
    """
    if sr <= 0:
        raise ValueError("Sample rate must be positive")
        
    fade_samples = int(fade_duration * sr)
    
    # Adjust fade duration if either segment is too short
    min_length = min(len(audio1), len(audio2))
    if min_length < fade_samples:
        fade_samples = min_length // 2
        if fade_samples == 0:
            raise ValueError("Audio segments too short for crossfade")
            
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    # Handle case where audio segments are shorter than fade_samples
    audio1_fade = audio1[-fade_samples:] if len(audio1) >= fade_samples else audio1
    audio2_fade = audio2[:fade_samples] if len(audio2) >= fade_samples else audio2
    
    # Apply fades
    audio1_fade *= fade_out[-len(audio1_fade):]
    audio2_fade *= fade_in[:len(audio2_fade)]
    
    return np.concatenate([
        audio1[:-fade_samples] if len(audio1) > fade_samples else np.array([]),
        audio1_fade + audio2_fade,
        audio2[fade_samples:] if len(audio2) > fade_samples else np.array([])
    ])

def select_model(
    models: dict[tuple[str, str, str], hmm.GaussianHMM],
    piece_name: Optional[str] = None,
    feature_type: Optional[str] = None,
    cluster_name: Optional[str] = None
) -> tuple[tuple[str, str, str], hmm.GaussianHMM]:
    """Select a specific model from the models dictionary."""
    for key, model in models.items():
        if (piece_name is None or key[0] == piece_name) and \
           (feature_type is None or key[1] == feature_type) and \
           (cluster_name is None or key[2] == cluster_name):
            return key, model
    raise ValueError(f"Model not found for piece: {piece_name}, feature: {feature_type}, cluster: {cluster_name}")

if __name__ == '__main__':
    config = StitchingConfig()
    try:
        # Validate configuration
        config.validate()
        
        logger.info("Starting audio stitching process...")
        logger.info(f"Configuration: {asdict(config)}")
        
        logger.info("Preparing dataset...")
        dataset = prepare_dataset(config)
        
        logger.info("Training HMM models...")
        models, state_to_chunks, model_path = train_hmm(dataset, config)
        if not models:
            raise AudioError("No models were successfully trained")
        logger.info(f"Model saved at: {model_path}")
        
        logger.info("Generating sequences...")
        sequences = generate_sequences(
            models=models,
            state_to_chunks=state_to_chunks,
            length=config.num_segments,
            config=config
        )
        
        if not sequences:
            raise AudioError("No sequences were generated")
        
        logger.info("Processing and saving sequences...")
        # Create output directory structure
        output_dir = Path(__file__).parent / 'hmm_out'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_key, model_sequences in sequences.items():
            piece_name, feature_type, cluster_name = model_key
            model = models[model_key]
            
            # Create subdirectory for this model
            model_dir = output_dir / f"{piece_name}_{feature_type}_{cluster_name}"
            model_dir.mkdir(exist_ok=True)
            
            # Process each sequence
            for seq_idx, sequence_data in enumerate(model_sequences):
                audio_sequence = sequence_data['chunks']
                features = sequence_data['features']
                
                if not audio_sequence:
                    logger.warning(f"Empty sequence {seq_idx} for {piece_name}/{feature_type}/{cluster_name}, skipping")
                    continue
                
                # Create crossfaded version
                crossfaded = audio_sequence[0]
                for i in range(1, len(audio_sequence)):
                    crossfaded = crossfade(
                        crossfaded,
                        audio_sequence[i],
                        fade_duration=config.crossfade_duration,
                        sr=config.sampling_rate
                    )
                
                # Create HMM-layered version
                layered = layer_audio_with_hmm(
                    audio_sequence,
                    model,
                    features,
                    sr=config.sampling_rate,
                    min_overlap=config.overlap_ratio * 0.5,  # Use config overlap as reference
                    max_overlap=config.overlap_ratio * 1.5
                )
                
                # Save both versions
                import soundfile as sf
                
                crossfaded_path = model_dir / f"crossfaded_seq_{seq_idx}.wav"
                sf.write(crossfaded_path, crossfaded, config.sampling_rate)
                logger.info(f"Saved crossfaded sequence to {crossfaded_path}")
                
                layered_path = model_dir / f"hmm_layered_seq_{seq_idx}.wav"
                sf.write(layered_path, layered, config.sampling_rate)
                logger.info(f"Saved HMM-layered sequence to {layered_path}")
            
        logger.info("Audio stitching process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
