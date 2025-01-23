import freesound
import librosa
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import logging
from dotenv import load_dotenv
from dataclasses import dataclass
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time  
import random  # Add at top with other imports

logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    
    # Audio parameters
    sr: Optional[int] = None
    mono: bool = True
    
    # MFCC parameters
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    
    # Search parameters
    duration_multiplier: float = 1.2
    max_parallel_searches: int = 4
    min_results_per_file: int = 5
    
    # Output parameters
    output_subfolder_format: str = r"{feature_type}/{piece_name}/cluster_{cluster_id}/chunk_{chunk_id}"
    
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

class FolderProcessor:
    """Process audio files in a folder structure."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.matcher = FreesoundMatcher(config=self.config)
    
    def get_relative_output_path(self, input_path: Path, file_path: Path) -> Tuple[str, str, str, str]:
        """Generate output path components preserving relative structure.
        
        Args:
            input_path: Base input directory path (Segmented_Audio_filtered)
            file_path: Full path to the audio file
            
        Returns:
            Tuple of (feature_type, piece_name, cluster_id, chunk_id) for organizing output
            
        Example path:
            input: Segmented_Audio_filtered/mfcc/piano_piece_1/cluster_0/chunk_29.wav
            returns: ("mfcc", "piano_piece_1", "cluster_0", "chunk_29")
        """
        # Get relative path from input directory to file
        rel_path = file_path.relative_to(input_path)
        parts = rel_path.parts
        # Always use file stem as chunk_id
        chunk_id = Path(file_path).stem
        
        # Handle different path structures
        if len(parts) >= 3:
            # Standard structure: feature_type/piece_name/cluster_X/audio.wav
            feature_type = parts[0]    # e.g., mfcc
            piece_name = parts[1]      # e.g., piano_piece_1
            cluster_id = parts[2]      # e.g., cluster_0
        elif len(parts) == 2:
            # Shallow structure: piece_name/audio.wav
            feature_type = "mixed_features"
            piece_name = parts[0]
            cluster_id = parts[1]
        elif len(parts) == 1:
            # Flat structure: audio.wav
            feature_type = "mixed_features"
            piece_name = "unknown_piece"
            cluster_id = "unknown_cluster"
        else:
            # Fallback for empty paths
            logger.warning(f"Unexpected path structure for {file_path}, using defaults")
            feature_type = "mixed_features"
            piece_name = rel_path.parent.name if rel_path.parent.name != "." else "unknown_piece"
            cluster_id = rel_path.parent.stem if rel_path.parent.stem != "." else "unknown_cluster"
            
       
            
        return feature_type, piece_name, cluster_id, chunk_id
    
    def process_folder(self, input_path: Union[str, Path], output_folder: Union[str, Path]) -> Dict[str, Any]:
        """Process audio files in the input path.
        
        Args:
            input_path: Path to either a single WAV file or a folder containing WAV files
            output_folder: Path where matched sounds will be saved
            
        Returns:
            Dictionary mapping input files to their matching results
        """
        input_path = Path(input_path)
        output_path = Path(output_folder)
        
        # Handle single file case
        if input_path.is_file() and input_path.suffix.lower() == '.wav':
            audio_files = [input_path]
            # Use parent as base path for relative path calculation
            base_path = input_path.parent
        else:
            # Handle folder case
            audio_files = list(input_path.rglob("*.wav"))
            base_path = input_path
            
        if not audio_files:
            raise ValueError(f"No WAV files found at {input_path}")
        
        results = {}
        with ThreadPoolExecutor(max_workers=self.config.max_parallel_searches) as executor:
            futures = []
            for file_path in audio_files:
                future = executor.submit(self._process_single_file, base_path, file_path, output_path)
                futures.append((file_path, future))
            
            # Process results with progress bar
            for file_path, future in tqdm(futures, desc="Processing files"):
                try:
                    file_results = future.result()
                    if file_results:
                        results[str(file_path)] = file_results
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        return results
    
    def _process_single_file(self, input_path: Path, file_path: Path, output_base: Path) -> List[Dict]:
        """Process a single audio file and save results."""
        try:
            # Generate output path components
            feature_type, piece_name, cluster_id, chunk_id = self.get_relative_output_path(input_path, file_path)
            
            # Find matching sounds
            matching_sounds = self.matcher.match_sounds(file_path)
            
            if matching_sounds:
                # Create output subfolder with proper cluster/chunk structure
                # Preserve original numeric IDs from input paths
                cluster_num = cluster_id.split('_')[-1]  # Get numeric part after last underscore
                chunk_num = chunk_id.split('_')[-1]      # Get numeric part after last underscore
                
                # Create output path using original directory structure
                output_subfolder = output_base / feature_type / piece_name / f"cluster_{cluster_num}" / f"chunk_{chunk_num}"
                output_subfolder.mkdir(parents=True, exist_ok=True)
                
                # Download sounds
                for sound in matching_sounds[:self.config.min_results_per_file]:
                    download_sound(sound, output_subfolder)
                
            return matching_sounds
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []

class AudioFeatureExtractor:
    """Handles audio feature extraction and processing."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize feature extractor."""
        self.config = config or AudioConfig()
        self.config.validate()

    def extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features from audio."""
        if sr is None:
            raise ValueError("Sample rate cannot be None")
        return librosa.feature.mfcc(
            y=audio, 
            sr=sr, 
            n_mfcc=self.config.n_mfcc,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )

    def extract_rms(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract RMS (Root Mean Square) energy from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            RMS energy for each frame
            
        Raises:
            ValueError: If audio is empty or invalid
        """
        if audio.size == 0:
            raise ValueError("Audio cannot be empty")
            
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
            
        try:
            return librosa.feature.rms(
                y=audio,
                frame_length=self.config.n_fft,
                hop_length=self.config.hop_length
            )
        except Exception as e:
            logger.error(f"Error extracting RMS: {e}")
            raise

    def extract_spectral_features(
        self, 
        audio: np.ndarray, 
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract spectral features from audio."""
        centroid = librosa.feature.spectral_centroid(
            y=audio, 
            sr=sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        flatness = librosa.feature.spectral_flatness(
            y=audio,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        return centroid, flatness

    def compute_statistics(
        self, 
        feature: np.ndarray, 
        axis: int = 1
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Compute mean and variance statistics of a feature."""
        mean = np.mean(feature, axis=axis)
        var = np.var(feature, axis=axis)
        return mean, var

class FreesoundMatcher:
    """Handles Freesound API interaction and sound matching."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[AudioConfig] = None
    ):
        """Initialize Freesound client.
        
        Args:
            api_key: Optional Freesound API key. If not provided, will attempt to load from environment.
            config: Optional audio configuration
        """
        self.client = freesound.FreesoundClient()
        self.client.set_token(api_key or self._load_api_key(), "token")
        self.config = config or AudioConfig()
        self.feature_extractor = AudioFeatureExtractor(self.config)

    def _load_api_key(self) -> str:
        """Load Freesound API key from environment variables.
        
        Returns:
            str: The API key from environment variables
            
        Raises:
            ValueError: If FREESOUND_API_KEY is not found in environment variables
        """
        load_dotenv()
        api_key = os.getenv('FREESOUND_API_KEY')
        if not api_key:
            raise ValueError("FREESOUND_API_KEY not found in environment variables")
        return api_key

    def format_feature_string(self, feature: Union[np.ndarray, float]) -> str:
        """Format feature values as string for API query."""
        if isinstance(feature, np.ndarray):
            return ",".join([f"{x:.3f}" for x in feature])
        return f"{feature:.3f}"

    def build_search_params(
        self, 
        features: Dict[str, Union[float, np.ndarray]], 
        duration: float
    ) -> Dict[str, str]:
        """Build search parameters for Freesound API."""
        search_target = " ".join([
            f"{name}:{self.format_feature_string(value)}"
            for name, value in features.items()
        ])
        
        return {
            "target": search_target,
            "fields": "id,name,previews,analysis,duration",
            "descriptors": ",".join(features.keys()),
            "filter": f"duration:[{duration} TO {duration * self.config.duration_multiplier}]"
        }

    def match_sounds(self, file_path: Union[str, Path]) -> List[Dict]:
        """Find matching sounds on Freesound."""
        # Add random delay between 1.3 and 2.1 seconds
        time.sleep(random.uniform(0.8, 1.3))
        
        # Load and process audio
        audio, sr = librosa.load(
            str(file_path), 
            sr=self.config.sr, 
            mono=self.config.mono
        )
        duration = len(audio) / sr
        
        # Extract features
        mfcc = self.feature_extractor.extract_mfcc(audio, sr)
        rms = self.feature_extractor.extract_rms(audio)
        centroid, flatness = self.feature_extractor.extract_spectral_features(audio, sr)
        
        # Compute statistics
        mfcc_mean, mfcc_var = self.feature_extractor.compute_statistics(mfcc)
        rms_mean, rms_var = self.feature_extractor.compute_statistics(rms)
        cent_mean, cent_var = self.feature_extractor.compute_statistics(centroid)
        flat_mean, flat_var = self.feature_extractor.compute_statistics(flatness)
        
        # Prepare features dictionary
        features = {
            "lowlevel.mfcc.mean": mfcc_mean,
            "lowlevel.mfcc.var": mfcc_var,
            "lowlevel.rms.mean": rms_mean,
            "lowlevel.rms.var": rms_var,
            "lowlevel.spectral_centroid.mean": cent_mean,
            "lowlevel.spectral_centroid.var": cent_var,
            "lowlevel.spectral_flatness.mean": flat_mean,
            "lowlevel.spectral_flatness.var": flat_var
        }
        
        # Search for matches using text_search
        search_params = self.build_search_params(features, duration)
        results = self.client.text_search(**search_params)
        
        return self._process_results(results)

    def _process_results(self, results: List) -> List[Dict]:
        """Process search results into standardized format."""
        processed_results = []
        for sound in results:
            try:
                processed_results.append({
                    'id': sound.id,
                    'name': sound.name,
                    'preview_url': sound.previews.preview_hq_mp3,
                    'duration': sound.duration
                })
                logger.info(f"Found sound: {sound.name} ({sound.duration}s)")
            except AttributeError as e:
                logger.warning(f"Skipping sound {sound.id}: {str(e)}")
                
        return processed_results

def download_sound(sound: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """Download a sound file."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Remove .mp3 extension since it's already in the filename
        # Create unique filename while preserving existing extension
        original_name = Path(sound['name'])
        # Use original chunk ID from processing
        clean_name = original_name.stem.replace(" ", "_").replace("-", "_")[:50]
        timestamp = int(time.time() * 1000)
        # Extract original IDs from input path components
        # Get original input path components from the source file structure
        input_parts = output_dir.parts[-4:]  # [feature_type, piece_name, cluster_X, chunk_Y]
        cluster_num = input_parts[2].split('_')[-1]  # Just the numeric cluster ID
        chunk_num = input_parts[3].split('chunk_')[-1]  # Just the numeric chunk ID
        output_file = output_dir / f"{chunk_num}_{sound['id']}_{clean_name}_{timestamp}{original_name.suffix}"
        # Add .mp3 if no extension present
        if not output_file.suffix:
            output_file = output_file.with_suffix(".mp3")
        
        response = requests.get(sound['preview_url'])
        if response.status_code == 200:
            if not output_file.exists():
                output_file.write_bytes(response.content)
                logger.info(f"Downloaded sound to: {output_file}")
            else:
                logger.warning(f"Skipping existing file: {output_file}")
            return output_file
    except Exception as e:
        logger.error(f"Failed to download sound: {str(e)}")
        return None

def search_and_download(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Optional[AudioConfig] = None,
    api_key: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Search for similar sounds on Freesound and download them.
    
    Args:
        input_path: Path to input folder or single WAV file
        output_path: Path where matched sounds will be saved
        config: Optional audio configuration parameters
        api_key: Optional Freesound API key. If not provided, will load from environment
        
    Returns:
        Dictionary mapping input files to their matching results
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Set up configuration
        if config is None:
            config = AudioConfig(
                sr=44100,
                n_mfcc=13,
                duration_multiplier=1.5,
                max_parallel_searches=4,
                min_results_per_file=1
            )
        
        # Process folder or file
        processor = FolderProcessor(config)
        results = processor.process_folder(input_path, output_path)
        
        # Log summary
        total_files = len(results)
        total_matches = sum(len(matches) for matches in results.values())
        logger.info(f"Processing complete:")
        logger.info(f"- Files processed: {total_files}")
        logger.info(f"- Total matches found: {total_matches}")
        if total_files > 0:
            logger.info(f"- Average matches per file: {total_matches/total_files:.1f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in Freesound search: {e}")
        raise

# if __name__ == "__main__":
#     # Example usage when run directly
#     try:
#         input_folder = Path(__file__).parent.parent / "tests/filtered_segments_dir_test"
#         output_folder = Path(__file__).parent.parent / "tests/Freesound_Matches_Test"
        
#         results = search_and_download(input_folder, output_folder)
        
#     except Exception as e:
#         logger.error(f"Error in main execution: {e}")
#         raise
