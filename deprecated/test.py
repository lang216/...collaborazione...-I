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

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    
    # Audio parameters
    sr: Optional[int] = None  # Sample rate (None for original)
    mono: bool = True
    
    # MFCC parameters
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    
    # Feature extraction parameters
    duration_multiplier: float = 1.5  # For search duration filter
    
    # API parameters
    api_search_fields: str = "id,name,previews,analysis,duration"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_mfcc <= 0:
            raise ValueError("n_mfcc must be positive")
        if self.n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.duration_multiplier <= 0:
            raise ValueError("duration_multiplier must be positive")

class AudioFeatureExtractor:
    """Handles audio feature extraction and processing."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize feature extractor.
        
        Args:
            config: Audio processing configuration
        """
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

def load_api_key() -> str:
    """Load Freesound API key from environment variables."""
    load_dotenv()
    api_key = os.getenv('FREESOUND_API_KEY')
    if not api_key:
        raise ValueError("FREESOUND_API_KEY not found in environment variables")
    return api_key

class FreesoundMatcher:
    """Handles Freesound API interaction and sound matching."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        config: Optional[AudioConfig] = None
    ):
        """Initialize Freesound client."""
        self.client = freesound.FreesoundClient()
        self.client.set_token(api_key or load_api_key())
        self.config = config or AudioConfig()
        self.feature_extractor = AudioFeatureExtractor(self.config)

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
            "fields": self.config.api_search_fields,
            "descriptors": ",".join(features.keys()),
            "filter": f"duration:[0 TO {duration * self.config.duration_multiplier}]"
        }

    def match_sounds(self, file_path: Union[str, Path]) -> List[Dict]:
        """Find matching sounds on Freesound."""
        # Load and process audio
        audio, sr = librosa.load(
            str(file_path), 
            sr=self.config.sr, 
            mono=self.config.mono
        )
        duration = len(audio) / sr
        
        # Extract features
        mfcc = self.feature_extractor.extract_mfcc(audio, sr)
        centroid, flatness = self.feature_extractor.extract_spectral_features(audio, sr)
        
        # Compute statistics
        mfcc_mean, mfcc_var = self.feature_extractor.compute_statistics(mfcc)
        cent_mean, cent_var = self.feature_extractor.compute_statistics(centroid)
        flat_mean, flat_var = self.feature_extractor.compute_statistics(flatness)
        
        # Prepare features dictionary
        features = {
            "lowlevel.mfcc.mean": mfcc_mean,
            "lowlevel.mfcc.var": mfcc_var,
            "lowlevel.spectral_centroid.mean": cent_mean,
            "lowlevel.spectral_centroid.var": cent_var,
            "lowlevel.spectral_flatness.mean": flat_mean,
            "lowlevel.spectral_flatness.var": flat_var
        }
        
        # Search for matches
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
        output_file = output_dir / f"{sound['id']}_{sound['name']}"
        
        response = requests.get(sound['preview_url'])
        if response.status_code == 200:
            output_file.write_bytes(response.content)
            logger.info(f"Downloaded sound to: {output_file}")
            return output_file
            
    except Exception as e:
        logger.error(f"Failed to download sound: {str(e)}")
        return None

def main() -> None:
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create configuration
        config = AudioConfig(
            sr=44100,  # Set specific sample rate
            n_mfcc=13,
            duration_multiplier=1.5
        )
        
        test_file = Path(__file__).parent / "Segmented_Audio_filtered/rms/piano_piece_1/cluster_0/chunk_29.wav"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
            
        matcher = FreesoundMatcher(config=config)
        matching_sounds = matcher.match_sounds(test_file)
        
        if matching_sounds:
            logger.info(f"Found {len(matching_sounds)} matching sounds")
            output_dir = Path(__file__).parent / "matched_sounds"
            download_sound(matching_sounds[0], output_dir)
        else:
            logger.warning("No matching sounds found")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()