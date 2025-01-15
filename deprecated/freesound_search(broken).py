import freesound
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_api_key() -> str:
    """Load API key from environment variables."""
    load_dotenv()
    api_key = os.getenv("FREESOUND_API_KEY")
    if not api_key:
        raise ValueError(
            "FREESOUND_API_KEY not found in environment variables. "
            "Please create a .env file with your API key."
        )
    return api_key

class FreesoundSearch:
    def __init__(self, api_key: str = None):
        """Initialize Freesound client."""
        self.client = freesound.FreesoundClient()
        self.client.set_token(api_key or load_api_key())

    def search_similar(
        self,
        sound_id: int,
        num_results: int = 10,
        fields: List[str] = ["id", "name", "tags", "previews", "duration"]
    ) -> List[Dict]:
        """Search for sounds similar to a given Freesound ID.
        
        Args:
            sound_id: Freesound ID of the reference sound (must be positive integer)
            num_results: Number of similar sounds to return (1-100)
            fields: Sound fields to retrieve
            
        Raises:
            ValueError: If input parameters are invalid
            freesound.FreesoundException: For API errors
        """
        if not isinstance(sound_id, int) or sound_id <= 0:
            raise ValueError("sound_id must be a positive integer")
        if not isinstance(num_results, int) or num_results < 1 or num_results > 100:
            raise ValueError("num_results must be between 1 and 100")
        if not all(isinstance(f, str) for f in fields):
            raise ValueError("fields must be a list of strings")
            
        try:
            # Set timeout and retry settings
            self.client.set_timeout(30)  # 30 second timeout
            # Get the reference sound
            sound = self.client.get_sound(sound_id)
            
            # Search for similar sounds
            results = sound.get_similar(
                fields=",".join(fields),
                page_size=num_results
            )
            
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "tags": s.tags,
                    "preview_url": s.previews.preview_hq_mp3,
                    "duration": s.duration
                }
                for s in results
            ]
            
        except Exception as e:
            logger.error(f"Error searching similar sounds: {str(e)}")
            return []

    def download_sounds(
        self,
        sounds: List[Dict],
        output_dir: str,
        max_retries: int = 3
    ) -> List[Path]:
        """Download sounds to the specified directory.
        
        Args:
            sounds: List of sound dictionaries to download
            output_dir: Directory to save downloaded files
            max_retries: Maximum number of download retry attempts
            
        Returns:
            List of Path objects for successfully downloaded files
            
        Raises:
            PermissionError: If output directory is not writable
            ValueError: If sounds list is empty
        """
        if not sounds:
            raise ValueError("sounds list cannot be empty")
            
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Test directory writability
            test_file = output_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Cannot write to output directory: {output_dir}") from e
            
        downloaded_files = []
        for sound in sounds:
            for attempt in range(max_retries):
                try:
                    # Get full sound object
                    s = self.client.get_sound(sound["id"])
                    
                    # Clean filename of invalid characters
                    clean_name = "".join(c for c in sound['name'] if c.isalnum() or c in (' ', '-', '_'))
                    filename = f"{sound['id']}_{clean_name}.mp3"
                    filepath = output_dir / filename
                    
                    # Download using requests
                    import requests
                    response = requests.get(sound["preview_url"])
                    if response.status_code == 200:
                        filepath.write_bytes(response.content)
                        downloaded_files.append(filepath)
                        logger.info(f"Successfully downloaded: {filename}")
                        break  # Success - exit retry loop
                    else:
                        logger.error(f"Failed to download {filename}: HTTP {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for sound {sound['id']}: {str(e)}")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"Failed to download sound {sound['id']} after {max_retries} attempts")
                
        return downloaded_files

    def extract_audio_features(self, file_path: Path) -> Dict[str, List[float]]:
        """Extract audio features from a file using librosa.
        
        Args:
            file_path: Path to audio file to analyze
            
        Returns:
            Dictionary of audio features including:
            - MFCC coefficients
            - Spectral contrast
            - Zero crossing rate
            - Spectral centroid
            - Spectral spread
            - Spectral flatness
            - Spectral rolloff
            - Spectral flux
            - RMS energy
            - Loudness
            
        Raises:
            ValueError: If file is invalid or feature extraction fails
        """
        import librosa
        try:
            # Validate audio file
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.warning(f"Skipping invalid file: {file_path}")
                raise ValueError(f"Invalid file: {file_path}")
            
            # Load audio file
            y, sr = librosa.load(file_path, sr=None, mono=True)
            if len(y) == 0:
                logger.warning(f"Empty audio data in file: {file_path}")
                raise ValueError(f"Empty audio data: {file_path}")
            
            # Extract low-level features
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            # Calculate MFCC statistics
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)
            mfcc_max = np.max(mfcc, axis=1)
            mfcc_min = np.min(mfcc, axis=1)
            
            # Calculate first and second derivatives
            mfcc_diff = np.diff(mfcc, axis=1)
            mfcc_diff2 = np.diff(mfcc_diff, axis=1)
            
            mfcc_dmean = np.mean(mfcc_diff, axis=1)
            mfcc_dmean2 = np.mean(mfcc_diff2, axis=1)
            mfcc_dvar = np.var(mfcc_diff, axis=1)
            mfcc_dvar2 = np.var(mfcc_diff2, axis=1)

            # Calculate features
            features = {
                "lowlevel.mfcc.dmean": [float(v) for v in mfcc_dmean],
                "lowlevel.mfcc.dmean2": [float(v) for v in mfcc_dmean2],
                "lowlevel.mfcc.dvar": [float(v) for v in mfcc_dvar],
                "lowlevel.mfcc.dvar2": [float(v) for v in mfcc_dvar2],
                "lowlevel.mfcc.max": [float(v) for v in mfcc_max],
                "lowlevel.mfcc.mean": [float(v) for v in mfcc_mean],
                "lowlevel.mfcc.min": [float(v) for v in mfcc_min],
                "lowlevel.mfcc.var": [float(v) for v in mfcc_var],
            }
            
            # Validate features
            for feature_name, values in features.items():
                if not isinstance(values, list) or not all(isinstance(v, (float, int)) for v in values):
                    raise ValueError(f"Invalid feature values for {feature_name}")
                
            logger.debug(f"Extracted features for {file_path}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error for {file_path}: {str(e)}")
            raise ValueError(f"Feature extraction failed: {str(e)}")

    def _validate_folders(self, input_path: Path, output_path: Path) -> None:
        """Validate input and output folders."""
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_path}")
            
        audio_files = list(input_path.rglob("*.wav")) + list(input_path.rglob("*.mp3"))
        if not audio_files:
            raise ValueError(f"No audio files found in: {input_path}")
            
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            test_file = output_path / ".test_write"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise PermissionError(f"Cannot write to output folder: {output_path}") from e

    def _process_single_file(
        self, 
        file_path: Path, 
        num_results: int,
        target_duration: Optional[float] = None,
        target_tags: Optional[List[str]] = None,
        min_similarity: float = 0.5
    ) -> tuple[str, list]:
        """Process a single audio file and return search results.
        
        Args:
            file_path: Path to audio file
            num_results: Number of results to return
            target_duration: Optional target duration in seconds
            target_tags: Optional list of tags to filter by
            min_similarity: Minimum similarity score (0.0-1.0)
        """
        try:
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.warning(f"Skipping invalid file: {file_path}")
                return str(file_path), []

            # Extract audio features
            logger.info(f"Extracting features from {file_path}")
            descriptors = self.extract_audio_features(file_path)
            logger.debug(f"Descriptors for {file_path}: {descriptors}")
            
            if not descriptors:
                logger.warning(f"No descriptors extracted from {file_path}")
                return str(file_path), []
                
            # Format MFCC descriptors for Freesound API
            mfcc_mean = descriptors["lowlevel.mfcc.mean"]
            mfcc_var = descriptors["lowlevel.mfcc.var"]
            
            # Format values as comma-separated strings
            m = ",".join(["%.3f" % x for x in mfcc_mean])
            v = ",".join(["%.3f" % x for x in mfcc_var])
            
            # Build search parameters
            search_params = {
                "target": f"lowlevel.mfcc.mean:{m} lowlevel.mfcc.var:{v}",
                "fields": "id,name,url,analysis",
                "descriptors": "lowlevel.mfcc.mean,lowlevel.mfcc.var",
                "filter": "duration:0 TO 3"
            }
            # Remove None values
            search_params = {k: v for k, v in search_params.items() if v is not None}
            
            # Perform content-based search
            logger.info(f"Searching for similar sounds to {file_path}")
            try:
                search_results = self.client.content_based_search(**search_params)
                
                # Process results with similarity scores
                valid_results = []
                for s in search_results:
                    try:
                        if hasattr(s, 'id') and hasattr(s, 'previews'):
                            # Get similarity score from analysis metadata
                            similarity = float(s.analysis['lowlevel']['mfcc']['similarity'])
                            if similarity >= min_similarity:
                                result = {
                                    "id": s.id,
                                    "name": s.name,
                                    "preview_url": s.previews.preview_hq_mp3,
                                    "similarity_score": similarity,
                                    "duration": s.duration
                                }
                                valid_results.append(result)
                                logger.debug(f"Found result: {result}")
                    except Exception as e:
                        logger.error(f"Error processing result: {str(e)}")
                        continue
            except Exception as api_error:
                logger.error(f"API error searching for {file_path}: {str(api_error)}")
                return str(file_path), []
            
            logger.info(f"Found {len(valid_results)} valid results for {file_path}")
            return str(file_path), valid_results
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            return str(file_path), []

    def search_by_content(
        self,
        input_folder: str,
        output_folder: str,
        num_results: int = 5,
        max_parallel: int = 4,
        target_duration: Optional[float] = None,
        target_tags: Optional[List[str]] = None,
        min_similarity: float = 0.5
    ) -> Dict[str, List[Dict]]:
        """Search for similar sounds based on audio content.
        
        Args:
            input_folder: Path to folder containing audio files to analyze
            output_folder: Path to folder where results should be saved
            num_results: Number of similar sounds to find per input file (1-15)
            max_parallel: Maximum parallel requests to Freesound API (1-8)
            target_duration: Optional target duration in seconds for results
            target_tags: Optional list of tags to filter results
            min_similarity: Minimum similarity score (0.0-1.0)
            
        Returns:
            Dictionary mapping input file paths to lists of similar sounds
            
        Raises:
            ValueError: If input parameters are invalid
            PermissionError: If output folder cannot be written to
        """
        # Validate parameters
        if not 1 <= num_results <= 15:
            raise ValueError("num_results must be between 1 and 15")
        if not 1 <= max_parallel <= 8:
            raise ValueError("max_parallel must be between 1 and 8")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")
            
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        self._validate_folders(input_path, output_path)
        
        results = {}
        processed_files = 0
        total_files = sum(1 for _ in input_path.rglob("*.wav"))
        
        logger.info(f"Starting content-based search with {total_files} files")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(
                    self._process_single_file, 
                    file, 
                    num_results,
                    target_duration,
                    target_tags,
                    min_similarity
                ): file
                for file in input_path.rglob("*.wav")
            }
            
            logger.info(f"Created {len(futures)} search tasks")
            
            for future in as_completed(futures):
                try:
                    file_path, similar_sounds = future.result()
                    processed_files += 1
                    
                    if similar_sounds:
                        logger.debug(f"Found {len(similar_sounds)} results for {file_path}")
                        relative_path = Path(file_path).relative_to(input_path)
                        output_subfolder = output_path / relative_path.parent
                        output_subfolder.mkdir(parents=True, exist_ok=True)
                        results[str(file_path)] = similar_sounds
                    else:
                        logger.debug(f"No results found for {file_path}")
                        
                    logger.info(
                        f"Processed {processed_files}/{total_files} files "
                        f"({processed_files/total_files:.1%})"
                    )
                    
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    continue
                    
        logger.info(f"Search completed. Found results for {len(results)}/{total_files} files")
        if len(results) == 0:
            logger.warning("No results found. Check input files and API connection")
        return results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Example usage
if __name__ == "__main__":
    try:
        # Load API key from environment variable
        API_KEY = load_api_key()
        fs = FreesoundSearch(API_KEY)
        
        # Search for sounds using content-based search
        input_folder = "Piano Piece with Electronics/Segmented_Audio_filtered/mfcc/piano_piece_0/cluster_2"  # Folder containing audio files to analyze
        output_folder = "test_output"  # Folder to save results
        num_results = 1  # Number of similar sounds to find per input file
        
        logger.info(f"Starting content-based search on {input_folder}")
        results = fs.search_by_content(
            input_folder=input_folder,
            output_folder=output_folder,
            num_results=num_results,
            max_parallel=4,
            # target_duration=5.0,  # Optional: target duration in seconds
            # target_tags=["piano", "electronic"],  # Optional: filter by tags
            min_similarity=0.5  # Minimum similarity score
        )
        
        # Process and download results
        downloaded_files = []
        for input_file, similar_sounds in results.items():
            if similar_sounds:
                logger.info(f"Found {len(similar_sounds)} results for {input_file}")
                downloaded_files.extend(
                    fs.download_sounds(similar_sounds, output_folder)
                )
        
        logger.info(f"Successfully downloaded {len(downloaded_files)} sounds to {output_folder}")
        
        # Print results to console
        if results:
            print("\nSearch Results:")
            for input_file, similar_sounds in results.items():
                print(f"\nInput file: {input_file}")
                for sound in similar_sounds:
                    print(f"- {sound['name']} (ID: {sound['id']}, Score: {sound['similarity_score']:.2f})")
        else:
            print("No matching sounds found.")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
