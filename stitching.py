import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from segmentation import AudioFeatureExtractor
from scipy.signal import fftconvolve
from pathlib import Path
import soundfile as sf
from hmmlearn import hmm
import shutil

# def filter_short_chunks(input_dir, min_duration=0.2):
#     """Filters audio files in directory, copying only those longer than min_duration.
#     Creates new directory with '_filtered' suffix and maintains original structure."""
#     input_path = Path(input_dir)
#     output_path = input_path.parent / f"{input_path.name}_filtered"

#     # Create output directory structure
#     output_path.mkdir(exist_ok=True)

#     # Walk through directory structure using pathlib
#     for item in input_path.rglob('*'):
#         if item.is_file() and item.suffix == '.wav':
#             try:
#                 # Get audio duration without loading the entire file
#                 duration = librosa.get_duration(path=item)
#                 if duration >= min_duration:
#                     # Create corresponding directory in output
#                     relative_path = item.relative_to(input_path)
#                     output_file_path = output_path / relative_path
#                     output_file_path.parent.mkdir(parents=True, exist_ok=True)
#                     # Copy file to corresponding location in output directory
#                     shutil.copy(item, output_file_path)
#             except Exception as e:
#                 print(f"Error processing {item}: {e}")


filter_short_chunks(r'Piano Piece with Electronics\Segmented_Audio')

def extract_features(file_path):
    """Extracts normalized sequences of MFCCs, Spectral Centroid, and Spectral Flatness."""
    try:
        # Use absolute path and updated librosa loading
        abs_path = os.path.abspath(file_path)
        audio, sr = librosa.load(abs_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {abs_path}: {e}")
        return None

    extractor = AudioFeatureExtractor()
    # Pass adjusted n_fft to feature extraction
    segment_features = extractor.extract_segment_features(audio, sr)

    mfcc = segment_features['mfcc']
    print(mfcc.shape)
    spectral_centroid = segment_features['spectral_centroid']
    print(spectral_centroid.shape)
    spectral_flatness = segment_features['spectral_flatness']
    print(spectral_flatness.shape)

    combined_features = np.hstack([mfcc, spectral_centroid, spectral_flatness])
    print(combined_features.shape)
    
    # Reshape combined_features to be 2D
    combined_features = np.array(combined_features).reshape(1, -1)  # Reshape to (1, n_features)
    combined_scaler = StandardScaler()
    # Now you can fit_transform the scaler
    combined_features = combined_scaler.fit_transform(combined_features)

    return combined_features

def prepare_dataset():
    """Prepares dataset for HMM training from audio files in cluster directories."""
    base_dir = Path(__file__).parent/'Segmented_Audio'
    dataset = {}
    for feature_type in os.listdir(base_dir):
        feature_dir = os.path.join(base_dir, feature_type)
        if os.path.isdir(feature_dir):
            for piano_piece in os.listdir(feature_dir):
                piano_piece_dir = os.path.join(feature_dir, piano_piece)
                if os.path.isdir(piano_piece_dir):
                    for cluster in os.listdir(piano_piece_dir):
                        cluster_dir = os.path.join(piano_piece_dir, cluster)
                        if os.path.isdir(cluster_dir):
                            file_paths = [os.path.join(cluster_dir, f) for f in os.listdir(cluster_dir) if f.endswith('.wav')]
                            if file_paths:
                                dataset[cluster_dir] = [extract_features(fp) for fp in file_paths]
    return dataset

def train_hmm(dataset, n_components=5):
    """Trains a Gaussian HMM on the prepared dataset and creates state-to-chunks mapping."""
    cluster_sequences = {}
    state_to_chunks = {}
    
    # Create mapping of states to chunks
    for cluster_dir, features_list in dataset.items():
        cluster_sequences[cluster_dir] = np.vstack(features_list)
        # Assign each cluster to a state
        state = len(state_to_chunks)  # Use sequential state numbers
        state_to_chunks[state] = [
            {'audio': librosa.load(fp, sr=None)[0], 'features': features}
            for fp, features in zip(
                [f for f in os.listdir(cluster_dir) if f.endswith('.wav')],
                features_list
            )
        ]
    
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    model.fit(np.vstack(list(cluster_sequences.values())))
    return model, state_to_chunks

def generate_sequence(model, state_to_chunks, length=10):
    """Generates a sequence of original audio chunks based on HMM states."""
    # Generate state sequence
    state_sequence = model.sample(length)[1]
    
    # Select chunks for each state
    selected_chunks = []
    for state in state_sequence:
        # Get all chunks for this state
        chunks = state_to_chunks[state]
        # Select random chunk from this state
        selected = np.random.choice(chunks)
        selected_chunks.append(selected['audio'])
    
    return selected_chunks

def crossfade(audio1, audio2, fade_duration=0.08, sr=22050):
    """Applies crossfade between two audio segments."""
    fade_samples = int(fade_duration * sr)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    # Apply crossfade
    audio1[-fade_samples:] *= fade_out
    audio2[:fade_samples] *= fade_in
    
    # Concatenate with overlap
    return np.concatenate([audio1[:-fade_samples], 
                         audio1[-fade_samples:] + audio2[:fade_samples], 
                         audio2[fade_samples:]])

def layer_audio(audio_chunks, overlap=0.2, sr=22050):
    """Layers multiple audio chunks with specified overlap."""
    overlap_samples = int(overlap * sr)
    max_length = max(len(chunk) for chunk in audio_chunks)
    layered = np.zeros(max_length + (len(audio_chunks)-1)*overlap_samples)
    
    for i, chunk in enumerate(audio_chunks):
        start = i * overlap_samples
        end = start + len(chunk)
        layered[start:end] += chunk
    
    return layered

def stitch_audio(model, state_to_chunks, num_segments=10, crossfade_duration=0.1):
    """Generates and stitches original audio chunks with crossfades and layering."""
    stitched_audio = []
    previous_audio = None
    
    # Generate sequence of original chunks
    chunks = generate_sequence(model, state_to_chunks, num_segments)
    
    for chunk in chunks:
        if previous_audio is not None:
            # Crossfade with previous segment
            chunk = crossfade(previous_audio, chunk, crossfade_duration)
        
        stitched_audio.append(chunk)
        previous_audio = chunk
    
    # Layer overlapping segments
    return layer_audio(stitched_audio)


# if __name__ == '__main__':
#     dataset = prepare_dataset()
#     model, state_to_chunks = train_hmm(dataset)
    
#     # Generate and save stitched audio using original chunks
#     stitched = stitch_audio(model, state_to_chunks, num_segments=10, crossfade_duration=0.08)
#     output_path = Path(__file__).parent/'stitched_output.wav'
#     sf.write(output_path, stitched, 48000)
#     print(f"Saved stitched audio to {output_path}")
