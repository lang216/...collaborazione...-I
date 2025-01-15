import numpy as np
from segmentation import AudioFeatureExtractor, load_audio
import librosa

def test_feature_extraction():
    # Load test audio
    audio, sr = load_audio("Piano Piece with Electronics/Raw Piano Materials/cheesy_piano_1.wav")
    
    # Extract 1-second segment
    segment = audio[:sr]  # First second of audio
    
    # Initialize feature extractor
    extractor = AudioFeatureExtractor()
    
    # Extract features
    features = extractor.extract_segment_features(segment, sr)
    
    # Verify results
    print("Extracted Features:")
    for feature_name, feature_data in features.items():
        if feature_data is not None:
            print(f"{feature_name}: shape={feature_data.shape}")
        else:
            print(f"{feature_name}: None")
    
    # Basic validation
    assert 'mfcc' in features, "MFCC features missing"
    # assert 'chroma' in features, "Chroma features missing"
    assert 'spectral_centroid' in features, "Spectral centroid missing"
    assert 'spectral_flatness' in features, "Spectral flatness missing"
    assert 'rms' in features, "RMS missing"
    
    print("\nAll feature types present")
    print("Feature extraction test passed!")

if __name__ == "__main__":
    test_feature_extraction()
