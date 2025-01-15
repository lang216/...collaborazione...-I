import pytest
import numpy as np
import os
from stitching import (
    StitchingConfig,
    load_audio,
    StitchingFeatureExtractor,
    extract_features,
    prepare_dataset,
    train_hmm,
    generate_sequence,
    crossfade,
    layer_audio,
    stitch_audio,
    AudioError
)
from pathlib import Path

@pytest.fixture
def config():
    return StitchingConfig(n_jobs=1)  # Use 1 job for consistent testing

@pytest.fixture
def test_audio_path():
    return str(Path(__file__).parent / "Raw Piano Materials" / "cheesy_piano_1.wav")

def test_load_audio(test_audio_path):
    """Test loading audio file with caching"""
    audio, sr = load_audio(test_audio_path)
    assert isinstance(audio, np.ndarray)
    assert sr > 0
    assert len(audio) > 0

def test_extract_segment_features(config, test_audio_path):
    """Test feature extraction from audio segment"""
    audio, sr = load_audio(test_audio_path)
    extractor = StitchingFeatureExtractor(config)
    features = extractor.extract_segment_features(audio, sr)
    
    assert isinstance(features, dict)
    assert 'mfcc' in features
    assert 'spectral_centroid' in features
    assert 'spectral_flatness' in features
    assert all(isinstance(v, np.ndarray) for v in features.values())

def test_extract_features(config, test_audio_path):
    """Test feature extraction wrapper"""
    features = extract_features(test_audio_path, config)
    assert isinstance(features, np.ndarray)
    assert features.shape[0] == 1  # Single feature vector

def test_crossfade():
    """Test crossfading between two audio segments"""
    # Use realistic audio lengths (200ms = 4410 samples at 22050Hz)
    audio1 = np.random.rand(4410)
    audio2 = np.random.rand(4410)
    sr = 22050
    
    # Test normal case
    result = crossfade(audio1, audio2, sr=sr)
    assert isinstance(result, np.ndarray)
    assert len(result) > max(len(audio1), len(audio2))
    assert np.all(result >= 0)  # No negative values after crossfade
    
    # Test with different length inputs
    short_audio = np.random.rand(1000)
    result = crossfade(audio1, short_audio, sr=sr)
    assert isinstance(result, np.ndarray)
    
    # Test with empty input
    with pytest.raises(ValueError):
        crossfade(np.array([]), audio2, sr=sr)
    
    # Test with invalid sample rate
    with pytest.raises(ValueError):
        crossfade(audio1, audio2, sr=0)

def test_layer_audio():
    """Test audio layering with overlap"""
    chunks = [np.random.rand(1000) for _ in range(3)]
    sr = 22050
    
    layered = layer_audio(chunks, sr=sr)
    assert isinstance(layered, np.ndarray)
    assert len(layered) > len(chunks[0])  # Should be longer than single chunk

def test_prepare_dataset(config):
    """Test dataset preparation with parallel processing and file locking"""
    dataset = prepare_dataset(config)
    assert isinstance(dataset, dict)
    assert len(dataset) > 0
    
    # Verify nested dataset structure
    for piece_name, piece_data in dataset.items():
        assert isinstance(piece_name, str)
        assert piece_name.startswith('piano_piece_')
        
        # Verify we have the expected feature types
        expected_features = {'mfcc', 'spectral_centroid', 'spectral_flatness', 'rms'}
        assert set(piece_data.keys()).issuperset(expected_features)
        
        # Verify cluster structure for each feature type
        for feature_type, clusters in piece_data.items():
            assert isinstance(clusters, dict)
            assert len(clusters) >= 4  # At least 4 clusters per feature
            
            for cluster_name, features in clusters.items():
                assert cluster_name.startswith('cluster_')
                assert isinstance(features, np.ndarray)
                assert features.ndim == 2
                assert features.shape[0] > 0  # At least one feature vector
                assert features.shape[1] > 0  # Features have multiple dimensions
                assert not np.isnan(features).any()
                assert not np.isinf(features).any()

def test_save_load_cluster_dataset(config, tmp_path):
    """Test saving and loading cluster datasets"""
    from stitching import save_cluster_dataset, load_cluster_dataset, AudioError
    
    # Create test cluster directory
    test_cluster_dir = tmp_path / "test_cluster"
    test_cluster_dir.mkdir()
    
    # Create test data
    test_data = np.random.rand(10, 20)
    
    # Test normal save/load cycle
    save_cluster_dataset(str(test_cluster_dir), test_data)
    
    # Verify the file was created
    dataset_files = list(test_cluster_dir.glob('*.npy'))
    assert len(dataset_files) == 1
    
    # Load the saved data
    loaded_data = load_cluster_dataset(str(test_cluster_dir))
    assert isinstance(loaded_data, np.ndarray)
    np.testing.assert_array_equal(test_data, loaded_data)
    
    # Test error handling for save
    with pytest.raises(AudioError) as exc_info:
        save_cluster_dataset(str(test_cluster_dir), "invalid_data")  # Invalid data type
    assert "Dataset save failed" in str(exc_info.value)
    
    # Test error handling for load with invalid path
    invalid_path = str(tmp_path / "nonexistent" / "invalid" / "path")
    with pytest.raises(AudioError) as exc_info:
        load_cluster_dataset(invalid_path)
    assert "Dataset file not found" in str(exc_info.value)
    
    # Test error handling for load with invalid file
    invalid_file = tmp_path / "invalid_file.npy"
    invalid_file.touch()
    with pytest.raises(AudioError) as exc_info:
        load_cluster_dataset(str(invalid_file))
    assert "Failed to load dataset" in str(exc_info.value)

def test_parallel_processing(config):
    """Test parallel processing with different worker counts"""
    
    # Test default worker count
    assert config.get_worker_count() == min(4, os.cpu_count())
    
    # Test explicit worker counts
    for n in [1, 2, 4, 8]:
        config.n_jobs = n
        assert config.get_worker_count() == min(n, os.cpu_count())
    
    # Test invalid worker counts
    config.n_jobs = 0
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        config.get_worker_count()
        
    config.n_jobs = -1
    assert config.get_worker_count() == (os.cpu_count() or 1)
        
    # Test parallel processing integrity
    config.n_jobs = 2
    dataset = prepare_dataset(config)
    assert isinstance(dataset, dict)
    assert len(dataset) > 0

# def test_train_hmm(config):
#     """Test HMM training"""
#     dataset = prepare_dataset(config)
#     model, state_to_chunks = train_hmm(dataset, config)
#     assert model is not None
#     assert isinstance(state_to_chunks, dict)
#     # Verify state_to_chunks structure
#     for state, chunks in state_to_chunks.items():
#         assert isinstance(state, int)
#         assert isinstance(chunks, list)
#         assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

# def test_stitch_audio(config):
#     """Test audio stitching with trained model"""
#     dataset = prepare_dataset(config)
#     model, state_to_chunks = train_hmm(dataset, config)
    
#     # Test stitching with different segment counts
#     for num_segments in [2, 3, 5]:
#         stitched = stitch_audio(model, state_to_chunks, num_segments=num_segments, crossfade_duration=config.crossfade_duration)
#         assert isinstance(stitched, np.ndarray)
#         assert len(stitched) > 0
#         assert np.all(stitched >= -1.0) and np.all(stitched <= 1.0)  # Valid audio range
        
#     # Test edge cases
#     with pytest.raises(ValueError):
#         stitch_audio(model, state_to_chunks, num_segments=0)  # Invalid segment count
        
#     with pytest.raises(ValueError):
#         stitch_audio(model, {}, num_segments=3)  # Empty state_to_chunks

if __name__ == "__main__":
    pytest.main()
