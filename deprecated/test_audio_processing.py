import os
from segmentation import process_audio_files
import numpy as np

def test_audio_processing():
    # Test parameters
    test_dir = "Piano Piece with Electronics/Raw Piano Materials"
    test_file = "cheesy_piano_1.wav"
    k = 3  # Number of clusters
    
    # Run processing
    results = process_audio_files(test_dir, k)
    
    # Verify results structure
    assert len(results) > 0, "No results returned"
    first_key = list(results.keys())[0]
    piece_data = results[first_key]
    
    # Verify required keys exist
    assert 'segments' in piece_data, "Segments missing"
    assert 'sr' in piece_data, "Sample rate missing"
    assert 'cluster_labels' in piece_data, "Cluster labels missing"
    
    # Verify segment data
    segments = piece_data['segments']
    assert len(segments) > 0, "No segments created"
    for segment in segments:
        assert isinstance(segment, np.ndarray), "Segment is not numpy array"
        assert len(segment) > 0, "Empty segment"
    
    # Verify cluster labels
    cluster_labels = piece_data['cluster_labels']
    assert len(cluster_labels) > 0, "No cluster labels"
    for feature, labels in cluster_labels.items():
        assert len(labels) == len(segments), "Label count mismatch"
        assert len(set(labels)) <= k, "Too many clusters"
    
    print("\nAudio processing test passed!")
    print(f"Processed {len(segments)} segments")
    print(f"Cluster counts per feature:")
    for feature, labels in cluster_labels.items():
        unique, counts = np.unique(labels, return_counts=True)
        print(f"{feature}: {dict(zip(unique, counts))}")

if __name__ == "__main__":
    test_audio_processing()
