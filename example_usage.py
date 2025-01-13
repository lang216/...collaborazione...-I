"""
Example Usage of Audio Segmentation Tool

This script demonstrates how to use the audio segmentation and feature analysis
functionality from test.py in your own projects.
"""

from segmentation import AudioConfig, process_audio_files, save_audio_chunks
from pathlib import Path

# Configuration
config = AudioConfig(
    sr=44100,  # Sample rate
    mono=True,  # Mono audio
    hop_length=512,  # Hop length for analysis
    n_mfcc=20,  # Number of MFCC coefficientss
    n_chroma=12  # Number of chroma features
)

# Input/output directories
input_dir = Path(__file__).parent / "Raw Piano Materials"
output_dir = Path("example_output")

# Process audio files
try:
    print("Processing audio files...")
    results = process_audio_files(str(input_dir), k=5)
    
    # Save results
    for piece_name, data in results.items():
        print(f"Saving {piece_name}...")
        segments = data['segments']
        sr = data['sr']
        
        # Save using each feature's clustering
        for feature_name, labels in data['cluster_labels'].items():
            feature_output_dir = output_dir / feature_name
            save_audio_chunks(segments, sr, str(feature_output_dir), piece_name, labels)
            
    print("Processing complete! Results saved to:", output_dir)
    
except Exception as e:
    print(f"Error processing audio: {str(e)}")
