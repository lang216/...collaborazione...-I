"""
Example Usage of Audio Segmentation Tool

This script demonstrates how to use the audio segmentation and feature analysis
functionality from segmentation.py in your own projects.

Usage:
    python example_usage.py

The script will:
1. Load audio files from Raw_Piano_Materials directory
2. Process them using the segmentation pipeline
3. Save segmented audio chunks to Segmented_Audio directory
4. Organize chunks by feature type and cluster labels

Configuration can be modified in the AudioConfig instance.
"""

from segmentation import AudioConfig, process_audio_files, save_audio_chunks
from pathlib import Path

# Configuration
config = AudioConfig(
    sr=None,  # Sample rate
    mono=True,  # Mono audio
    hop_length=512,  # Hop length for analysis
    n_mfcc=20,  # Number of MFCC coefficients
    n_chroma=12  # Number of chroma features
)

# Input/output directories
input_dir = Path(__file__).parent / "Raw Piano Materials"
output_dir = Path(__file__).parent / "Segmented_Audio"

# Process audio files
try:
    print("Processing audio files...")
    results = process_audio_files(str(input_dir), k=8)
    
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
    
except FileNotFoundError as e:
    print(f"File not found: {str(e)}")
except ValueError as e:
    print(f"Invalid configuration: {str(e)}")
except RuntimeError as e:
    print(f"Processing error: {str(e)}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
