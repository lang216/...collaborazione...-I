"""
Main Entry Point for Audio Segmentation System

This script provides the main interface for processing piano audio files
using the segmentation and feature analysis functionality.

Usage:
    python main.py

The script will:
1. Load audio files from Raw Piano Materials directory
2. Process them using the segmentation pipeline
3. Save and organize audio chunks by feature type and cluster labels
4. Filter out short audio chunks
5. Search for similar sounds on Freesound for each chunk

Configuration is handled through config.json.
"""

from segmentation import AudioFeatureExtractor, process_audio_files, save_audio_chunks
from freesound_search import search_and_download
from config_utils import load_config, create_audio_config
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    # Load configuration
    config_data = load_config()
    config = create_audio_config(config_data)
    
    # Get paths from config
    input_dir = Path(__file__).parent.parent / config_data["paths"]["input_dir"]
    segments_dir = Path(__file__).parent.parent / config_data["paths"]["segments_dir"]
    filtered_dir = Path(__file__).parent.parent / config_data["paths"]["filtered_segments_dir"]
    freesound_dir = Path(__file__).parent.parent / config_data["paths"]["freesound_dir"]
    print("Processing audio files...")
    results = process_audio_files(str(input_dir), k=config_data["segmentation"]["k_clusters"])
    
    # Save results first
    for piece_name, data in results.items():
        print(f"Saving {piece_name}...")
        segments = data['segments']
        sr = data['sr']
        
        # Save using each feature's clustering
        for feature_name, labels in data['cluster_labels'].items():
            feature_output_dir = segments_dir / feature_name
            save_audio_chunks(segments, sr, str(feature_output_dir), piece_name, labels)
            
    # Filter short chunks after saving
    print("Filtering short audio chunks...")
    feature_extractor = AudioFeatureExtractor(config)
    feature_extractor.filter_chunk_durations(
        str(segments_dir), 
        min_duration=config_data["segmentation"]["min_duration"],
        max_duration=config_data["segmentation"]["max_duration"],
    )
    
    print(f"Processing complete! Results saved to: {segments_dir}")
    print(f"Filtered chunks saved to: {filtered_dir}")
    
    # Search for similar sounds on Freesound using filtered chunks
    print("\nSearching Freesound for similar sounds...")
    freesound_results = search_and_download(
        input_path=filtered_dir,  # Use filtered chunks for Freesound search
        output_path=freesound_dir,
        config=config
    )
    
    # Print summary
    total_files = len(freesound_results)
    total_matches = sum(len(matches) for matches in freesound_results.values())
    print(f"\nFreesound search complete:")
    print(f"- Files processed: {total_files}")
    print(f"- Total matches found: {total_matches}")
    print(f"- Average matches per file: {total_matches/total_files if total_files > 0 else 0:.1f}")
    print(f"- Results saved to: {freesound_dir}")
    
except FileNotFoundError as e:
    print(f"File not found: {str(e)}")
    exit(1)
except ValueError as e:
    print(f"Invalid configuration: {str(e)}")
    exit(1)
except RuntimeError as e:
    print(f"Processing error: {str(e)}")
    exit(1)
