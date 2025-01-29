# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary sounds

## Key Features

- **Audio Segmentation**: 
  - Automatic segmentation using advanced onset detection with Audioflux
  - Smart fade handling for clean segment transitions
  - Memory-efficient processing with caching
  - Parallel processing optimization

- **Feature Extraction**: Comprehensive audio feature analysis including:
  - MFCC (Mel-frequency cepstral coefficients)
  - Spectral Centroid
  - Spectral Flatness
  - RMS Energy
  - Real-time feature visualization
  - Parallel feature extraction with joblib

- **Clustering**: 
  - Hierarchical clustering of audio segments by feature similarity
  - Automatic cluster validation
  - Minimum file count enforcement per cluster
  - Duration-based filtering

- **MIDI Generation (Spray Notes)**:
  - Probabilistic note generation
  - Configurable frequency ranges
  - Adjustable note density and duration
  - Velocity randomization
  - Automatic MIDI file creation

- **Chord Processing**:
  - Automatic chord extraction from source audio
  - High-precision pitch shifting with formant preservation
  - Support for up to 4 voices by default
  - Time-stretching capabilities
  - Comprehensive metadata generation

- **Freesound Integration**: 
  - Automated search for complementary sounds
  - Duration-based filtering
  - Feature-based matching

- **Performance Optimization**:
  - Memory caching for repeated operations
  - Multi-core processing using joblib
  - Efficient audio loading with librosa
  - Progress tracking with tqdm

## System Workflow

1. **Input Processing**:
   - Raw piano recordings (WAV format)
   - MIDI sources
   - Audio chord materials

2. **Audio Analysis**:
   - Onset detection using Audioflux
   - Multi-feature extraction in parallel
   - Automatic segmentation with fade handling
   - Duration-based filtering

3. **Generation & Transformation**:
   - Probabilistic MIDI note generation
   - Chord extraction and building
   - Feature-based clustering
   - Complementary sound matching

4. **Output**:
   - Organized audio chunks by feature type
   - Generated MIDI files
   - Processed chord stems and mixes
   - Matched Freesound samples

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lang216/yuseok_piano_piece.git
   cd yuseok_piano_piece
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Freesound API credentials in `.env`:
   ```bash
   FREESOUND_API_KEY=your_api_key
   ```

## Configuration Management

The system uses a combination of `config.json` and `config_utils.py` for configuration:

1. **Static Configuration** (`config.json`):
   - Contains all user-modifiable parameters
   - Organized into sections for paths, segmentation, and audio processing settings
   - Automatically validated on startup

2. **Configuration Utilities** (`config_utils.py`):
   - Provides helper functions for:
     - Validating configuration values
     - Loading and saving configuration
     - Generating default configurations
   - Ensures type safety through schema validation
   - Handles configuration versioning and migrations

Edit `config.json` to customize processing parameters:

```json
{
  "paths": {
    "input_dir": "audio/Audio_Raw_Materials",
    "segments_dir": "audio/Segmented_Audio",
    "filtered_segments_dir": "audio/Segmented_Audio_Filtered",
    "freesound_dir": "tests/Freesound_Matches_Test",
    "chord_output_dir": "./output",
    "spray_notes_dir": "./midi_output"
  },
  "segmentation": {
    "k_clusters": 5,
    "min_duration": 1.0,
    "max_duration": 5.0,
    "fade_duration": 0.02
  },
  "spray_notes": {
    "density_notes_per_second": 4.0,
    "total_duration": 60,
    "min_note_duration": 0.1,
    "max_note_duration": 2.0,
    "min_velocity": 60,
    "max_velocity": 100,
    "lower_freq": 20,
    "upper_freq": 20000
  },
  "freesound": {
    "max_results": 5,
    "min_duration": 1.0,
    "max_duration": 5.0
  },
  "chord_builder": {
    "normalize_output": true,
    "generate_metadata": true
  }
}
```

## Usage

### Main Processing Pipeline
```bash
python src/main.py
```

This will process audio files through the complete pipeline.

### MIDI Note Generation
```bash
python src/spray_notes.py
```

Generates probabilistic MIDI notes based on configured parameters.

### Chord Generation and Extraction
```bash
# Manual chord building
python src/chord_builder.py input.wav 6000 6500 6700 --detect-pitch

# Automatic chord extraction
python src/chord_builder.py input.wav --extract-chords 10 --chord-source source.wav
```

See [Chord Builder Documentation](docs/chord_builder_README.md) for detailed usage.

## Output Structure

```
audio/
├── Audio_Raw_Materials/          # Input piano recordings
├── Audio_Chord_Materials/        # Chord source and component files
├── Segmented_Audio/             # Initial segmentation results
└── Segmented_Audio_Filtered/    # Filtered segments after clustering

output/                          # Generated chord files
└── chord_[timestamp]/
    ├── stem_[note].wav         # Individual chord stems
    ├── mixed_chord.wav         # Mixed chord output
    └── metadata.json           # Processing details

midi_output/                     # Generated MIDI files
└── spray_notes_[timestamp].mid # Probabilistic note sequences

tests/
├── filtered_segments_dir_test/  # Test filtered segments
└── Freesound_Matches_Test/     # Freesound search results
```

## Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public methods
- Keep functions small and focused

### Performance Considerations

- Use memory caching for repeated operations
- Implement parallel processing where appropriate
- Monitor memory usage with large audio files
- Use progress bars for long-running operations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
