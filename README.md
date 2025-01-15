# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary electronic sounds

## Key Features

- **Audio Segmentation**: Automatic segmentation of piano recordings using onset detection
- **Feature Extraction**: Comprehensive audio feature analysis including:
  - MFCC (Mel-frequency cepstral coefficients)
  - Spectral Centroid
  - Spectral Flatness
  - RMS Energy
- **Clustering**: Hierarchical clustering of audio segments by feature similarity
- **Freesound Integration**: Automated search for complementary sounds from Freesound.org
- **Parallel Processing**: Optimized for multi-core CPUs using joblib
- **Structured Output**: Organized storage of processed audio chunks by feature type and cluster

## System Workflow

1. **Input**: Raw piano recordings in WAV format
2. **Processing**:
   - Audio segmentation using onset detection
   - Feature extraction and clustering
   - Duration-based filtering of audio chunks
3. **Output**:
   - Organized audio chunks by feature type and cluster
   - Filtered audio chunks meeting duration requirements
   - Complementary sounds from Freesound

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
   - Organized into sections for paths, segmentation, and freesound settings
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
    "input_dir": "Raw Piano Materials",
    "segments_dir": "Segmented_Audio",
    "filtered_segments_dir": "Segmented_Audio_Filtered",
    "freesound_dir": "Freesound_Results"
  },
  "segmentation": {
    "k_clusters": 5,
    "min_duration": 1.0,
    "max_duration": 5.0
  },
  "freesound": {
    "max_results": 5,
    "min_duration": 1.0,
    "max_duration": 5.0
  }
}
```

## Usage

Run the main processing pipeline:

```bash
python main.py
```

This will:
1. Process all WAV files in the `Raw Piano Materials` directory
2. Save segmented audio chunks organized by feature type and cluster
3. Filter chunks based on duration requirements
4. Search Freesound for complementary sounds using filtered chunks

## Output Structure

Processed audio is organized in the following directory structure:

```
Segmented_Audio/
├── mfcc/
│   ├── piano_piece_0/
│   │   ├── cluster_0/
│   │   ├── cluster_1/
│   │   └── ...
│   └── piano_piece_1/
│       ├── cluster_0/
│       └── ...
├── spectral_centroid/
├── spectral_flatness/
└── rms/
```

Filtered audio chunks are stored in `Segmented_Audio_Filtered` with the same structure.

Freesound results are stored in `Freesound_Results` with subdirectories matching the input file names.

## Development

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public methods
- Keep functions small and focused

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
