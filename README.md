# 🎹 Audio Segmentation Tool for a Piece Written for Piano+Electronics 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Audio segmentation and feature analysis tool for Piano+Electronics composition

## Table of Contents
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Configuration Options](#configuration-options)
  - [Command Line Usage](#command-line-usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contributing](#contributing)

## Features
| Feature | Description |
|---------|-------------|
| Audio Processing | Load and process audio files with caching |
| Feature Extraction | Extract MFCC, spectral, and chroma features |
| Parallel Processing | Optimized for multi-core CPUs |
| Error Handling | Comprehensive validation and error recovery |
| Output Organization | Structured output of processed audio chunks |

## Quick Start

```python
from segmentation import AudioProcessor

# Initialize processor with default settings
processor = AudioProcessor()

# Process audio file
results = processor.analyze("path/to/audio.wav")

# Save segmented audio
processor.save_segments(results, "output_directory")
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/audio-segmentation.git
   cd audio-segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create cache directory:
   ```bash
   mkdir .cache
   ```

## Usage

### Basic Usage

```python
from segmentation import process_audio_files, save_audio_chunks, AudioConfig

# Configure processing
config = AudioConfig(
    sr=44100,
    mono=True,
    hop_length=512,
    n_mfcc=20,
    n_chroma=12,
    max_workers=4
)

# Process audio files
results = process_audio_files("path/to/audio/files", k=8)

# Save results
for piece_name, data in results.items():
    save_audio_chunks(
        data['segments'],
        data['sr'],
        "output/directory",
        piece_name,
        data['cluster_labels']['mfcc']
    )
```

### Configuration Options

The `AudioConfig` class provides validated configuration options:

- `sr`: Sample rate (default: None)
- `mono`: Convert to mono (default: True)
- `hop_length`: Hop length for analysis (default: 512)
- `n_mfcc`: Number of MFCC coefficients (default: 20)
- `n_chroma`: Number of chroma bins (default: 12)
- `max_workers`: Maximum parallel workers (default: CPU count - 1)
- `max_memory`: Maximum memory usage in MB (default: None)

### Command Line Usage

Run the example script:

```bash
python example_usage.py
```

## Development

### Setting Up Development Environment
1. Clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Running Tests
```bash
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public methods
- Keep functions small and focused

## Troubleshooting

### Resource Management

The system provides real-time resource monitoring and validation:

1. Memory Usage:
   - Monitors total and used memory
   - Logs memory usage before/after processing
   - Validates against `max_memory` setting

2. CPU Usage:
   - Tracks CPU utilization
   - Adjusts worker count based on usage

3. Processing Time:
   - Tracks total processing time
   - Provides performance metrics

If you encounter resource issues:
1. Check resource logs in console
2. Reduce `max_workers`
3. Set appropriate `max_memory` in `AudioConfig`
4. Clear cache directory: `rm -rf .cache/*`

### Error Handling

The system provides comprehensive error handling:

1. Audio Loading:
   - Verifies file existence
   - Validates file format (WAV recommended)
   - Checks file permissions
   - Detects corrupted files

2. Configuration:
   - Validates all configuration parameters
   - Provides clear error messages for invalid values

3. Processing:
   - Tracks progress with detailed logging
   - Provides resource usage warnings
   - Preserves partial results on failure

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## Workflow

### Generating the "Unwanted" Piano Piece
Objective: Use AI to generate a cheesy, pop-style piano piece as raw material.
Tool: Suno (or any AI music generator).
Prompt: Generate multiple versions if necessary to find the most "cheesy" result.

### Splitting the Piano Music into Fragments
Objective: Segment the generated piano music into meaningful fragments using content-aware methods.
Criteria for Splitting:
- Onset Detection: Split at note or chord changes (e.g., using librosa.onset.onset_detect).
- Novelty Detection: Identify transitions between contrasting sections (e.g., via self-similarity matrices).
- Spectral Features: Split based on timbral changes (e.g., spectral centroid or flatness).
- Energy-Based Segmentation: Divide based on loudness levels (e.g., RMS energy).
- Harmonic vs. Percussive Decomposition: Separate harmonic and percussive components (e.g., HPSS in librosa).
- Silence Detection: Split at pauses or low-energy regions (e.g., pydub.silence.detect_nonsilent).
- Beat and Tempo Analysis: Align splits with rhythmic patterns or tempo changes.

Tools:
- Python libraries: Librosa, Essentia, madmom, or pydub.
- Spectral analysis tools like Sonic Visualiser for manual refinement.

### Filtering the Fragments
Objective: Reduce redundancy and select fragments that align with your aesthetic goals.
Filtering Criteria:
- Timbre-Based Filtering: Use MFCCs or spectral centroid to group fragments by timbre.
- Spectral Complexity: Retain fragments with high or low harmonic richness.
- Dynamic Range: Focus on fragments with strong dynamic contrasts.
- Rhythmic Patterns: Select fragments with regular or irregular rhythmic structures.
- Pitch Content: Choose fragments based on tonal stability, dissonance, or microtonality.
- Novelty Scores: Retain unique fragments with high novelty values.

Clustering for Filtering:
Use clustering algorithms to organize fragments by feature similarity:
- K-Means for grouping by timbre or rhythm.
- DBSCAN for identifying outliers or unique fragments.
- Hierarchical clustering for nested relationships between fragments.
- Spectral clustering for subtle textural differences.

Tools for Clustering:
- Python libraries: sklearn.cluster (K-Means, DBSCAN), Librosa (feature extraction), matplotlib/seaborn (visualization).
- Dimensionality reduction techniques like t-SNE or UMAP for visualizing clusters.

### Enriching Fragments Using Freesound API
Objective: Retrieve similar sounds from Freesound.org to layer with your piano fragments.
Steps:
- Query Freesound API using features extracted from each fragment:
  - Spectral descriptors (e.g., pitch, timbre, spectral centroid).
  - Tags related to emotional quality or texture (e.g., "bright," "metallic").
  - Duration constraints to match fragment lengths.

Tools:
- Freesound API documentation and Python SDK for querying sounds programmatically.

Combining Sounds:
- Layer retrieved sounds with piano fragments using time-stretching, pitch-shifting, or spectral morphing techniques.

### Morphing Sounds
Objective: Blend piano fragments with retrieved sounds to create hybrid textures.
Recommended Tools for Morphing:
- Zynaptiq MORPH (plugin): Real-time audio morphing with multiple algorithms.
- MeldaProduction MMorph (plugin): Spectral morphing based on harmonic features.
- CDP (Composers' Desktop Project): Command-line tools for experimental sound transformations.
- iZotope Iris: Spectral editing and layering of audio components.

### Developing Piano Materials Using Spectral Techniques
Analyze combined sounds to extract spectral features such as harmonic content, inharmonicity, or formant structures.
Translate these features into playable piano material:
- Microtonal clusters derived from partials.
- Rhythmic patterns inspired by transient behavior in electronic textures.

Tools:
- SPEAR (Sinusoidal Partial Editing Analysis and Resynthesis) for detailed spectral analysis.
- Python libraries (Librosa, Essentia) for extracting spectral data programmatically.

### Structuring the Composition
Develop a structure where electronic textures interact dynamically with live piano material:
- Start with fragmented electronic textures that gradually merge with live piano playing.
- Alternate between sections dominated by electronics and those focusing on acoustic piano.
- Consider spatialization techniques to enhance interaction between acoustic and electronic elements:
  - Assign electronic sounds to different speakers in a multi-channel setup.
