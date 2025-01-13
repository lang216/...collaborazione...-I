# 🎹 Piano+Electronics Audio Segmentation Tool

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
| Resource Monitoring | Real-time CPU and memory monitoring |
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
