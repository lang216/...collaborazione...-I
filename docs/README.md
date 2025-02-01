# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Audio segmentation and microtonal processing system for contemporary classical composition

## Core Architecture

```python
.
├── config.json              # seg_match_main configuration
├── src/
│   ├── seg_match_main.py              # Primary processing pipeline
│   ├── chord_builder.py     # Microtonal chord construction
│   ├── chord_extract.py     # Chord sequence extraction
│   ├── spray_notes.py       # Probabilistic MIDI generation
│   ├── segmentation.py      # Audio segmentation logic
│   └── config_utils.py      # Configuration management
├── utils/
│   ├── frequency_converter.py # Pitch/frequency conversions
│   ├── pitch_converter.py   # OpenMusic notation tools
│   └── audio_renamer.py     # File organization utilities
├── audio/                   # Audio material hierarchy
│   ├── Audio_Raw_Materials/
│   ├── Segmented_Audio/
│   └── Segmented_Audio_Filtered/
└── docs/                    # Documentation
```

## Enhanced Features

### Microtonal Processing (src/chord_extract.py)
- Automatic chord sequence extraction from audio sources
- Formant-preserving pitch shifting using RubberBand
- Dynamic voice allocation (1-8 voices)
- Spectral analysis-driven chord selection

### Audio Segmentation
- Multivariate onset detection (energy + spectral flux)
- Adaptive fade curves based on segment content
- Parallelized processing pipeline

### Configuration Management
```python
# config_utils.py
class ConfigManager:
    def __init__(self):
        self.schema = {
            "paths": {
                "input_dir": {"type": "str", "required": True},
                "chord_output_dir": {"type": "str", "default": "./output"}
            },
            "segmentation": {
                "k_clusters": {"type": "int", "min": 2, "max": 10}
            }
        }
    
    def validate(self, config):
        # Type checking and constraint validation
        ...
```

## Installation & Setup

```bash
# Clone with audio materials submodule
git clone --recurse-submodules https://github.com/lang216/yuseok_piano_piece.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with audio processing extras
pip install -r requirements.txt[audio]
```

## Processing Pipeline

1. Configure paths in config.json
2. Run seg_match_main segmentation process:
```bash
python src/seg_match_main.py --input-dir audio/Audio_Raw_Materials --bpm 72
```
3. Generate chord sequences from segmented audio:
```bash
python src/chord_extract.py --source audio/Segmented_Audio --output output/chords
```
4. Create MIDI spray notes:
```bash
python src/spray_notes.py --duration 300 --density 4
```

## Key Configuration Options

| Section         | Parameter              | Type    | Default      | Description                  |
|-----------------|------------------------|---------|--------------|------------------------------|
| paths           | input_dir              | string  | (required)   | Raw audio source directory   |
| segmentation    | k_clusters             | integer | 5            | Feature clusters for grouping|
| spray_notes     | density_notes_per_sec  | float   | 4.0          | MIDI event density           |
| chord_builder   | normalize_output       | boolean | True         | Peak normalization           |

## Contribution Guidelines

1. Branch naming: feature/description or fix/issue
2. Type hints required for all new code
3. Audio processing tests in tests/audio_processing
4. Documentation updates in docs/ with:
```bash
python utils/generate_docs.py --update
```

## Analysis Tools

```python
# utils/frequency_converter.py
def analyze_spectral_content(file_path):
    """Extract harmonic profile for microtonal processing"""
    y, sr = librosa.load(file_path)
    S = np.abs(librosa.stft(y))
    harmonics = librosa.effects.harmonic(y)
    return {
        'fundamental': librosa.yin(y, fmin=20, fmax=2000),
        'inharmonicity': np.mean(S - harmonics)
    }
