# Chord Builder Usage Guide

## Requirements
- Python 3.8+
- Required packages: `librosa`, `numpy`, `soundfile`, `pyrubberband`
- Install dependencies: `pip install -r requirements.txt`

## Basic Usage

### Manual Chord Building
```bash
python src/chord_builder.py input.wav 6000 6500 6700 --detect-pitch
```

### Automatic Chord Extraction
```bash
python src/chord_builder.py input.wav --extract-chords 10 --chord-source source.wav --detect-pitch
```

## Arguments
- `input.wav`: Path to input audio file (WAV, AIFF, or FLAC)
- `6000 6500 6700`: List of chord notes in OpenMusic format (100 = 1 semitone)
- `--detect-pitch`: Auto-detect the original pitch of the input audio
- `--original_note`: Manually specify the original note (e.g., 6000)
- `--output_dir`: Output directory (default: configured in config.json)
- `--sr`: Sample rate (choices: 44100, 48000, 96000)
- `--extract-chords`: Number of chords to extract automatically
- `--chord-source`: Audio file to extract chords from (default: audio/Audio_Chord_Materials/sources/output.wav)
- `--num-voices`: Number of voices for chord extraction (default: 4)

## Example 1: Auto-detect pitch with manual chords
```bash
python src/chord_builder.py piano.wav 6000 6500 6700 --detect-pitch
```

## Example 2: Automatic chord extraction
```bash
python src/chord_builder.py input.wav --extract-chords 10 --chord-source source.wav --detect-pitch
```

## Example 3: Custom output directory and sample rate
```bash
python src/chord_builder.py guitar.wav 6000 6500 6700 --detect-pitch --output_dir ./output --sr 48000
```

## Programmatic Usage
```python
from chord_builder import run_chord_builder

# Example with manual chords
run_chord_builder(
    input_audio="piano.wav",
    chord_notes=[6000, 6500, 6700],
    detect_pitch=True
)

# Example with automatic chord extraction
run_chord_builder(
    input_audio="input.wav",
    extract_chords=10,
    chord_source="source.wav",
    detect_pitch=True,
    num_voices=4
)
```

## Output Files
The script creates:
- Individual stems for each chord note (e.g., `stem_06000.wav`)
- Mixed chord (`mixed_chord.wav`)
- Metadata file (`metadata.json`) containing processing details

## Configuration
Edit `config.json` to set default values:
```json
{
  "audio": {
    "sr": 44100
  },
  "paths": {
    "chord_output_dir": "./output"
  },
  "chord_builder": {
    "normalize_output": true,
    "generate_metadata": true
  }
}
```

## Features
- High-precision pitch shifting using RubberBand with formant preservation
- Automatic pitch detection using YIN algorithm
- Supports microtonal chord generation
- Automatic chord extraction from source audio
- Time-stretching to match target durations
- 24-bit WAV output for high quality
- Comprehensive metadata generation

## Troubleshooting
- Ensure input audio is mono
- Use high-quality source material for best results
- If pitch detection fails, manually specify --original_note
- For automatic chord extraction, ensure the chord source audio is clear and well-defined
- Check the metadata.json file for processing details if issues occur
