# Chord Builder Usage Guide

## Requirements
- Python 3.8+
- Required packages: `librosa`, `numpy`, `soundfile`, `pyrubberband`
- Install dependencies: `pip install -r requirements.txt`

## Basic Usage
```bash
python src/chord_builder.py input.wav 6000 6500 6700 --detect-pitch
```

## Arguments
- `input.wav`: Path to input audio file (WAV, AIFF, or FLAC)
- `6000 6500 6700`: List of chord notes in OpenMusic format (100 = 1 semitone)
- `--detect-pitch`: Auto-detect the original pitch of the input audio
- `--original_note`: Manually specify the original note (e.g., 6000)
- `--output_dir`: Output directory (default: configured in config.json)
- `--sr`: Sample rate (default: 44100)

## Example 1: Auto-detect pitch
```bash
python src/chord_builder.py piano.wav 6000 6500 6700 --detect-pitch
```

## Example 2: Specify original note
```bash
python src/chord_builder.py violin.wav 6000 6500 6700 --original_note 6000
```

## Example 3: Custom output directory
```bash
python src/chord_builder.py guitar.wav 6000 6500 6700 --detect-pitch --output_dir ./output
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

## Troubleshooting
- Ensure input audio is mono
- Use high-quality source material for best results
- If pitch detection fails, manually specify --original_note
