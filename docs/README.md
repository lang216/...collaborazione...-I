# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary sounds

## Key Features

- **Audio Segmentation**:
  - Automatic audio segmentation using advanced onset detection algorithms with Audioflux library.
  - Smart fade handling for clean audio segment transitions using linear fades.
  - Memory-efficient processing with joblib caching for intermediate results.
  - Parallel processing capabilities for faster segmentation of multiple audio files.

- **Feature Extraction**:
  - Comprehensive audio feature analysis including MFCCs, Spectral Centroid, Spectral Flatness, and RMS Energy.
  - Utilizes Librosa for robust feature extraction.
  - Real-time feature visualization (planned feature, not currently implemented).
  - Parallel feature extraction using joblib to speed up processing.

- **Clustering**:
  - Hierarchical clustering of audio segments based on feature similarity using Agglomerative Clustering.
  - Automatic cluster validation to ensure meaningful groupings.
  - Minimum file count enforcement per cluster to filter out noise clusters.
  - Duration-based filtering to focus on segments within specified time ranges.

- **MIDI Generation (Spray Notes)**:
  - Probabilistic MIDI note generation for creating ambient textures and soundscapes.
  - Configurable frequency ranges to constrain note pitches to desired musical ranges.
  - Adjustable note density and duration to control the sparsity and rhythmic feel of the generated MIDI.
  - Velocity randomization for more natural and expressive MIDI outputs.
  - Automatic MIDI file creation in specified output directory.

- **Chord Processing**:
  - Automatic chord extraction from source audio using a microtonal approach.
  - High-precision pitch shifting with RubberBand library, preserving formants for natural sound.
  - Supports processing of up to 4 voices by default for complex chord structures.
  - Time-stretching capabilities to adjust chord durations without affecting pitch.
  - Generation of comprehensive metadata for each processed chord, detailing parameters and processing steps.

- **Sine Wave Generation**:
  - Highly customizable sine tone generation for creating synthetic audio elements.
  - Amplitude jitter modulation to add subtle variations and warmth to sine tones.
  - Configurable frequency, duration, and sample rate for precise tone control.
  - Smart normalization to prevent audio clipping and ensure consistent loudness.
  - Amplitude modulation capabilities for creating tremolo and other effects.

- **Musical Notation Tools**:
  - OpenMusic note number to musical note conversion for integration with visual music environments.
  - Supports conversion to both sharp and flat notations based on user preference.
  - Batch processing of note numbers for efficient conversion of large datasets.
  - Command-line interface for quick, scriptable musical notation conversions.

- **Freesound Integration**:
  - Automated search for complementary sounds on Freesound based on extracted audio features.
  - Duration-based filtering to find sounds matching the length of audio segments.
  - Feature-based matching using MFCCs, spectral centroid, flatness, and RMS energy to find perceptually similar sounds.
  - Download and organization of Freesound samples into project directories.

- **Performance Optimization**:
  - Memory caching using joblib.Memory to cache intermediate audio processing results and speed up repeated operations.
  - Multi-core processing using joblib.Parallel and ThreadPoolExecutor to leverage multiple CPU cores for parallel tasks.
  - Efficient audio loading and processing with Librosa and Soundfile libraries.
  - Progress tracking with tqdm progress bars for monitoring long-running operations.

## System Workflow

1. **Configuration Loading**:
   - Load system configuration from `config.json` using `config_utils.py`.
   - Validate configuration parameters to ensure correct setup.

2. **Audio Input and Segmentation**:
   - Load raw piano audio recordings from the input directory specified in config.
   - Perform automatic audio segmentation using onset detection with Audioflux.
   - Apply fade-in/fade-out to segments for smooth transitions.
   - Cache intermediate segmentation results for efficiency.

3. **Feature Extraction and Clustering**:
   - Extract audio features (MFCC, Spectral Centroid, Spectral Flatness, RMS) from each segment in parallel.
   - Cluster audio segments based on feature similarity using hierarchical clustering.
   - Organize segments into clusters based on different feature types.

4. **Filtered Segment Processing**:
   - Filter out short audio chunks based on minimum and maximum duration thresholds defined in config.
   - Save filtered audio segments into designated directories, maintaining cluster structure.

5. **Freesound Integration and Sound Matching**:
   - Search for complementary sounds on Freesound for each filtered audio chunk.
   - Use extracted audio features to guide Freesound searches for perceptually similar sounds.
   - Download and organize matched Freesound samples into output directories, categorized by feature type and cluster.

6. **Output Generation**:
   - Save segmented audio chunks organized by feature type and cluster labels.
   - Generate MIDI files using probabilistic note generation (Spray Notes).
   - Process and output chord notes and mixes from chord processing scripts.
   - Output matched Freesound samples downloaded from Freesound API.
   - Generate sine wave tones based on specified parameters.
   - Provide musical notation conversion utilities for OpenMusic note numbers.

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
  },
  "sine_tone_generator": {
    "frequency": 440.0,
    "duration": 5.0,
    "amplitude_jitter_amount": 0.0,
    "jitter_frequency": 5.0,
    "output_filename": "generated_tone.wav"
  }
}
```

## Usage

### Main Processing Pipeline (`src/main.py`)
```bash
python src/main.py
```
- **Description**: Executes the main audio processing pipeline. Loads configuration, segments audio files from the input directory, extracts features, performs clustering, filters segments by duration, and searches Freesound for similar sounds.
- **Arguments**: Takes no command-line arguments. All parameters are configured in `config.json`.

### MIDI Note Generation (`src/spray_notes.py`)
```bash
python src/spray_notes.py
```
- **Description**: Generates a MIDI file containing probabilistic, randomly generated notes. Useful for creating ambient textures or background音響.
- **Arguments**: No command-line arguments. Configuration parameters such as note density, duration, frequency range, and velocity are set in the `spray_notes` section of `config.json`.

### Chord Generation and Extraction (`src/chord_builder.py`)

#### Manual Chord Building
```bash
python src/chord_builder.py <input_audio_path> <note1> <note2> <note3> ... --detect-pitch --output-dir <output_directory>
```
- **Description**: Manually builds microtonal chords from a single input audio file by pitch-shifting it to specified notes.
- **Arguments**:
    - `<input_audio_path>`: Path to the input audio file (e.g., `audio/Audio_Raw_Materials/input.wav`).
    - `<note1> <note2> <note3> ...`: List of target notes in OpenMusic note format (e.g., `6000 6500 7000`).
    - `--detect-pitch`: (Optional) Automatically detect the original pitch of the input audio.
    - `--original-note <stem_value>`: (Optional, required if `--detect-pitch` is not used) Specify the original pitch in OpenMusic note format.
    - `--output-dir <output_directory>`: (Optional) Specify the output directory for generated chord files. Defaults to `output/chord_[timestamp]`.

#### Automatic Chord Sequence Extraction
```bash
python src/chord_builder.py --chord-source <chord_source_audio> --extract-chords <num_chords> --output-dir <output_directory>
```
- **Description**: Automatically extracts a sequence of microtonal chords from a source audio file. Segments the source audio and extracts chord voicings from each segment.
- **Arguments**:
    - `--chord-source <chord_source_audio>`: Path to the source audio file to extract chords from (e.g., `audio/Audio_Chord_Materials/source.wav`).
    - `--extract-chords <num_chords>`: Number of chords to extract from the source audio.
    - `--num-voices <num_voices>`: (Optional) Number of voices to extract per chord (default: 4).
    - `--output-dir <output_directory>`: (Optional) Specify the output directory for generated chord sequence files. Defaults to `output/chord_sequence_[timestamp]`.

See [Chord Builder Documentation](docs/chord_builder_README.md) for detailed usage and advanced options.

### Sine Tone Generation (`src/sine_tone_generator.py`)
```bash
python src/sine_tone_generator.py --frequency <frequency_hz> --duration <duration_sec> --amplitude-jitter <jitter_amount> --output-filename <filename.wav>
```
- **Description**: Generates a sine wave tone with customizable parameters, including frequency, duration, and amplitude jitter.
- **Arguments**:
    - `--frequency <frequency_hz>`: Frequency of the sine tone in Hz (default: 440.0 Hz, A4).
    - `--duration <duration_sec>`: Duration of the sine tone in seconds (default: 5.0 seconds).
    - `--amplitude-jitter <jitter_amount>`: Amount of amplitude jitter (random variation) to add, between 0.0 (none) and 1.0 (full random amplitude) (default: 0.0).
    - `--jitter-frequency <jitter_frequency_hz>`: Frequency of amplitude jitter modulation in Hz (default: 5.0 Hz).
    - `--output-filename <filename.wav>`: Output filename for the generated sine tone WAV file (default: `generated_tone.wav`).

### Musical Notation Conversion (`utils/pitch_converter.py`)
```bash
python utils/pitch_converter.py
```
- **Description**: Interactive command-line tool for converting OpenMusic note numbers to musical note notation. Prompts for note numbers and displays corresponding musical notes.
- **Arguments**: No command-line arguments. Runs in interactive mode.

### Frequency Conversion (`utils/frequency_converter.py`)
```bash
python utils/frequency_converter.py
```
- **Description**: Interactive command-line tool for converting between musical pitch notations/stem numbers and frequencies.
- **Arguments**: No command-line arguments. Runs in interactive mode, prompting for input values.

## Utility Scripts

The `utils` directory contains several utility scripts for audio file manipulation, format conversion, and data processing.

### Audio Converter (`utils/audio_converter.py`)
```bash
python utils/audio_converter.py <input_directory> <output_directory>
```
- **Description**: Converts audio files in a directory (and subdirectories) to WAV format. Supports various input formats like MP3, M4A, FLAC, etc.
- **Arguments**:
    - `<input_directory>`: Path to the input directory containing audio files to convert.
    - `<output_directory>`: Path to the output directory where converted WAV files will be saved.
    - `--preserve-original`: (Optional) If specified, preserves the original files instead of replacing them (currently not implemented in script).
    - `--log <log_file>`: (Optional) Path to the log file for logging script operations (default: `utils.log`).

### Audio Renamer (`utils/audio_renamer.py`)
```bash
python utils/audio_renamer.py <source_directory> <target_directory>
```
- **Description**: Renames audio files in a directory by detecting their actual file type and adding the correct extension. Useful for fixing files with incorrect or missing extensions.
- **Arguments**:
    - `<source_directory>`: Path to the source directory containing audio files to rename.
    - `<target_directory>`: Path to the target directory where renamed files will be saved, maintaining the original directory structure.
    - `--log <log_file>`: (Optional) Path to the log file for logging script operations (default: `utils.log`).

### Duplicate Remover (`utils/duplicate_remover.py`)
```bash
python utils/duplicate_remover.py <directory_to_scan>
```
- **Description**: Finds and removes duplicate audio files within a directory. Identifies duplicates based on MD5 hash and copies only unique files to a new directory.
- **Arguments**:
    - `<directory_to_scan>`: Path to the directory to scan for duplicate audio files.
    - `--log <log_file>`: (Optional) Path to the log file for logging script operations (default: `utils.log`).

### Frequency Converter (`utils/frequency_converter.py`)
```bash
python utils/frequency_converter.py
```
- **Description**: Interactive command-line tool for converting between musical pitch notations/stem numbers and frequencies.
- **Arguments**: No command-line arguments. Runs in interactive mode, prompting for input values.

### Pitch Converter (`utils/pitch_converter.py`)
```bash
python utils/pitch_converter.py
```
- **Description**: Interactive command-line tool for converting OpenMusic note numbers to musical note notation (e.g., stem numbers to "C4", "F#5").
- **Arguments**: No command-line arguments. Runs in interactive mode, prompting for input values.

### Shared Utilities (`utils/shared_utils.py`)
- **Description**: Contains shared utility functions and classes used by other scripts, such as logging setup, directory validation, and argument parsing. Not intended to be run directly.
- **Arguments**: N/A
# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary sounds

## Key Features

- **Audio Segmentation**:
  - Automatic audio segmentation using advanced onset detection algorithms with Audioflux library.
  - Smart fade handling for clean audio segment transitions using linear fades.
  - Memory-efficient processing with joblib caching for intermediate results.
  - Parallel processing capabilities for faster segmentation of multiple audio files.

- **Feature Extraction**:
  - Comprehensive audio feature analysis including MFCCs, Spectral Centroid, Spectral Flatness, and RMS Energy.
  - Utilizes Librosa for robust feature extraction.
  - Real-time feature visualization (planned feature, not currently implemented).
  - Parallel feature extraction using joblib to speed up processing.

- **Clustering**:
  - Hierarchical clustering of audio segments based on feature similarity using Agglomerative Clustering.
  - Automatic cluster validation to ensure meaningful groupings.
  - Minimum file count enforcement per cluster to filter out noise clusters.
  - Duration-based filtering to focus on segments within specified time ranges.

- **MIDI Generation (Spray Notes)**:
  - Probabilistic MIDI note generation for creating ambient textures and soundscapes.
  - Configurable frequency ranges to constrain note pitches to desired musical ranges.
  - Adjustable note density and duration to control the sparsity and rhythmic feel of the generated MIDI.
  - Velocity randomization for more natural and expressive MIDI outputs.
  - Automatic MIDI file creation in specified output directory.

- **Chord Processing**:
  - Automatic chord extraction from source audio using a microtonal approach.
  - High-precision pitch shifting with RubberBand library, preserving formants for natural sound.
  - Supports processing of up to 4 voices by default for complex chord structures.
  - Time-stretching capabilities to adjust chord durations without affecting pitch.
  - Generation of comprehensive metadata for each processed chord, detailing parameters and processing steps.

- **Sine Wave Generation**:
  - Highly customizable sine tone generation for creating synthetic audio elements.
  - Amplitude jitter modulation to add subtle variations and warmth to sine tones.
  - Configurable frequency, duration, and sample rate for precise tone control.
  - Smart normalization to prevent audio clipping and ensure consistent loudness.
  - Amplitude modulation capabilities for creating tremolo and other effects.

- **Musical Notation Tools**:
  - OpenMusic note number to musical note conversion for integration with visual music environments.
  - Supports conversion to both sharp and flat notations based on user preference.
  - Batch processing of note numbers for efficient conversion of large datasets.
  - Command-line interface for quick, scriptable musical notation conversions.

- **Freesound Integration**:
  - Automated search for complementary sounds on Freesound based on extracted audio features.
  - Duration-based filtering to find sounds matching the length of audio segments.
  - Feature-based matching using MFCCs, spectral centroid, flatness, and RMS energy to find perceptually similar sounds.
  - Download and organization of Freesound samples into project directories.

- **Performance Optimization**:
  - Memory caching using joblib.Memory to cache intermediate audio processing results and speed up repeated operations.
  - Multi-core processing using joblib.Parallel and ThreadPoolExecutor to leverage multiple CPU cores for parallel tasks.
  - Efficient audio loading and processing with Librosa and Soundfile libraries.
  - Progress tracking with tqdm progress bars for monitoring long-running operations.

## System Workflow

1. **Configuration Loading**:
   - Load system configuration from `config.json` using `config_utils.py`.
   - Validate configuration parameters to ensure correct setup.

2. **Audio Input and Segmentation**:
   - Load raw piano audio recordings from the input directory specified in config.
   - Perform automatic audio segmentation using onset detection with Audioflux.
   - Apply fade-in/fade-out to segments for smooth transitions.
   - Cache intermediate segmentation results for efficiency.

3. **Feature Extraction and Clustering**:
   - Extract audio features (MFCC, Spectral Centroid, Spectral Flatness, RMS) from each segment in parallel.
   - Cluster audio segments based on feature similarity using hierarchical clustering.
   - Organize segments into clusters based on different feature types.

4. **Filtered Segment Processing**:
   - Filter out short audio chunks based on minimum and maximum duration thresholds defined in config.
   - Save filtered audio segments into designated directories, maintaining cluster structure.

5. **Freesound Integration and Sound Matching**:
   - Search for complementary sounds on Freesound for each filtered audio chunk.
   - Use extracted audio features to guide Freesound searches for perceptually similar sounds.
   - Download and organize matched Freesound samples into output directories, categorized by feature type and cluster.

6. **Output Generation**:
   - Save segmented audio chunks organized by feature type and cluster labels.
   - Generate MIDI files using probabilistic note generation (Spray Notes).
   - Process and output chord notes and mixes from chord processing scripts.
   - Output matched Freesound samples downloaded from Freesound API.
   - Generate sine wave tones based on specified parameters.
   - Provide musical notation conversion utilities for OpenMusic note numbers.

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
  },
  "sine_tone_generator": {
    "frequency": 440.0,
    "duration": 5.0,
    "amplitude_jitter_amount": 0.0,
    "jitter_frequency": 5.0,
    "output_filename": "generated_tone.wav"
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

### Sine Tone Generation
```bash
python src/sine_tone_generator.py
```

Generates sine wave tones with optional amplitude jitter based on configuration.

### Musical Notation Conversion
```bash
python utils/pitch_converter.py
```

Interactive tool for converting OpenMusic note numbers to musical notation.

## Output Structure

```
audio/
├── Audio_Raw_Materials/         # Input piano recordings
├── Audio_Chord_Materials/       # Chord source and component files
├── Segmented_Audio/            # Initial segmentation results
└── Segmented_Audio_Filtered/   # Filtered segments after clustering

output/
└── chord_[timestamp]/         # Generated chord files
    ├── note_[note].wav        # Individual chord notes
    ├── mixed_chord.wav        # Mixed chord output
    └── metadata.json          # Processing details

midi_output/                    # Generated MIDI files
└── spray_notes_[timestamp].mid # Probabilistic note sequences

tests/
├── filtered_segments_dir_test/ # Test filtered segments
└── Freesound_Matches_Test/     # Freesound search results
```
```
audio/
├── Audio_Raw_Materials/         # Input piano recordings
├── Audio_Chord_Materials/       # Chord source and component files
├── Segmented_Audio/            # Initial segmentation results
└── Segmented_Audio_Filtered/   # Filtered segments after clustering

output/
└── chord_[timestamp]/         # Generated chord files
    ├── note_[note].wav        # Individual chord notes
    ├── mixed_chord.wav        # Mixed chord output
    └── metadata.json          # Processing details

midi_output/                    # Generated MIDI files
└── spray_notes_[timestamp].mid # Probabilistic note sequences

tests/
├── filtered_segments_dir_test/ # Test filtered segments
└── Freesound_Matches_Test/     # Freesound search results
```
# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary sounds

## Key Features

- **Audio Segmentation**:
  - Automatic audio segmentation using advanced onset detection algorithms with Audioflux library.
  - Smart fade handling for clean audio segment transitions using linear fades.
  - Memory-efficient processing with joblib caching for intermediate results.
  - Parallel processing capabilities for faster segmentation of multiple audio files.

- **Feature Extraction**:
  - Comprehensive audio feature analysis including MFCCs, Spectral Centroid, Spectral Flatness, and RMS Energy.
  - Utilizes Librosa for robust feature extraction.
  - Real-time feature visualization (planned feature, not currently implemented).
  - Parallel feature extraction using joblib to speed up processing.

- **Clustering**:
  - Hierarchical clustering of audio segments based on feature similarity using Agglomerative Clustering.
  - Automatic cluster validation to ensure meaningful groupings.
  - Minimum file count enforcement per cluster to filter out noise clusters.
  - Duration-based filtering to focus on segments within specified time ranges.

- **MIDI Generation (Spray Notes)**:
  - Probabilistic MIDI note generation for creating ambient textures and soundscapes.
  - Configurable frequency ranges to constrain note pitches to desired musical ranges.
  - Adjustable note density and duration to control the sparsity and rhythmic feel of the generated MIDI.
  - Velocity randomization for more natural and expressive MIDI outputs.
  - Automatic MIDI file creation in specified output directory.

- **Chord Processing**:
  - Automatic chord extraction from source audio using a microtonal approach.
  - High-precision pitch shifting with RubberBand library, preserving formants for natural sound.
  - Supports processing of up to 4 voices by default for complex chord structures.
  - Time-stretching capabilities to adjust chord durations without affecting pitch.
  - Generation of comprehensive metadata for each processed chord, detailing parameters and processing steps.

- **Sine Wave Generation**:
  - Highly customizable sine tone generation for creating synthetic audio elements.
  - Amplitude jitter modulation to add subtle variations and warmth to sine tones.
  - Configurable frequency, duration, and sample rate for precise tone control.
  - Smart normalization to prevent audio clipping and ensure consistent loudness.
  - Amplitude modulation capabilities for creating tremolo and other effects.

- **Musical Notation Tools**:
  - OpenMusic note number to musical note conversion for integration with visual music environments.
  - Supports conversion to both sharp and flat notations based on user preference.
  - Batch processing of note numbers for efficient conversion of large datasets.
  - Command-line interface for quick, scriptable musical notation conversions.

- **Freesound Integration**:
  - Automated search for complementary sounds on Freesound based on extracted audio features.
  - Duration-based filtering to find sounds matching the length of audio segments.
  - Feature-based matching using MFCCs, spectral centroid, flatness, and RMS energy to find perceptually similar sounds.
  - Download and organization of Freesound samples into project directories.

- **Performance Optimization**:
  - Memory caching using joblib.Memory to cache intermediate audio processing results and speed up repeated operations.
  - Multi-core processing using joblib.Parallel and ThreadPoolExecutor to leverage multiple CPU cores for parallel tasks.
  - Efficient audio loading and processing with Librosa and Soundfile libraries.
  - Progress tracking with tqdm progress bars for monitoring long-running operations.

## System Workflow

1. **Configuration Loading**:
   - Load system configuration from `config.json` using `config_utils.py`.
   - Validate configuration parameters to ensure correct setup.

2. **Audio Input and Segmentation**:
   - Load raw piano audio recordings from the input directory specified in config.
   - Perform automatic audio segmentation using onset detection with Audioflux.
   - Apply fade-in/fade-out to segments for smooth transitions.
   - Cache intermediate segmentation results for efficiency.

3. **Feature Extraction and Clustering**:
   - Extract audio features (MFCC, Spectral Centroid, Spectral Flatness, RMS) from each segment in parallel.
   - Cluster audio segments based on feature similarity using hierarchical clustering.
   - Organize segments into clusters based on different feature types.

4. **Filtered Segment Processing**:
   - Filter out short audio chunks based on minimum and maximum duration thresholds defined in config.
   - Save filtered audio segments into designated directories, maintaining cluster structure.

5. **Freesound Integration and Sound Matching**:
   - Search for complementary sounds on Freesound for each filtered audio chunk.
   - Use extracted audio features to guide Freesound searches for perceptually similar sounds.
   - Download and organize matched Freesound samples into output directories, categorized by feature type and cluster.

6. **Output Generation**:
   - Save segmented audio chunks organized by feature type and cluster labels.
   - Generate MIDI files using probabilistic note generation (Spray Notes).
   - Process and output chord notes and mixes from chord processing scripts.
   - Output matched Freesound samples downloaded from Freesound API.
   - Generate sine wave tones based on specified parameters.
   - Provide musical notation conversion utilities for OpenMusic note numbers.

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
  },
  "sine_tone_generator": {
    "frequency": 440.0,
    "duration": 5.0,
    "amplitude_jitter_amount": 0.0,
    "jitter_frequency": 5.0,
    "output_filename": "generated_tone.wav"
  }
}
```

## Usage

### Main Processing Pipeline (`src/main.py`)
```bash
python src/main.py
```
- **Description**: Executes the main audio processing pipeline. Loads configuration, segments audio files from the input directory, extracts features, performs clustering, filters segments by duration, and searches Freesound for similar sounds.
- **Arguments**: Takes no command-line arguments. All parameters are configured in `config.json`.

### MIDI Note Generation (`src/spray_notes.py`)
```bash
python src/spray_notes.py
```
- **Description**: Generates a MIDI file containing probabilistic, randomly generated notes. Useful for creating ambient textures or background音響.
- **Arguments**: No command-line arguments. Configuration parameters such as note density, duration, frequency range, and velocity are set in the `spray_notes` section of `config.json`.

### Chord Generation and Extraction (`src/chord_builder.py`)

#### Manual Chord Building
```bash
python src/chord_builder.py <input_audio_path> <note1> <note2> <note3> ... --detect-pitch --output-dir <output_directory>
```
- **Description**: Manually builds microtonal chords from a single input audio file by pitch-shifting it to specified notes.
- **Arguments**:
    - `<input_audio_path>`: Path to the input audio file (e.g., `audio/Audio_Raw_Materials/input.wav`).
    - `<note1> <note2> <note3> ...`: List of target notes in OpenMusic note format (e.g., `6000 6500 7000`).
    - `--detect-pitch`: (Optional) Automatically detect the original pitch of the input audio.
    - `--original-note <stem_value>`: (Optional, required if `--detect-pitch` is not used) Specify the original pitch in OpenMusic note format.
    - `--output-dir <output_directory>`: (Optional) Specify the output directory for generated chord files. Defaults to `output/chord_[timestamp]`.

#### Automatic Chord Sequence Extraction
```bash
python src/chord_builder.py --chord-source <chord_source_audio> --extract-chords <num_chords> --output-dir <output_directory>
```
- **Description**: Automatically extracts a sequence of microtonal chords from a source audio file. Segments the source audio and extracts chord voicings from each segment.
- **Arguments**:
    - `--chord-source <chord_source_audio>`: Path to the source audio file to extract chords from (e.g., `audio/Audio_Chord_Materials/source.wav`).
    - `--extract-chords <num_chords>`: Number of chords to extract from the source audio.
    - `--num-voices <num_voices>`: (Optional) Number of voices to extract per chord (default: 4).
    - `--output-dir <output_directory>`: (Optional) Specify the output directory for generated chord sequence files. Defaults to `output/chord_sequence_[timestamp]`.

See [Chord Builder Documentation](docs/chord_builder_README.md) for detailed usage and advanced options.

### Sine Tone Generation (`src/sine_tone_generator.py`)
```bash
python src/sine_tone_generator.py --frequency <frequency_hz> --duration <duration_sec> --amplitude-jitter <jitter_amount> --output-filename <filename.wav>
```
- **Description**: Generates a sine wave tone with customizable parameters, including frequency, duration, and amplitude jitter.
- **Arguments**:
    - `--frequency <frequency_hz>`: Frequency of the sine tone in Hz (default: 440.0 Hz, A4).
    - `--duration <duration_sec>`: Duration of the sine tone in seconds (default: 5.0 seconds).
    - `--amplitude-jitter <jitter_amount>`: Amount of amplitude jitter (random variation) to add, between 0.0 (none) and 1.0 (full random amplitude) (default: 0.0).
    - `--jitter-frequency <jitter_frequency_hz>`: Frequency of amplitude jitter modulation in Hz (default: 5.0 Hz).
    - `--output-filename <filename.wav>`: Output filename for the generated sine tone WAV file (default: `generated_tone.wav`).

### Musical Notation Conversion (`utils/pitch_converter.py`)
```bash
python utils/pitch_converter.py
```
- **Description**: Interactive command-line tool for converting OpenMusic note numbers to musical note notation. Prompts for note numbers and displays corresponding musical notes.
- **Arguments**: No command-line arguments. Runs in interactive mode.

### Frequency Conversion (`utils/frequency_converter.py`)
```bash
python utils/frequency_converter.py
```
- **Description**: Interactive command-line tool for converting between musical pitch notations/stem numbers and frequencies.
- **Arguments**: No command-line arguments. Runs in interactive mode, prompting for input values.
# 🎹 Piano + Electronics Audio Processing System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Audio segmentation and feature analysis system for processing piano recordings and finding complementary sounds

## Key Features

- **Audio Segmentation**:
  - Automatic audio segmentation using advanced onset detection algorithms with Audioflux library.
  - Smart fade handling for clean audio segment transitions using linear fades.
  - Memory-efficient processing with joblib caching for intermediate results.
  - Parallel processing capabilities for faster segmentation of multiple audio files.

- **Feature Extraction**:
  - Comprehensive audio feature analysis including MFCCs, Spectral Centroid, Spectral Flatness, and RMS Energy.
  - Utilizes Librosa for robust feature extraction.
  - Real-time feature visualization (planned feature, not currently implemented).
  - Parallel feature extraction using joblib to speed up processing.

- **Clustering**:
  - Hierarchical clustering of audio segments based on feature similarity using Agglomerative Clustering.
  - Automatic cluster validation to ensure meaningful groupings.
  - Minimum file count enforcement per cluster to filter out noise clusters.
  - Duration-based filtering to focus on segments within specified time ranges.

- **MIDI Generation (Spray Notes)**:
  - Probabilistic MIDI note generation for creating ambient textures and soundscapes.
  - Configurable frequency ranges to constrain note pitches to desired musical ranges.
  - Adjustable note density and duration to control the sparsity and rhythmic feel of the generated MIDI.
  - Velocity randomization for more natural and expressive MIDI outputs.
  - Automatic MIDI file creation in specified output directory.

- **Chord Processing**:
  - Automatic chord extraction from source audio using a microtonal approach.
  - High-precision pitch shifting with RubberBand library, preserving formants for natural sound.
  - Supports processing of up to 4 voices by default for complex chord structures.
  - Time-stretching capabilities to adjust chord durations without affecting pitch.
  - Generation of comprehensive metadata for each processed chord, detailing parameters and processing steps.

- **Sine Wave Generation**:
  - Highly customizable sine tone generation for creating synthetic audio elements.
  - Amplitude jitter modulation to add subtle variations and warmth to sine tones.
  - Configurable frequency, duration, and sample rate for precise tone control.
  - Smart normalization to prevent audio clipping and ensure consistent loudness.
  - Amplitude modulation capabilities for creating tremolo and other effects.

- **Musical Notation Tools**:
  - OpenMusic note number to musical note conversion for integration with visual music environments.
  - Supports conversion to both sharp and flat notations based on user preference.
  - Batch processing of note numbers for efficient conversion of large datasets.
  - Command-line interface for quick, scriptable musical notation conversions.

- **Freesound Integration**:
  - Automated search for complementary sounds on Freesound based on extracted audio features.
  - Duration-based filtering to find sounds matching the length of audio segments.
  - Feature-based matching using MFCCs, spectral centroid, flatness, and RMS energy to find perceptually similar sounds.
  - Download and organization of Freesound samples into project directories.

- **Performance Optimization**:
  - Memory caching using joblib.Memory to cache intermediate audio processing results and speed up repeated operations.
  - Multi-core processing using joblib.Parallel and ThreadPoolExecutor to leverage multiple CPU cores for parallel tasks.
  - Efficient audio loading and processing with Librosa and Soundfile libraries.
  - Progress tracking with tqdm progress bars for monitoring long-running operations.

## System Workflow

1. **Configuration Loading**:
   - Load system configuration from `config.json` using `config_utils.py`.
   - Validate configuration parameters to ensure correct setup.

2. **Audio Input and Segmentation**:
   - Load raw piano audio recordings from the input directory specified in config.
   - Perform automatic audio segmentation using onset detection with Audioflux.
   - Apply fade-in/fade-out to segments for smooth transitions.
   - Cache intermediate segmentation results for efficiency.

3. **Feature Extraction and Clustering**:
   - Extract audio features (MFCC, Spectral Centroid, Spectral Flatness, RMS) from each segment in parallel.
   - Cluster audio segments based on feature similarity using hierarchical clustering.
   - Organize segments into clusters based on different feature types.

4. **Filtered Segment Processing**:
   - Filter out short audio chunks based on minimum and maximum duration thresholds defined in config.
   - Save filtered audio segments into designated directories, maintaining cluster structure.

5. **Freesound Integration and Sound Matching**:
   - Search for complementary sounds on Freesound for each filtered audio chunk.
   - Use extracted audio features to guide Freesound searches for perceptually similar sounds.
   - Download and organize matched Freesound samples into output directories, categorized by feature type and cluster.

6. **Output Generation**:
   - Save segmented audio chunks organized by feature type and cluster labels.
   - Generate MIDI files using probabilistic note generation (Spray Notes).
   - Process and output chord notes and mixes from chord processing scripts.
   - Output matched Freesound samples downloaded from Freesound API.
   - Generate sine wave tones based on specified parameters.
   - Provide musical notation conversion utilities for OpenMusic note numbers.

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
  },
  "sine_tone_generator": {
    "frequency": 440.0,
    "duration": 5.0,
    "amplitude_jitter_amount": 0.0,
    "jitter_frequency": 5.0,
    "output_filename": "generated_tone.wav"
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

### Sine Tone Generation
```bash
python src/sine_tone_generator.py
```

Generates sine wave tones with optional amplitude jitter based on configuration.

### Musical Notation Conversion
```bash
python utils/pitch_converter.py
```

Interactive tool for converting OpenMusic note numbers to musical notation.

## Output Structure

```
audio/
├── Audio_Raw_Materials/         # Input piano recordings
├── Audio_Chord_Materials/       # Chord source and component files
├── Segmented_Audio/            # Initial segmentation results
└── Segmented_Audio_Filtered/   # Filtered segments after clustering

output/
└── chord_[timestamp]/         # Generated chord files
    ├── note_[note].wav        # Individual chord notes (e.g., note_6000.wav, note_6500.wav)
    ├── mixed_chord.wav        # Mixed chord output
    └── metadata.json          # Processing details

midi_output/                    # Generated MIDI files
└── spray_notes_[timestamp].mid # Probabilistic note sequences

tests/
├── filtered_segments_dir_test/ # Test filtered segments
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
