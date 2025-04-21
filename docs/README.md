# 🎹 Piano + Electronics Audio Processing System

A collection of Python scripts for audio processing, generation, and manipulation, tailored for contemporary classical composition workflows involving piano and electronics. Provides tools for microtonal chord generation, TTS measure counting, sine tone synthesis, MIDI generation, and audio file management.
```



## Configuration (`config.json`)

This file centralizes settings used by various scripts. Create `config.json` in the project root if it doesn't exist.

```json
{
  "audio": {
    "sr": 44100,         // Default sample rate for processing (e.g., 44100, 48000, 96000)
    "tts_sr": 24000      // Sample rate for TTS output
  },
  "paths": {
    "input_dir": "audio/Audio_Raw_Materials", // Default input for some scripts
    "output_tone_file": "audio/Sine_Tones",   // Default output for sine tones
    "measure_audio_files": "audio/Measure_TTS", // Default output for measure TTS
    "chord_output_dir": "output/chords",      // Base output for chord_builder
    "chord_input_dir": "audio/Chord_Sources"  // Default input for chord_builder source extraction
  },
  "sine_tone_generator": {
    "duration": 5.0,
    "amplitude_jitter_amount": 0.1, // 0.0 to 1.0
    "jitter_frequency": 5.0         // Hz
  },
  "chord_builder": {
    "default_bit_depth": 24,       // e.g., 16, 24, 32
    "normalize_output": true,      // Normalize generated stems/mixes
    "generate_metadata": true      // Create metadata.json for each run
  }
  // Add sections for other configurable scripts as needed
}
```

The `src/config_utils.py` script provides functions to load and validate this configuration.

## Core Tools (`src/`)

### `chord_builder.py`
Generates microtonal chords by pitch-shifting a single input audio note.
- Can generate a single chord based on `--chord-notes` (list of target OpenMusic stem numbers).
- Can extract a sequence of chords (notes + durations) from a `--chord-source` audio file using `chord_extract.py` logic, then generate stems/mixes for each segment.
- Uses high-quality formant-preserving pitch shifting (RubberBand).
- Features parallel processing, caching, optional pitch detection (`--detect-pitch`), and metadata generation.
- Output is saved in timestamped subdirectories within the path specified by `config.json` (`paths.chord_output_dir`).

**Usage (Single Chord):**
```bash
python src/chord_builder.py <input_audio> --original-note <stem> --chord-notes <stem1> <stem2> ... [options]
```
**Usage (Chord Sequence Extraction):**
```bash
python src/chord_builder.py <input_audio> --original-note <stem> --chord-source <source_audio> --extract-chords <count> [--num-voices <N>] [options]
```
*Common Options:*
  * `--sr <hz>`: Processing sample rate (default from config).
  * `--detect-pitch`: Auto-detect original note from `<input_audio>` instead of using `--original-note`.
  * `--output-dir <path>`: Override base output directory from config.

*Example (Single Chord):* Generate a C major chord (stems 6000, 6400, 6700) from `input.wav` (assuming it's Middle C, stem 6000):
```bash
python src/chord_builder.py audio/input.wav --original-note 6000 --chord-notes 6000 6400 6700
```
*Example (Sequence):* Extract 5 4-voice chords from `source.wav` and build them using `input.wav` (auto-detect pitch):
```bash
python src/chord_builder.py audio/input.wav --detect-pitch --chord-source audio/Chord_Sources/source.wav --extract-chords 5 --num-voices 4
```

### `chord_extract.py`
Extracts sequences of microtonal chords from audio sources. Used internally by `chord_builder.py` but can potentially be used standalone.
- Detects pitches with confidence scoring.
- Returns `ChordSegment` objects containing notes (stems) and durations.

**Usage (if standalone):**
*(Requires understanding its library functions or if it has a direct CLI)*
```bash
# Example (Conceptual - verify actual usage if needed)
python src/chord_extract.py --source <input_audio> --num-chords <count> --num-voices <N> --output <output_json>
```

### `measure_tts_generator.py`
Generates spoken measure numbers using Text-to-Speech (Kokoro TTS).
- Applies special formatting for numbers > 99 for clarity (e.g., 101 -> "1 O 1", 110 -> "1 ten").
- Time-stretches audio to fit within a beat duration defined by BPM.
- Output directory configurable via `config.json` (`paths.measure_audio_files`). Saves as `measure_XXX.wav`.

**Usage:**
```bash
python src/measure_tts_generator.py --num_measures <count> --starting_measure <start_num> --bpm <tempo>
```
*Example:* Generate audio for measures 141 to 150 at 120 BPM:
```bash
python src/measure_tts_generator.py --num_measures 10 --starting_measure 141 --bpm 120
```

### `sine_tone_generator.py`
Generates sine wave tones from frequency inputs.
- Accepts input as OpenMusic stem numbers (e.g., `6000`) or standard pitch notation (e.g., `A4`, `C#5`).
- Supports configurable duration, sample rate, amplitude jitter, and output directory (via args or `config.json`). Saves as `tone_<input_value>.wav`.

**Usage:**
```bash
python src/sine_tone_generator.py <input_values...> [options]
```
*Options:*
  * `--duration <sec>`: Tone duration (default from config).
  * `--sample-rate <hz>`: Sample rate (default from config).
  * `--jitter-amount <0.0-1.0>`: Amplitude jitter (default from config).
  * `--jitter-frequency <hz>`: Jitter frequency (default from config).
  * `--output-dir <path>`: Output directory (default from config `paths.output_tone_file`).
*Example:* Generate 3-second tones for Middle C (stem 6000) and A4:
```bash
python src/sine_tone_generator.py 6000 A4 --duration 3.0
```

### `spray_notes.py`
Generates probabilistic MIDI note sequences ("sprays").
*(Further details on input/parameters might be needed for full documentation)*

**Usage (based on previous README):**
```bash
python src/spray_notes.py [--duration <sec>] [--density <notes_per_sec>] [other_options]
```
*Example:* Generate a 300-second sequence with an average density of 4 notes per second:
```bash
python src/spray_notes.py --duration 300 --density 4
```


