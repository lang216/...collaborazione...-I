import argparse
import os
import numpy as np
import librosa
import soundfile as sf
import pyrubberband as rb
from datetime import datetime
from config_utils import load_config
import json
from pathlib import Path
import sys
from chord_extract import extract_microtonal_sequence, ChordSegment

def detect_original_note(audio_path: str, sr: int) -> int:
    """Detect the original note using improved pitch detection."""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Optimized parameters for musical pitch detection
    f0 = librosa.yin(y, fmin=27.5, fmax=4186,  # A0 to C8 piano range
                    sr=sr, frame_length=4096)
    f0 = f0[f0 > 0]
    if len(f0) == 0:
        raise ValueError("No pitch detected in the audio.")
    
    # Use median for more robust frequency estimation
    median_f0 = np.median(f0)
    midi_note = librosa.hz_to_midi(median_f0)
    cents = (midi_note - int(midi_note)) * 100
    return int(round(int(midi_note) * 100 + cents))

def write_metadata(args, stem_paths: list[str], output_dir: str, duration: float = None) -> None:
    """Generate comprehensive metadata file."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "input_file": os.path.abspath(args.input_audio),
        "original_note": args.original_note,
        "chord_notes": args.chord_notes,
        "sample_rate": args.sr,
        "stems": [os.path.abspath(p) for p in stem_paths],
        "processing_chain": {
            "pitch_detection": "librosa YIN" if args.detect_pitch else "manual",
            "pitch_shifting": "RubberBand with formant preservation"
        }
    }
    if duration is not None:
        metadata["duration"] = duration
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def validate_chord_notes(notes: list[int]) -> None:
    """Validate OpenMusic format with correct MIDI range."""
    for note in notes:
        if note < 0 or note > 12799:  # Correct upper bound (MIDI 127 + 99 cents)
            raise ValueError(f"Invalid chord note: {note} (0-12799 allowed)")

def precise_rubberband_shift(y: np.ndarray, sr: int, shift_cents: float) -> np.ndarray:
    """High-precision pitch shifting with optimized RubberBand settings."""
    return rb.pyrb.pitch_shift(
        y=y, sr=sr,
        n_steps=shift_cents / 100,
        rbargs={
            "--formant":"",  # Preserve spectral envelope
            "--fine":"",     # High-precision mode
            "-c":"0", # Higher frequency resolution
            "--threads":""  # 
        }
    )

def extract_and_build_chords(y, native_sr, args, config):
    """
    Extract chords using chord_extract and build stems.
    """
    num_chords_to_extract = args.extract_chords
    num_voices = args.num_voices if hasattr(args, 'num_voices') and args.num_voices else 4
    chord_segments = extract_microtonal_sequence(args.chord_source, num_chords_to_extract, num_voices=num_voices)
    
    all_stems = []
    for i, segment in enumerate(chord_segments):
        output_dir = create_output_dir(args.output_dir, base_dir_name=f"chord_sequence_{i+1}")
        args.chord_notes = segment.notes
        stems = process_chord_notes(y, native_sr, args, output_dir, target_duration=segment.duration)
        if stems:
            mix_and_save(stems, y, native_sr, output_dir, config, target_duration=segment.duration)
            if config['chord_builder']['generate_metadata']:
                write_metadata(args, stems, output_dir, duration=segment.duration)
        all_stems.extend(stems)
        
    return chord_segments

def process_chord_notes(y: np.ndarray, native_sr: int, args, output_dir: str, target_duration: float = None) -> list:
    """
    Process all chord notes and return stem paths.
    If target_duration is provided, time-stretch the output to match that duration.
    """
    stems = []
    original_midi = args.original_note / 100.0
    
    # Time stretch the input audio if target_duration is provided and if difference is significant
    if target_duration is not None:
        current_duration = len(y) / native_sr
        if abs(current_duration - target_duration) > 0.01:  # More reasonable tolerance (10ms)
            time_scale = target_duration / current_duration
            if abs(time_scale - 1.0) > 0.01:  # Only stretch if change is significant
                y = rb.pyrb.time_stretch(
                    y, native_sr,
                    rate=time_scale,
                    rbargs={
                        "--formant": "",
                        "--fine": "",
                        "-c": "0",
                        "--threads": ""
                    }
                )
    
    for note in args.chord_notes:
        target_midi = note / 100.0
        shift_cents = (target_midi - original_midi) * 100
        
        try:
            y_shifted = precise_rubberband_shift(y, native_sr, shift_cents)
            config = load_config()
            if config['chord_builder']['normalize_output']:
                y_shifted = librosa.util.normalize(y_shifted)
                
            stem_path = save_stem(y_shifted, native_sr, note, output_dir)
            stems.append(stem_path)
            
        except Exception as e:
            print(f"Failed to process {note}: {str(e)}")
    
    return stems

def save_stem(y: np.ndarray, native_sr: int, note: int, output_dir: str) -> str:
    """Save individual stem with proper formatting"""
    stem_path = os.path.join(output_dir, f'stem_{note:05d}.wav')
    sf.write(stem_path, y, native_sr, subtype='PCM_24')  # Higher quality 24-bit
    print(f"Saved stem: {stem_path}")
    return stem_path

def mix_and_save(stems: list, original: np.ndarray, native_sr: int, 
                output_dir: str, config: dict, target_duration: float = None) -> None:
    """Mix stems with targeted duration and gain staging"""
    # Initialize with target length if provided, otherwise use original length
    if target_duration is not None:
        target_samples = int(target_duration * native_sr)
        max_length = target_samples
    else:
        max_length = len(original)
    
    mixed = np.zeros(max_length)
    
    for stem_path in stems:
        # Load stem
        y, _ = librosa.load(stem_path, sr=native_sr)
        
        # Time stretch to match target duration if needed and if difference is significant
        if target_duration is not None:
            current_duration = len(y) / native_sr
            if abs(current_duration - target_duration) > 0.01:  # More reasonable tolerance (10ms)
                time_scale = target_duration / current_duration
                if abs(time_scale - 1.0) > 0.01:  # Only stretch if change is significant
                    y = rb.pyrb.time_stretch(
                        y, native_sr,
                        rate=time_scale,
                        rbargs={
                            "--formant": "",
                            "--fine": "",
                            "-c": "0",
                            "--threads": ""
                        }
                    )
        
        # Ensure stem matches mixed buffer length
        y_padded = librosa.util.fix_length(y, size=len(mixed))
        mixed += y_padded * 0.8  # Headroom for summation
    
    if config['chord_builder']['normalize_output']:
        mixed = librosa.util.normalize(mixed)
    
    mixed_path = os.path.join(output_dir, 'mixed_chord.wav')
    sf.write(mixed_path, mixed, native_sr, subtype='PCM_24')
    print(f"Saved mix: {mixed_path}")

def create_output_dir(base_path: str, base_dir_name="chord") -> str:
    """Create unique output directory with a base directory name"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_path, f"{base_dir_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    return output_path

def run_chord_builder(
    input_audio: str,
    chord_notes: list[int] = None,
    original_note: int = None,
    detect_pitch: bool = False,
    output_dir: str = None,
    sr: int = 44100,  
    extract_chords: int = None,
    chord_source: str = "audio/Audio_Chord_Materials/sources/output.wav",
    num_voices: int = 4
) -> None:
    # In the command args construction:
    if sr != 44100:  # Only add if not default
        cmd_args.extend(['--sr', str(sr)])

    """
    Run chord builder programmatically.

    Args:
        input_audio: Path to input audio file
        chord_notes: List of chord notes in OpenMusic format (e.g. [6000, 6400]) - required if extract_chords is None
        original_note: Original note in OpenMusic format (optional if detect_pitch=True)
        detect_pitch: Whether to auto-detect the pitch
        output_dir: Custom output directory (optional)
        sr: Sample rate (44100, 48000, or 96000)
        extract_chords: Number of chords to extract automatically (optional, if provided, chord_notes is ignored)
        chord_source: Path to audio file to extract chords from (optional, defaults to "audio/Audio_Chord_Materials/sources/output.wav")
        num_voices: Number of voices to extract for automatic chord extraction (optional, defaults to 4)
    """
    # Convert args to command line format
    cmd_args = [sys.argv[0]]  # Script name
    cmd_args.append(str(Path(input_audio).absolute()))

    if extract_chords is None:
        if chord_notes is None:
            raise ValueError("chord_notes must be provided if extract_chords is None")
        cmd_args.extend([str(note) for note in chord_notes])
    else:
        cmd_args.extend(['--extract-chords', str(extract_chords)])
        if chord_source:
            cmd_args.extend(['--chord-source', str(Path(chord_source).absolute())])
        if num_voices != 4: # Only add if it's not the default value
            cmd_args.extend(['--num-voices', str(num_voices)])

    if original_note is not None:
        cmd_args.extend(['--original_note', str(original_note)])
    if detect_pitch:
        cmd_args.append('--detect-pitch')
    if output_dir:
        cmd_args.extend(['--output_dir', str(Path(output_dir).absolute())])

    # Temporarily replace sys.argv
    old_argv = sys.argv
    sys.argv = cmd_args

    try:
        main()
    finally:
        # Restore original sys.argv
        sys.argv = old_argv

def main():
    config = load_config()

    # Configure argument parser with safer defaults
    parser = argparse.ArgumentParser(
        description="Microtonal Chord Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('input_audio', type=str,
                      help="Input audio file path")
    parser.add_argument('--original_note', type=int,
                      help="Original note in OpenMusic format")
    parser.add_argument('--detect-pitch', action='store_true',
                      help="Auto-detect original pitch")
    parser.add_argument('--output_dir', type=str,
                      default=config['paths']['chord_output_dir'],
                      help="Output directory root")
    parser.add_argument('--chord-source', type=str,
                        help="Audio file to extract chords from", default="audio/Audio_Chord_Materials/sources/output.wav")
    parser.add_argument('chord_notes', type=int, nargs='*', help="OpenMusic chord notes (required unless --extract-chords is used)")
    parser.add_argument('--extract-chords', type=int, help="Number of chords to extract automatically")
    parser.add_argument('--num-voices', type=int, default=4, help="Number of voices to extract for automatic chord extraction")
    parser.add_argument('--sr', type=int, choices=[44100, 48000, 96000],
                        default=44100, help="Target sample rate")
    
    # Add mutual exclusivity check in code
    args = parser.parse_args()
    if not args.extract_chords and not args.chord_notes:
        parser.error("Either provide chord_notes or use --extract-chords")

    # Validation pipeline
    try:
        if not os.path.isfile(args.input_audio):
            raise FileNotFoundError(f"Audio file not found: {args.input_audio}")

        if args.detect_pitch:
            args.original_note = detect_original_note(args.input_audio, 44100)
            print(f"Detected pitch: {args.original_note}")
        elif not args.original_note:
            raise ValueError("Must specify --original_note or --detect-pitch")

        validate_chord_notes([args.original_note] + args.chord_notes)

    except Exception as e:
        print(f"Validation error: {str(e)}")
        return 1

    # Audio processing
    try:
        # Load audio with native sample rate
        y, native_sr = librosa.load(args.input_audio, sr=None, mono=True)
        output_dir = create_output_dir(args.output_dir)

        if args.extract_chords:
            chord_segments = extract_and_build_chords(y, native_sr, args, config)
        else: # process_chord_notes
            stems = process_chord_notes(y, native_sr, args, output_dir)
            if stems:
                mix_and_save(stems, y, native_sr, output_dir, config)
                if config['chord_builder']['generate_metadata']:
                    write_metadata(args, stems, output_dir)

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return 1

# Example usage for automatic chord extraction
input_file_extract_chords = "audio/Audio_Chord_Materials/components/piano_3.wav"
run_chord_builder(
    input_audio=input_file_extract_chords,
    extract_chords=10,
    chord_source="audio/Audio_Chord_Materials/sources/M4.wav",
    detect_pitch=True,
    num_voices=4
)

# if __name__ == "__main__":
#     main()
