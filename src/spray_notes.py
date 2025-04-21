import pretty_midi
import random
import math
import os
import argparse  # Import argparse
from pathlib import Path
from config_utils import load_config, create_spray_notes_config, SprayNotesConfig
from typing import List, Dict, Any, Optional


def probabilistic_round(target: float) -> int:
    """Rounds a float to the nearest integer probabilistically.

    The integer part is always included. The fractional part determines the
    probability of rounding up. For example, 4.3 has a 30% chance of being
    rounded to 5 and a 70% chance of being rounded to 4.

    Args:
        target: The float number to round.

    Returns:
        The probabilistically rounded integer.
    """
    integer_part: int = math.floor(target)
    fractional_part: float = target - integer_part
    return integer_part + (1 if random.random() < fractional_part else 0)


def freq_to_midi(freq: float) -> float:
    """Convert a frequency in Hertz (Hz) to a MIDI note number.

    Uses A4 (440 Hz) as the reference frequency (MIDI note 69).

    Args:
        freq: The frequency in Hz.

    Returns:
        The corresponding MIDI note number (can be fractional for microtones).
    """
    return 12 * math.log2(freq / 440.0) + 69


def main() -> None:
    """Generates a MIDI file with randomly 'sprayed' notes based on config.

    Loads configuration from 'config.json', validates the 'spray_notes' section,
    calculates the number of notes based on density and duration, determines
    valid MIDI pitches within the specified frequency range, and generates
    random notes (pitch, start time, duration, velocity) within the configured
    constraints. The resulting notes are saved to a MIDI file specified in the
    config. Prints generation summary and errors to the console.

    Raises:
        ValueError: If configuration is invalid or no valid MIDI notes exist
                    within the specified frequency range.
        Exception: For file I/O errors or other unexpected issues.
    """
    try:
        # Load base configuration
        config: Optional[Dict[str, Any]] = load_config()
        if config is None:
            print("Warning: Failed to load config.json. Using defaults.")
            config = {}  # Use empty dict if config fails

        # Get defaults from config or hardcoded values
        spray_defaults = config.get("spray_notes", {})
        paths_defaults = config.get("paths", {})

        default_density = spray_defaults.get("density_notes_per_second", 10)
        default_total_duration = spray_defaults.get("total_duration", 60)
        default_lower_freq = spray_defaults.get("lower_freq", 20.0)
        default_upper_freq = spray_defaults.get("upper_freq", 20000.0)
        default_min_note_duration = spray_defaults.get("min_note_duration", 0.05)
        default_max_note_duration = spray_defaults.get("max_note_duration", 0.5)
        default_min_velocity = spray_defaults.get("min_velocity", 30)
        default_max_velocity = spray_defaults.get("max_velocity", 100)
        default_output_filename = spray_defaults.get("output_filename", "spray.mid")
        default_output_dir = paths_defaults.get("spray_notes_dir", "audio/Spray_Notes")

        # Setup argparse
        parser = argparse.ArgumentParser(
            description="Generate a MIDI file with randomly 'sprayed' notes.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--density",
            type=int,
            default=default_density,
            help="Average number of notes per second.",
        )
        parser.add_argument(
            "--duration",
            type=int,
            default=default_total_duration,
            help="Total duration of the MIDI file in seconds.",
        )
        parser.add_argument(
            "--min-freq",
            type=float,
            default=default_lower_freq,
            help="Minimum frequency (Hz) for generated notes.",
        )
        parser.add_argument(
            "--max-freq",
            type=float,
            default=default_upper_freq,
            help="Maximum frequency (Hz) for generated notes.",
        )
        parser.add_argument(
            "--min-note-dur",
            type=float,
            default=default_min_note_duration,
            help="Minimum duration (seconds) for a single note.",
        )
        parser.add_argument(
            "--max-note-dur",
            type=float,
            default=default_max_note_duration,
            help="Maximum duration (seconds) for a single note.",
        )
        parser.add_argument(
            "--min-vel",
            type=int,
            default=default_min_velocity,
            help="Minimum MIDI velocity (0-127).",
        )
        parser.add_argument(
            "--max-vel",
            type=int,
            default=default_max_velocity,
            help="Maximum MIDI velocity (0-127).",
        )
        parser.add_argument(
            "--output-file",
            default=default_output_filename,
            help="Name of the output MIDI file.",
        )
        parser.add_argument(
            "--output-dir",
            default=default_output_dir,
            help="Directory to save the output MIDI file.",
        )

        args = parser.parse_args()

        # Validate arguments (basic checks, more complex in SprayNotesConfig if needed)
        if args.min_freq >= args.max_freq:
            raise ValueError("max-freq must be greater than min-freq")
        if args.min_note_dur >= args.max_note_dur:
            raise ValueError("max-note-dur must be greater than min-note-dur")
        if not (0 <= args.min_vel <= 127) or not (0 <= args.max_vel <= 127):
            raise ValueError("Velocity must be between 0 and 127")
        if args.min_vel >= args.max_vel:
            raise ValueError("max-vel must be greater than min-vel")

        # Generate valid MIDI notes within frequency bounds
        valid_notes: List[int] = []
        for midi_note in range(0, 128):
            note_freq: float = 440 * (2 ** ((midi_note - 69) / 12))
            if args.min_freq <= note_freq <= args.max_freq:
                valid_notes.append(midi_note)

        if not valid_notes:
            raise ValueError(
                "No MIDI notes exist within the specified frequency range."
            )

        # Float-aware note count calculation
        total_notes: int = probabilistic_round(args.density * args.duration)
        start_times: List[float] = sorted(
            [random.uniform(0, args.duration) for _ in range(total_notes)]
        )

        # Create MIDI structure
        midi_data: pretty_midi.PrettyMIDI = pretty_midi.PrettyMIDI()
        piano: pretty_midi.Instrument = pretty_midi.Instrument(
            program=0
        )  # Acoustic Grand Piano

        for start in start_times:
            # Random note parameters
            pitch: int = random.choice(valid_notes)
            note_duration: float = random.uniform(args.min_note_dur, args.max_note_dur)
            velocity: int = random.randint(args.min_vel, args.max_vel)

            # Create note
            note: pretty_midi.Note = pretty_midi.Note(
                velocity=velocity, pitch=pitch, start=start, end=start + note_duration
            )
            piano.notes.append(note)

        midi_data.instruments.append(piano)

        # Ensure output directory exists and construct full output path
        output_dir_path: Path = Path(args.output_dir)
        os.makedirs(output_dir_path, exist_ok=True)
        output_path: Path = output_dir_path / args.output_file

        midi_data.write(str(output_path))

        print(f"Generated {total_notes} notes in '{output_path}'.")
        print(f"Note density: {args.density}/sec")
        print(f"Frequency range: {args.min_freq}-{args.max_freq} Hz")
        print(f"Duration range: {args.min_note_dur}-{args.max_note_dur} sec")
        print(f"Velocity range: {args.min_vel}-{args.max_vel}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
