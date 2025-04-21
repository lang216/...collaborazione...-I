import numpy as np
import scipy.io.wavfile as wavfile
import json
import os
import re
import argparse  # Import argparse
from typing import List, Union, Final, Dict, Optional, Any
from enum import Enum


class Accidental(Enum):
    """Enumeration for musical accidental preference."""

    SHARP: str = "#"
    FLAT: str = "b"

    def __str__(self) -> str:
        return self.value


# Constants
MIDDLE_C_STEM: Final[int] = 6000  # OpenMusic stem number for middle C
MIDDLE_C_MIDI: Final[int] = 60  # MIDI note number for middle C
SEMITONE_UNIT: Final[int] = 100  # Number of units per semitone in stem notation
NOTES_PER_OCTAVE: Final[int] = 12
A4_FREQUENCY: Final[float] = 440.0  # Standard concert pitch A4 in Hz
A4_MIDI: Final[int] = 69  # MIDI note number for A4

# Note to MIDI number mapping (relative to C0)
NOTE_TO_MIDI: Final[Dict[str, int]] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

__slots__ = []  # Optimize memory usage


def validate_pitch_notation(pitch: str) -> None:
    """Validate that the pitch notation string conforms to the expected format.

    The expected format is Note[Accidental]Octave, where Note is A-G,
    Accidental is optional (# or b), and Octave is an integer (-1 to 9).

    Args:
        pitch: The pitch notation string to validate (e.g., 'A4', 'C#5', 'Bb3').

    Raises:
        ValueError: If the pitch notation is invalid.
    """
    pattern = r"^[A-G](#|b)?(-1|[0-9])$"
    if not re.match(pattern, pitch):
        raise ValueError(
            f"Invalid pitch notation: {pitch}. "
            "Format should be note[accidental]octave (e.g., 'A4', 'C#5', 'Bb3')"
        )


def validate_stem(stem: Union[int, float]) -> None:
    """Validate that the stem number (OpenMusic format) is within a reasonable range.

    The range -6000 to 18000 (inclusive) is checked, representing approximately
    10 octaves centered around Middle C (6000), covering the typical range of
    musical instruments and beyond.

    Args:
        stem: The stem number to validate.

    Raises:
        ValueError: If the stem number is outside the reasonable range.
    """
    if not -6000 <= stem <= 18000:
        raise ValueError(
            f"Stem number {stem} is outside the valid range (-6000 to 18000). "
            "This range covers approximately 10 octaves around middle C."
        )


def pitch_to_frequency(pitch: str) -> float:
    """Convert a standard musical pitch notation string to its frequency in Hz.

    Uses A4 (440 Hz) as the reference frequency.

    Args:
        pitch: The pitch notation string (e.g., 'A4', 'C#5', 'Bb3').

    Returns:
        float: The frequency of the note in Hz, rounded to 3 decimal places.

    Raises:
        ValueError: If the pitch notation is invalid.
    """
    validate_pitch_notation(pitch)

    # Split pitch into note and octave
    match = re.match(r"([A-G][#b]?)(-?\d+)", pitch)
    note, octave = match.groups()
    octave = int(octave)

    # Get MIDI note number
    base_midi = NOTE_TO_MIDI[note]
    midi_note = base_midi + ((octave + 1) * NOTES_PER_OCTAVE)

    # Convert MIDI note to frequency using A4 (440 Hz) as reference
    frequency: float = A4_FREQUENCY * 2 ** ((midi_note - A4_MIDI) / NOTES_PER_OCTAVE)
    return round(frequency, 3)


def stem_to_frequency(stem: Union[int, float]) -> float:
    """Convert an OpenMusic stem number to its corresponding frequency in Hz.

    Uses Middle C (MIDI 60, stem 6000) and A4 (MIDI 69, 440 Hz) as references.
    Stem numbers represent MIDI note * 100 + cents deviation.

    Args:
        stem: The OpenMusic stem number (integer or float) to convert.

    Returns:
        float: The frequency of the note in Hz, rounded to 3 decimal places.

    Raises:
        ValueError: If the stem number is outside the reasonable range.
    """
    validate_stem(stem)

    # Calculate the semitone offset and corresponding MIDI note
    semitone_offset: float = round((stem - MIDDLE_C_STEM) / SEMITONE_UNIT)
    midi_note: float = MIDDLE_C_MIDI + semitone_offset

    # Convert MIDI note number to frequency using A4 (440 Hz) as reference
    frequency: float = A4_FREQUENCY * 2 ** ((midi_note - A4_MIDI) / NOTES_PER_OCTAVE)

    return round(frequency, 3)


def parse_input(input_str: str) -> List[float]:
    """Parse a string containing space-separated stem numbers or pitch notations.

    Converts each valid input value (stem or pitch) into its corresponding
    frequency in Hz.

    Args:
        input_str: A string containing space-separated numeric stem values
                   (e.g., "6000 6200") or standard pitch notations
                   (e.g., "C4 E4 G4"). Mixed inputs are allowed.

    Returns:
        List[float]: List of frequencies in Hz.

    Raises:
        ValueError: If the input contains invalid values.
    """
    values = input_str.split()
    frequencies = []

    for value in values:
        try:
            # Try parsing as a numeric value first
            frequencies.append(stem_to_frequency(float(value)))
        except ValueError:
            try:
                # If that fails, try parsing as pitch notation
                frequencies.append(pitch_to_frequency(value))
            except ValueError as e:
                raise ValueError(
                    f"Invalid input: {value}. Please enter either numeric stem values "
                    "or pitch notation (e.g., 'A4', 'C#5', 'Bb3') separated by spaces."
                ) from e

    return frequencies


def generate_sine_tone(
    frequency,
    duration,
    sample_rate=44100,
    amplitude=0.8,
    amplitude_jitter_amount=0.0,
    jitter_frequency: float = 5.0,
) -> np.ndarray:
    """Generate a sine wave tone with optional smooth amplitude jitter.

    Applies amplitude modulation based on interpolated random jitter values
    to create a more natural-sounding variation. Includes clipping prevention
    by normalizing only if the generated tone exceeds the [-1, 1] range.

    Args:
        frequency: The frequency of the sine wave in Hz.
        duration: The duration of the tone in seconds.
        sample_rate: The sample rate in Hz (default: 44100).
        amplitude: The target base amplitude (default: 0.8).
        amplitude_jitter_amount: The maximum amount of jitter to apply, scaled
                                 between 0 (no jitter) and 1 (max jitter).
                                 (default: 0.0).
        jitter_frequency: The approximate frequency (in Hz) of the amplitude
                          variations (default: 5.0).

    Returns:
        A NumPy array containing the generated sine tone audio data, normalized
        to the range [-1, 1] only if necessary to prevent clipping.
    """
    # Create time array
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Generate the base sine tone with higher amplitude
    base_tone = amplitude * np.sin(2 * np.pi * frequency * time)

    if amplitude_jitter_amount > 0:
        # Determine number of jitter control points based on the jitter frequency
        num_jitter_points = int(duration * jitter_frequency) + 1

        # Times at which jitter values are defined
        jitter_times = np.linspace(0, duration, num_jitter_points)

        # Generate random jitter values between -1 and 1 for each control point
        jitter_values = np.random.uniform(-1, 1, num_jitter_points)

        # Smoothly interpolate these jitter values over the entire duration
        amplitude_modulation = np.interp(time, jitter_times, jitter_values)

        # Scale modulation and shift so that the base amplitude remains around 1
        amplitude_modulation = 1 + amplitude_jitter_amount * amplitude_modulation

        # Apply the smooth amplitude modulation to the base tone
        tone = base_tone * amplitude_modulation
    else:
        tone = base_tone

    # Only normalize if we're actually clipping
    max_val = np.max(np.abs(tone))
    if max_val > 1.0:
        tone = tone / max_val

    return tone


def save_wav(tone: np.ndarray, filename: str, sample_rate: int = 44100) -> None:
    """Save the given audio data as a 16-bit WAV file.

    Creates the output directory if it doesn't exist.

    Args:
        tone: NumPy array containing the audio data (should be in range [-1, 1]).
        filename: The desired path for the output WAV file.
        sample_rate: The sample rate of the audio data (default: 44100).

    Side Effects:
        - Creates directories as needed for the output file path.
        - Writes a WAV file to the specified `filename`.
        - Prints a confirmation message to the console.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Scale to full 16-bit range
    int_tone = np.int16(tone * 32767)
    wavfile.write(filename, sample_rate, int_tone)
    print(f"Wave file saved as: {filename}")


def load_config(config_file_path: str = "config.json") -> Optional[Dict[str, Any]]:
    """Load configuration settings from a specified JSON file.

    Args:
        config_file_path: The path to the JSON configuration file
                          (default: "config.json").

    Returns:
        A dictionary containing the loaded configuration, or None if loading fails
        due to file not found or JSON decoding errors. Prints error messages
        to the console on failure.
    """
    try:
        with open(config_file_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_file_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{config_file_path}'.")
        return None


def main() -> None:
    """Main function for the command-line sine tone generator.

    Loads configuration, parses command-line arguments for input values
    (stem numbers or pitch notations) and generation parameters, generates
    sine tones, and saves them as WAV files. Handles errors gracefully.
    """
    config = load_config()

    # Set defaults from config or hardcoded values if config fails
    default_sr = 44100
    default_duration = 5.0
    default_jitter_amount = 0.0
    default_jitter_freq = 5.0
    default_output_dir = "audio/Sine_Tones"

    if config:
        default_sr = config.get("audio", {}).get("sr", default_sr)
        sine_config = config.get("sine_tone_generator", {})
        default_duration = sine_config.get("duration", default_duration)
        default_jitter_amount = sine_config.get(
            "amplitude_jitter_amount", default_jitter_amount
        )
        default_jitter_freq = sine_config.get("jitter_frequency", default_jitter_freq)
        default_output_dir = config.get("paths", {}).get(
            "output_tone_file", default_output_dir
        )

    parser = argparse.ArgumentParser(
        description="Generate sine wave tones from stem numbers or pitch notation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_values",
        nargs="+",
        help="Space-separated stem numbers (e.g., 6000 6100) or pitch notations (e.g., A4 C#5 Bb3).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=default_duration,
        help="Duration of each tone in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=default_sr,
        help="Sample rate in Hz.",
    )
    parser.add_argument(
        "--jitter-amount",
        type=float,
        default=default_jitter_amount,
        help="Amplitude jitter amount (0.0 to 1.0).",
    )
    parser.add_argument(
        "--jitter-frequency",
        type=float,
        default=default_jitter_freq,
        help="Frequency of amplitude jitter in Hz.",
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory to save the generated WAV files.",
    )

    args = parser.parse_args()

    try:
        # Join the list of input values back into a space-separated string for parsing
        input_str = " ".join(args.input_values)
        frequencies = parse_input(input_str)

        print("\nGenerating tones:")
        print(f"{'Input':>10} | {'Frequency (Hz)':^15} | {'Output File':<40}")
        print("-" * 70)

        for i, value in enumerate(args.input_values):
            frequency = frequencies[i]

            # Sanitize filename
            safe_value = value.replace("#", "sharp").replace("b", "flat")
            filename = f"tone_{safe_value}.wav"
            full_path = os.path.join(args.output_dir, filename)

            print(f"{value:>10} | {frequency:^15.3f} | {full_path:<40}")

            tone = generate_sine_tone(
                frequency=frequency,
                duration=args.duration,
                sample_rate=args.sample_rate,
                amplitude_jitter_amount=args.jitter_amount,
                jitter_frequency=args.jitter_frequency,
            )
            save_wav(tone, full_path, sample_rate=args.sample_rate)

        print("\nSine tones generated and saved successfully!")

    except ValueError as e:
        print(f"\nError: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
