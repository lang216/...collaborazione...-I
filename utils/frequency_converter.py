"""
Module for converting OpenMusic stem numbers and musical pitch notation to frequencies.
"""
from typing import List, Union, Final, Dict, Optional
from enum import Enum
import re

class Accidental(Enum):
    """Enumeration for musical accidental preference."""
    SHARP: str = '#'
    FLAT: str = 'b'

    def __str__(self) -> str:
        return self.value

# Constants
MIDDLE_C_STEM: Final[int] = 6000    # OpenMusic stem number for middle C
MIDDLE_C_MIDI: Final[int] = 60      # MIDI note number for middle C
SEMITONE_UNIT: Final[int] = 100     # Number of units per semitone in stem notation
NOTES_PER_OCTAVE: Final[int] = 12
A4_FREQUENCY: Final[float] = 440.0  # Standard concert pitch A4 in Hz
A4_MIDI: Final[int] = 69           # MIDI note number for A4

# Note to MIDI number mapping (relative to C0)
NOTE_TO_MIDI: Final[Dict[str, int]] = {
    'C': 0, 'C#': 1, 'Db': 1,
    'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8,
    'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11
}

__slots__ = []  # Optimize memory usage

def validate_pitch_notation(pitch: str) -> None:
    """
    Validate that the pitch notation is properly formatted.
    
    Args:
        pitch: The pitch notation to validate (e.g., 'A4', 'C#5', 'Bb3').
        
    Raises:
        ValueError: If the pitch notation is invalid.
    """
    pattern = r'^[A-G](#|b)?(-1|[0-9])$'
    if not re.match(pattern, pitch):
        raise ValueError(
            f"Invalid pitch notation: {pitch}. "
            "Format should be note[accidental]octave (e.g., 'A4', 'C#5', 'Bb3')"
        )

def validate_stem(stem: Union[int, float]) -> None:
    """
    Validate that the stem number is within a reasonable range.
    
    The range -6000 to 18000 represents approximately 10 octaves around middle C,
    which covers the entire range of most musical instruments.
    
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
    """
    Convert a musical pitch notation to its corresponding frequency.
    
    Args:
        pitch: The pitch notation (e.g., 'A4', 'C#5', 'Bb3').
    
    Returns:
        float: The frequency of the note in Hz, rounded to 3 decimal places.
        
    Raises:
        ValueError: If the pitch notation is invalid.
    """
    validate_pitch_notation(pitch)
    
    # Split pitch into note and octave
    match = re.match(r'([A-G][#b]?)(-?\d+)', pitch)
    note, octave = match.groups()
    octave = int(octave)
    
    # Get MIDI note number
    base_midi = NOTE_TO_MIDI[note]
    midi_note = base_midi + ((octave + 1) * NOTES_PER_OCTAVE)
    
    # Convert MIDI note to frequency using A4 (440 Hz) as reference
    frequency: float = A4_FREQUENCY * 2 ** ((midi_note - A4_MIDI) / NOTES_PER_OCTAVE)
    return round(frequency, 3)

def stem_to_frequency(stem: Union[int, float]) -> float:
    """
    Convert a stem number to its corresponding musical frequency.
    
    Args:
        stem: The stem number to convert.
    
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

def process_stems(stems: List[Union[int, float]]) -> List[float]:
    """
    Convert a list of stem numbers to their corresponding musical frequencies.
    
    Args:
        stems: List of stem numbers.
    
    Returns:
        List[float]: List of frequencies in Hz.
        
    Raises:
        ValueError: If any stem number is invalid.
    """
    return [stem_to_frequency(stem) for stem in stems]

def parse_input(input_str: str) -> List[Union[str, float]]:
    """
    Parse space-separated numbers or pitch notations into a list of frequencies.
    
    Args:
        input_str: String containing space-separated numbers or pitch notations.
        
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

def main() -> None:
    """Command-line interface for stem/pitch to frequency conversion."""
    try:
        print("Enter values separated by spaces.")
        print("Accepted formats:")
        print("  - Stem numbers (e.g., 6000 6100 6200)")
        print("  - Pitch notation (e.g., A4 C#5 Bb3)")
        user_input = input("\nEnter values: ")
        
        frequencies = parse_input(user_input)
        
        print("\nResults:")
        print(f"{'Input':>8} | {'Frequency (Hz)':^15}")
        print("-" * 30)
        for value, frequency in zip(user_input.split(), frequencies):
            print(f"{value:>8} | {frequency:^15.3f}")
            
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

if __name__ == '__main__':
    main()