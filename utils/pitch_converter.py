#!/usr/bin/env python3
"""
Module for converting between OpenMusic stem numbers and musical note notation.
Converts numerical pitch representations to standard musical notation using
either sharps or flats as specified by the user.
"""
from typing import List, Union
from enum import Enum

class Accidental(Enum):
    """Enumeration for musical accidental preference."""
    SHARP = '#'
    FLAT = 'b'

# Constants
MIDDLE_C_STEM = 6000  # OpenMusic stem number for middle C
MIDDLE_C_MIDI = 60    # MIDI note number for middle C
SEMITONE_UNIT = 100   # Number of units per semitone in stem notation
NOTES_PER_OCTAVE = 12

# Note mappings
NOTES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_FLAT  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

def validate_stem(stem: Union[int, float]) -> None:
    """
    Validate that the stem number is within a reasonable range.
    
    Args:
        stem: The stem number to validate.
        
    Raises:
        ValueError: If the stem number is outside the reasonable range.
    """
    # Assuming a reasonable range is from -6000 to 18000 (about 10 octaves around middle C)
    if not -6000 <= stem <= 18000:
        raise ValueError(f"Stem number {stem} is outside the reasonable range (-6000 to 18000)")

def stem_to_note(stem: Union[int, float], accidental: Accidental = Accidental.SHARP) -> str:
    """
    Convert a stem number to a musical note.
    
    The conversion is as follows:
      - 6000 units corresponds to middle C (C4, MIDI note 60)
      - Each 100 units equals one semitone.
      - Calculate the semitone offset: (stem - 6000)/100 (rounded to the nearest integer).
      - Add the offset to the MIDI note for middle C.
      - Convert the MIDI note number to a note name (with the chosen accidental) and octave.
    
    Args:
        stem: The stem number to convert.
        accidental: Whether to use sharps or flats (default is sharp).
    
    Returns:
        The musical note with octave (e.g., "C4", "F#3", "Bb2").
        
    Raises:
        ValueError: If the stem number is outside the reasonable range.
    """
    validate_stem(stem)
    
    # Calculate the semitone offset and corresponding MIDI note
    semitone_offset = round((stem - MIDDLE_C_STEM) / SEMITONE_UNIT)
    midi_note = MIDDLE_C_MIDI + semitone_offset
    
    # Choose the note mapping based on the accidental preference
    note_names = NOTES_SHARP if accidental == Accidental.SHARP else NOTES_FLAT
    
    # Compute note name and octave
    note_name = note_names[midi_note % NOTES_PER_OCTAVE]
    octave = (midi_note // NOTES_PER_OCTAVE) - 1

    return f"{note_name}{octave}"

def process_stems(
    stems: List[Union[int, float]], 
    accidental: Accidental = Accidental.SHARP
) -> List[str]:
    """
    Convert a list of stem numbers to their corresponding musical notes.
    
    Args:
        stems: List of stem numbers.
        accidental: Whether to use sharps or flats in the notation.
    
    Returns:
        List of musical notes.
        
    Raises:
        ValueError: If any stem number is invalid.
    """
    return [stem_to_note(stem, accidental) for stem in stems]

def parse_input(input_str: str) -> List[float]:
    """
    Parse space-separated numbers from a string into a list of floats.
    
    Args:
        input_str: String containing space-separated numbers.
        
    Returns:
        List of parsed float values.
        
    Raises:
        ValueError: If the input contains non-numeric values.
    """
    try:
        return [float(num) for num in input_str.strip().split()]
    except ValueError:
        raise ValueError("Invalid input. Please enter numbers separated by spaces.")

def main() -> None:
    """Command-line interface for stem to note conversion."""
    try:
        user_input = input("Enter stem numbers separated by spaces: ")
        stems = parse_input(user_input)
        
        # acc_input = input("Enter accidental preference ('#' for sharp, 'b' for flat): ").strip()
        # if acc_input == 'b':
        #     accidental = Accidental.FLAT
        # else:
        #     accidental = Accidental.SHARP

        notes = process_stems(stems, Accidental.FLAT)
        
        print("\nResults:")
        print(f"{'Stem':>8} | {'Note':^6}")
        print("-" * 18)
        for stem, note in zip(stems, notes):
            print(f"{stem:>8.0f} | {note:^6}")
            
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")

if __name__ == '__main__':
    main()
