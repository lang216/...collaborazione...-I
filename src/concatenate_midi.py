import os
from midiutil import MIDIFile
from typing import List
import pretty_midi
from tqdm import tqdm


def get_all_midi_files(root_folder: str) -> List[str]:
    """
    Recursively find all MIDI files in the root folder and its subfolders.
    
    Args:
        root_folder (str): Path to the root folder containing MIDI files and subfolders
        
    Returns:
        List[str]: List of full paths to all MIDI files found
    """
    midi_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
    
    midi_files.sort()  # Sort for consistent ordering
    return midi_files

def trim_silence(pm: pretty_midi.PrettyMIDI, MAX_SILENCE: float=0.2) -> pretty_midi.PrettyMIDI:
    """
    Trim silence from a MIDI file by removing gaps longer than MAX_SILENCE.
    
    Args:
        pm (pretty_midi.PrettyMIDI): MIDI object to process
        MAX_SILENCE (float, optional): Maximum allowed silence duration in seconds. Defaults to 0.2.
        
    Returns:
        pretty_midi.PrettyMIDI: Processed MIDI object with long silences removed
    """
    if not pm.instruments or not any(inst.notes for inst in pm.instruments):
        return pm
    
    # Get all notes across all instruments
    all_notes = []
    for instrument in pm.instruments:
        all_notes.extend(instrument.notes)
    
    if not all_notes:
        return pm
    
    # Sort notes by start time
    all_notes.sort(key=lambda x: x.start)
    
    # Find gaps longer than MAX_SILENCE
    time_shifts = []
    current_shift = 0
    last_end = all_notes[0].start
    
    for note in all_notes:
        gap = note.start - last_end
        if gap > MAX_SILENCE:
            # Accumulate the excess silence
            current_shift += gap - MAX_SILENCE
        time_shifts.append(current_shift)
        last_end = max(last_end, note.end)
    
    # Apply time shifts to remove long silences
    for instrument in pm.instruments:
        for note in instrument.notes:
            # Find the appropriate time shift for this note
            shift_index = next(i for i, n in enumerate(all_notes) if n.start >= note.start)
            shift = time_shifts[shift_index]
            note.start -= shift
            note.end -= shift
    
    return pm

def concatenate_midi_files_from_paths(midi_file_paths: List[str], output_file: str) -> None:
    """
    Concatenate multiple MIDI files from a list of file paths into a single MIDI file,
    placing each piece one after another in a single track.
    
    Args:
        midi_file_paths (List[str]): List of paths to MIDI files
        output_file (str): Path where the concatenated MIDI file will be saved
    """
    if not midi_file_paths:
        raise ValueError("No MIDI files provided")

    # Initialize the combined MIDI with a single piano track
    combined_midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    combined_track = pretty_midi.Instrument(program=piano_program)
    current_time = 0.0

    # Create progress bar
    pbar = tqdm(midi_file_paths, desc="Processing MIDI files", unit="file")

    # Process each MIDI file
    for midi_path in pbar:
        try:
            # Update progress bar description with current file
            pbar.set_description(f"Processing {os.path.basename(midi_path)}")
            
            # Load and trim the MIDI file
            pm = pretty_midi.PrettyMIDI(midi_path)
            pm = trim_silence(pm)
            
            # Collect all notes from all instruments
            for instrument in pm.instruments:
                for note in instrument.notes:
                    new_note = pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start + current_time,
                        end=note.end + current_time
                    )
                    combined_track.notes.append(new_note)
            
            # Update the current time for the next file
            current_time += pm.get_end_time()
            # Add a small gap between pieces
            current_time += MAX_SILENCE
            
        except Exception as e:
            print(f"\nError processing {midi_path}: {str(e)}")
            continue

    # Add the combined track to the MIDI file
    combined_midi.instruments.append(combined_track)
    
    # Save the combined MIDI file
    print("\nSaving combined MIDI file...")
    combined_midi.write(output_file)
    print(f"Successfully created concatenated MIDI file: {output_file}")

if __name__ == "__main__":
    # Use the actual path to your classical music MIDI files
    input_folder = "RAVE_Train_GEN/midi_files/raw_midi/classical-music-midi"
    output_file = "RAVE_Train_GEN/midi_files/concatenated_output/concatenated_classical_music.mid"
    
    # Get all MIDI files from the folder and its subfolders
    print("Scanning for MIDI files...")
    midi_files = get_all_midi_files(input_folder)
    print(f"Found {len(midi_files)} MIDI files")
    
    # Concatenate all found MIDI files
    concatenate_midi_files_from_paths(midi_files, output_file)
