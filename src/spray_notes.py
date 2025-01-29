import pretty_midi
import random
import math
import os
from pathlib import Path
from config_utils import load_config, create_spray_notes_config
import math

def probabilistic_round(target):
    integer_part = math.floor(target)
    fractional_part = target - integer_part
    return integer_part + (1 if random.random() < fractional_part else 0)

def freq_to_midi(freq):
    """Convert frequency in Hz to MIDI note number."""
    return 12 * math.log2(freq / 440.0) + 69

def main():
    """Generate random MIDI notes based on configuration."""
    try:
        # Load and validate configuration
        config = load_config()
        spray_config = create_spray_notes_config(config)
        spray_config.validate()

        # Generate valid MIDI notes within frequency bounds
        valid_notes = []
        for midi_note in range(0, 128):
            note_freq = 440 * (2 ** ((midi_note - 69) / 12))
            if spray_config.lower_freq <= note_freq <= spray_config.upper_freq:
                valid_notes.append(midi_note)

        if not valid_notes:
            raise ValueError("No MIDI notes exist within the specified frequency range.")
        
        # Float-aware note count calculation
        total_notes = probabilistic_round(spray_config.density_notes_per_second * spray_config.total_duration)  # Rounded to nearest integer
        start_times = sorted([random.uniform(0, spray_config.total_duration) 
                            for _ in range(total_notes)])

        # Create MIDI structure
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        for start in start_times:
            # Random note parameters
            pitch = random.choice(valid_notes)
            duration = random.uniform(spray_config.min_note_duration, 
                                   spray_config.max_note_duration)
            velocity = random.randint(spray_config.min_velocity, 
                                   spray_config.max_velocity)
            
            # Create note
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=start + duration
            )
            piano.notes.append(note)

        midi.instruments.append(piano)
        
        # Ensure output directory exists and construct full output path
        spray_notes_dir = Path(config["paths"]["spray_notes_dir"])
        os.makedirs(spray_notes_dir, exist_ok=True)
        output_path = spray_notes_dir / spray_config.output_filename
        
        midi.write(str(output_path))

        print(f"Generated {total_notes} notes in '{output_path}'.")
        print(f"Note density: {spray_config.density_notes_per_second}/sec")
        print(f"Frequency range: {spray_config.lower_freq}-{spray_config.upper_freq} Hz")
        print(f"Duration range: {spray_config.min_note_duration}-"
              f"{spray_config.max_note_duration} sec")
        print(f"Velocity range: {spray_config.min_velocity}-{spray_config.max_velocity}")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
