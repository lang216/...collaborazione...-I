import librosa
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass

@dataclass
class ChordSegment:
    """Class to store chord notes and duration information."""
    notes: list[int]  # OpenMusic format notes
    duration: float   # Duration in seconds

def extract_microtonal_sequence(audio_path, num_chords, num_voices=4, min_freq=40, max_freq=12000) -> list[ChordSegment]:
    """
    Extract a sequence of chords from audio with duration information.
    Returns a list of ChordSegment objects containing notes and durations.
    """
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract MFCC features for agglomerative clustering
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = librosa.util.normalize(mfcc, axis=1)
    
    # Perform agglomerative clustering
    boundaries = librosa.segment.agglomerative(
        mfcc, 
        k=num_chords,
    )
    
    # Create segments based on cluster boundaries
    segments = []
    prev_boundary = 0
    for boundary in boundaries:
        start = librosa.frames_to_samples(prev_boundary)
        end = librosa.frames_to_samples(boundary)
        segments.append(y[start:end])
        prev_boundary = boundary
    # Add final segment
    segments.append(y[librosa.frames_to_samples(prev_boundary):])
    
    sequence = []
    
    # Convert boundaries to sample indices and durations
    boundary_samples = [librosa.frames_to_samples(b) for b in boundaries]
    boundary_samples.append(len(y))  # Add end of audio
    
    for i, (start, end) in enumerate(zip(boundary_samples[:-1], boundary_samples[1:])):
        seg = y[start:end]
        duration = (end - start) / sr  # Duration in seconds
        if len(seg) < 10:  # Skip empty segments
            sequence.append([])
            continue
            
        # Advanced spectral analysis with windowing
        window = np.blackman(len(seg))
        fft = np.fft.rfft(seg * window)
        magnitudes = np.abs(fft)
        freqs = np.fft.rfftfreq(len(seg), d=1/sr)
        
        # Find significant peaks with noise floor adaptation
        noise_floor = np.median(magnitudes)
        peaks, props = find_peaks(magnitudes, 
                                height=noise_floor*2, 
                                distance=50,
                                prominence=noise_floor*1.5)
        
        # Filter frequency range and sort peaks
        valid_peaks = [(freqs[p], magnitudes[p]) for p in peaks 
                      if min_freq <= freqs[p] <= max_freq]
        sorted_peaks = sorted(valid_peaks, 
                            key=lambda x: x[1], 
                            reverse=True)[:num_voices]
        
        # Convert to OpenMusic MIDI with cent precision
        om_notes = []
        for freq, mag in sorted_peaks:
            if freq <= 0:
                continue
            midi = 69 + 12 * np.log2(freq/440)  # Exact MIDI calculation
            om = int(round(midi * 100))  # OpenMusic format
            om_notes.append(om)
        
        sequence.append(ChordSegment(notes=om_notes, duration=duration))
    
    return sequence

# # Usage
# if __name__ == "__main__":
#     audio_file = "your_audio.wav"
#     num_chords = 12  # Number of sequential harmonic groupings
    
#     om_sequence = extract_microtonal_sequence(audio_file, num_chords)
    
#     print("Microtonal Chord Sequence (OpenMusic MIDI):")
#     for idx, chord in enumerate(om_sequence):
#         print(f"Chord {idx+1}: {sorted(chord)}")
