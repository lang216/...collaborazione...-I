"""
Text-to-Speech generator using Kokoro TTS pipeline.
Generates and saves audio files from input text, splitting on newlines.

This script is specifically designed to generate spoken measure numbers for
musical scores, with special formatting rules for different number ranges to
ensure clear pronunciation.

Features:
- Custom number formatting for measures above 100
- Configurable output directory via config.json
- Support for multiple TTS voices and languages
- Individual WAV file generation for each measure number
"""

# Standard library imports
import os
from typing import Iterator, Tuple, Dict, Any, Optional, List
import argparse

# Third-party imports
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import numpy as np
import pyrubberband as rb

# Local imports
from config_utils import load_config

# Initialize configuration
config: Optional[Dict[str, Any]] = load_config()
if config is None:
    # Handle error: Maybe exit or use default values
    print("Error: Failed to load configuration. Using default sample rate.")
    SAMPLE_RATE: int = 24000
else:
    SAMPLE_RATE: int = config.get("audio", {}).get("tts_sr", 24000)
print(f"Using Sample Rate: {SAMPLE_RATE}")


def format_measure_number(num: int) -> str:
    """Format measure number for speech synthesis using specific rules.

    Applies special formatting for numbers above 99 to improve clarity
    when spoken by a TTS engine.

    Formatting rules:
    - Numbers 1-99: Spoken as is
    - Numbers 100, 200, 300, etc: Spoken as "one hundred", "two hundred", etc.
    - Numbers ending in 0 (110, 120, etc): Spoken as "1 ten", "1 twenty", etc.
    - All other numbers over 100: Spoken digit by digit with "O" for zero

    Examples:
    - 101 -> "1 O 1"
    - 110 -> "1 ten"
    - 120 -> "1 twenty"
    - 203 -> "2 O 3"
    - 350 -> "3 fifty"

    Args:
        num: The measure number to format.

    Returns:
        The formatted string representation of the measure number for TTS input.
    """
    if num <= 99:
        return str(num)

    hundreds: int = num // 100
    remainder: int = num % 100

    # Handle exact hundreds
    if remainder == 0:
        hundreds_map: Dict[int, str] = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }
        return f"{hundreds_map[hundreds]} hundred"

    # Handle numbers ending in 0 (110, 120, etc)
    if remainder % 10 == 0:
        tens_map: Dict[int, str] = {
            10: "ten",
            20: "twenty",
            30: "thirty",
            40: "forty",
            50: "fifty",
            60: "sixty",
            70: "seventy",
            80: "eighty",
            90: "ninety",
        }
        return f"{hundreds} {tens_map[remainder]}"

    # All other numbers use digit-by-digit pronunciation with "O" for zero
    digits: str = str(num)
    return " ".join("O" if d == "0" else d for d in digits)


def apply_fade(
    audio: np.ndarray, fade_duration: float = 0.005, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """
    Apply fade in and fade out to audio to prevent clipping.

    Args:
        audio: Audio data as numpy array
        fade_duration: Duration of fade in seconds (default=5ms)
        sample_rate: Audio sample rate in Hz

    Returns:
        A new NumPy array representing the audio with fades applied.
    """
    fade_length: int = int(fade_duration * sample_rate)
    fade_in: np.ndarray = np.linspace(0, 1, fade_length)
    fade_out: np.ndarray = np.linspace(1, 0, fade_length)

    audio_copy: np.ndarray = audio.copy()
    # Apply fade in
    audio_copy[:fade_length] *= fade_in
    # Apply fade out
    audio_copy[-fade_length:] *= fade_out

    return audio_copy


def detect_silence_edges(
    audio: np.ndarray, threshold: float = 0.01, window_size: int = 1024
) -> tuple[int, int]:
    """
    Detect the start and end of non-silent portions in audio.

    Args:
        audio: Audio data as numpy array
        threshold: RMS energy threshold below which is considered silence
        window_size: Size of windows to analyze for RMS energy

    Returns:
        A tuple containing the start and end sample indices of the
        first detected non-silent portion.
    """
    # Calculate RMS energy in windows
    audio_abs: np.ndarray = np.abs(audio)
    start_idx: int = 0
    end_idx: int = len(audio)

    # Find start (first window above threshold)
    for i in range(0, len(audio), window_size):
        window: np.ndarray = audio_abs[i: i + window_size]
        if len(window) > 0 and np.mean(window) > threshold: # Check window not empty
            start_idx = i
            break

    # Find end (last window above threshold)
    for i in range(len(audio) - window_size, start_idx, -window_size):
        window: np.ndarray = audio_abs[i: i + window_size]
        if len(window) > 0 and np.mean(window) > threshold: # Check window not empty
            end_idx = min(i + window_size, len(audio))
            break

    return start_idx, end_idx


def generate_audio(
    text: str, lang_code: str = "a", voice: str = "af_heart", speed: float = 1.0
) -> Iterator[Tuple[str, str, List[float]]]:  # Hint list content as float
    """
    Generate audio from text using Kokoro TTS.

    The function splits input text on newlines and processes each segment separately.
    Each segment generates a tuple containing:
    - The original text (graphemes)
    - The phonetic representation (phonemes)
    - The audio data as a list of samples

    Args:
        text: Input text to convert to speech. Newlines act as segment separators.
        lang_code: Language code for TTS ('a'=US English, 'b'=UK English,
                   'j'=Japanese, 'z'=Mandarin).
        voice: Specific voice model to use (e.g., 'af_heart').
        speed: Speech speed multiplier (1.0 is normal speed).

    Yields:
        Tuples, one for each processed text segment (split by newline),
        containing:
            - graphemes (str): The original text segment.
            - phonemes (str): The phonetic representation.
            - audio (List[float]): The generated audio samples as a list of floats.
    """
    pipeline = KPipeline(lang_code=lang_code)
    # Assuming the pipeline yields tuples matching the signature
    return pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")


def process_and_save_audio(
    num_measures: int,
    starting_measure: int = 1,
    output_dir: Optional[str] = None,  # Allow None
    bpm: int = 120,
    **kwargs: Any,  # Accept any keyword args for generate_audio
) -> None:
    """
    Process text and save resulting audio files.

    Args:
        num_measures: Number of measure numbers to generate
        starting_measure: First measure number (default=1)
        output_dir: Directory to save the generated WAV files. If None, uses the
                    path specified in the 'measure_audio_files' key under 'paths'
                    in the global config.
        bpm: Beats per minute, used to calculate the target maximum duration
             (one beat) for each audio file.
        **kwargs: Additional keyword arguments passed directly to the
                  `generate_audio` function (e.g., `lang_code`, `voice`, `speed`).

    Side Effects:
        - Creates the output directory if it doesn't exist.
        - Prints processing information to the console.
        - Saves generated audio as WAV files in the output directory.
        - Displays the generated audio using IPython.display.Audio (if in a
          compatible environment).

    Processing Steps per Measure Number:
        1. Format the number using `format_measure_number`.
        2. Generate audio using `generate_audio`.
        3. Remove leading/trailing silence using `detect_silence_edges`.
        4. Time-stretch the audio using `pyrubberband` if its duration exceeds
           one beat length (calculated from `bpm`).
        5. Apply fade-in/fade-out using `apply_fade`.
        6. Save the processed audio as a WAV file (e.g., "measure_001.wav").
    """
    if config is None:
        print("Error: Configuration not loaded. Cannot determine output directory.")
        return  # Or raise an error

    final_output_dir: str
    if output_dir is None:
        # Ensure config and necessary keys exist before accessing
        if "paths" in config and "measure_audio_files" in config["paths"]:
            final_output_dir = config["paths"]["measure_audio_files"]
        else:
            print("Error: Output directory not specified and not found in config.")
            return  # Or raise an error
    else:
        final_output_dir = output_dir

    os.makedirs(final_output_dir, exist_ok=True)

    beat_duration: float = 60 / bpm
    print(f"Beat duration: {beat_duration:.2f} seconds (BPM: {bpm})")

    # Generate measure numbers text with special formatting
    measure_numbers_text: str = ""
    for i in range(starting_measure, starting_measure + num_measures):
        measure_numbers_text += format_measure_number(i) + "\n"

    # Type hint for the loop variables
    graphemes: str
    phonemes: str
    audio: List[float]  # Matches generate_audio yield type
    audio_generator = generate_audio(measure_numbers_text, **kwargs)

    for i, (graphemes, phonemes, audio) in enumerate(audio_generator):
        print(f"Processing segment {i + 1}:")
        print(f"Text: {graphemes}")
        print(f"Phonemes: {phonemes}\n")

        # Convert to numpy array
        audio_array: np.ndarray = np.array(audio, dtype=np.float32)  # Specify dtype

        # Calculate original duration
        original_duration: float = len(audio_array) / SAMPLE_RATE
        print(f"Original duration: {original_duration:.2f}s")

        # Remove silence from start and end
        start_idx: int
        end_idx: int
        start_idx, end_idx = detect_silence_edges(audio_array)
        trimmed_audio: np.ndarray = audio_array[start_idx:end_idx]

        # Calculate duration after silence removal
        current_duration: float = len(trimmed_audio) / SAMPLE_RATE
        print(f"Duration after silence removal: {current_duration:.2f}s")
        print(f"Beat duration: {beat_duration:.2f}s")

        # Time-stretch if needed using rubberband
        if current_duration > beat_duration:
            time_ratio: float = current_duration / beat_duration
            print(f"Time ratio for compression: {time_ratio:.2f}")
            # Ensure trimmed_audio is float for rubberband
            trimmed_audio = rb.time_stretch(
                trimmed_audio.astype(np.float32), SAMPLE_RATE, time_ratio
            )
            stretched_duration: float = len(trimmed_audio) / SAMPLE_RATE
            print(f"Duration after time stretch: {stretched_duration:.2f}s")

        # Apply fade in/out
        trimmed_audio = apply_fade(trimmed_audio)

        output_file: str = os.path.join(
            final_output_dir, f"measure_{starting_measure + i:03d}.wav"
        )
        # Check if display is available before calling
        try:
            display(Audio(data=trimmed_audio, autoplay=i == 0, rate=SAMPLE_RATE))
        except NameError:
            print("(IPython display not available, skipping audio preview)")
        sf.write(output_file, trimmed_audio, SAMPLE_RATE)


if __name__ == "__main__":
    try:
        parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description="Generate measure number audio files with BPM control.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Add formatter
        )
        parser.add_argument(
            "--bpm", type=int, default=127, help="Beats per minute"
        )
        parser.add_argument(
            "--num_measures",
            type=int,
            default=1,
            help="Number of measures to generate",
        )
        parser.add_argument(
            "--starting_measure",
            type=int,
            default=141,
            help="First measure number",
        )
        args: argparse.Namespace = parser.parse_args()

        process_and_save_audio(
            num_measures=args.num_measures,
            starting_measure=args.starting_measure,
            bpm=args.bpm,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        # Optionally, exit with a non-zero status code
        # import sys
        # sys.exit(1)
