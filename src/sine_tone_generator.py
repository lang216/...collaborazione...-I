import numpy as np
import scipy.io.wavfile as wavfile
import json

def generate_sine_tone(frequency, duration, sample_rate=44100, amplitude=0.5, amplitude_jitter_amount=0.0, jitter_frequency=5.0):
    """
    Generates a sine wave tone with smoother amplitude jitter and robust clipping prevention.
    """
    time = np.linspace(0, duration, int(sample_rate * duration), False)
    base_tone = amplitude * np.sin(2 * np.pi * frequency * time)

    if amplitude_jitter_amount > 0:
        jitter_rate = int(sample_rate / jitter_frequency)
        num_jitter_points = int(len(time) / jitter_rate) + 1
        jitter_envelope = np.random.uniform(-1, 1, num_jitter_points)
        amplitude_modulation = np.repeat(jitter_envelope, jitter_rate)[:len(time)]
        amplitude_modulation = 1 + amplitude_jitter_amount * amplitude_modulation
        tone = base_tone * amplitude_modulation
    else:
        tone = base_tone

    # Robust Normalization: Always normalize AFTER jitter to prevent clipping
    max_val = np.max(np.abs(tone))
    if max_val > 0: # Avoid division by zero if tone is silent
        tone = tone / max_val

    return tone

def save_wav(tone, filename, sample_rate=44100):
    """
    Saves the given tone as a WAV file.
    """
    int_tone = np.int16(tone * 32767)
    wavfile.write(filename, sample_rate, int_tone)
    print(f"Wave file saved as: {filename}")

def load_config(config_file_path="config.json"):
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at '{config_file_path}'.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{config_file_path}'.")
        return None

if __name__ == "__main__":
    config = load_config()

    if config:
        try:
            sample_rate = config.get("audio", {}).get("sr", 44100)
            sine_config = config.get("sine_tone_generator", {})
            frequency = sine_config.get("frequency", 440.0)
            duration = sine_config.get("duration", 5.0)
            amplitude_jitter_amount = sine_config.get("amplitude_jitter_amount", 0.0)
            jitter_frequency = sine_config.get("jitter_frequency", 5.0)
            filename = sine_config.get("output_filename", "generated_tone_config.wav")

            tone = generate_sine_tone(
                frequency=frequency,
                duration=duration,
                sample_rate=sample_rate,
                amplitude_jitter_amount=amplitude_jitter_amount,
                jitter_frequency=jitter_frequency
            )
            save_wav(tone, filename, sample_rate=sample_rate)

            print("Sine tone generated and saved successfully using config file settings!")

        except KeyError as e:
            print(f"Error: Missing configuration key: {e}.")
        except ValueError as e:
            print(f"Error: Invalid value in config file: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")