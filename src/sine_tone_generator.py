import numpy as np
import scipy.io.wavfile as wavfile
import json
import os

def generate_sine_tone(frequency, duration, sample_rate=44100, amplitude=0.8, 
                         amplitude_jitter_amount=0.0, jitter_frequency=5.0):
    """
    Generates a sine wave tone with smoother amplitude jitter and robust clipping prevention.
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

def save_wav(tone, filename, sample_rate=44100):
    """
    Saves the given tone as a WAV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Scale to full 16-bit range
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
            # Retrieve configuration values with defaults if not provided
            sample_rate = config.get("audio", {}).get("sr", 44100)
            sine_config = config.get("sine_tone_generator", {})
            frequency = sine_config.get("frequency", 440.0)
            duration = sine_config.get("duration", 5.0)
            amplitude_jitter_amount = sine_config.get("amplitude_jitter_amount", 0.0)
            jitter_frequency = sine_config.get("jitter_frequency", 5.0)
            
            # Use the output directory from config and join with filename
            output_dir = config.get("paths", {}).get("output_tone_file", "audio/Sine_Tones")
            filename = sine_config.get("output_filename", "generated_tone.wav")
            full_path = os.path.join(output_dir, filename)

            # Generate the tone with the optimized smoother jitter
            tone = generate_sine_tone(
                frequency=frequency,
                duration=duration,
                sample_rate=sample_rate,
                amplitude_jitter_amount=amplitude_jitter_amount,
                jitter_frequency=jitter_frequency
            )
            save_wav(tone, full_path, sample_rate=sample_rate)

            print("Sine tone generated and saved successfully using config file settings!")
            print(f"Full path: {os.path.abspath(full_path)}")

        except KeyError as e:
            print(f"Error: Missing configuration key: {e}.")
        except ValueError as e:
            print(f"Error: Invalid value in config file: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
