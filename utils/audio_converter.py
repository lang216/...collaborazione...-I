from pydub import AudioSegment
import os
import logging
from pathlib import Path
from tqdm import tqdm
from .shared_utils import (
    setup_logging,
    validate_directory,
    get_audio_files,
    AudioScriptArgumentParser
)

def convert_to_wav(input_file, output_dir):
    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(input_file))
    
    # Load the audio file
    audio = AudioSegment.from_file(input_file, format=file_extension[1:])
    
    # Define the output file name
    output_file = os.path.join(output_dir, f"{file_name}.wav")
    
    # Export the audio file as .wav
    audio.export(output_file, format="wav")
    
    logging.info(f"File converted to {output_file}")

def process_directory(input_dir, output_dir):
    """Process all audio files in the directory and its subdirectories"""
    audio_files = get_audio_files(input_dir)
    
    for file_path in tqdm(audio_files, desc="Processing files"):
        try:
            convert_to_wav(file_path, output_dir)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

def main():
    parser = AudioScriptArgumentParser(
        description='Convert audio files to .wav format'
    )
    parser.add_argument('output_dir', type=str, help='Directory to save the converted .wav files')
    parser.add_argument('--preserve-original', action='store_true', help='Preserve original files')
    
    args = parser.parse_args()
    
    # Validate directories
    try:
        args.directory = validate_directory(args.directory)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Directory validation failed: {str(e)}")
        return
    
    # Configure logging
    setup_logging(args.log)
    logging.info(f"Starting audio conversion process")
    logging.info(f"Input directory: {args.directory}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Process directory
    process_directory(args.directory, args.output_dir)
    
    logging.info("Conversion complete!")

if __name__ == "__main__":
    main()
