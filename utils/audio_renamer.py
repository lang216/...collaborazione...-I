import os
import shutil
import filetype
import logging
from pathlib import Path
from tqdm import tqdm
from .shared_utils import (
    setup_logging,
    validate_directory,
    get_audio_files,
    AudioScriptArgumentParser
)

# Supported audio formats
AUDIO_FORMATS = {
    'audio/wav': '.wav',
    'audio/x-wav': '.wav',
    'audio/mpeg': '.mp3',
    'audio/x-m4a': '.m4a',
    'audio/flac': '.flac',
    'audio/aac': '.aac',
    'audio/ogg': '.ogg'
}

def detect_file_type(file_path):
    """Detect the actual file type using filetype library"""
    kind = filetype.guess(file_path)
    return kind.mime if kind else None

def create_mirror_structure(source_dir, target_dir):
    """Create mirrored directory structure"""
    for root, dirs, _ in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        new_path = os.path.join(target_dir, relative_path)
        os.makedirs(new_path, exist_ok=True)

def process_audio_files(source_dir, target_dir):
    """Process audio files and rename with correct extensions"""
    processed_files = 0
    errors = 0
    
    # Get list of all audio files
    audio_files = get_audio_files(source_dir)
    
    # Process files with progress bar
    for file_path in tqdm(audio_files, desc="Processing files"):
        try:
            # Detect actual file type
            mime_type = detect_file_type(file_path)
            
            if mime_type and mime_type in AUDIO_FORMATS:
                # Create new file path with correct extension
                relative_path = os.path.relpath(file_path, source_dir)
                new_extension = AUDIO_FORMATS[mime_type]
                new_filename = os.path.splitext(relative_path)[0] + new_extension
                new_path = os.path.join(target_dir, new_filename)
                
                # Copy file to new location
                shutil.copy2(file_path, new_path)
                
                # Log the change
                logging.info(f"Renamed: {file_path} -> {new_path}")
                processed_files += 1
            else:
                logging.warning(f"Unsupported format: {file_path} ({mime_type})")
                errors += 1
        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            errors += 1
    
    return processed_files, errors

def main():
    parser = AudioScriptArgumentParser(
        description='Rename audio files with correct extensions'
    )
    parser.add_argument('target_dir', help='Directory to save renamed files')
    
    args = parser.parse_args()
    
    # Validate and setup directories
    validate_directory(args.directory)
    Path(args.target_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    setup_logging(args.log)
    logging.info(f"Starting audio renaming process in {args.directory}")
    
    # Create target directory structure
    logging.info("Creating directory structure...")
    create_mirror_structure(args.directory, args.target_dir)
    
    # Process files
    logging.info("Processing audio files...")
    processed, errors = process_audio_files(args.directory, args.target_dir)
    
    # Log summary
    logging.info(f"Processing complete!")
    logging.info(f"Files processed: {processed}")
    logging.info(f"Errors encountered: {errors}")

if __name__ == "__main__":
    main()
