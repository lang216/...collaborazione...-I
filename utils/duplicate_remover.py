import os
import hashlib
import logging
import shutil
from collections import defaultdict
from .shared_utils import (
    setup_logging,
    validate_directory,
    AudioScriptArgumentParser
)

def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 hash of a file"""
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception as e:
        logging.error(f"Error reading {filepath}: {str(e)}")
        return None

def find_duplicates(directory):
    """Find duplicate files in directory"""
    files_by_size = defaultdict(list)
    duplicates = defaultdict(list)
    
    # First group files by size
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                file_size = os.path.getsize(filepath)
                files_by_size[file_size].append(filepath)
            except Exception as e:
                logging.warning(f"Could not get size of {filepath}: {str(e)}")
    
    # Then compare hashes for files with same size
    for size, filepaths in files_by_size.items():
        if len(filepaths) > 1:
            hashes = defaultdict(list)
            for filepath in filepaths:
                file_hash = calculate_md5(filepath)
                if file_hash:
                    hashes[file_hash].append(filepath)
            
            for hash_val, paths in hashes.items():
                if len(paths) > 1:
                    # Keep first file as original, others as duplicates
                    duplicates[paths[0]] = paths[1:]
    
    return duplicates

def copy_uniques(directory, duplicates):
    """Copy unique files to new folder with flat structure"""
    # Create output directory
    base_dir = os.path.dirname(str(directory).rstrip(os.sep))
    dir_name = os.path.basename(str(directory).rstrip(os.sep))
    output_dir = os.path.join(base_dir, f"{dir_name}_unique")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Created output directory: {output_dir}")
    except Exception as e:
        logging.error(f"Error creating output directory: {str(e)}")
        return
    
    # Get all files in directory
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # Filter out duplicates
    unique_files = set(all_files) - set([copy for copies in duplicates.values() for copy in copies])
    
    # Copy unique files
    copied_count = 0
    for filepath in unique_files:
        try:
            filename = os.path.basename(filepath)
            dest_path = os.path.join(output_dir, filename)
            
            # Handle filename conflicts
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                counter += 1
                
            shutil.copy2(filepath, dest_path)
            copied_count += 1
            logging.info(f"Copied {filepath} to {dest_path}")
        except Exception as e:
            logging.error(f"Error copying {filepath}: {str(e)}")
    
    logging.info(f"Total unique files copied: {copied_count}")

def main():
    parser = AudioScriptArgumentParser(
        description='Find duplicates and copy unique files to new folder'
    )
    args = parser.parse_args()
    
    # Validate directory
    try:
        validate_directory(args.directory)
    except Exception as e:
        logging.error(f"Directory validation failed: {str(e)}")
        return
    
    # Configure logging
    setup_logging(args.log)
    logging.info(f"Starting duplicate scan")
    logging.info(f"Directory: {args.directory}")
    
    duplicates = find_duplicates(args.directory)
    if duplicates:
        logging.info(f"Found {len(duplicates)} sets of duplicates")
        copy_uniques(args.directory, duplicates)
    else:
        logging.info("No duplicates found")
        # Copy all files since there are no duplicates
        copy_uniques(args.directory, {})
    
    logging.info("Duplicate removal process complete")

if __name__ == '__main__':
    main()
