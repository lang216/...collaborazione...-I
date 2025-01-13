# Audio Segmentation and Feature Analysis

This project provides tools for segmenting audio files based on musical features and clustering similar segments. It is specifically designed for a Piano+Electronics piece written for Yuseok Seol, aiming to analyze and organize the audio material for further composition and sound design.

## Features

- Audio segmentation based on onset detection
- Parallel feature extraction (MFCC, spectral features, chroma)
- Hierarchical clustering of audio segments
- Automatic organization of output files
- Memory-efficient processing with caching
- Progress tracking and logging

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-segmentation.git
cd audio-segmentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Place your audio files in the `input_audio` directory (WAV format recommended).

Run the segmentation:
```bash
python segmentation.py
```

Output will be saved in `output_segments` organized by:
- Piece name
- Feature type
- Cluster number

## Configuration

Edit `segmentation.py` to adjust:
- Audio processing parameters (sample rate, hop length, etc.)
- Number of clusters
- Input/output directories

## Dependencies

- Python 3.8+
- librosa
- numpy
- scikit-learn
- joblib
- soundfile
- tqdm

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
