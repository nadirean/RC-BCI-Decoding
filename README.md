# RC-BCI-Decoding 

Reservoir Computing for Brain-Computer Interface motor imagery classification.

## Setup
```bash
uv sync
```

## Usage

Run comparison of different reservoir sizes:
```bash
python bci_reservoir_computing.py
```

## Project Structure

- `bci_reservoir_computing.py` - Main script for training ESN and comparing reservoir sizes
- `explore_data.ipynb` - Dataset exploration notebook
- `assets/` - Generated results (confusion matrices, performance plots)

## Dataset

BNCI2014-001 (BCI Competition IV 2a):
- 9 subjects, 22 EEG channels, 4 classes (left hand, right hand, feet, tongue)
- Automatically downloaded on first run (~200MB)

## Results

The script trains Echo State Networks with different reservoir sizes (100, 200, 500, 1000 units) and generates:
- Confusion matrices
- Accuracy comparison plots
- Classification reports

## Authors
- Wojciech Bartoszek
- Dawid Woźniak
- Mateusz Oracz
- Jerzy Boksa

