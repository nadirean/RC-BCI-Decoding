# RC-BCI-Decoding

Reservoir Computing for Brain-Computer Interface motor imagery classification. Echo State
Networks (ReservoirPy) decode 4-class motor imagery (left hand, right hand, feet, tongue) from
EEG epochs of the BNCI2014-001 dataset.

PDF report: [reports/report.pdf](reports/report.pdf)

## Setup

```bash
uv sync
```

## Usage

Run the main experiment (single ESN plus a reservoir-size comparison):
```bash
python bci_reservoir_computing.py
```

Hyperparameter tuning with Optuna (60/20/20 split, TPE sampler, median pruning) lives in
`notebooks/optuna_hyperparameter_tuning.ipynb`; dataset exploration in
`notebooks/explore_data.ipynb`.

## Project Structure

- `bci_reservoir_computing.py` - Main script: trains an ESN and compares reservoir sizes (500-2000 units)
- `notebooks/explore_data.ipynb` - Dataset exploration
- `notebooks/optuna_hyperparameter_tuning.ipynb` - Optuna hyperparameter search and final tuned model
- `assets/` - Generated results (confusion matrices, performance plots)
- `reports/report.pdf` - Project report

## Dataset

BNCI2014-001 (BCI Competition IV 2a):
- 9 subjects, 22 EEG channels, 4 classes (left hand, right hand, feet, tongue)
- Automatically downloaded via MOABB on first run (~200MB)
- License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Results

Subject 1, 576 trials, 80/20 stratified split (chance level 25%):

| Model | Test accuracy |
|---|---|
| ESN, 1000 units (lr=0.05, sr=0.99) | 39.66% |
| Optuna-tuned ESN (1400 units, sr=0.72) | 46.55% |

Both models decode tongue best and confuse the hand classes with each other and with feet;
full analysis in the [report](reports/report.pdf), figures in `assets/`.

## Authors

- Wojciech Bartoszek
- Dawid Woźniak
- Mateusz Oracz
- Jerzy Boksa
