# Census Income Model

This repository contains a small Python project for working with a Census income dataset.

It includes:
- `main.py`: training, running, and tuning a Random Forest model using pre-split train/test CSV data
- `add_csv_headers.py`: utility for applying column headers from `census_income_metadata.txt` to CSV files
- `census_income_learn.csv` and `census_income_test.csv`: dataset files
- `census_income_metadata.txt`: metadata file with column names

## Features

- load and preprocess census CSV data
- drop low-impact columns based on mutual information
- one-hot encode string and selected numeric columns
- train a `RandomForestClassifier`
- save trained model and generated visualizations
- tune hyperparameters via `RandomizedSearchCV`
- load and run a saved model on test data

## Requirements

Recommended Python 3.10+.

Install dependencies with:

```bash
python3 -m pip install pandas numpy matplotlib plotly scikit-learn joblib
```

If you use the included `env/` virtual environment, activate it first.

## Usage

### Train the model

```bash
python3 main.py --mode train
```

### Run the saved model on the test dataset

```bash
python3 main.py --mode run
```

### Tune hyperparameters

```bash
python3 main.py --mode tune --n-iter 20 --cv 3
```

### Custom files and label column

```bash
python3 main.py --mode train --train-file census_income_learn.csv --test-file census_income_test.csv --label-col label
```

## Column header utility

If your CSV files are missing headers, use `add_csv_headers.py` to prepend headers from the metadata file.

```bash
python3 add_csv_headers.py
```

## Output files

Trained models and visualizations are written to `model_outputs/`.

## Notes

- The project currently uses `main.py` as the main entry point.
- The `--mode` argument controls whether the script trains, runs, or tunes the model.
