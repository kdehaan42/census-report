# Census Income Model

Tools for preprocessing, exploring, training and tuning classifiers on the Census income dataset.

Includes:
- [main.py](main.py): preprocessing, training, running, and tuning (Random Forest + Logistic Regression)
- [add_csv_headers.py](add_csv_headers.py): apply column headers from `census_income_metadata.txt` to CSVs
- census CSVs: `census_income_learn.csv`, `census_income_test.csv`

**Features**
- Load and preprocess CSV data (`load_data`, `one_hot_encode_strings`, `one_hot_encode_numeric_columns`).
- Group `education` into broader categories with `group_education_levels` (maps exact CSV values to four groups).
- Drop low-impact columns using mutual information (`drop_low_impact_columns`).
- Train models:
	- Random Forest (`train_random_forest`) — saves model, an example tree image, feature importances, and metadata.
	- Logistic Regression (`train_logistic_regression`) — saves model, coefficient plot, and metadata.
- Hyperparameter tuning via `RandomizedSearchCV`:
	- Random Forest: `tune_random_forest_hyperparams`
	- Logistic Regression: `tune_logistic_regression_hyperparams`
- Run a saved model on test data and export predictions to `model_outputs/run_predictions.csv` (`run_saved_random_forest`).
- Generate per-column distribution plots and stats saved to the `data_review/` folder (`show_column_distributions`).

**Outputs (examples)**
- `model_outputs/random_forest_model.joblib`, `random_forest_model_metadata.json`
- `model_outputs/random_forest_tuned_model.joblib`, `random_search_results.csv`
- `model_outputs/logistic_regression_model.joblib`, `logistic_regression_model_metadata.json`
- `model_outputs/logistic_regression_tuned_model.joblib`, `logistic_random_search_results.csv`
- `model_outputs/run_predictions.csv` (original inputs + predictions + true labels when available)
- `data_review/` (per-column PNGs/JSON and `column_stats_summary.csv`)

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
python3 -m pip install pandas numpy matplotlib plotly scikit-learn joblib
```

## Usage examples

- Train both Random Forest and Logistic Regression (default label column unless overridden):

```bash
python3 main.py --mode train --train-file census_income_learn.csv --test-file census_income_test.csv
```

- Run saved Random Forest model on test file and write predictions:

```bash
python3 main.py --mode run --test-file census_income_test.csv
```

- Tune Random Forest hyperparameters:

```bash
python3 main.py --mode tune --model rf --n-iter 50 --cv 3
```

- Tune Logistic Regression hyperparameters:

```bash
python3 main.py --mode tune --model logistic --n-iter 30 --cv 3
```

## About `group_education_levels`

The function maps exact `education` strings in the provided CSV to broader groups:

- "Didn't finish High School": Less than 1st grade, 1st 2nd 3rd or 4th grade, 5th or 6th grade, 7th and 8th grade, 9th grade, 10th grade, 11th grade, 12th grade no diploma, Children
- "Finished High School": High school graduate, Some college but no degree
- "Finished Associates": Associates degree-academic program, Associates degree-occup /vocational
- "Finished Bachelors": Bachelors degree(BA AB BS)
- "Has Grad Degree": Masters degree(MA MS MEng MEd MSW MBA), Doctorate degree(PhD EdD), Prof school degree (MD DDS DVM LLB JD)

If you want different groupings (e.g., treat "Some college but no degree" differently), edit `group_education_levels` in [main.py](main.py).

## Notes

- Use `--label-col` to specify a different target column name.
- Use `--model` with `--mode tune` to select which model to tune (`rf` or `logistic`).
- Tuning uses `RandomizedSearchCV` and may try parameter combinations that are incompatible; failed fits are recorded but do not stop the overall search.
- Output files and plots are written to `model_outputs/` and `data_review/`.

If you'd like, I can tighten the logistic search space to avoid invalid solver/penalty combinations.
