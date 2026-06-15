import argparse
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def plot_label_correlations(data, label_col="label"):
    """Plot correlation of each column against the label using Plotly."""
    if data is None or data.empty:
        print("No data available to plot.")
        return None

    if label_col not in data.columns:
        label_col = data.columns[-1]
        print(f"Label column not found; using last column '{label_col}' instead.")

    df = data.copy()
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        df[label_col] = pd.factorize(df[label_col])[0]

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]

    corr = df.corr()[label_col].drop(label_col, errors="ignore")
    corr = corr.sort_values(key=lambda values: values.abs(), ascending=False)

    fig = px.bar(
        x=corr.index,
        y=corr.values,
        labels={"x": "Feature", "y": "Correlation with label"},
        title=f"Feature correlations with '{label_col}'",
        text=corr.round(3),
    )
    fig.update_layout(yaxis_title="Correlation", xaxis_tickangle=-45)
    fig.update_traces(
        marker_color=["crimson" if value < 0 else "steelblue" for value in corr.values]
    )
    fig.show()
    return fig


def show_column_distributions(data, include_columns=None, max_categories=20, save_dir="data_review"):
    """Show column distributions and save histogram + stats per column."""
    if data is None or data.empty:
        print("No data available to plot distributions.")
        return None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    include_columns = include_columns or data.columns.tolist()
    plot_columns = [col for col in include_columns if col in data.columns]
    if not plot_columns:
        print("No valid columns found to plot.")
        return None

    saved_files = []
    stats_list = []

    for col in plot_columns:
        series = data[col]
        if series.empty:
            print(f"Column '{col}' is empty.")
            continue

        series_non_null = series.dropna()
        stats = {
            "column": col,
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "unique": int(series_non_null.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series_non_null):
            describe = series_non_null.describe()
            stats.update(
                {
                    "mean": float(describe.get("mean", np.nan)),
                    "std": float(describe.get("std", np.nan)),
                    "min": float(describe.get("min", np.nan)),
                    "25%": float(describe.get("25%", np.nan)),
                    "50%": float(describe.get("50%", np.nan)),
                    "75%": float(describe.get("75%", np.nan)),
                    "max": float(describe.get("max", np.nan)),
                }
            )
            fig = go.Figure(
                data=[go.Histogram(x=series_non_null, marker_color="steelblue")],
                layout=dict(
                    title=f"{col} distribution",
                    xaxis_title=col,
                    yaxis_title="count",
                ),
            )
        else:
            counts = series_non_null.value_counts().nlargest(max_categories)
            top_values = counts.to_dict()
            stats.update({f"top_{i+1}": k for i, k in enumerate(top_values.keys())})
            stats.update({f"top_{i+1}_count": int(v) for i, v in enumerate(top_values.values())})
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=[str(v) for v in counts.index],
                        y=counts.values,
                        marker_color="steelblue",
                    )
                ],
                layout=dict(
                    title=f"{col} value counts",
                    xaxis_title=col,
                    yaxis_title="count",
                ),
            )

        fig_file = save_path / f"{col.replace(' ', '_')}_distribution.png"
        fig.write_image(str(fig_file), format="png")
        print(f"Saved distribution figure for '{col}' to {fig_file}")

        stats_file = save_path / f"{col.replace(' ', '_')}_stats.json"
        with stats_file.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats for '{col}' to {stats_file}")

        saved_files.append({
            "column": col,
            "figure": str(fig_file),
            "stats": str(stats_file),
        })
        stats_list.append(stats)

    summary_csv = save_path / "column_stats_summary.csv"
    pd.DataFrame(stats_list).to_csv(summary_csv, index=False)
    print(f"Saved column stats summary to {summary_csv}")

    return saved_files


def determine_feature_impact(data, label_col="label", method="mutual_info"):
    """Determine which columns have impact on the label column."""
    if data is None or data.empty:
        print("No data available for impact analysis.")
        return None

    if label_col not in data.columns:
        label_col = data.columns[-1]
        print(f"Label column not found; using last column '{label_col}' instead.")

    df = data.copy()

    # Encode all columns for analysis
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]

    X = df.drop(columns=[label_col])
    y = df[label_col]

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    if method == "mutual_info":
        # Mutual information: works for both numeric and categorical features
        mi_scores = mutual_info_classif(X, y, random_state=42)
        impact_df = pd.DataFrame(
            {"feature": X.columns, "mutual_information": mi_scores}
        ).sort_values("mutual_information", ascending=False)
        print("Feature Impact Analysis (Mutual Information):")
        print(impact_df.to_string(index=False))

        fig = px.bar(
            impact_df,
            x="mutual_information",
            y="feature",
            orientation="h",
            title="Feature Impact on Label (Mutual Information)",
            labels={"mutual_information": "Mutual Information Score"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        # fig.show()
        return impact_df

    elif method == "correlation":
        # Correlation: only for numeric features
        corr = df.corr()[label_col].drop(label_col, errors="ignore")
        corr = corr.sort_values(key=lambda values: values.abs(), ascending=False)
        impact_df = pd.DataFrame({"feature": corr.index, "correlation": corr.values})
        print("Feature Impact Analysis (Correlation):")
        print(impact_df.to_string(index=False))

        fig = px.bar(
            impact_df,
            x="correlation",
            y="feature",
            orientation="h",
            title="Feature Impact on Label (Correlation)",
            labels={"correlation": "Correlation with Label"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        # fig.show()
        return impact_df

    else:
        print(f"Unknown method: {method}. Use 'mutual_info' or 'correlation'.")
        return None


def drop_low_impact_columns(
    train_data, test_data, label_col="label", threshold=0.02, method="mutual_info"
):
    """Drop columns from train/test with low impact on the label."""
    if train_data is None or train_data.empty:
        raise ValueError("No train data provided.")
    if test_data is None or test_data.empty:
        raise ValueError("No test data provided.")

    impact_df = determine_feature_impact(train_data, label_col=label_col, method=method)
    if impact_df is None or impact_df.empty:
        print("No impact data returned; skipping drop.")
        return train_data.copy(), test_data.copy(), []

    if method == "mutual_info":
        if "mutual_information" not in impact_df.columns:
            raise ValueError("Impact DataFrame missing 'mutual_information' column.")
        low_impact = impact_df[impact_df["mutual_information"] < threshold][
            "feature"
        ].tolist()
    elif method == "correlation":
        if "correlation" not in impact_df.columns:
            raise ValueError("Impact DataFrame missing 'correlation' column.")
        low_impact = impact_df[impact_df["correlation"].abs() < threshold][
            "feature"
        ].tolist()
    else:
        raise ValueError(f"Unknown method: {method}")

    if not low_impact:
        print(f"No columns below {threshold} impact threshold.")
        return train_data.copy(), test_data.copy(), []

    print(f"Dropping low-impact columns: {low_impact}")
    train_clean = drop_columns(train_data.copy(), low_impact)
    test_clean = drop_columns(test_data.copy(), low_impact)

    return train_clean, test_clean, low_impact


def tune_random_forest_hyperparams(
    train_data,
    test_data,
    label_col="label",
    n_iter=50,
    cv=3,
    random_state=42,
    param_distributions=None,
    save_dir="model_outputs",
):
    """Run randomized search to tune Random Forest hyperparameters.

    Returns: best_estimator, best_params, results_df, model_path
    """
    if train_data is None or train_data.empty:
        raise ValueError("No train data provided for tuning.")

    if label_col not in train_data.columns:
        label_col = train_data.columns[-1]

    # combine to ensure consistent encoding
    combined = (
        pd.concat([train_data, test_data], axis=0, ignore_index=True)
        if test_data is not None
        else train_data.copy()
    )
    combined_encoded = one_hot_encode_numeric_columns(
        combined,
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    combined_encoded = one_hot_encode_strings(
        combined_encoded, exclude_cols=[label_col]
    )

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()

    if label_col not in train_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X = train_encoded.drop(columns=[label_col])
    y = train_encoded[label_col]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    # Default search space
    if param_distributions is None:
        param_distributions = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [None, "sqrt", "log2"],
        }

    clf = RandomForestClassifier(random_state=random_state)
    rsearch = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    rsearch.fit(X, y)

    best = rsearch.best_estimator_
    results_df = pd.DataFrame(rsearch.cv_results_)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "random_forest_tuned_model.joblib"
    joblib.dump(best, model_path)
    results_csv = out_dir / "random_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    print(f"Saved tuned model to {model_path}")
    print(f"Saved search results to {results_csv}")

    return best, rsearch.best_params_, results_df, model_path


def tune_logistic_regression_hyperparams(
    train_data,
    test_data,
    label_col="label",
    n_iter=20,
    cv=3,
    random_state=42,
    param_distributions=None,
    save_dir="model_outputs",
):
    """Run randomized search to tune Logistic Regression hyperparameters.

    Returns: best_estimator, best_params, results_df, model_path
    """
    if train_data is None or train_data.empty:
        raise ValueError("No train data provided for tuning.")

    if label_col not in train_data.columns:
        label_col = train_data.columns[-1]

    # combine to ensure consistent encoding
    combined = (
        pd.concat([train_data, test_data], axis=0, ignore_index=True)
        if test_data is not None
        else train_data.copy()
    )
    combined_encoded = one_hot_encode_numeric_columns(
        combined,
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    combined_encoded = one_hot_encode_strings(combined_encoded, exclude_cols=[label_col])

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()

    if label_col not in train_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X = train_encoded.drop(columns=[label_col])
    y = train_encoded[label_col]
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    # Default search space
    if param_distributions is None:
        param_distributions = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l2", "l1", "elasticnet", "none"],
            "solver": ["liblinear", "saga", "lbfgs"],
            "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
            "max_iter": [200, 500, 1000],
        }

    clf = LogisticRegression(random_state=random_state)
    # Some parameter combinations are invalid for certain solvers/penalties.
    # Use error_score=np.nan so failed fits are recorded and search continues.
    rsearch = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        error_score=np.nan,
    )

    rsearch.fit(X, y)

    best = rsearch.best_estimator_
    results_df = pd.DataFrame(rsearch.cv_results_)

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "logistic_regression_tuned_model.joblib"
    joblib.dump(best, model_path)
    results_csv = out_dir / "logistic_random_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    print(f"Saved tuned logistic regression model to {model_path}")
    print(f"Saved search results to {results_csv}")

    return best, rsearch.best_params_, results_df, model_path


def one_hot_encode_strings(data, exclude_cols=None):
    """One-hot encode all string columns in the DataFrame."""
    if data is None or data.empty:
        print("No data available to encode.")
        return data

    exclude_cols = exclude_cols or []
    obj_cols = [
        col
        for col in data.columns
        if col not in exclude_cols
        and (pd.api.types.is_string_dtype(data[col]) or data[col].dtype == object)
    ]
    if not obj_cols:
        print("No string columns found to one-hot encode.")
        return data

    print(f"One-hot encoding columns: {obj_cols}")
    encoded = pd.get_dummies(data, columns=obj_cols, dummy_na=False, drop_first=False)
    return encoded


def group_education_levels(df, col="education", new_col="education_group"):
    """Group education values into broader categories.

    Categories:
    - "Didn't finish High School"
    - "Finished High School"
    - "Finished Associates"
    - "Finished Bachelors"
    - "Has Grad Degree"

    Adds a new column `new_col` to the DataFrame and returns the DataFrame.
    """
    if df is None:
        return df
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")

    # Map exact values from the CSV to the requested groups
    dont_finish = {
        "less than 1st grade",
        "1st 2nd 3rd or 4th grade",
        "5th or 6th grade",
        "7th and 8th grade",
        "9th grade",
        "10th grade",
        "11th grade",
        "12th grade no diploma",
        "children",
    }

    finished_high = {"high school graduate", "some college but no degree"}

    finished_associates = {
        "associates degree-academic program",
        "associates degree-occup /vocational",
    }

    finished_bachelors = {"bachelors degree(ba ab bs)"}

    grad = {
        "masters degree(ma ms meng med msw mba)",
        "doctorate degree(phd edd)",
        "prof school degree (md dds dvm llb jd)",
    }

    def _map_edu(val):
        if pd.isna(val):
            return np.nan
        v = str(val).strip()
        key = v.lower()

        if key in dont_finish:
            return "Didn't finish High School"
        if key in finished_high:
            return "Finished High School"
        if key in finished_associates:
            return "Finished Associates"
        if key in finished_bachelors:
            return "Finished Bachelors"
        if key in grad:
            return "Has Grad Degree"

        # fallback heuristics
        return "Finished High School"

    df[new_col] = df[col].apply(_map_edu)
    return df


def one_hot_encode_numeric_columns(data, numeric_cols=None, exclude_cols=None):
    """One-hot encode only the specified numeric columns in the DataFrame."""
    if data is None or data.empty:
        print("No data available to encode.")
        return data

    if numeric_cols is None:
        print("No numeric columns provided to one-hot encode.")
        return data

    exclude_cols = exclude_cols or []
    selected_cols = [
        col for col in numeric_cols if col in data.columns and col not in exclude_cols
    ]
    if not selected_cols:
        print("No matching numeric columns found to one-hot encode.")
        return data

    print(f"One-hot encoding numeric columns: {selected_cols}")
    encoded = pd.get_dummies(
        data, columns=selected_cols, dummy_na=False, drop_first=False
    )
    return encoded


def drop_columns(data, columns_to_drop):
    """Drop a list of columns from the DataFrame."""
    if data is None or data.empty:
        print("No data available to drop columns from.")
        return data

    if not columns_to_drop:
        print("No columns specified for dropping.")
        return data

    existing_cols = [col for col in columns_to_drop if col in data.columns]
    if not existing_cols:
        print("No matching columns found to drop.")
        return data

    print(f"Dropping columns: {existing_cols}")
    return data.drop(columns=existing_cols)


def train_random_forest(
    train_data, test_data, label_col="label", random_state=42, n_estimators=100
):
    """Train a Random Forest classifier using pre-split train and test DataFrames."""
    if train_data is None or train_data.empty:
        raise ValueError("No train data available to train.")
    if test_data is None or test_data.empty:
        raise ValueError("No test data available to evaluate.")

    if label_col not in train_data.columns:
        if label_col in test_data.columns:
            print(
                f"Label column '{label_col}' not found in train data but present in test data."
            )
        else:
            label_col = train_data.columns[-1]
            print(
                f"Label column not found; using last train column '{label_col}' instead."
            )

    combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    combined_encoded = one_hot_encode_numeric_columns(
        combined,
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    combined_encoded = one_hot_encode_strings(
        combined_encoded, exclude_cols=[label_col]
    )

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()
    test_encoded = combined_encoded.iloc[len(train_data) :].copy()

    if label_col not in train_encoded.columns or label_col not in test_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X_train = train_encoded.drop(columns=[label_col])
    y_train = train_encoded[label_col]
    X_test = test_encoded.drop(columns=[label_col])
    y_test = test_encoded[label_col]

    label_mapping = None
    if not pd.api.types.is_numeric_dtype(y_train):
        combined_labels = pd.concat([y_train, y_test], ignore_index=True)
        combined_codes, uniques = pd.factorize(combined_labels)
        y_train = pd.Series(combined_codes[: len(y_train)], index=y_train.index)
        y_test = pd.Series(combined_codes[len(y_train) :], index=y_test.index)
        label_mapping = [str(value) for value in uniques]

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    model_dir = Path("model_outputs")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path}")

    tree_image_path = model_dir / "random_forest_tree.png"
    fig = plt.figure(figsize=(24, 16))
    plot_tree(
        model.estimators_[0],
        feature_names=X_train.columns,
        class_names=[str(c) for c in np.unique(y_train)],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("Random Forest Example Tree")
    plt.tight_layout()
    fig.savefig(tree_image_path, dpi=150)
    plt.close(fig)
    print(f"Saved tree visualization to {tree_image_path}")

    importance_image_path = model_dir / "feature_importances.png"
    feature_importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)
    fig2, ax = plt.subplots(figsize=(10, 14))
    feature_importances.tail(25).plot(kind="barh", ax=ax)
    ax.set_title("Top 25 Feature Importances")
    ax.set_xlabel("Importance")
    fig2.tight_layout()
    fig2.savefig(importance_image_path, dpi=150)
    plt.close(fig2)
    print(f"Saved feature importance plot to {importance_image_path}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    metadata = {
        "feature_columns": X_train.columns.tolist(),
        "label_col": label_col,
        "label_mapping": label_mapping,
    }
    metadata_path = model_dir / "random_forest_model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    print(f"Random Forest accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    return (
        model,
        accuracy,
        report,
        model_path,
        tree_image_path,
        importance_image_path,
        metadata_path,
    )


def train_logistic_regression(
    train_data, test_data, label_col="label", random_state=42, max_iter=1000
):
    """Train a Logistic Regression classifier using pre-split train and test DataFrames."""
    if train_data is None or train_data.empty:
        raise ValueError("No train data available to train.")
    if test_data is None or test_data.empty:
        raise ValueError("No test data available to evaluate.")

    if label_col not in train_data.columns:
        if label_col in test_data.columns:
            print(
                f"Label column '{label_col}' not found in train data but present in test data."
            )
        else:
            label_col = train_data.columns[-1]
            print(
                f"Label column not found; using last train column '{label_col}' instead."
            )

    combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    combined_encoded = one_hot_encode_numeric_columns(
        combined,
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    combined_encoded = one_hot_encode_strings(
        combined_encoded, exclude_cols=[label_col]
    )

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()
    test_encoded = combined_encoded.iloc[len(train_data) :].copy()

    if label_col not in train_encoded.columns or label_col not in test_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X_train = train_encoded.drop(columns=[label_col])
    y_train = train_encoded[label_col]
    X_test = test_encoded.drop(columns=[label_col])
    y_test = test_encoded[label_col]

    label_mapping = None
    if not pd.api.types.is_numeric_dtype(y_train):
        combined_labels = pd.concat([y_train, y_test], ignore_index=True)
        combined_codes, uniques = pd.factorize(combined_labels)
        y_train = pd.Series(combined_codes[: len(y_train)], index=y_train.index)
        y_test = pd.Series(combined_codes[len(y_train) :], index=y_test.index)
        label_mapping = [str(value) for value in uniques]

    model = LogisticRegression(
        max_iter=max_iter, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)

    model_dir = Path("model_outputs")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "logistic_regression_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained logistic regression model to {model_path}")

    # Save coefficients as a visualization
    coefficients = pd.Series(
        np.abs(model.coef_[0]) if model.coef_.shape[0] == 1 else np.abs(model.coef_).mean(axis=0),
        index=X_train.columns,
    ).sort_values(ascending=True)
    coef_image_path = model_dir / "logistic_regression_coefficients.png"
    fig, ax = plt.subplots(figsize=(10, 14))
    coefficients.tail(25).plot(kind="barh", ax=ax)
    ax.set_title("Top 25 Feature Coefficients (Logistic Regression)")
    ax.set_xlabel("Absolute Coefficient Value")
    fig.tight_layout()
    fig.savefig(coef_image_path, dpi=150)
    plt.close(fig)
    print(f"Saved coefficient plot to {coef_image_path}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    metadata = {
        "feature_columns": X_train.columns.tolist(),
        "label_col": label_col,
        "label_mapping": label_mapping,
    }
    metadata_path = model_dir / "logistic_regression_model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    print(f"Logistic Regression accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    return (
        model,
        accuracy,
        report,
        model_path,
        coef_image_path,
        metadata_path,
    )


def train_xgboost(
    train_data,
    test_data,
    label_col="label",
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
):
    """Train an XGBoost classifier using pre-split train and test DataFrames."""
    if xgb is None:
        raise ImportError(
            "XGBoost is not installed. Install xgboost to use this training function."
        )
    if train_data is None or train_data.empty:
        raise ValueError("No train data available to train.")
    if test_data is None or test_data.empty:
        raise ValueError("No test data available to evaluate.")

    if label_col not in train_data.columns:
        if label_col in test_data.columns:
            print(
                f"Label column '{label_col}' not found in train data but present in test data."
            )
        else:
            label_col = train_data.columns[-1]
            print(
                f"Label column not found; using last train column '{label_col}' instead."
            )

    combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)

    combined_encoded = one_hot_encode_numeric_columns(
        combined,
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    combined_encoded = one_hot_encode_strings(
        combined_encoded, exclude_cols=[label_col]
    )

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()
    test_encoded = combined_encoded.iloc[len(train_data) :].copy()

    if label_col not in train_encoded.columns or label_col not in test_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X_train = train_encoded.drop(columns=[label_col])
    y_train = train_encoded[label_col]
    X_test = test_encoded.drop(columns=[label_col])
    y_test = test_encoded[label_col]

    sanitized_columns = sanitize_feature_names(X_train.columns.tolist())
    X_train.columns = sanitized_columns
    X_test.columns = sanitized_columns

    label_mapping = None
    if not pd.api.types.is_numeric_dtype(y_train):
        combined_labels = pd.concat([y_train, y_test], ignore_index=True)
        combined_codes, uniques = pd.factorize(combined_labels)
        y_train = pd.Series(combined_codes[: len(y_train)], index=y_train.index)
        y_test = pd.Series(combined_codes[len(y_train) :], index=y_test.index)
        label_mapping = [str(value) for value in uniques]

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    model_dir = Path("model_outputs")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained XGBoost model to {model_path}")

    importance_image_path = model_dir / "xgboost_feature_importances.png"
    feature_importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 14))
    feature_importances.tail(25).plot(kind="barh", ax=ax)
    ax.set_title("Top 25 Feature Importances (XGBoost)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(importance_image_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {importance_image_path}")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    metadata = {
        "feature_columns": X_train.columns.tolist(),
        "label_col": label_col,
        "label_mapping": label_mapping,
        "model_type": "xgboost",
    }
    metadata_path = model_dir / "xgboost_model_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to {metadata_path}")

    print(f"XGBoost accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    return (
        model,
        accuracy,
        report,
        model_path,
        importance_image_path,
        metadata_path,
    )


def sanitize_feature_names(columns):
    """Return a sanitized list of feature names safe for XGBoost."""
    sanitized = []
    seen = {}
    for idx, col in enumerate(columns):
        name = str(col)
        name = name.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
        name = name.replace(" ", "_")
        if name == "":
            name = f"feature_{idx}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        sanitized.append(name)
    return sanitized


def load_model_and_metadata(model_path=None, metadata_path=None):
    """Load a saved Random Forest model and its feature metadata."""
    model_path = Path(model_path or "model_outputs/random_forest_model.joblib")
    metadata_path = Path(
        metadata_path or "model_outputs/random_forest_model_metadata.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Saved model metadata not found at {metadata_path}")

    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def align_features(data, feature_columns):
    """Align test data columns to the saved feature set."""
    if data is None or data.empty:
        return pd.DataFrame(columns=feature_columns)
    return data.reindex(columns=feature_columns, fill_value=0)


def run_saved_random_forest(
    test_data, model_path=None, metadata_path=None, label_col="label", output_path=None
):
    """Load a saved model and run it on provided test data."""
    if test_data is None or test_data.empty:
        raise ValueError("No test data provided for running the saved model.")

    original_input = test_data.reset_index(drop=True).copy()

    model, metadata = load_model_and_metadata(model_path, metadata_path)
    feature_columns = metadata.get("feature_columns")
    saved_label_col = metadata.get("label_col", label_col)
    label_mapping = metadata.get("label_mapping")

    if label_col not in test_data.columns and saved_label_col in test_data.columns:
        label_col = saved_label_col
    elif label_col not in test_data.columns:
        label_col = test_data.columns[-1]
        print(f"Label column not found; using last column '{label_col}' instead.")

    test_encoded = one_hot_encode_numeric_columns(
        test_data.copy(),
        numeric_cols=["industry code", "occupation code", "veterans benefits"],
        exclude_cols=[label_col],
    )
    test_encoded = one_hot_encode_strings(test_encoded, exclude_cols=[label_col])

    if label_col in test_encoded.columns:
        y_test_raw = test_encoded[label_col]
        X_test = test_encoded.drop(columns=[label_col], errors="ignore")
    else:
        y_test_raw = None
        X_test = test_encoded.copy()

    X_test = align_features(X_test, feature_columns)
    if X_test.empty:
        raise ValueError("No aligned test features available for prediction.")

    predictions = model.predict(X_test)
    results = {"predictions": predictions.tolist()}

    if label_mapping is not None:
        mapped_predictions = [
            label_mapping[int(pred)] if 0 <= int(pred) < len(label_mapping) else pred
            for pred in predictions
        ]
        results["predicted_labels"] = mapped_predictions

    output_path = Path(output_path or "model_outputs/run_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = original_input
    output_df["prediction"] = predictions
    if label_mapping is not None:
        output_df["predicted_label"] = mapped_predictions
    if y_test_raw is not None:
        output_df["true_label"] = y_test_raw.reset_index(drop=True)

    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    results["output_path"] = str(output_path)

    if y_test_raw is not None:
        if label_mapping is not None:
            mapping = {str(value): index for index, value in enumerate(label_mapping)}
            y_test = y_test_raw.astype(str).map(lambda x: mapping.get(x, np.nan))
        else:
            y_test = y_test_raw

        if y_test.isna().any():
            print(
                "Warning: some test labels were not in the saved label mapping; accuracy may not be available."
            )

        if pd.api.types.is_numeric_dtype(y_test) and not y_test.isna().any():
            accuracy = accuracy_score(y_test.astype(int), predictions)
            report = classification_report(
                y_test.astype(int), predictions, zero_division=0
            )
            results.update({"accuracy": accuracy, "report": report})
            print(f"Saved model accuracy on test data: {accuracy:.4f}")
            print("Classification report:\n", report)
        else:
            print(
                "Cannot compute accuracy because test labels are non-numeric or contain unseen classes."
            )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train, run, or tune machine learning models on the census dataset."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "run", "tune"],
        default="train",
        help="Choose whether to train, run, or tune the model.",
    )
    parser.add_argument(
        "--train-file",
        default="census_income_learn.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--test-file",
        default="census_income_test.csv",
        help="Path to the test CSV file.",
    )
    parser.add_argument(
        "--label-col", default="label", help="Name of the label column."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators when training random forest or XGBoost.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV iterations when tuning.",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "logistic", "xgboost"],
        default="rf",
        help="Which model to train or tune: 'rf', 'logistic', or 'xgboost'.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds when tuning.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=6,
        help="Maximum tree depth for XGBoost.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="Subsample ratio for XGBoost.",
    )
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=0.8,
        help="Colsample by tree ratio for XGBoost.",
    )
    args = parser.parse_args()

    # show_column_distributions(load_data(args.train_file))

    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)
    group_education_levels(train_data, col="education", new_col="education_group")
    group_education_levels(test_data, col="education", new_col="education_group")
    columns_to_drop = [
        "instance weight",
        "total person income",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        "wage per hour",
        "education"
    ]
    train_data = drop_columns(train_data, columns_to_drop=columns_to_drop)
    test_data = drop_columns(test_data, columns_to_drop=columns_to_drop)
    train_data, test_data, dropped_columns = drop_low_impact_columns(
        train_data,
        test_data,
        label_col=args.label_col,
        threshold=0.02,
        method="mutual_info",
    )

    if args.mode == "train":
        if args.model == "rf":
            train_random_forest(
                train_data,
                test_data,
                label_col=args.label_col,
                n_estimators=args.n_estimators,
            )
        elif args.model == "logistic":
            train_logistic_regression(
                train_data,
                test_data,
                label_col=args.label_col,
                max_iter=args.max_iter,
            )
        elif args.model == "xgboost":
            train_xgboost(
                train_data,
                test_data,
                label_col=args.label_col,
                n_estimators=args.n_estimators,
                max_depth=args.xgb_max_depth,
                learning_rate=args.xgb_learning_rate,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
            )
    elif args.mode == "run":
        run_saved_random_forest(
            test_data,
            label_col=args.label_col,
            output_path="model_outputs/run_predictions.csv",
        )
    elif args.mode == "tune":
        if args.model == "rf":
            tune_random_forest_hyperparams(
                train_data,
                test_data,
                label_col=args.label_col,
                n_iter=args.n_iter,
                cv=args.cv,
            )
        elif args.model == "logistic":
            tune_logistic_regression_hyperparams(
                train_data,
                test_data,
                label_col=args.label_col,
                n_iter=args.n_iter,
                cv=args.cv,
            )
        else:
            print(
                "Tuning for xgboost is not implemented yet. Use --mode train --model xgboost to train XGBoost."
            )
