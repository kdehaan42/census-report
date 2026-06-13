import argparse
import joblib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import plot_tree


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


def show_column_distributions(data, include_columns=None, max_categories=20):
    """Show column distributions and help identify outliers."""
    if data is None or data.empty:
        print("No data available to plot distributions.")
        return None

    include_columns = include_columns or data.columns.tolist()
    for col in include_columns:
        if col not in data.columns:
            print(f"Skipping missing column: {col}")
            continue

        series = data[col].dropna()
        if series.empty:
            print(f"Column '{col}' is empty after dropping nulls.")
            continue

        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = series[(series < lower) | (series > upper)]
            print(
                f"{col}: numeric distribution, {len(outliers)} outliers, "
                f"IQR={iqr:.3f}, lower={lower:.3f}, upper={upper:.3f}"
            )
            fig = px.histogram(
                data,
                x=col,
                nbins=50,
                marginal="box",
                title=f"{col} distribution with box plot",
                labels={col: col},
            )
            fig.update_layout(bargap=0.1)
            # fig.show()

        else:
            counts = series.value_counts().nlargest(max_categories)
            print(f"{col}: categorical distribution, {series.nunique()} unique values")
            fig = px.bar(
                x=counts.index.astype(str),
                y=counts.values,
                labels={"x": col, "y": "count"},
                title=f"{col} value counts (top {len(counts)})",
            )
            fig.update_layout(xaxis_tickangle=-45)
            # fig.show()
    return True


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
        description="Train or run the saved Random Forest model."
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
        help="Number of estimators when training the Random Forest.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of RandomizedSearchCV iterations when tuning.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds when tuning.",
    )
    args = parser.parse_args()

    train_data = load_data(args.train_file)
    test_data = load_data(args.test_file)
    columns_to_drop = [
        "instance weight",
        "total person income",
        "capital gains",
        "capital losses",
        "divdends from stocks",
        "wage per hour",
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
        train_random_forest(
            train_data,
            test_data,
            label_col=args.label_col,
            n_estimators=args.n_estimators,
        )
    elif args.mode == "run":
        run_saved_random_forest(
            test_data,
            label_col=args.label_col,
            output_path="model_outputs/run_predictions.csv",
        )
    elif args.mode == "tune":
        tune_random_forest_hyperparams(
            train_data,
            test_data,
            label_col=args.label_col,
            n_iter=args.n_iter,
            cv=args.cv,
        )
