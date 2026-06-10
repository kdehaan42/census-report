import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


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
    fig.update_traces(marker_color=["crimson" if value < 0 else "steelblue" for value in corr.values])
    fig.show()
    return fig


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


def train_random_forest(train_data, test_data, label_col="label", random_state=42, n_estimators=100):
    """Train a Random Forest classifier using pre-split train and test DataFrames."""
    if train_data is None or train_data.empty:
        raise ValueError("No train data available to train.")
    if test_data is None or test_data.empty:
        raise ValueError("No test data available to evaluate.")

    if label_col not in train_data.columns:
        if label_col in test_data.columns:
            print(f"Label column '{label_col}' not found in train data but present in test data.")
        else:
            label_col = train_data.columns[-1]
            print(f"Label column not found; using last train column '{label_col}' instead.")

    combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)
    combined_encoded = one_hot_encode_strings(combined, exclude_cols=[label_col])

    train_encoded = combined_encoded.iloc[: len(train_data)].copy()
    test_encoded = combined_encoded.iloc[len(train_data) :].copy()

    if label_col not in train_encoded.columns or label_col not in test_encoded.columns:
        raise KeyError(f"Label column '{label_col}' not found after encoding.")

    X_train = train_encoded.drop(columns=[label_col])
    y_train = train_encoded[label_col]
    X_test = test_encoded.drop(columns=[label_col])
    y_test = test_encoded[label_col]

    if not pd.api.types.is_numeric_dtype(y_train):
        y_combined = pd.factorize(pd.concat([y_train, y_test], ignore_index=True))[0]
        y_train = pd.Series(y_combined[: len(y_train)], index=y_train.index)
        y_test = pd.Series(y_combined[len(y_train) :], index=y_test.index)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Random Forest accuracy: {accuracy:.4f}")
    print("Classification report:\n", report)
    return model, accuracy, report


if __name__ == "__main__":
    train_data = load_data("census_income_learn.csv")
    test_data = load_data("census_income_test.csv")
    train_data.info()
    # plot_label_correlations(train_data)
    encoded_train = one_hot_encode_strings(train_data)
    train_random_forest(train_data, test_data)