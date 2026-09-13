import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# EXPECTED DATASET COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y"
]


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):
    """
    Clean and standardize the uploaded Bank Marketing dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset uploaded through Streamlit.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataset.
    """

    df = df.copy()

    # Remove accidental quotation marks from column names
    df.columns = (
        df.columns
        .astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )

    # Remove accidental quotation marks from string values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace('"', '', regex=False)
            .str.strip()
        )

    # Convert age to numeric
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Convert numerical columns to numeric
    numerical_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]

    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing values
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Validate that the uploaded dataset contains the
    columns required by the project.

    Returns
    -------
    valid : bool
    missing_columns : list
    """

    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    valid = len(missing_columns) == 0

    return valid, missing_columns


# ============================================================
# TARGET + MODEL FEATURES
# ============================================================

def prepare_model_data(df):
    """
    Prepare the uploaded dataset for prediction modelling.

    Follows the preprocessing logic used in the notebook:

    - Remove duration
    - Remove y_num
    - Remove age_group
    - Remove balance_group
    - Convert y into 0/1
    - Separate X and y
    """

    df = df.copy()

    # Create numerical target if target exists
    if "y" in df.columns:
        df["y_num"] = (
            df["y"]
            .astype(str)
            .str.lower()
            .map({"no": 0, "yes": 1})
        )

    # Remove variables not used for prediction
    columns_to_drop = [
        "duration",
        "y_num",
        "age_group",
        "balance_group"
    ]

    model_df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # Separate target
    if "y" in model_df.columns:

        y = (
            model_df["y"]
            .astype(str)
            .str.lower()
            .map({"no": 0, "yes": 1})
        )

        X = model_df.drop(columns=["y"])

    else:

        # Allows preprocessing to work for datasets
        # where the target is not supplied.
        y = None
        X = model_df.copy()

    return X, y


# ============================================================
# ONE-HOT ENCODING
# ============================================================

def encode_features(X):
    """
    One-hot encode categorical variables.

    This follows the notebook implementation:
    pd.get_dummies(..., drop_first=True, dtype=int)
    """

    X = X.copy()

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    numerical_cols = X.select_dtypes(
        exclude=["object", "category"]
    ).columns

    X_transformed = pd.get_dummies(
        X,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    # Ensure all modelling values are numeric
    X_transformed = X_transformed.astype("float32")

    return X_transformed


# ============================================================
# SCALING
# ============================================================

def scale_features(X_transformed):
    """
    Standardize the encoded features.

    Returns both the scaled dataframe and the fitted scaler.
    """

    scaler = StandardScaler()

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_transformed),
        columns=X_transformed.columns,
        index=X_transformed.index
    )

    return X_scaled, scaler


# ============================================================
# COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset(df):
    """
    Complete preprocessing pipeline.

    Used by Streamlit after the user uploads a dataset.

    Returns
    -------
    result : dict
        Contains:

        cleaned_df
        X
        y
        X_transformed
        X_scaled
        scaler
    """

    # 1. Clean
    cleaned_df = clean_data(df)

    # 2. Validate
    valid, missing_columns = validate_dataset(cleaned_df)

    if not valid:
        raise ValueError(
            "Uploaded dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # 3. Prepare modelling data
    X, y = prepare_model_data(cleaned_df)

    # 4. One-hot encode
    X_transformed = encode_features(X)

    # 5. Scale
    X_scaled, scaler = scale_features(X_transformed)

    return {
        "cleaned_df": cleaned_df,
        "X": X,
        "y": y,
        "X_transformed": X_transformed,
        "X_scaled": X_scaled,
        "scaler": scaler
    }