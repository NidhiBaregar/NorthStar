import pandas as pd
import numpy as np
import joblib


def load_segmentation_model(model_path):
    """
    Load the trained K-Means segmentation model.
    """
    return joblib.load(model_path)


def load_segmentation_scaler(scaler_path):
    """
    Load the scaler fitted during K-Means training.
    """
    return joblib.load(scaler_path)


def load_pca_model(pca_path):
    """
    Load the PCA model used for visualization.
    """
    return joblib.load(pca_path)


def align_features(X, model_features):
    """
    Make sure uploaded data has exactly the same
    features used when training K-Means.
    """
    X = X.copy()

    for feature in model_features:
        if feature not in X.columns:
            X[feature] = 0

    X_aligned = X[model_features]

    return X_aligned


def prepare_segmentation_features(
    X_transformed,
    segmentation_features
):
    """
    Prepare the one-hot encoded features for K-Means.

    The notebook excludes duration and day because
    they are not used for customer segmentation.
    """

    X_segmentation = X_transformed.copy()

    X_segmentation = X_segmentation.drop(
        columns=["y_num", "y", "duration", "day"],
        errors="ignore"
    )

    X_segmentation = align_features(
        X_segmentation,
        segmentation_features
    )

    return X_segmentation


def predict_segments(
    kmeans_model,
    segmentation_scaler,
    X_segmentation
):
    """
    Assign each customer to a K-Means cluster.
    """

    X_scaled = segmentation_scaler.transform(
        X_segmentation
    )

    cluster_labels = kmeans_model.predict(
        X_scaled
    )

    return cluster_labels, X_scaled


def add_segments_to_customer_data(
    df,
    cluster_labels
):
    """
    Add the predicted customer segment
    to the original customer data.
    """

    customer_data = df.copy()

    customer_data["Cluster"] = cluster_labels

    return customer_data


def add_pca_coordinates(
    customer_data,
    pca_model,
    X_scaled
):
    """
    Add PCA coordinates for visualization.
    """

    X_pca = pca_model.transform(X_scaled)

    customer_data = customer_data.copy()

    customer_data["PCA1"] = X_pca[:, 0]
    customer_data["PCA2"] = X_pca[:, 1]

    return customer_data


def run_segmentation_pipeline(
    df,
    X_transformed,
    kmeans_model,
    segmentation_scaler,
    segmentation_features,
    pca_model=None
):
    """
    Complete segmentation pipeline.

    Returns customer data containing:
    - Cluster
    - PCA1
    - PCA2 (if PCA model is supplied)
    """

    X_segmentation = prepare_segmentation_features(
        X_transformed=X_transformed,
        segmentation_features=segmentation_features
    )

    cluster_labels, X_scaled = predict_segments(
        kmeans_model=kmeans_model,
        segmentation_scaler=segmentation_scaler,
        X_segmentation=X_segmentation
    )

    customer_data = add_segments_to_customer_data(
        df=df,
        cluster_labels=cluster_labels
    )

    if pca_model is not None:
        customer_data = add_pca_coordinates(
            customer_data=customer_data,
            pca_model=pca_model,
            X_scaled=X_scaled
        )

    return customer_data