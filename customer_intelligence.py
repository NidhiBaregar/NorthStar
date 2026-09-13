import pandas as pd
import numpy as np


# ---------------------------------------------------------
# SEGMENT DEFINITIONS
# Based on the 4 K-Means clusters from the notebook
# ---------------------------------------------------------

SEGMENT_INFO = {
    0: {
        "segment_name": "Low-Engagement, Low-Value",
        "segment_type": "Low Priority",
        "description": "Lower-balance customers with low response propensity and limited prior engagement."
    },

    1: {
        "segment_name": "Moderate-Value, Low-Response",
        "segment_type": "Nurture",
        "description": "Moderate-value customers with repeated campaign exposure but relatively low response."
    },

    2: {
        "segment_name": "High-Potential, Engaged",
        "segment_type": "High Priority",
        "description": "Customers with the highest response propensity and stronger evidence of previous engagement."
    },

    3: {
        "segment_name": "High-Value, Untapped",
        "segment_type": "High Opportunity",
        "description": "Higher-balance customers with limited previous engagement and meaningful targeting potential."
    }
}


def add_segment_information(df):
    """
    Convert K-Means cluster numbers into
    business-friendly segment information.
    """

    customer_data = df.copy()

    customer_data["segment_name"] = (
        customer_data["Cluster"]
        .map(lambda x: SEGMENT_INFO.get(
            int(x),
            {}
        ).get("segment_name", "Unknown"))
    )

    customer_data["segment_type"] = (
        customer_data["Cluster"]
        .map(lambda x: SEGMENT_INFO.get(
            int(x),
            {}
        ).get("segment_type", "Unknown"))
    )

    customer_data["segment_description"] = (
        customer_data["Cluster"]
        .map(lambda x: SEGMENT_INFO.get(
            int(x),
            {}
        ).get("description", ""))
    )

    return customer_data


def calculate_customer_value(df):
    """
    Create customer-value indicators.

    For this dataset, balance is used as the
    primary monetary/value indicator.
    """

    customer_data = df.copy()

    customer_data["customer_value"] = customer_data["balance"]

    # Percentile rank makes customers easier to compare
    # across the entire uploaded dataset.
    customer_data["value_percentile"] = (
        customer_data["balance"]
        .rank(pct=True)
        .round(4)
    )

    return customer_data


def calculate_engagement_indicators(df):
    """
    Create simple behavioral indicators from
    campaign and previous-contact information.
    """

    customer_data = df.copy()

    if "campaign" in customer_data.columns:
        customer_data["campaign_intensity"] = (
            customer_data["campaign"]
        )

    if "previous" in customer_data.columns:
        customer_data["previous_engagement"] = (
            customer_data["previous"]
        )

    if "poutcome" in customer_data.columns:
        customer_data["previous_campaign_status"] = (
            customer_data["poutcome"]
        )

    return customer_data


def calculate_opportunity_score(df):
    """
    Combine predicted response probability and
    customer value into a normalized opportunity score.

    This is NOT the final marketing action.
    It is an analytical opportunity indicator.
    """

    customer_data = df.copy()

    probability_score = (
        customer_data["response_probability"]
        .clip(0, 1)
    )

    value_score = (
        customer_data["value_percentile"]
        .clip(0, 1)
    )

    customer_data["opportunity_score"] = (
        0.60 * probability_score +
        0.40 * value_score
    ) * 100

    customer_data["opportunity_score"] = (
        customer_data["opportunity_score"]
        .round(2)
    )

    return customer_data


def classify_opportunity(df):
    """
    Convert opportunity score into simple
    business priority bands.
    """

    customer_data = df.copy()

    customer_data["opportunity_band"] = pd.cut(
        customer_data["opportunity_score"],
        bins=[-np.inf, 33, 66, np.inf],
        labels=[
            "Low Opportunity",
            "Medium Opportunity",
            "High Opportunity"
        ]
    )

    return customer_data


def create_customer_intelligence_table(df):
    """
    Create the complete customer intelligence layer.

    Expected input columns:
        - Cluster
        - response_probability
        - predicted_response
        - balance
        - campaign
        - previous
        - poutcome

    Returns:
        Enriched customer-level intelligence dataframe.
    """

    customer_data = df.copy()

    customer_data = add_segment_information(
        customer_data
    )

    customer_data = calculate_customer_value(
        customer_data
    )

    customer_data = calculate_engagement_indicators(
        customer_data
    )

    customer_data = calculate_opportunity_score(
        customer_data
    )

    customer_data = classify_opportunity(
        customer_data
    )

    return customer_data


def run_customer_intelligence_pipeline(
    df,
    predictions,
    segmentation_data
):
    """
    Combine prediction and segmentation outputs
    into a single customer intelligence table.
    """

    customer_data = df.copy()

    # Add prediction results
    customer_data["response_probability"] = (
        predictions["response_probability"]
    )

    customer_data["predicted_response"] = (
        predictions["predicted_response"]
    )

    # Add segmentation results
    customer_data["Cluster"] = (
        segmentation_data["Cluster"]
    )

    # Add PCA coordinates if available
    if "PCA1" in segmentation_data.columns:
        customer_data["PCA1"] = (
            segmentation_data["PCA1"]
        )

    if "PCA2" in segmentation_data.columns:
        customer_data["PCA2"] = (
            segmentation_data["PCA2"]
        )

    # Build intelligence layer
    customer_data = create_customer_intelligence_table(
        customer_data
    )

    return customer_data