import pandas as pd
import numpy as np


# ============================================================
# 1. SEGMENT-BASED BUSINESS RULES
# ============================================================

SEGMENT_ACTIONS = {
    0: {
        "priority_weight": 0.35,
        "action": "Low-cost digital nurture",
        "channel": "Digital",
        "strategy": "Use automated and low-cost communication."
    },

    1: {
        "priority_weight": 0.50,
        "action": "Refine offer and messaging",
        "channel": "Email / Digital",
        "strategy": "Improve the proposition before increasing contact frequency."
    },

    2: {
        "priority_weight": 1.00,
        "action": "Priority sales follow-up",
        "channel": "Phone",
        "strategy": "Prioritize personalized follow-up because of strong response potential."
    },

    3: {
        "priority_weight": 0.85,
        "action": "Personalized high-value offer",
        "channel": "Phone / Relationship",
        "strategy": "Use personalized communication focused on higher-value customers."
    }
}


# ============================================================
# 2. CONTACT-FATIGUE RISK
# ============================================================

def calculate_fatigue_risk(df):
    """
    Estimate contact-fatigue risk using campaign intensity.

    campaign = number of contacts during the current campaign.

    Higher campaign frequency indicates greater risk of
    over-contacting the customer.
    """

    customer_data = df.copy()

    if "campaign" not in customer_data.columns:
        customer_data["fatigue_score"] = 0
        customer_data["fatigue_risk"] = "Unknown"
        return customer_data

    campaign = customer_data["campaign"].fillna(0)

    # Convert campaign intensity into a 0-100 score.
    fatigue_score = np.select(
        [
            campaign <= 2,
            campaign <= 4,
            campaign <= 6,
            campaign > 6
        ],
        [
            10,
            35,
            65,
            90
        ],
        default=10
    )

    customer_data["fatigue_score"] = fatigue_score

    customer_data["fatigue_risk"] = np.select(
        [
            customer_data["fatigue_score"] < 40,
            customer_data["fatigue_score"] < 70,
            customer_data["fatigue_score"] >= 70
        ],
        [
            "Low",
            "Medium",
            "High"
        ],
        default="Low"
    )

    return customer_data


# ============================================================
# 3. CONTACT RECOMMENDATION
# ============================================================

def calculate_contact_recommendation(df):
    """
    Recommend whether the customer should be contacted now,
    nurtured, or temporarily deprioritized.
    """

    customer_data = df.copy()

    customer_data["contact_recommendation"] = np.select(
        [
            (
                (customer_data["fatigue_risk"] == "High") &
                (customer_data["response_probability"] < 0.60)
            ),

            (
                (customer_data["fatigue_risk"] == "High") &
                (customer_data["response_probability"] >= 0.60)
            ),

            (
                customer_data["response_probability"] >= 0.60
            ),

            (
                customer_data["response_probability"] >= 0.30
            )
        ],
        [
            "Pause / Reduce Contact",
            "Use Selective Follow-up",
            "Contact Now",
            "Nurture"
        ],
        default="Low Priority"
    )

    return customer_data


# ============================================================
# 4. PRIORITY SCORE
# ============================================================

def calculate_priority_score(df):
    """
    Calculate an overall customer priority score.

    Components:
        Response probability = 50%
        Customer value       = 30%
        Segment opportunity  = 20%

    Fatigue is handled separately so that a valuable customer
    is not automatically treated as low-value simply because
    they have already received several contacts.
    """

    customer_data = df.copy()

    response_score = (
        customer_data["response_probability"]
        .clip(0, 1)
        * 100
    )

    value_score = (
        customer_data["value_percentile"]
        .clip(0, 1)
        * 100
    )

    segment_weight = (
        customer_data["Cluster"]
        .map(
            lambda x: SEGMENT_ACTIONS
            .get(int(x), {})
            .get("priority_weight", 0.50)
        )
    )

    segment_score = segment_weight * 100

    customer_data["priority_score"] = (
        0.50 * response_score +
        0.30 * value_score +
        0.20 * segment_score
    ).round(2)

    return customer_data


# ============================================================
# 5. PRIORITY TIER
# ============================================================

def classify_priority(df):
    """
    Convert priority score into business-friendly tiers.
    """

    customer_data = df.copy()

    customer_data["priority_tier"] = pd.cut(
        customer_data["priority_score"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=[
            "Low Priority",
            "Medium Priority",
            "High Priority"
        ]
    )

    return customer_data


# ============================================================
# 6. NEXT-BEST-ACTION ENGINE
# ============================================================

def calculate_next_best_action(df):
    """
    Generate a business recommendation for each customer.

    Fatigue takes precedence where repeated contact makes
    additional outreach undesirable.
    """

    customer_data = df.copy()

    conditions = [
        # High fatigue + low propensity
        (
            (customer_data["fatigue_risk"] == "High") &
            (customer_data["response_probability"] < 0.60)
        ),

        # High propensity
        (
            customer_data["response_probability"] >= 0.60
        ),

        # High-value customer
        (
            (customer_data["value_percentile"] >= 0.75) &
            (customer_data["response_probability"] >= 0.40)
        ),

        # Medium propensity
        (
            customer_data["response_probability"] >= 0.30
        )
    ]

    actions = [
        "Pause contact and avoid additional outreach",
        "Priority sales follow-up",
        "Personalized high-value offer",
        "Nurture with improved messaging"
    ]

    customer_data["next_best_action"] = np.select(
        conditions,
        actions,
        default="Low-cost digital nurture"
    )

    return customer_data


# ============================================================
# 7. RECOMMENDED CHANNEL
# ============================================================

def calculate_recommended_channel(df):
    """
    Recommend a communication channel based on
    the customer's next-best action and available
    contact information.
    """

    customer_data = df.copy()

    if "contact" in customer_data.columns:

        customer_data["recommended_channel"] = np.select(
            [
                (
                    customer_data["next_best_action"]
                    == "Pause contact and avoid additional outreach"
                ),

                (
                    customer_data["next_best_action"]
                    == "Priority sales follow-up"
                ),

                (
                    customer_data["next_best_action"]
                    == "Personalized high-value offer"
                )
            ],
            [
                "No immediate contact",
                "Phone",
                "Phone / Relationship"
            ],
            default="Digital"
        )

    else:

        customer_data["recommended_channel"] = np.select(
            [
                (
                    customer_data["next_best_action"]
                    == "Pause contact and avoid additional outreach"
                ),

                (
                    customer_data["next_best_action"]
                    == "Priority sales follow-up"
                ),

                (
                    customer_data["next_best_action"]
                    == "Personalized high-value offer"
                )
            ],
            [
                "No immediate contact",
                "Phone",
                "Phone / Relationship"
            ],
            default="Digital"
        )

    return customer_data


# ============================================================
# 8. COMPLETE BUSINESS LOGIC PIPELINE
# ============================================================

def run_business_logic(df):
    """
    Run the complete business decision layer.

    Input:
        Customer intelligence dataframe

    Output:
        Enriched dataframe containing:

        - fatigue_score
        - fatigue_risk
        - contact_recommendation
        - priority_score
        - priority_tier
        - next_best_action
        - recommended_channel
    """

    customer_data = df.copy()

    customer_data = calculate_fatigue_risk(
        customer_data
    )

    customer_data = calculate_contact_recommendation(
        customer_data
    )

    customer_data = calculate_priority_score(
        customer_data
    )

    customer_data = classify_priority(
        customer_data
    )

    customer_data = calculate_next_best_action(
        customer_data
    )

    customer_data = calculate_recommended_channel(
        customer_data
    )

    return customer_data