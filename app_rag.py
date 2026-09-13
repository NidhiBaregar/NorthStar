
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from preprocessing import preprocess_dataset
from prediction import predict_response_probability
from segmentation import (
    prepare_segmentation_features,
    predict_segments,
)
from customer_intelligence import (
    create_customer_intelligence_table,
    SEGMENT_INFO,
)
from business_logic import run_business_logic, SEGMENT_ACTIONS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f6f7f9;
        color: #172033;
    }

    [data-testid="stSidebar"] {
        background: #294765;
    }

    [data-testid="stSidebar"] * {
        color: #e7eef5;
    }

    [data-testid="stHeader"] {
        background: #ffffff;
        height: 64px;
        border-bottom: 1px solid #e6eaf0;
    }

    [data-testid="stHeader"] button {
        color: #294765;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    .workspace-bar {
        background: #ffffff;
        border: 1px solid #e3e8ee;
        border-radius: 12px;
        padding: 10px 14px 4px 14px;
        margin: 0 0 24px 0;
    }

    .workspace-label {
        color: #52657a;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 1px;
        margin: 0 0 3px 2px;
    }

    .page-title {
        font-size: 32px;
        font-weight: 750;
        letter-spacing: -0.7px;
        margin-bottom: 4px;
        color: #111827;
    }

    .page-subtitle {
        color: #667085;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #172033;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .card {
        background: white;
        border: 1px solid #eaecf0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }

    .decision-card {
        background: white;
        border: 1px solid #dfe3e8;
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0 18px 0;
    }

    .decision-card-green {
        background: #f2f8f4;
        border: 1px solid #2f7d55;
        border-left: 5px solid #2f7d55;
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0 18px 0;
    }

    .decision-card-green .decision-label {
        color: #397153;
    }

    .decision-card-green .decision-value {
        color: #205c3b;
    }

    .segment-1 { border-top: 4px solid #55738f; }
    .segment-2 { border-top: 4px solid #75648c; }
    .segment-3 { border-top: 4px solid #4f8068; }
    .segment-4 { border-top: 4px solid #a1814f; }

    [data-testid="stFileUploader"] section {
        background: #f4f7fa;
        border-color: #9bb0c4;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] span {
        color: #c8d6e3 !important;
    }

    [data-testid="stFileUploader"] button {
        color: #dce6ef !important;
        border-color: #b8c7d5 !important;
    }

    .decision-label {
        color: #667085;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .decision-value {
        color: #111827;
        font-size: 22px;
        font-weight: 750;
    }

    .recommendation {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        border-radius: 10px;
        padding: 16px 18px;
        margin-top: 12px;
    }

    .recommendation-title {
        font-size: 13px;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: .5px;
        color: #1d4ed8;
        margin-bottom: 5px;
    }

    .recommendation-text {
        color: #172033;
        font-size: 15px;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        background: #eef2f6;
        color: #344054;
        margin-right: 5px;
    }

    .metric-label {
        color: #667085;
        font-size: 12px;
        margin-bottom: 2px;
    }

    .metric-value {
        color: #111827;
        font-size: 25px;
        font-weight: 750;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 14px;
        margin: 8px 0 18px 0;
    }

    .kpi-card-green {
        background: #f2f8f4;
        border: 1px solid #2f7d55;
    }

    .kpi-card-green .kpi-label,
    .kpi-card-green .kpi-value {
        color: #205c3b;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 14px;
        padding: 17px 18px;
        min-width: 0;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    }

    .kpi-label {
        color: #667085;
        font-size: 12px;
        font-weight: 650;
        line-height: 1.25;
        white-space: normal;
    }

    .kpi-value {
        color: #111827;
        font-size: 26px;
        line-height: 1.15;
        font-weight: 750;
        margin-top: 8px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    @media (max-width: 1000px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    .upload-box {
        background: #ffffff;
        border: 1px dashed #b8c0cc;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin-top: 20px;
    }

    .small-note {
        color: #667085;
        font-size: 13px;
        line-height: 1.5;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #eaecf0;
        padding: 14px;
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 9px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL ASSETS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

REQUIRED_FILES = [
    "xgboost_model.pkl",
    "model_features.pkl",
    "kmeans_model.pkl",
    "segmentation_scaler.pkl",
    "segmentation_features.pkl",
]

ASSETS = {}

try:
    ASSETS["model"] = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    ASSETS["model_features"] = joblib.load(MODEL_DIR / "model_features.pkl")
    ASSETS["kmeans"] = joblib.load(MODEL_DIR / "kmeans_model.pkl")
    ASSETS["seg_scaler"] = joblib.load(MODEL_DIR / "segmentation_scaler.pkl")
    ASSETS["seg_features"] = joblib.load(MODEL_DIR / "segmentation_features.pkl")
except Exception as e:
    st.error(
        "The model files could not be loaded. Please make sure the complete "
        "`models` folder is beside `app.py`."
    )
    st.exception(e)
    st.stop()


# ============================================================
# DATA VALIDATION / PIPELINE
# ============================================================

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y"
]


def validate_uploaded_data(df):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return False, missing
    return True, []


def build_customer_intelligence(df):
    processed = preprocess_dataset(df)

    probabilities = predict_response_probability(
        ASSETS["model"],
        processed["X_transformed"],
        ASSETS["model_features"],
    )

    X_seg = prepare_segmentation_features(
        processed["X_transformed"],
        ASSETS["seg_features"],
    )

    clusters, _ = predict_segments(
        ASSETS["kmeans"],
        ASSETS["seg_scaler"],
        X_seg,
    )

    customer_data = processed["cleaned_df"].copy()
    customer_data["response_probability"] = probabilities
    customer_data["predicted_response"] = (
        probabilities >= 0.50
    ).astype(int)
    customer_data["Cluster"] = clusters

    customer_data = create_customer_intelligence_table(customer_data)
    customer_data = run_business_logic(customer_data)

    return customer_data


def predict_kmeans_robust(kmeans_model, X_scaled):
    """
    Predict using the exact floating-point dtype used by the saved
    K-Means cluster centers. This avoids sklearn/Cython dtype mismatch
    errors when a model was trained/saved with float32 centers.
    """
    X_array = np.asarray(X_scaled)

    if hasattr(kmeans_model, "cluster_centers_"):
        center_dtype = np.asarray(kmeans_model.cluster_centers_).dtype
        if center_dtype.kind == "f":
            X_array = X_array.astype(center_dtype, copy=False)

    return kmeans_model.predict(X_array)


def format_pct(value):
    return f"{float(value) * 100:.1f}%"


def segment_number(cluster):
    return str (int(cluster) + 1)


def action_for_new_customer(probability, balance, campaign):
    """
    Apply the same business interpretation used in the interface to a
    newly entered customer. Model prediction remains the primary signal.
    """
    if campaign >= 7 and probability < 0.60:
        return (
            "Pause contact and avoid additional outreach",
            "No immediate contact",
            "High",
        )

    if probability >= 0.60:
        return (
            "Priority sales follow-up",
            "Phone",
            "Low" if campaign <= 2 else "Medium",
        )

    if balance >= 0 and probability >= 0.40:
        return (
            "Personalized high-value offer",
            "Phone / Relationship",
            "Low" if campaign <= 2 else "Medium",
        )

    if probability >= 0.30:
        return (
            "Nurture with improved messaging",
            "Digital",
            "Low" if campaign <= 2 else "Medium",
        )

    return (
        "Low-cost digital nurture",
        "Digital",
        "Low" if campaign <= 2 else "Medium",
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## Customer Intelligence")
    st.caption("AI-powered marketing decision support")

    st.markdown("---")

    uploaded = st.file_uploader(
        "1. Bank Marketing customer data",
        type=["csv"],
        help=(
            "Upload the semicolon-separated Bank Marketing customer dataset. "
            "This dataset powers customer prediction, segmentation and targeting."
        ),
        key="bank_marketing_upload",
    )

    st.markdown("---")

    campaign_uploaded = st.file_uploader(
        "2. Illustrative campaign data",
        type=["csv"],
        help=(
            "Upload the campaign-level CSV used by the Campaigns workspace."
        ),
        key="illustrative_campaign_upload",
    )


# ============================================================
# WORKSPACE NAVIGATION
# ============================================================

workspace_options = [
    "TARGET",
    "CUSTOMERS",
    "SEGMENTS",
    "CAMPAIGNS",
    "AI STRATEGIST",
]

st.markdown('<div class="workspace-bar">', unsafe_allow_html=True)
st.markdown('<div class="workspace-label">WORKSPACE</div>', unsafe_allow_html=True)
navigation = st.radio(
    "WORKSPACE",
    workspace_options,
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# UPLOAD-FIRST STATE
# ============================================================

if uploaded is None or campaign_uploaded is None:
    st.markdown('<div class="page-title">Load your data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Upload both datasets to activate the complete customer and campaign intelligence workspace."
        "</div>",
        unsafe_allow_html=True,
    )

    missing = []
    if uploaded is None:
        missing.append("Bank Marketing customer data")
    if campaign_uploaded is None:
        missing.append("Illustrative Campaign Data")

    st.info("Please upload: " + " and ".join(missing) + ".")
    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

try:
    raw_df = pd.read_csv(
        uploaded,
        sep=";",
        quotechar='"',
        quoting=3,
    )
except Exception as e:
    st.error("The uploaded CSV could not be read.")
    st.exception(e)
    st.stop()

raw_df.columns = raw_df.columns.astype(str).str.replace('"', '', regex=False).str.strip()

valid, missing_columns = validate_uploaded_data(raw_df)

if not valid:
    st.error("This dataset does not match the Bank Marketing model.")
    st.write("Missing required columns:")
    st.code(", ".join(missing_columns))
    st.stop()

try:
    customer_df = build_customer_intelligence(raw_df)
except Exception as e:
    st.error(
        "The uploaded dataset was recognized, but the model pipeline could not "
        "process it. Check that the values follow the Bank Marketing dataset format."
    )
    st.exception(e)
    st.stop()

# Campaign-level illustrative data is intentionally kept separate from the
# customer-level Bank Marketing dataset.
try:
    campaign_df = pd.read_csv(campaign_uploaded)
    campaign_df.columns = (
        campaign_df.columns.astype(str)
        .str.replace('"', '', regex=False)
        .str.strip()
    )
except Exception as e:
    st.error("The Illustrative Campaign Data could not be read.")
    st.exception(e)
    st.stop()

REQUIRED_CAMPAIGN_COLUMNS = [
    "campaign_id", "campaign_date", "campaign_name", "channel",
    "customers_contacted", "responses", "response_rate",
    "avg_contacts_per_customer", "high_fatigue_pct",
    "campaign_budget", "cost_per_response", "campaign_status",
]
missing_campaign = [c for c in REQUIRED_CAMPAIGN_COLUMNS if c not in campaign_df.columns]
if missing_campaign:
    st.error("The Illustrative Campaign Data is missing required columns.")
    st.code(", ".join(missing_campaign))
    st.stop()


# ============================================================
# TARGET
# ============================================================

def render_target():
    st.markdown('<div class="page-title">Target</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Turn a marketing objective into a prioritized audience."
        "</div>",
        unsafe_allow_html=True,
    )

    # Same decision layout for all four marketing objectives.
    controls_left, controls_right = st.columns([1.05, 1.95])

    with controls_left:
        objective = st.selectbox(
            "Marketing objective",
            [
                "Maximize conversions",
                "Acquire high-value customers",
                "Re-engage customers",
                "Reduce wasted outreach",
            ],
        )

    with controls_right:
        threshold = st.slider(
            "Minimum response propensity",
            min_value=0.10,
            max_value=0.90,
            value=0.40,
            step=0.05,
            format="%.2f",
        )

    exclude_fatigue = st.checkbox(
        "Exclude high-fatigue customers",
        value=True,
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    all_customers = customer_df.copy()
    eligible = all_customers.copy()

    if exclude_fatigue:
        eligible = eligible[eligible["fatigue_risk"] != "High"].copy()

    if objective == "Maximize conversions":
        work = eligible[eligible["response_probability"] >= threshold].copy()
        sort_cols = ["response_probability", "priority_score"]
        ascending = [False, False]
        decision_label = "Target customers"
        secondary_label = "Not selected"
        table_title = "Target customers"
        recommendation = (
            "Target customers with the highest predicted response propensity. "
            "Increase the threshold when campaign capacity is limited and "
            "conversion efficiency is more important than reach."
        )

    elif objective == "Acquire high-value customers":
        work = eligible[
            (eligible["response_probability"] >= threshold)
            & (eligible["value_percentile"] >= 0.75)
        ].copy()
        sort_cols = ["value_percentile", "response_probability"]
        ascending = [False, False]
        decision_label = "High-value customers"
        secondary_label = "Not selected"
        table_title = "Target customers"
        recommendation = (
            "Target customers in the top 25% of the balance distribution who "
            "also meet the response-propensity threshold. This prioritizes "
            "higher-value opportunities over broad reach."
        )

    elif objective == "Re-engage customers":
        work = eligible[
            (eligible["response_probability"] >= threshold)
            & (eligible["previous"] > 0)
        ].copy()
        sort_cols = ["response_probability", "previous"]
        ascending = [False, False]
        decision_label = "Customers to re-engage"
        secondary_label = "Not selected"
        table_title = "Customers to re-engage"
        recommendation = (
            "Re-engage customers with previous campaign engagement who still "
            "show sufficient response propensity. This focuses follow-up on "
            "customers with evidence of prior engagement."
        )

    else:
        # For wasted outreach, the primary decision is explicitly who NOT to contact.
        work = eligible[eligible["response_probability"] >= threshold].copy()
        sort_cols = ["response_probability", "priority_score"]
        ascending = [True, True]
        decision_label = "Customers not to target"
        secondary_label = "Customers to target"
        table_title = "Customers not to target"
        recommendation = (
            "Do not target customers who fall below the selected response "
            "threshold or are excluded because of high contact fatigue. "
            "Use the remaining audience for the campaign."
        )

    work = work.sort_values(sort_cols, ascending=ascending)

    if objective == "Reduce wasted outreach":
        decision_df = all_customers.loc[
            ~all_customers.index.isin(work.index)
        ].copy()

        decision_df["suppression_reason"] = np.select(
            [
                decision_df["fatigue_risk"] == "High",
                decision_df["response_probability"] < threshold,
            ],
            [
                "High contact fatigue",
                "Low response propensity",
            ],
            default="Below targeting criteria",
        )

        decision_df = decision_df.sort_values(
            ["fatigue_score", "response_probability"],
            ascending=[False, True],
        )

        target_audience = work
        decision_count = len(decision_df)
        secondary_count = len(target_audience)
        table_df = decision_df
    else:
        decision_df = work
        decision_count = len(decision_df)
        secondary_count = len(all_customers) - len(decision_df)
        target_audience = work
        table_df = decision_df

    expected_responders = (
        target_audience["response_probability"].sum()
        if len(target_audience) else 0
    )
    avg_propensity = (
        target_audience["response_probability"].mean()
        if len(target_audience) else 0
    )

    # Four KPI cards remain visually identical for every marketing objective.
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card kpi-card-green">
                <div class="kpi-label">{decision_label}</div>
                <div class="kpi-value">{decision_count:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">{secondary_label}</div>
                <div class="kpi-value">{secondary_count:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Expected responders</div>
                <div class="kpi-value">{expected_responders:,.0f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg response propensity</div>
                <div class="kpi-value">{format_pct(avg_propensity)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="recommendation">
            <div class="recommendation-title">Target recommendation</div>
            <div class="recommendation-text">{recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-title">{table_title}</div>',
        unsafe_allow_html=True,
    )

    if len(table_df) == 0:
        st.warning("No customers meet the selected marketing criteria.")
        return

    if objective == "Reduce wasted outreach":
        display = table_df[
            [
                "age", "job", "balance", "response_probability",
                "priority_score", "fatigue_risk", "Cluster",
                "suppression_reason"
            ]
        ].head(100).copy()

        display["Segment"] = display["Cluster"].apply(
            lambda x: str(int(x) + 1)
        )
        display["Response propensity"] = display[
            "response_probability"
        ].map(format_pct)
        display["Priority"] = display["priority_score"].round(1)

        display = display.rename(
            columns={
                "age": "Age",
                "job": "Job",
                "balance": "Balance",
                "fatigue_risk": "Fatigue",
                "suppression_reason": "Reason",
            }
        )

        display = display[
            [
                "Age", "Job", "Balance", "Response propensity",
                "Priority", "Fatigue", "Segment", "Reason"
            ]
        ]

    else:
        display = table_df[
            [
                "age", "job", "balance", "response_probability",
                "priority_score", "fatigue_risk", "Cluster",
                "next_best_action", "recommended_channel"
            ]
        ].head(100).copy()

        display["Segment"] = display["Cluster"].apply(
            lambda x: str(int(x) + 1)
        )
        display["Response propensity"] = display[
            "response_probability"
        ].map(format_pct)
        display["Priority"] = display["priority_score"].round(1)

        display = display.rename(
            columns={
                "age": "Age",
                "job": "Job",
                "balance": "Balance",
                "fatigue_risk": "Fatigue",
                "next_best_action": "Next best action",
                "recommended_channel": "Channel",
            }
        )

        display = display[
            [
                "Age", "Job", "Balance", "Response propensity",
                "Priority", "Fatigue", "Segment",
                "Next best action", "Channel"
            ]
        ]

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )



# ============================================================
# CUSTOMERS
# ============================================================

def render_existing_customer():
    st.markdown('<div class="section-title">Existing customer</div>', unsafe_allow_html=True)

    customer_index = st.selectbox(
        "Select customer",
        options=list(customer_df.index),
        format_func=lambda x: f"Customer {int(x) + 1}",
    )

    customer = customer_df.loc[customer_index]

    a, b, c, d = st.columns(4)
    a.metric("Response propensity", format_pct(customer["response_probability"]))
    b.metric("Financial value proxy", f"{customer['balance']:,.0f}")
    c.metric("Priority score", f"{customer['priority_score']:.1f}")
    d.metric("Fatigue risk", str(customer["fatigue_risk"]))

    st.markdown(
        f"""
        <div class="decision-card-green">
            <div class="decision-label">Recommended next action</div>
            <div class="decision-value">{customer['next_best_action']}</div>
            <span class="badge">{customer['recommended_channel']}</span>
            <span class="badge">{segment_number(customer['Cluster'])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)

    profile_cols = [
        "age", "job", "marital", "education", "balance",
        "housing", "loan", "contact", "campaign",
        "previous", "poutcome"
    ]

    profile = pd.DataFrame(
        {
            "Attribute": [c.replace("_", " ").title() for c in profile_cols],
            "Value": [customer[c] for c in profile_cols],
        }
    )

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"""
        <div class="recommendation">
            <div class="recommendation-title">Decision rationale</div>
            <div class="recommendation-text">
                The customer has a predicted response propensity of
                <b>{format_pct(customer['response_probability'])}</b>,
                belongs to <b>{segment_number(customer['Cluster'])}</b>,
                and currently carries <b>{customer['fatigue_risk'].lower()}</b>
                contact-fatigue risk. The recommended action is therefore
                <b>{customer['next_best_action'].lower()}</b> through
                <b>{customer['recommended_channel']}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def assess_new_customer():
    st.markdown('<div class="section-title">Assess a new customer</div>', unsafe_allow_html=True)

    with st.form("new_customer_form"):
        p1, p2, p3 = st.columns(3)

        with p1:
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox(
                "Job",
                [
                    "admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown"
                ],
            )
            marital = st.selectbox(
                "Marital status",
                ["married", "single", "divorced"],
            )
            education = st.selectbox(
                "Education",
                ["primary", "secondary", "tertiary", "unknown"],
            )
            default = st.selectbox("Default", ["no", "yes"])

        with p2:
            balance = st.number_input(
                "Account balance",
                value=1000,
                step=100,
            )
            housing = st.selectbox("Housing loan", ["no", "yes"])
            loan = st.selectbox("Personal loan", ["no", "yes"])
            contact = st.selectbox(
                "Contact type",
                ["cellular", "telephone", "unknown"],
            )
            day = st.number_input("Contact day", 1, 31, 15)
            month = st.selectbox(
                "Contact month",
                ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"],
            )

        with p3:
            duration = st.number_input(
                "Call duration",
                min_value=0,
                value=180,
                step=10,
            )
            campaign = st.number_input(
                "Contacts in current campaign",
                min_value=1,
                value=2,
                step=1,
            )
            pdays = st.number_input(
                "Days since previous contact",
                value=-1,
                step=1,
            )
            previous = st.number_input(
                "Previous contacts",
                min_value=0,
                value=0,
                step=1,
            )
            poutcome = st.selectbox(
                "Previous campaign outcome",
                ["unknown", "failure", "other", "success"],
            )

        submitted = st.form_submit_button(
            "Assess customer",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    new_row = pd.DataFrame(
        [
            {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan,
                "contact": contact,
                "day": day,
                "month": month,
                "duration": duration,
                "campaign": campaign,
                "pdays": pdays,
                "previous": previous,
                "poutcome": poutcome,
                "y": "no",
            }
        ]
    )

    try:
        processed = preprocess_dataset(new_row)

        probability = float(
            predict_response_probability(
                ASSETS["model"],
                processed["X_transformed"],
                ASSETS["model_features"],
            )[0]
        )

        X_seg = prepare_segmentation_features(
            processed["X_transformed"],
            ASSETS["seg_features"],
        )

        X_seg_scaled = ASSETS["seg_scaler"].transform(X_seg)
        clusters = predict_kmeans_robust(
            ASSETS["kmeans"],
            X_seg_scaled,
        )

        cluster = int(clusters[0])

        # Value percentile relative to the uploaded dataset.
        value_percentile = float(
            (customer_df["balance"] <= balance).mean()
        )

        segment_weight = SEGMENT_ACTIONS.get(
            cluster, {}
        ).get("priority_weight", 0.50)

        priority_score = (
            0.50 * probability * 100
            + 0.30 * value_percentile * 100
            + 0.20 * segment_weight * 100
        )

        action, channel, fatigue = action_for_new_customer(
            probability,
            balance,
            campaign,
        )

        st.session_state["new_customer_result"] = {
            "probability": probability,
            "cluster": cluster,
            "priority_score": priority_score,
            "fatigue": fatigue,
            "action": action,
            "channel": channel,
            "balance_percentile": value_percentile,
        }

    except Exception as e:
        st.error("The new customer could not be assessed.")
        st.exception(e)
        return

    result = st.session_state["new_customer_result"]

    st.markdown("---")
    st.markdown('<div class="section-title">Model assessment</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Response propensity", format_pct(result["probability"]))
    r2.metric("Segment", segment_number(result["cluster"]))
    r3.metric("Priority score", f"{result['priority_score']:.1f}")
    r4.metric("Fatigue risk", result["fatigue"])

    st.markdown(
        f"""
        <div class="decision-card-green">
            <div class="decision-label">Best way to target this customer</div>
            <div class="decision-value">{result['action']}</div>
            <span class="badge">{result['channel']}</span>
            <span class="badge">{segment_number(result['cluster'])}</span>
        </div>

        <div class="recommendation">
            <div class="recommendation-title">Why this recommendation?</div>
            <div class="recommendation-text">
                The model estimates a response propensity of
                <b>{format_pct(result['probability'])}</b>.
                The customer's balance falls around the
                <b>{result['balance_percentile'] * 100:.0f}th percentile</b>
                of the uploaded customer base. Combined with the assigned
                segment and current campaign contact level, the interface
                recommends <b>{result['action'].lower()}</b> using
                <b>{result['channel']}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_customers():
    st.markdown('<div class="page-title">Customers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Understand an individual customer and decide how to target them."
        "</div>",
        unsafe_allow_html=True,
    )

    customer_mode = st.radio(
        "CUSTOMER WORKSPACE",
        ["Existing Customer", "Assess New Customer"],
        horizontal=True,
    )

    if customer_mode == "Existing Customer":
        render_existing_customer()
    else:
        assess_new_customer()


# ============================================================
# SEGMENTS
# ============================================================

def render_segments():
    st.markdown('<div class="page-title">Segments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Understand who each customer group is and how marketing should approach them."
        "</div>",
        unsafe_allow_html=True,
    )

    summary = (
        customer_df
        .groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Avg_Balance=("balance", "mean"),
            Avg_Response=("response_probability", "mean"),
            Avg_Campaign=("campaign", "mean"),
        )
        .reset_index()
    )

    cols = st.columns(4)

    for i, cluster in enumerate(sorted(customer_df["Cluster"].unique())):
        row = summary[summary["Cluster"] == cluster].iloc[0]

        with cols[i]:
            st.markdown(
                f"""
                <div class="card segment-{int(cluster) + 1}">
                    <div class="decision-label">{segment_number(cluster)}</div>
                    <div style="font-size:18px;font-weight:750;margin-bottom:12px;">
                        {SEGMENT_INFO.get(int(cluster), {}).get("segment_type", "Segment")}
                    </div>
                    <div class="small-note">
                        {SEGMENT_INFO.get(int(cluster), {}).get("description", "")}
                    </div>
                    <hr style="border:none;border-top:1px solid #eaecf0;margin:15px 0;">
                    <b>{int(row['Customers']):,}</b> customers<br>
                    <b>{format_pct(row['Avg_Response'])}</b> avg response propensity<br>
                    <b>{row['Avg_Balance']:,.0f}</b> avg balance
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Segment comparison</div>', unsafe_allow_html=True)

    comparison = (
        customer_df
        .groupby("Cluster")
        .agg(
            Customers=("Cluster", "size"),
            Average_Balance=("balance", "mean"),
            Response_Propensity=("response_probability", "mean"),
            Campaign_Contacts=("campaign", "mean"),
            Previous_Contacts=("previous", "mean"),
        )
        .reset_index()
    )

    comparison["Segment"] = comparison["Cluster"].apply(segment_number)
    comparison["Response Propensity"] = comparison["Response_Propensity"].map(format_pct)
    comparison["Average Balance"] = comparison["Average_Balance"].round(0)
    comparison["Campaign Contacts"] = comparison["Campaign_Contacts"].round(1)
    comparison["Previous Contacts"] = comparison["Previous_Contacts"].round(1)

    comparison = comparison[
        [
            "Segment", "Customers", "Average Balance",
            "Response Propensity", "Campaign Contacts",
            "Previous Contacts"
        ]
    ]

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="recommendation">
            <div class="recommendation-title">How to use segments</div>
            <div class="recommendation-text">
                Use segments to adapt the marketing treatment, not simply to
                describe customers. The response model determines propensity;
                segmentation adds context for deciding how aggressively and
                through which channel to engage.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CAMPAIGNS
# ============================================================

def render_campaigns():
    st.markdown('<div class="page-title">Campaigns</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Review campaign performance and identify where outreach should change."
        "</div>",
        unsafe_allow_html=True,
    )

    # No campaign is pre-selected. Each filter is a true multi-select dropdown.
    f1, f2, f3 = st.columns([1.4, 1, 1])

    campaign_options = sorted(campaign_df["campaign_name"].dropna().unique().tolist())
    channel_options = sorted(campaign_df["channel"].dropna().unique().tolist())
    status_options = sorted(campaign_df["campaign_status"].dropna().unique().tolist())

    with f1:
        selected_campaigns = st.multiselect(
            "Campaign",
            campaign_options,
            default=[],
            placeholder="Select one or more campaigns",
            key="campaign_filter",
        )
    with f2:
        selected_channels = st.multiselect(
            "Channel",
            channel_options,
            default=[],
            placeholder="Select one or more channels",
            key="channel_filter",
        )
    with f3:
        selected_statuses = st.multiselect(
            "Status",
            status_options,
            default=[],
            placeholder="Select one or more statuses",
            key="status_filter",
        )

    filtered = campaign_df.copy()

    if selected_campaigns:
        filtered = filtered[filtered["campaign_name"].isin(selected_campaigns)]
    if selected_channels:
        filtered = filtered[filtered["channel"].isin(selected_channels)]
    if selected_statuses:
        filtered = filtered[filtered["campaign_status"].isin(selected_statuses)]

    if filtered.empty:
        st.warning("No campaign records match the selected filters.")
        return

    total_contacted = int(filtered["customers_contacted"].sum())
    total_responses = int(filtered["responses"].sum())
    overall_response_rate = total_responses / total_contacted if total_contacted else 0
    high_fatigue_contacts = int(
        round((filtered["customers_contacted"] * filtered["high_fatigue_pct"]).sum())
    )

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Customers contacted</div>
                <div class="kpi-value">{total_contacted:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Responses</div>
                <div class="kpi-value">{total_responses:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Response rate</div>
                <div class="kpi-value">{format_pct(overall_response_rate)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">High-fatigue contacts</div>
                <div class="kpi-value">{high_fatigue_contacts:,}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Campaign performance</div>', unsafe_allow_html=True)

    performance = (
        filtered.groupby(["campaign_name", "channel"], as_index=False)
        .agg(
            Customers_Contacted=("customers_contacted", "sum"),
            Responses=("responses", "sum"),
            Avg_Contacts=("avg_contacts_per_customer", "mean"),
            High_Fatigue=("high_fatigue_pct", "mean"),
            Budget=("campaign_budget", "sum"),
        )
    )
    performance["Response Rate"] = (
        performance["Responses"] / performance["Customers_Contacted"]
    )
    performance["Cost / Response"] = performance["Budget"] / performance["Responses"].replace(0, np.nan)
    performance["High Fatigue"] = performance["High_Fatigue"].map(format_pct)
    performance["Response Rate"] = performance["Response Rate"].map(format_pct)
    performance["Avg Contacts"] = performance["Avg_Contacts"].round(1)
    performance["Budget"] = performance["Budget"].round(0)
    performance["Cost / Response"] = performance["Cost / Response"].round(2)
    performance = performance.rename(
        columns={
            "campaign_name": "Campaign",
            "channel": "Channel",
            "Customers_Contacted": "Customers Contacted",
            "Responses": "Responses",
        }
    )
    performance = performance[
        ["Campaign", "Channel", "Customers Contacted", "Responses",
         "Response Rate", "Avg Contacts", "High Fatigue", "Budget", "Cost / Response"]
    ].sort_values("Response Rate", ascending=False)

    st.dataframe(performance, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">Contact-frequency diagnosis</div>', unsafe_allow_html=True)

    freq = filtered.copy()
    freq["Contact band"] = pd.cut(
        freq["avg_contacts_per_customer"],
        bins=[0, 2, 4, 6, np.inf],
        labels=["1–2 contacts", "3–4 contacts", "5–6 contacts", "7+ contacts"],
        include_lowest=True,
    )
    frequency = (
        freq.groupby("Contact band", observed=False)
        .agg(
            Customers=("customers_contacted", "sum"),
            Responses=("responses", "sum"),
            Avg_Contacts=("avg_contacts_per_customer", "mean"),
            High_Fatigue=("high_fatigue_pct", "mean"),
        )
        .reset_index()
    )
    frequency["Response Rate"] = frequency["Responses"] / frequency["Customers"].replace(0, np.nan)
    frequency["Response Rate"] = frequency["Response Rate"].fillna(0).map(format_pct)
    frequency["High Fatigue"] = frequency["High_Fatigue"].map(format_pct)
    frequency["Avg Contacts"] = frequency["Avg_Contacts"].round(1)
    st.dataframe(
        frequency[["Contact band", "Customers", "Responses", "Response Rate", "Avg Contacts", "High Fatigue"]],
        use_container_width=True,
        hide_index=True,
    )

    # Business diagnosis: compare response improvement with fatigue increase.
    numeric_freq = frequency.copy()
    numeric_freq["response_numeric"] = (
        numeric_freq["Responses"] / numeric_freq["Customers"].replace(0, np.nan)
    ).fillna(0)
    numeric_freq["fatigue_numeric"] = freq.groupby("Contact band", observed=False)["high_fatigue_pct"].mean().reindex(numeric_freq["Contact band"]).fillna(0).to_numpy()

    if len(numeric_freq) >= 2:
        first = numeric_freq.iloc[0]
        last = numeric_freq.iloc[-1]
        response_change = last["response_numeric"] - first["response_numeric"]
        fatigue_change = last["fatigue_numeric"] - first["fatigue_numeric"]

        if fatigue_change > 0.10 and response_change < 0.03:
            diagnosis = (
                "Higher contact frequency is increasing fatigue faster than response performance. "
                "Tighten targeting before adding more outreach."
            )
        elif response_change >= 0.03:
            diagnosis = (
                "Higher contact frequency is associated with a meaningful response improvement. "
                "Use additional outreach selectively rather than increasing contacts across the full audience."
            )
        else:
            diagnosis = (
                "The selected campaign mix does not show a strong response improvement from additional contact frequency. "
                "Focus on targeting and message quality before increasing outreach."
            )
    else:
        diagnosis = "Select a broader set of campaign records to compare contact-frequency patterns."

    st.markdown(
        f"""
        <div class="recommendation">
            <div class="recommendation-title">Campaign diagnosis</div>
            <div class="recommendation-text">{diagnosis}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Campaign history shown here is illustrative and synthetic; it is separate from the customer-level Bank Marketing dataset.")


# ============================================================
# AI STRATEGIST
# ============================================================

def render_ai_strategist():
    st.markdown('<div class="page-title">AI Strategist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">'
        "Use customer intelligence, campaign performance and approved marketing knowledge to make a decision."
        "</div>",
        unsafe_allow_html=True,
    )

    # Import lazily so the rest of the dashboard can still load if
    # the RAG dependencies are not installed.
    try:
        from rag_engine import generate_strategic_answer
    except Exception as e:
        st.error("The AI Strategist could not load.")
        st.code("pip install -r requirements_rag.txt")
        st.caption(str(e))
        return

    mode = st.selectbox(
        "Strategist mode",
        [
            "Explain a customer recommendation",
            "Diagnose campaign performance",
            "Develop a marketing strategy",
        ],
    )

    data_context = ""

    # --------------------------------------------------------
    # CUSTOMER MODE
    # --------------------------------------------------------
    if mode == "Explain a customer recommendation":
        customer_index = st.selectbox(
            "Customer",
            options=list(customer_df.index),
            format_func=lambda x: f"Customer {int(x) + 1}",
            key="ai_customer_select",
        )

        customer = customer_df.loc[customer_index]

        data_context = f"""
Customer: Customer {int(customer_index) + 1}
Age: {customer['age']}
Job: {customer['job']}
Marital status: {customer['marital']}
Education: {customer['education']}
Balance: {customer['balance']}
Housing loan: {customer['housing']}
Personal loan: {customer['loan']}
Contact type: {customer['contact']}
Current campaign contacts: {customer['campaign']}
Previous contacts: {customer['previous']}
Previous campaign outcome: {customer['poutcome']}

Response propensity: {format_pct(customer['response_probability'])}
Priority score: {customer['priority_score']:.1f}
Fatigue risk: {customer['fatigue_risk']}
Segment: {segment_number(customer['Cluster'])}
Next best action: {customer['next_best_action']}
Recommended channel: {customer['recommended_channel']}
"""

        default_question = (
            "Why is this customer receiving this recommendation, "
            "and how should I approach them?"
        )

        st.caption(
            "You can ask for a framework-based analysis, for example: "
            "\"Do a SWOT analysis for this customer.\""
        )

    # --------------------------------------------------------
    # CAMPAIGN MODE
    # --------------------------------------------------------
    elif mode == "Diagnose campaign performance":
        campaign_options = sorted(
            campaign_df["campaign_name"].dropna().unique().tolist()
        )

        selected_campaigns = st.multiselect(
            "Campaigns to analyse",
            campaign_options,
            default=[],
            placeholder="Select one or more campaigns",
            key="ai_campaign_select",
        )

        filtered_ai = campaign_df.copy()

        if selected_campaigns:
            filtered_ai = filtered_ai[
                filtered_ai["campaign_name"].isin(selected_campaigns)
            ]

        grouped = (
            filtered_ai.groupby("campaign_name", as_index=False)
            .agg(
                Customers_Contacted=("customers_contacted", "sum"),
                Responses=("responses", "sum"),
                Avg_Contacts=("avg_contacts_per_customer", "mean"),
                High_Fatigue=("high_fatigue_pct", "mean"),
                Budget=("campaign_budget", "sum"),
            )
        )

        grouped["Response_Rate"] = (
            grouped["Responses"] / grouped["Customers_Contacted"].replace(0, np.nan)
        )
        grouped["Cost_Per_Response"] = (
            grouped["Budget"] / grouped["Responses"].replace(0, np.nan)
        )

        data_context = (
            "The campaign dataset is illustrative/synthetic.\n\n"
            + grouped.to_string(index=False)
        )

        default_question = (
            "What is working, what is inefficient, and what should "
            "the marketing manager change?"
        )

        st.caption(
            "Campaign analysis uses the illustrative/synthetic campaign dataset."
        )

    # --------------------------------------------------------
    # STRATEGY MODE
    # --------------------------------------------------------
    else:
        segment_summary = (
            customer_df.groupby("Cluster")
            .agg(
                Customers=("Cluster", "size"),
                Avg_Balance=("balance", "mean"),
                Avg_Response=("response_probability", "mean"),
                Avg_Campaign=("campaign", "mean"),
            )
            .reset_index()
        )

        campaign_summary = (
            campaign_df.groupby("channel")
            .agg(
                Customers_Contacted=("customers_contacted", "sum"),
                Responses=("responses", "sum"),
                Avg_Contacts=("avg_contacts_per_customer", "mean"),
                High_Fatigue=("high_fatigue_pct", "mean"),
            )
            .reset_index()
        )

        campaign_summary["Response_Rate"] = (
            campaign_summary["Responses"]
            / campaign_summary["Customers_Contacted"].replace(0, np.nan)
        )

        data_context = (
            "SEGMENT SUMMARY\n"
            + segment_summary.to_string(index=False)
            + "\n\nCAMPAIGN CHANNEL SUMMARY\n"
            + campaign_summary.to_string(index=False)
            + "\n\nNOTE: Campaign data is illustrative/synthetic."
        )

        default_question = (
            "How should we improve conversion while controlling "
            "unnecessary outreach and customer fatigue?"
        )

    question = st.text_area(
        "What would you like the AI Strategist to answer?",
        value=default_question,
        height=100,
        key="ai_strategist_question",
    )

    if st.button(
        "Generate strategic recommendation",
        type="primary",
        use_container_width=True,
        key="generate_ai_recommendation",
    ):
        if not question.strip():
            st.warning("Enter a question for the AI Strategist.")
            return

        with st.spinner(
            "Analysing platform data and retrieving relevant marketing knowledge..."
        ):
            try:
                answer, sources = generate_strategic_answer(
                    question,
                    data_context,
                )
            except Exception as e:
                st.error("The AI Strategist could not generate a response.")
                st.exception(e)
                return

        st.markdown(
            '<div class="section-title">Strategic recommendation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(answer)

        if sources:
            st.markdown(
                '<div class="section-title">Knowledge retrieved</div>',
                unsafe_allow_html=True,
            )
            for source in sources:
                st.markdown(f"- `{source}`")
        else:
            st.caption("No knowledge-base source was retrieved.")

        st.caption(
            "The AI Strategist combines live platform data with approved "
            "project knowledge retrieved through RAG before generating its response."
        )


# ============================================================
# ROUTING
# ============================================================

if navigation == "TARGET":
    render_target()

elif navigation == "CUSTOMERS":
    render_customers()

elif navigation == "SEGMENTS":
    render_segments()

elif navigation == "CAMPAIGNS":
    render_campaigns()

elif navigation == "AI STRATEGIST":
    render_ai_strategist()
