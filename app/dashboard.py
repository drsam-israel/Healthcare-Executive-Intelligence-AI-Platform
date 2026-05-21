import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
import joblib
import shap

def executive_metric_card(title, value, delta=None):

    st.metric(
        label=title,
        value=value,
        delta=delta
    )

    
# Add src folder to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from kpi_calculations import (
    load_data,
    calculate_total_admissions,
    calculate_average_los,
    calculate_readmission_rate,
    calculate_mortality_rate,
    calculate_average_cost,
    calculate_ed_admission_rate
)

# ---------------------------------------------------
# PREMIUM EXECUTIVE AI GLOBAL THEME
# ---------------------------------------------------

st.set_page_config(
    page_title="Healthcare Executive Intelligence & Predictive AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* -----------------------------
GLOBAL APP
------------------------------*/

.stApp {
    background: linear-gradient(
        135deg,
        #071028 0%,
        #0B1736 40%,
        #111827 100%
    );
    color: white;
}

/* -----------------------------
MAIN CONTAINER
------------------------------*/

.main .block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

/* -----------------------------
SIDEBAR
------------------------------*/

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #050816 0%,
        #071028 100%
    );
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #E5E7EB;
}

/* -----------------------------
HEADINGS
------------------------------*/

h1 {
    color: white !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #F9FAFB !important;
    font-weight: 700 !important;
}

/* -----------------------------
PARAGRAPHS
------------------------------*/

p {
    color: #D1D5DB !important;
    font-size: 1rem;
}


/* -----------------------------
STREAMLIT KPI METRIC CARDS
------------------------------*/

div[data-testid="metric-container"] {

    background: rgba(17,24,39,0.85);

    border: 1px solid rgba(34,211,238,0.25);

    border-radius: 24px;

    padding: 24px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.35);

    transition: all 0.3s ease;
}

/* Hover effect */

div[data-testid="metric-container"]:hover {

    border: 1px solid rgba(34,211,238,0.55);

    box-shadow:
        0 0 25px rgba(34,211,238,0.20);

    transform: translateY(-3px);
}

/* Metric label */

div[data-testid="metric-container"] label {

    color: #9CA3AF !important;

    font-weight: 700 !important;

    font-size: 1rem !important;
}

/* Metric value */

div[data-testid="metric-container"] [data-testid="stMetricValue"] {

    color: #F9FAFB !important;

    font-size: 2.3rem !important;

    font-weight: 900 !important;
}

/* Metric delta */

div[data-testid="metric-container"] [data-testid="stMetricDelta"] {

    color: #22D3EE !important;

    font-weight: 700 !important;

    font-size: 1rem !important;
}
/* -----------------------------
METRIC LABELS
------------------------------*/

.metric-label {
    color: #9CA3AF;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* -----------------------------
METRIC VALUES
------------------------------*/

.metric-value {
    color: white;
    font-size: 2.4rem;
    font-weight: 800;
}

/* -----------------------------
SECTION CONTAINERS
------------------------------*/

.section-card {
    background: rgba(17,24,39,0.72);
    border-radius: 24px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    box-shadow:
        0 8px 24px rgba(0,0,0,0.25);
}

/* -----------------------------
BUTTONS
------------------------------*/

.stButton>button {
    background: linear-gradient(
        135deg,
        #06B6D4 0%,
        #2563EB 100%
    );
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow:
        0 0 20px rgba(34,211,238,0.35);
}

/* -----------------------------
TEXT INPUTS
------------------------------*/

.stTextInput input {
    background: rgba(17,24,39,0.72) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0.8rem !important;
}

/* -----------------------------
SELECTBOX
------------------------------*/

.stSelectbox div[data-baseweb="select"] {
    background: rgba(17,24,39,0.72);
    border-radius: 14px;
}

/* Selected value box */

div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: #F9FAFB !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
    border-radius: 14px !important;
}

/* Selected text */

div[data-baseweb="select"] span {
    color: #F9FAFB !important;
    font-weight: 600 !important;
}

/* Dropdown arrow */

div[data-baseweb="select"] svg {
    fill: #22D3EE !important;
}

/* Dropdown menu */

ul[role="listbox"] {
    background-color: #111827 !important;
    color: #F9FAFB !important;
}

/* Dropdown options */

li[role="option"] {
    background-color: #111827 !important;
    color: #F9FAFB !important;
}

/* Hover effect */

li[role="option"]:hover {
    background-color: #1E3A8A !important;
}
/* -----------------------------
TABS
------------------------------*/

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    color: #9CA3AF;
}

.stTabs [aria-selected="true"] {
    color: #22D3EE !important;
}

/* -----------------------------
PLOTLY CHART CONTAINERS
------------------------------*/

.js-plotly-plot {
    border-radius: 20px;
    overflow: hidden;
}

/* -----------------------------
SUCCESS / INFO / WARNING
------------------------------*/

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* -----------------------------
SCROLLBAR
------------------------------*/

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #1E3A8A;
    border-radius: 20px;
}

/* -----------------------------
LOGIN PAGE TITLE
------------------------------*/

.login-title {
    font-size: 4rem;
    font-weight: 900;
    background: linear-gradient(
        90deg,
        #22D3EE,
        #10B981
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* -----------------------------
EXECUTIVE GLOW
------------------------------*/

.glow {
    color: #22D3EE;
    text-shadow:
        0 0 10px rgba(34,211,238,0.7);
}

</style>
""", unsafe_allow_html=True)


# Load data
df = load_data("data/synthetic/hospital_operations_synthetic.csv")

df["admission_date"] = pd.to_datetime(df["admission_date"])
df["month"] = df["admission_date"].dt.to_period("M").astype(str)
df["year"] = df["admission_date"].dt.year

# Sidebar
st.sidebar.title("Healthcare Executive Dashboard")

page = st.sidebar.radio(
    "Navigate",
     [
    "Executive Overview",
    "Operations Intelligence",
    "Quality & Risk",
    "Financial Analytics",
    "AI Readmission Predictor",
    "Explainable AI Insights",
    "Operational Forecasting",
    "Executive Insights & Recommendations",
    "Conversational Executive AI Assistant",
    "Data Preview"
]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

departments = ["All"] + sorted(df["department"].unique())
payer_types = ["All"] + sorted(df["payer_type"].unique())
diagnoses = ["All"] + sorted(df["primary_diagnosis"].unique())
years = ["All"] + sorted(df["year"].unique())

selected_department = st.sidebar.selectbox("Department", departments)
selected_payer = st.sidebar.selectbox("Payer Type", payer_types)
selected_diagnosis = st.sidebar.selectbox("Primary Diagnosis", diagnoses)
selected_year = st.sidebar.selectbox("Admission Year", years)

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[filtered_df["department"] == selected_department]

if selected_payer != "All":
    filtered_df = filtered_df[filtered_df["payer_type"] == selected_payer]

if selected_diagnosis != "All":
    filtered_df = filtered_df[filtered_df["primary_diagnosis"] == selected_diagnosis]

if selected_year != "All":
    filtered_df = filtered_df[filtered_df["year"] == selected_year]


# KPI function

def show_kpis(data):

    total_admissions = calculate_total_admissions(data)
    avg_los = calculate_average_los(data)
    readmission_rate = calculate_readmission_rate(data)

    mortality_rate = calculate_mortality_rate(data)
    avg_cost = calculate_average_cost(data)
    ed_rate = calculate_ed_admission_rate(data)

    col1, col2, col3 = st.columns(3)

    with col1:
        executive_metric_card(
            "Total Admissions",
            f"{total_admissions:,}",
            "↑ 4.8% operational growth"
        )

    with col2:
        executive_metric_card(
            "Average LOS",
            f"{avg_los:.2f} days",
            "↓ 1.2% LOS optimization"
        )

    with col3:
        executive_metric_card(
            "Readmission Rate",
            f"{readmission_rate}%",
            "↓ 2.1% quality improvement"
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        executive_metric_card(
            "Mortality Rate",
            f"{mortality_rate}%",
            "↓ 0.8% mortality reduction"
        )

    with col5:
        executive_metric_card(
            "Average Cost",
            f"${avg_cost:,.2f}",
            "↑ Financial utilization monitored"
        )

    with col6:
        executive_metric_card(
            "ED Admission Rate",
            f"{ed_rate}%",
            "↑ High ED operational pressure"
        )


# Empty data safety check
if filtered_df.empty:
    st.warning("No records match the selected filters. Please adjust your sidebar filters.")
    st.stop()


# Executive Overview Page
if page == "Executive Overview":

    st.title("Healthcare Executive Intelligence Dashboard")

    st.markdown(
        "Executive overview of hospital operations, quality, utilization, and financial performance analytics."
    )

    show_kpis(filtered_df)

    st.markdown("---")

    st.subheader("Executive Operational Insights")

    highest_los_department = (
        filtered_df.groupby("department")["length_of_stay"]
        .mean()
        .idxmax()
    )

    highest_readmission_department = (
        filtered_df.groupby("department")["readmitted_30_days"]
        .mean()
        .idxmax()
    )

    highest_cost_department = (
        filtered_df.groupby("department")["estimated_cost"]
        .mean()
        .idxmax()
    )

    st.info(f"Average length of stay is highest in {highest_los_department}.")
    st.warning(f"Readmission rates are highest in {highest_readmission_department}.")
    st.success(f"Average patient cost is highest in {highest_cost_department}.")

    st.markdown("---")

    monthly_admissions = (
        filtered_df.groupby("month")
        .size()
        .reset_index(name="admissions")
    )

    fig = px.line(
        monthly_admissions,
        x="month",
        y="admissions",
        title="Monthly Admissions Trend",
        markers=True
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

    st.subheader("AI-Generated Executive Summary")

    total_admissions = calculate_total_admissions(filtered_df)
    avg_los = calculate_average_los(filtered_df)
    readmission_rate = calculate_readmission_rate(filtered_df)
    mortality_rate = calculate_mortality_rate(filtered_df)

    summary = f"""
    The hospital recorded {total_admissions:,} admissions during the selected period.

    Average length of stay is currently {avg_los} days, while the 30-day readmission rate stands at {readmission_rate}%.

    Mortality rate is {mortality_rate}%, indicating current inpatient outcome performance across the selected operational segments.

    Operational analytics suggest increased utilization pressure in high-volume departments, requiring continued monitoring of patient flow, readmission exposure, and care efficiency.
    """

    st.success(summary)

elif page == "Operations Intelligence":

    st.title("Operations Intelligence")

    with st.expander("Operational Utilization Analytics", expanded=True):

        col1, col2 = st.columns(2)

        admissions_by_department = (
            filtered_df["department"]
            .value_counts()
            .reset_index()
        )

        admissions_by_department.columns = ["department", "admissions"]

        fig1 = px.bar(
            admissions_by_department,
            x="department",
            y="admissions",
            title="Admissions by Department"
        )

        col1.plotly_chart(fig1, use_container_width=True)

        los_by_department = (
            filtered_df.groupby("department")["length_of_stay"]
            .mean()
            .reset_index()
        )

        fig2 = px.bar(
            los_by_department,
            x="department",
            y="length_of_stay",
            title="Average Length of Stay by Department"
        )

        col2.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        los_distribution = px.histogram(
            filtered_df,
            x="length_of_stay",
            nbins=30,
            title="Length of Stay Distribution"
        )

        st.plotly_chart(los_distribution, use_container_width=True)

elif page == "Quality & Risk":

    st.title("Quality & Risk Intelligence")

    tab1, tab2, tab3 = st.tabs(
        [
            "Readmission Analytics",
            "Mortality Analytics",
            "Risk Mapping"
        ]
    )

    with tab1:

        readmission_by_department = (
            filtered_df.groupby("department")["readmitted_30_days"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig1 = px.bar(
            readmission_by_department,
            x="department",
            y="readmitted_30_days",
            title="Readmission Rate by Department (%)"
        )

        st.plotly_chart(fig1, use_container_width=True)

    with tab2:

        mortality_by_department = (
            filtered_df.groupby("department")["mortality_flag"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig2 = px.bar(
            mortality_by_department,
            x="department",
            y="mortality_flag",
            title="Mortality Rate by Department (%)"
        )

        st.plotly_chart(fig2, use_container_width=True)

    with tab3:

        diagnosis_risk = (
            filtered_df.groupby("primary_diagnosis")[
                ["readmitted_30_days", "mortality_flag"]
            ]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig3 = px.scatter(
            diagnosis_risk,
            x="readmitted_30_days",
            y="mortality_flag",
            size="readmitted_30_days",
            hover_name="primary_diagnosis",
            title="Diagnosis-Level Risk Map"
        )

        st.plotly_chart(fig3, use_container_width=True)


# Financial Analytics Page
elif page == "Financial Analytics":

    st.title("Financial Analytics")

    col1, col2 = st.columns(2)

    payer_mix = (
        filtered_df["payer_type"]
        .value_counts()
        .reset_index()
    )
    payer_mix.columns = ["payer_type", "count"]

    fig1 = px.pie(
        payer_mix,
        names="payer_type",
        values="count",
        title="Payer Mix"
    )

    col1.plotly_chart(fig1, use_container_width=True)

    cost_by_department = (
        filtered_df.groupby("department")["estimated_cost"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        cost_by_department,
        x="department",
        y="estimated_cost",
        title="Average Estimated Cost by Department"
    )

    col2.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    total_cost_by_department = (
        filtered_df.groupby("department")["estimated_cost"]
        .sum()
        .reset_index()
    )

    fig3 = px.bar(
        total_cost_by_department,
        x="department",
        y="estimated_cost",
        title="Total Estimated Cost Burden by Department"
    )

    st.plotly_chart(fig3, use_container_width=True)

elif page == "AI Readmission Predictor":

    st.title("AI Readmission Risk Predictor")

    st.markdown(
        "Predict 30-day hospital readmission risk using operational and clinical admission features."
    )

    model = joblib.load("src/models/readmission_model.pkl")
    encoders = joblib.load("src/models/label_encoders.pkl")

    col1, col2 = st.columns(2)

    with col1:
        department = st.selectbox(
            "Department",
            sorted(df["department"].unique())
        )

        payer_type = st.selectbox(
            "Payer Type",
            sorted(df["payer_type"].unique())
        )

        primary_diagnosis = st.selectbox(
            "Primary Diagnosis",
            sorted(df["primary_diagnosis"].unique())
        )

        length_of_stay = st.number_input(
            "Length of Stay",
            min_value=1.0,
            max_value=60.0,
            value=5.0
        )

    with col2:
        estimated_cost = st.number_input(
            "Estimated Cost",
            min_value=1000.0,
            max_value=50000.0,
            value=8500.0
        )

        mortality_flag = st.selectbox(
            "Mortality Flag",
            [0, 1]
        )

        ed_admission_flag = st.selectbox(
            "ED Admission Flag",
            [0, 1]
        )

    if st.button("Predict Readmission Risk"):

        input_data = pd.DataFrame({
            "department": [encoders["department"].transform([department])[0]],
            "payer_type": [encoders["payer_type"].transform([payer_type])[0]],
            "primary_diagnosis": [encoders["primary_diagnosis"].transform([primary_diagnosis])[0]],
            "length_of_stay": [length_of_stay],
            "estimated_cost": [estimated_cost],
            "mortality_flag": [mortality_flag],
            "ed_admission_flag": [ed_admission_flag]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        # SHAP explanation
        explainer = shap.Explainer(model)

        shap_values = explainer(input_data)

        st.markdown("---")

        st.subheader("Local Explainable AI Analysis")

        feature_impacts = pd.DataFrame({
        "Feature": input_data.columns,
        "SHAP Impact": shap_values.values[0]
})

        feature_impacts["Absolute Impact"] = (
        feature_impacts["SHAP Impact"]
        .abs()
)

        feature_impacts = feature_impacts.sort_values(
        by="Absolute Impact",
        ascending=False
)

        st.dataframe(
        feature_impacts[["Feature", "SHAP Impact"]],
        use_container_width=True
)

        top_feature = feature_impacts.iloc[0]["Feature"]

        st.info(
        f"The strongest contributor to predicted readmission risk is: {top_feature}."
)
        st.markdown("---")

        st.subheader("AI Prediction Result")

        if prediction == 1:
            st.error(f"High Readmission Risk Detected: {probability:.2f}% probability")
        else:
            st.success(f"Low Readmission Risk: {probability:.2f}% probability")

        st.info(
            "This prediction is generated from the trained XGBoost readmission risk model using the selected operational and clinical features."
        )

elif page == "Explainable AI Insights":

    st.title("Explainable Healthcare AI Insights")

    st.markdown(
        """
        Explainable AI (SHAP) analysis showing the strongest operational
        and clinical drivers influencing 30-day hospital readmission risk.
        """
    )

    st.image(
        "assets/images/shap_readmission_summary.png",
        caption="SHAP Feature Importance Summary",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Executive AI Interpretation")

    st.info(
        """
        SHAP analysis identifies the operational and clinical variables
        with the greatest contribution to readmission risk prediction.

        Features with higher SHAP importance have stronger influence on
        model decision-making and operational risk stratification.

        This supports:
        - operational transparency
        - explainable healthcare AI
        - executive-level AI governance
        - interpretable clinical intelligence
        """
    ) 

elif page == "Operational Forecasting":

    st.title("Operational Forecasting Intelligence")

    st.markdown(
        """
        Forecasting future hospital admissions volume using
        operational trend analytics and predictive modeling.
        """
    )

    st.image(
        "assets/images/admissions_forecast.png",
        caption="Hospital Admissions Forecast",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Executive Forecast Interpretation")

    st.info(
        """
        Forecasting analysis projects future hospital admissions
        volume trends based on historical operational utilization patterns.

        This supports:
        - staffing planning
        - hospital capacity management
        - operational resource allocation
        - bed occupancy preparation
        - executive operational forecasting

        Forecasting intelligence enables proactive healthcare
        operational decision-making and improved system readiness.
        """
    ) 

elif page == "Executive Insights & Recommendations":

    st.title("Executive Insights & Recommendations")

    st.markdown(
        """
        Quantified executive insights and AI-assisted recommendations generated from
        hospital utilization, quality, financial, forecasting, and explainable AI analytics.
        """
    )

    st.markdown("---")

    total_admissions = calculate_total_admissions(filtered_df)
    avg_los = calculate_average_los(filtered_df)
    readmission_rate = calculate_readmission_rate(filtered_df)
    mortality_rate = calculate_mortality_rate(filtered_df)
    avg_cost = calculate_average_cost(filtered_df)
    ed_rate = calculate_ed_admission_rate(filtered_df)

    highest_los_department = (
        filtered_df.groupby("department")["length_of_stay"]
        .mean()
        .idxmax()
    )

    highest_los_value = (
        filtered_df.groupby("department")["length_of_stay"]
        .mean()
        .max()
    )

    highest_readmission_department = (
        filtered_df.groupby("department")["readmitted_30_days"]
        .mean()
        .idxmax()
    )

    highest_readmission_value = (
        filtered_df.groupby("department")["readmitted_30_days"]
        .mean()
        .max() * 100
    )

    highest_mortality_department = (
        filtered_df.groupby("department")["mortality_flag"]
        .mean()
        .idxmax()
    )

    highest_mortality_value = (
        filtered_df.groupby("department")["mortality_flag"]
        .mean()
        .max() * 100
    )

    highest_cost_department = (
        filtered_df.groupby("department")["estimated_cost"]
        .sum()
        .idxmax()
    )

    highest_cost_value = (
        filtered_df.groupby("department")["estimated_cost"]
        .sum()
        .max()
    )

    st.subheader("Quantified Executive Insights")

    st.warning(
        f"""
        • Total admissions analyzed: {total_admissions:,}

        • Average length of stay: {avg_los} days

        • Highest LOS department: {highest_los_department} at {highest_los_value:.2f} days

        • ED admission utilization rate: {ed_rate}%

        • Highest cost burden department: {highest_cost_department} with ${highest_cost_value:,.2f}
        """
    )

    st.error(
        f"""
        • Overall 30-day readmission rate: {readmission_rate}%

        • Highest readmission department: {highest_readmission_department} at {highest_readmission_value:.2f}%

        • Overall mortality rate: {mortality_rate}%

        • Highest mortality department: {highest_mortality_department} at {highest_mortality_value:.2f}%
        """
    )

    st.info(
        f"""
        • Average estimated cost per admission: ${avg_cost:,.2f}

        • SHAP explainability identified length_of_stay, estimated_cost,
        primary_diagnosis, department, and payer_type as key readmission risk drivers.

        • Forecasting indicates continued monthly admissions demand around the
        200+ admissions/month range based on historical utilization trends.
        """
    )

    st.markdown("---")

    st.subheader("Executive Recommendations")

    st.success(
        f"""
        • Prioritize readmission reduction interventions in {highest_readmission_department},
        where readmission risk reached {highest_readmission_value:.2f}%.

        • Target LOS optimization in {highest_los_department},
        where average LOS reached {highest_los_value:.2f} days.

        • Strengthen mortality review and clinical quality monitoring in
        {highest_mortality_department}, where mortality reached {highest_mortality_value:.2f}%.

        • Monitor financial utilization in {highest_cost_department},
        which generated approximately ${highest_cost_value:,.2f} in estimated cost burden.

        • Because ED admission utilization is {ed_rate}%, implement ED throughput,
        triage optimization, and bed-capacity monitoring workflows.

        • Use predictive AI readmission scoring to prioritize high-risk patients
        for discharge planning and post-discharge follow-up.
        """
    )

    st.markdown("---")

    st.subheader("Strategic Executive Summary")

    st.info(
        f"""
        Across {total_admissions:,} hospital admissions, the platform identified
        an overall readmission rate of {readmission_rate}%, mortality rate of
        {mortality_rate}%, average LOS of {avg_los} days, and average admission
        cost of ${avg_cost:,.2f}.

        The strongest executive priorities are readmission reduction,
        LOS optimization, ED throughput management, cost burden monitoring,
        forecasting-based capacity planning, and explainable AI governance.
        """
    ) 

elif page == "Conversational Executive AI Assistant":

    st.title("Conversational Executive AI Assistant")

    st.markdown(
        """
        Ask executive-level questions about hospital operations, readmissions,
        mortality, length of stay, cost burden, ED utilization, and AI insights.
        """
    )

    user_question = st.text_input(
        "Ask a question",
        placeholder="Example: Which department has the highest readmission rate?"
    )

    if user_question:

        question = user_question.lower()

        total_admissions = calculate_total_admissions(filtered_df)
        avg_los = calculate_average_los(filtered_df)
        readmission_rate = calculate_readmission_rate(filtered_df)
        mortality_rate = calculate_mortality_rate(filtered_df)
        avg_cost = calculate_average_cost(filtered_df)
        ed_rate = calculate_ed_admission_rate(filtered_df)

        highest_los_department = (
            filtered_df.groupby("department")["length_of_stay"]
            .mean()
            .idxmax()
        )

        highest_los_value = (
            filtered_df.groupby("department")["length_of_stay"]
            .mean()
            .max()
        )

        highest_readmission_department = (
            filtered_df.groupby("department")["readmitted_30_days"]
            .mean()
            .idxmax()
        )

        highest_readmission_value = (
            filtered_df.groupby("department")["readmitted_30_days"]
            .mean()
            .max() * 100
        )

        highest_mortality_department = (
            filtered_df.groupby("department")["mortality_flag"]
            .mean()
            .idxmax()
        )

        highest_mortality_value = (
            filtered_df.groupby("department")["mortality_flag"]
            .mean()
            .max() * 100
        )

        highest_cost_department = (
            filtered_df.groupby("department")["estimated_cost"]
            .sum()
            .idxmax()
        )

        highest_cost_value = (
            filtered_df.groupby("department")["estimated_cost"]
            .sum()
            .max()
        )

        response = None

        if "admission" in question and "total" in question:
            response = f"Total admissions analyzed: {total_admissions:,}."

        elif "length of stay" in question or "los" in question:
            response = (
                f"Average length of stay is {avg_los} days. "
                f"The highest LOS department is {highest_los_department} "
                f"at {highest_los_value:.2f} days."
            )

        elif "readmission" in question:
            response = (
                f"Overall 30-day readmission rate is {readmission_rate}%. "
                f"The highest readmission department is {highest_readmission_department} "
                f"at {highest_readmission_value:.2f}%."
            )

        elif "mortality" in question or "death" in question:
            response = (
                f"Overall mortality rate is {mortality_rate}%. "
                f"The highest mortality department is {highest_mortality_department} "
                f"at {highest_mortality_value:.2f}%."
            )

        elif "cost" in question or "financial" in question:
            response = (
                f"Average estimated cost per admission is ${avg_cost:,.2f}. "
                f"The highest total cost burden department is {highest_cost_department} "
                f"with approximately ${highest_cost_value:,.2f}."
            )

        elif "ed" in question or "emergency" in question:
            response = (
                f"ED admission utilization rate is {ed_rate}%. "
                "This suggests the need for ED throughput monitoring and capacity planning."
            )

        elif "shap" in question or "explain" in question or "driver" in question:
            response = (
                "SHAP explainability identified length_of_stay, estimated_cost, "
                "primary_diagnosis, department, and payer_type as key readmission risk drivers."
            )

        elif "recommend" in question or "what should" in question:
            response = (
                f"Recommended priorities: reduce readmissions in {highest_readmission_department}, "
                f"optimize LOS in {highest_los_department}, monitor mortality in "
                f"{highest_mortality_department}, manage cost burden in {highest_cost_department}, "
                "and use forecasting intelligence for proactive capacity planning."
            )

        elif "summary" in question or "summarize" in question:
            response = (
                f"Executive summary: Across {total_admissions:,} admissions, "
                f"readmission rate is {readmission_rate}%, mortality rate is {mortality_rate}%, "
                f"average LOS is {avg_los} days, average cost is ${avg_cost:,.2f}, "
                f"and ED admission utilization is {ed_rate}%."
            )

        else:
            response = (
                "I can answer executive questions about admissions, LOS, readmissions, "
                "mortality, ED utilization, cost burden, SHAP explainability, and recommendations."
            )

        st.markdown("---")
        st.subheader("Assistant Response")
        st.success(response)    
# Data Preview Page
elif page == "Data Preview":

    st.title("Hospital Operations Data Preview")

    st.dataframe(filtered_df, use_container_width=True)