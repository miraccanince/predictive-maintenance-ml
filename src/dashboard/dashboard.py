"""
SentinelX CEO Dashboard
=======================

Executive visualization of predictive maintenance value.

Key Metrics (Paper 3 Reference):
- $120.9M annual savings potential
- Alert Fatigue: 1.4% FP vs 54% industry baseline
- Human-AI Trust: Transparent SHAP explanations

Launch: streamlit run dashboard.py
Docker: Included in docker-compose.yml
"""

import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text

# Load environment
load_dotenv()

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Database connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sentinelx")
DB_USER = os.getenv("DB_USER", "sentinelx")
DB_PASS = os.getenv("DB_PASS", "sentinelx_secure_2024")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# API endpoint
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Cost assumptions (from Paper 3 / evaluation_results.json)
COST_PER_FALSE_POSITIVE = 500  # Cost of unnecessary maintenance
COST_PER_FALSE_NEGATIVE = 5000  # Cost of undetected failure
COST_PER_TRUE_POSITIVE = 500  # Maintenance cost (but saves $5000)
NET_SAVINGS_PER_TP = 4500  # $5000 - $500 = $4500 saved per caught failure

# Industry benchmarks (Paper 3)
INDUSTRY_FP_RATE = 54.0  # 54% false positive rate
SENTINELX_FP_RATE = 1.44  # Our achieved rate
ANNUAL_PREDICTIONS = 2_600_000  # Predictions per year (realistic for large operation)


# =============================================================================
# 2. DATABASE FUNCTIONS
# =============================================================================


@st.cache_resource
def get_db_engine():
    """Create cached database engine."""
    try:
        engine = create_engine(DATABASE_URL)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


def query_alerts(engine, days: int = 30) -> pd.DataFrame:
    """Query alerts from database."""
    if engine is None:
        return pd.DataFrame()

    query = text("""
        SELECT
            id, machine_id, timestamp, failure_prob,
            risk_level, root_cause, confidence,
            acknowledged, false_positive, created_at
        FROM alerts
        WHERE created_at >= NOW() - INTERVAL ':days days'
        ORDER BY created_at DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"days": days})
        return df
    except Exception:
        # Return empty DataFrame if table doesn't exist yet
        return pd.DataFrame()


def get_alert_counts(engine) -> dict[str, int]:
    """Get alert statistics."""
    if engine is None:
        return {"total": 0, "high_risk": 0, "acknowledged": 0}

    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM alerts")).scalar() or 0
            high_risk = (
                conn.execute(
                    text("SELECT COUNT(*) FROM alerts WHERE risk_level IN ('high', 'critical')")
                ).scalar()
                or 0
            )
            acknowledged = (
                conn.execute(text("SELECT COUNT(*) FROM alerts WHERE acknowledged = true")).scalar()
                or 0
            )
        return {"total": total, "high_risk": high_risk, "acknowledged": acknowledged}
    except Exception:
        return {"total": 0, "high_risk": 0, "acknowledged": 0}


def get_root_cause_distribution(engine) -> pd.DataFrame:
    """Get distribution of root causes."""
    if engine is None:
        return pd.DataFrame()

    query = text("""
        SELECT root_cause, COUNT(*) as count
        FROM alerts
        WHERE root_cause IS NOT NULL
        GROUP BY root_cause
        ORDER BY count DESC
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


def get_risk_level_distribution(engine) -> pd.DataFrame:
    """Get distribution of risk levels."""
    if engine is None:
        return pd.DataFrame()

    query = text("""
        SELECT risk_level, COUNT(*) as count
        FROM alerts
        GROUP BY risk_level
        ORDER BY
            CASE risk_level
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'moderate' THEN 3
                WHEN 'low' THEN 4
                WHEN 'nominal' THEN 5
            END
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


def get_hourly_alerts(engine, hours: int = 24) -> pd.DataFrame:
    """Get alerts grouped by hour."""
    if engine is None:
        return pd.DataFrame()

    query = text("""
        SELECT
            DATE_TRUNC('hour', created_at) as hour,
            COUNT(*) as alert_count,
            AVG(failure_prob) as avg_probability
        FROM alerts
        WHERE created_at >= NOW() - INTERVAL ':hours hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY hour
    """)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"hours": hours})
        return df
    except Exception:
        return pd.DataFrame()


# =============================================================================
# 3. CALCULATION FUNCTIONS
# =============================================================================


def calculate_savings(alerts_df: pd.DataFrame) -> dict[str, Any]:
    """
    Calculate cost savings based on alert data.

    Paper 3 Logic:
    - Each True Positive (high-risk alert that was real) saves $4,500
    - Each False Positive costs $500 (unnecessary maintenance)
    - Industry loses $2.7M per 100K predictions at 54% FP rate
    - SentinelX saves $120.9M annually at 1.4% FP rate
    """
    if alerts_df.empty:
        # Return projected savings based on model performance
        return calculate_projected_savings()

    total_alerts = len(alerts_df)

    # Count by risk level
    high_risk = len(alerts_df[alerts_df["risk_level"].isin(["high", "critical"])])

    # For labeled data
    labeled = alerts_df[alerts_df["false_positive"].notna()]
    if len(labeled) > 0:
        fp_count = len(labeled[labeled["false_positive"]])
        tp_count = len(labeled[~labeled["false_positive"]])
        fp_rate = (fp_count / len(labeled)) * 100 if len(labeled) > 0 else SENTINELX_FP_RATE
    else:
        # Use model benchmarks
        fp_rate = SENTINELX_FP_RATE
        tp_count = int(high_risk * 0.986)  # 98.6% are true positives
        fp_count = high_risk - tp_count

    # Calculate savings
    savings_from_tp = tp_count * NET_SAVINGS_PER_TP
    cost_from_fp = fp_count * COST_PER_FALSE_POSITIVE
    net_savings = savings_from_tp - cost_from_fp

    # Compare to industry baseline
    industry_fp_at_scale = int(high_risk * (INDUSTRY_FP_RATE / 100))
    industry_tp_at_scale = high_risk - industry_fp_at_scale
    industry_savings = (industry_tp_at_scale * NET_SAVINGS_PER_TP) - (
        industry_fp_at_scale * COST_PER_FALSE_POSITIVE
    )

    improvement = net_savings - industry_savings if industry_savings > 0 else net_savings

    return {
        "total_alerts": total_alerts,
        "high_risk_alerts": high_risk,
        "true_positives": tp_count,
        "false_positives": fp_count,
        "fp_rate": fp_rate,
        "gross_savings": savings_from_tp,
        "fp_cost": cost_from_fp,
        "net_savings": net_savings,
        "vs_industry_improvement": improvement,
        "annual_projection": project_annual_savings(net_savings, total_alerts),
    }


def calculate_projected_savings() -> dict[str, Any]:
    """Calculate projected savings when no real data available."""
    # Based on Paper 3 evaluation results
    # At 2.6M predictions/year with 5.2% failure rate = 135,200 failures
    total_failures = int(ANNUAL_PREDICTIONS * 0.052)  # 5.2% failure rate

    # SentinelX catches 95% of failures (recall)
    sentinelx_tp = int(total_failures * 0.95)
    sentinelx_fp = int(ANNUAL_PREDICTIONS * (SENTINELX_FP_RATE / 100))

    # Industry catches 46% (100% - 54% FP means poor recall)
    industry_tp = int(total_failures * 0.46)
    industry_fp = int(ANNUAL_PREDICTIONS * (INDUSTRY_FP_RATE / 100))

    sentinelx_savings = (sentinelx_tp * NET_SAVINGS_PER_TP) - (
        sentinelx_fp * COST_PER_FALSE_POSITIVE
    )
    industry_savings = (industry_tp * NET_SAVINGS_PER_TP) - (industry_fp * COST_PER_FALSE_POSITIVE)

    return {
        "total_alerts": 0,
        "high_risk_alerts": 0,
        "true_positives": 0,
        "false_positives": 0,
        "fp_rate": SENTINELX_FP_RATE,
        "gross_savings": 0,
        "fp_cost": 0,
        "net_savings": 0,
        "vs_industry_improvement": sentinelx_savings - industry_savings,
        "annual_projection": sentinelx_savings,
    }


def project_annual_savings(current_savings: float, current_alerts: int) -> float:
    """Project savings to annual scale."""
    if current_alerts == 0:
        return 120_900_000  # Paper 3 benchmark

    # Scale based on current alert rate
    alerts_per_day = current_alerts / 7  # Assume 7-day window
    annual_alerts = alerts_per_day * 365

    savings_per_alert = current_savings / current_alerts if current_alerts > 0 else 0
    return savings_per_alert * annual_alerts


# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================


def create_savings_gauge(value: float, title: str, max_value: float = 150_000_000) -> go.Figure:
    """Create a gauge chart for savings visualization."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={"reference": 0, "valueformat": ",.0f"},
            title={"text": title, "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, max_value], "tickformat": ",.0f"},
                "bar": {"color": "#00D084"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, max_value * 0.33], "color": "#FFE5E5"},
                    {"range": [max_value * 0.33, max_value * 0.66], "color": "#FFF3E5"},
                    {"range": [max_value * 0.66, max_value], "color": "#E5FFE5"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 120_900_000,  # Paper 3 target
                },
            },
        )
    )

    fig.update_layout(
        height=300,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#333", "family": "Arial"},
    )

    return fig


def create_fp_comparison_chart() -> go.Figure:
    """Create Alert Fatigue comparison chart (Paper 3)."""
    categories = ["Industry Baseline", "SentinelX"]
    fp_rates = [INDUSTRY_FP_RATE, SENTINELX_FP_RATE]
    colors = ["#FF6B6B", "#00D084"]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categories,
            y=fp_rates,
            marker_color=colors,
            text=[f"{rate:.1f}%" for rate in fp_rates],
            textposition="outside",
            textfont={"size": 24, "color": "#333"},
        )
    )

    # Add reduction annotation
    reduction = ((INDUSTRY_FP_RATE - SENTINELX_FP_RATE) / INDUSTRY_FP_RATE) * 100

    fig.update_layout(
        title={"text": f"Alert Fatigue Reduction: {reduction:.0f}%", "font": {"size": 20}},
        yaxis_title="False Positive Rate (%)",
        yaxis={"range": [0, 70]},
        height=350,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_root_cause_pie(df: pd.DataFrame) -> go.Figure:
    """Create root cause distribution pie chart."""
    if df.empty:
        # Demo data
        df = pd.DataFrame(
            {
                "root_cause": [
                    "mechanical_wear",
                    "software_stress",
                    "thermal_overload",
                    "network_congestion",
                    "power_anomaly",
                ],
                "count": [35, 28, 18, 12, 7],
            }
        )

    # Format labels
    df["label"] = df["root_cause"].str.replace("_", " ").str.title()

    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#DDA0DD"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=df["label"],
                values=df["count"],
                hole=0.4,
                marker_colors=colors,
                textinfo="label+percent",
                textfont={"size": 12},
            )
        ]
    )

    fig.update_layout(
        title={"text": "Root Cause Distribution", "font": {"size": 18}},
        height=350,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """Create risk level distribution chart."""
    if df.empty:
        df = pd.DataFrame(
            {
                "risk_level": ["nominal", "low", "moderate", "high", "critical"],
                "count": [45, 25, 18, 10, 2],
            }
        )

    risk_colors = {
        "nominal": "#95E1D3",
        "low": "#A8E6CF",
        "moderate": "#FFE66D",
        "high": "#FFB347",
        "critical": "#FF6B6B",
    }

    df["color"] = df["risk_level"].map(risk_colors)

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["risk_level"].str.title(),
                y=df["count"],
                marker_color=df["color"],
                text=df["count"],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title={"text": "Risk Level Distribution", "font": {"size": 18}},
        xaxis_title="Risk Level",
        yaxis_title="Alert Count",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Create alert timeline chart."""
    if df.empty:
        # Generate demo timeline
        hours = pd.date_range(end=datetime.now(), periods=24, freq="H")
        df = pd.DataFrame(
            {
                "hour": hours,
                "alert_count": np.random.poisson(5, 24),
                "avg_probability": np.random.uniform(0.1, 0.4, 24),
            }
        )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=df["hour"], y=df["alert_count"], name="Alert Count", marker_color="#4ECDC4"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["hour"],
            y=df["avg_probability"],
            name="Avg Probability",
            line={"color": "#FF6B6B", "width": 2},
            mode="lines+markers",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title={"text": "Alert Activity (Last 24 Hours)", "font": {"size": 18}},
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )

    fig.update_yaxes(title_text="Alert Count", secondary_y=False)
    fig.update_yaxes(title_text="Avg Probability", secondary_y=True)

    return fig


# =============================================================================
# 5. MAIN DASHBOARD
# =============================================================================


def main():
    # Page configuration
    st.set_page_config(
        page_title="SentinelX | Executive Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS - Light theme with high contrast for visibility
    st.markdown(
        """
        <style>
        /* Force light background throughout */
        .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
            background-color: #ffffff !important;
        }

        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1a1a2e !important;
            text-align: center;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #555 !important;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            color: white;
            text-align: center;
        }
        .savings-highlight {
            font-size: 3rem;
            font-weight: 700;
            color: #00D084;
        }

        /* Metrics with light background and dark text */
        [data-testid="stMetric"] {
            background-color: #f8f9fa !important;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        [data-testid="stMetricLabel"] {
            color: #333 !important;
        }
        [data-testid="stMetricValue"] {
            color: #1a1a2e !important;
        }
        [data-testid="stMetricDelta"] {
            color: #666 !important;
        }

        /* All text should be dark on light background */
        p, span, div, label {
            color: #333 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #1a1a2e !important;
        }

        /* Sidebar - light gray background */
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div {
            background-color: #f0f2f6 !important;
        }

        /* Tables */
        .stDataFrame, .dataframe {
            background-color: #ffffff !important;
        }
        .stDataFrame td, .stDataFrame th {
            color: #333 !important;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            color: #333 !important;
            background-color: #f8f9fa !important;
        }
        .streamlit-expanderContent {
            background-color: #ffffff !important;
            color: #333 !important;
        }

        /* Info/Success/Warning boxes */
        .stAlert > div {
            color: #333 !important;
        }

        /* Plotly chart backgrounds */
        .js-plotly-plot .plotly {
            background-color: #ffffff !important;
        }

        /* Input elements - ensure visibility with light backgrounds */
        .stTextInput > div > div > input,
        .stTextInput input,
        input[type="text"] {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
            border: 2px solid #ddd !important;
            border-radius: 4px !important;
        }
        .stTextInput label,
        .stTextInput > label {
            color: #333 !important;
        }

        /* Sliders - ensure visibility */
        .stSlider > div > div > div {
            color: #333 !important;
        }
        .stSlider label,
        .stSlider > label {
            color: #333 !important;
        }
        .stSlider [data-testid="stTickBarMin"],
        .stSlider [data-testid="stTickBarMax"],
        .stSlider span {
            color: #333 !important;
        }
        /* Slider thumb value */
        .stSlider [data-testid="stThumbValue"] {
            color: #1a1a2e !important;
            background-color: #f0f0f0 !important;
        }

        /* Number inputs and selectboxes */
        .stNumberInput > div > div > input,
        .stNumberInput input,
        .stSelectbox > div > div > div,
        .stSelectbox select,
        select {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
            border: 2px solid #ddd !important;
        }

        /* All form inputs */
        [data-baseweb="input"] input,
        [data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #1a1a2e !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #667eea !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #5a6fd6 !important;
        }

        /* Code blocks - light background */
        .stCodeBlock, code, pre,
        [data-testid="stCodeBlock"] {
            background-color: #f8f8f8 !important;
            color: #1a1a2e !important;
            border: 1px solid #e0e0e0 !important;
        }

        /* Markdown text */
        .stMarkdown, .stMarkdown p, .stMarkdown span {
            color: #333 !important;
        }

        /* Expander content text */
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span,
        [data-testid="stExpander"] div {
            color: #333 !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        '<p class="main-header">🛡️ SentinelX Predictive Maintenance</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Industrial Intelligence | Real-Time Value Tracking</p>',
        unsafe_allow_html=True,
    )

    # Initialize database connection
    engine = get_db_engine()

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/1a1a2e/ffffff?text=SentinelX", width=200)
        st.markdown("---")

        st.markdown("### ⚙️ Configuration")
        st.selectbox("Auto-refresh", ["Off", "30s", "1min", "5min"], index=0)
        time_range = st.selectbox(
            "Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days"], index=1
        )

        st.markdown("---")
        st.markdown("### 📊 Data Source")

        # API health check
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.warning("⚠️ API Degraded")
        except Exception:
            st.error("❌ API Offline")

        # DB status
        if engine:
            st.success("✅ Database Connected")
        else:
            st.error("❌ Database Offline")

        st.markdown("---")
        st.markdown("### 📚 References")
        st.markdown("""
        - [Paper 3: Alert Fatigue](# )
        - [AWS Architecture](# )
        - [API Documentation](/docs)
        """)

    # Get data
    days_map = {"Last 24 Hours": 1, "Last 7 Days": 7, "Last 30 Days": 30}
    alerts_df = query_alerts(engine, days=days_map.get(time_range, 7))
    savings_data = calculate_savings(alerts_df)

    # ==========================================================================
    # Row 1: Key Metrics
    # ==========================================================================
    st.markdown("## 💰 Value Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📈 Annual Savings Projection",
            value=f"${savings_data['annual_projection']:,.0f}",
            delta=f"+${savings_data['vs_industry_improvement']:,.0f} vs Industry",
        )

    with col2:
        st.metric(
            label="🎯 False Positive Rate",
            value=f"{savings_data['fp_rate']:.1f}%",
            delta=f"-{INDUSTRY_FP_RATE - savings_data['fp_rate']:.1f}% vs Industry",
            delta_color="inverse",
        )

    with col3:
        st.metric(
            label="🔔 Total Alerts",
            value=f"{savings_data['total_alerts']:,}",
            delta=f"{savings_data['high_risk_alerts']} High Risk",
        )

    with col4:
        st.metric(
            label="✅ True Positives",
            value=f"{savings_data['true_positives']:,}",
            delta=f"${savings_data['gross_savings']:,.0f} Saved",
        )

    # ==========================================================================
    # Row 2: Main Visualizations
    # ==========================================================================
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💵 Savings Counter")
        fig = create_savings_gauge(savings_data["annual_projection"], "Projected Annual Savings")
        st.plotly_chart(fig, use_container_width=True)

        # Breakdown
        st.markdown("""
        | Metric | Value |
        |--------|-------|
        | Failures Caught | {:,} |
        | Cost per Failure | $5,000 |
        | Maintenance Cost | $500 |
        | **Net per Detection** | **$4,500** |
        """.format(savings_data["true_positives"]))

    with col2:
        st.markdown("### 📉 Alert Fatigue (Paper 3)")
        fig = create_fp_comparison_chart()
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **Paper 3 Finding:** Industry systems have 54% false positive rates,
        causing operators to ignore 73% of real alerts. SentinelX achieves
        1.4% FP rate through SHAP-driven explainability.
        """)

    # ==========================================================================
    # Row 3: Distributions
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 🔍 Alert Intelligence")

    col1, col2, col3 = st.columns(3)

    with col1:
        root_cause_df = get_root_cause_distribution(engine)
        fig = create_root_cause_pie(root_cause_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_df = get_risk_level_distribution(engine)
        fig = create_risk_distribution(risk_df)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        hourly_df = get_hourly_alerts(engine)
        fig = create_timeline_chart(hourly_df)
        st.plotly_chart(fig, use_container_width=True)

    # ==========================================================================
    # Row 4: Recent Alerts Table
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 📋 Recent High-Risk Alerts")

    if not alerts_df.empty:
        high_risk = alerts_df[alerts_df["risk_level"].isin(["high", "critical"])].head(10)
        if not high_risk.empty:
            display_df = high_risk[
                [
                    "timestamp",
                    "machine_id",
                    "risk_level",
                    "root_cause",
                    "failure_prob",
                    "acknowledged",
                ]
            ]
            display_df.columns = [
                "Timestamp",
                "Machine",
                "Risk",
                "Root Cause",
                "Probability",
                "Acknowledged",
            ]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No high-risk alerts in the selected time range.")
    else:
        st.info(
            "No alert data available. Run predictions through the API to populate this dashboard."
        )

    # ==========================================================================
    # Row 5: Live Prediction Demo
    # ==========================================================================
    st.markdown("---")
    st.markdown("## 🧪 Live Prediction Demo")

    with st.expander("Submit Test Prediction"):
        # Scenario presets
        scenarios = {
            "normal": {
                "air_temp": 305.0,
                "torque": 40.0,
                "tool_wear": 100.0,
                "vibration": 18.0,
                "error_rate": 2.0,
                "cpu_util": 55.0,
                "disk_io": 8.0,
                "http_5xx": 0,
                "queue_depth": 15,
                "api_latency": 120.0,
                "network_latency": 20.0,
            },
            "stress": {
                "air_temp": 330.0,
                "torque": 70.0,
                "tool_wear": 200.0,
                "vibration": 35.0,
                "error_rate": 15.0,
                "cpu_util": 88.0,
                "disk_io": 30.0,
                "http_5xx": 20,
                "queue_depth": 100,
                "api_latency": 400.0,
                "network_latency": 60.0,
            },
            "failure": {
                "air_temp": 365.0,
                "torque": 100.0,
                "tool_wear": 280.0,
                "vibration": 48.0,
                "error_rate": 28.0,
                "cpu_util": 98.0,
                "disk_io": 48.0,
                "http_5xx": 80,
                "queue_depth": 400,
                "api_latency": 750.0,
                "network_latency": 120.0,
            },
        }

        # Initialize session state with normal defaults
        if "air_temp" not in st.session_state:
            for key, val in scenarios["normal"].items():
                st.session_state[key] = val

        # Quick scenario buttons
        st.markdown("**Quick Test Scenarios:**")
        scenario_col1, scenario_col2, scenario_col3 = st.columns(3)

        with scenario_col1:
            if st.button("🟢 Normal Operation"):
                for key, val in scenarios["normal"].items():
                    st.session_state[key] = val
                st.rerun()
        with scenario_col2:
            if st.button("🟡 High Stress"):
                for key, val in scenarios["stress"].items():
                    st.session_state[key] = val
                st.rerun()
        with scenario_col3:
            if st.button("🔴 Imminent Failure"):
                for key, val in scenarios["failure"].items():
                    st.session_state[key] = val
                st.rerun()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**System Metrics**")
            machine_id = st.text_input("Machine ID", value="M1")
            air_temp = st.slider("Air Temperature (K)", 280.0, 400.0, key="air_temp")
            torque = st.slider("Torque (Nm)", 0.0, 150.0, key="torque")
            tool_wear = st.slider("Tool Wear (min)", 0.0, 300.0, key="tool_wear")
            vibration = st.slider("Vibration (mm/s)", 0.0, 60.0, key="vibration")
            network_latency = st.slider("Network Latency (ms)", 5.0, 200.0, key="network_latency")

        with col2:
            st.markdown("**Application Metrics**")
            error_rate = st.slider("Error Rate (%)", 0.0, 50.0, key="error_rate")
            cpu_util = st.slider("CPU Utilization (%)", 0.0, 100.0, key="cpu_util")
            disk_io = st.slider("Disk I/O Wait (ms)", 0.0, 60.0, key="disk_io")
            http_5xx = st.slider("HTTP 5xx Errors", 0, 200, key="http_5xx")
            queue_depth = st.slider("Queue Depth", 0, 500, key="queue_depth")
            api_latency = st.slider("API Latency (ms)", 50.0, 1000.0, key="api_latency")

        if st.button("🔮 Predict", type="primary"):
            payload = {
                "system": {
                    "machine_id": machine_id,
                    "air_temperature_K": air_temp,
                    "process_temperature_K": air_temp + 10,
                    "rotational_speed_rpm": 1500 + (torque * 10),  # Higher torque = faster speed
                    "torque_Nm": torque,
                    "tool_wear_min": tool_wear,
                    "vibration_mm_s": vibration,
                    "pressure_psi": 100.0 + (torque * 0.5),  # Pressure scales with torque
                    "network_latency_ms": network_latency,  # Critical for thermal throttle detection
                    "edge_processing_time_ms": network_latency * 0.5,  # Correlates with network
                    "fuzzy_pid_output": min(1.0, 0.3 + (error_rate / 50.0)),  # Higher with errors
                },
                "application": {
                    "error_rate_pct": error_rate,
                    "cpu_utilization_pct": cpu_util,
                    "memory_utilization_pct": cpu_util * 0.85,  # Memory correlates with CPU
                    "disk_io_wait_ms": disk_io,
                    "packet_loss_pct": error_rate * 0.3,  # Packet loss correlates with errors
                    "api_response_latency_ms": api_latency,
                    "http_5xx_count": http_5xx,
                    "queue_depth": queue_depth,
                    "request_throughput_rps": max(
                        10, 150 - queue_depth * 0.3
                    ),  # Throughput drops with queue
                },
            }

            # Debug: Show actual values being sent
            st.markdown("**📋 Debug - Values Sent to API:**")
            st.code(
                f"air_temp={air_temp}, torque={torque}, tool_wear={tool_wear}, vibration={vibration}\n"
                f"network_latency={network_latency}, error_rate={error_rate}, cpu={cpu_util}, disk_io={disk_io}"
            )

            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()

                    risk_color = {
                        "critical": "🔴",
                        "high": "🟠",
                        "moderate": "🟡",
                        "low": "🟢",
                        "nominal": "⚪",
                    }

                    st.success(f"""
                    **Prediction Complete!**

                    {risk_color.get(result['risk_level'], '⚪')} **Risk Level:** {result['risk_level'].upper()}

                    📊 **Failure Probability:** {result['failure_probability']:.2%}

                    🔍 **Root Cause:** {result['root_cause']}

                    💾 **Alert ID:** {result['alert_id']}
                    """)

                    # Show full diagnostic report with SHAP analysis
                    st.markdown("### 📊 Full Diagnostic Report")
                    st.markdown(result["markdown_report"])
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"API Error: {e}")

    # ==========================================================================
    # Footer
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>SentinelX v1.0 | Powered by XGBoost + SHAP Explainability</p>
        <p>Paper 3 Compliance: Human-AI Trust Through Transparent Diagnostics</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
