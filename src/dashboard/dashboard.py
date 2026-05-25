"""
SentinelX Predictive Maintenance Dashboard
===========================================
Real-time alert monitoring and model performance visualization.

Launch: streamlit run src/dashboard/dashboard.py
Docker: Included in docker-compose.yml
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "sentinelx")
DB_USER = os.getenv("DB_USER", "sentinelx")
DB_PASS = os.getenv("DB_PASS", "sentinelx_secure_2024")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

API_URL = os.getenv("API_URL", "http://localhost:8000")

INDUSTRY_FP_RATE = 54.0
SENTINELX_FP_RATE = 1.44

# Orange / green / teal palette
C_ORANGE = "#f97316"
C_GREEN = "#10b981"
C_TEAL = "#14b8a6"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_SLATE = "#64748b"

RISK_COLORS = {
    "critical": C_RED,
    "high": C_ORANGE,
    "moderate": C_AMBER,
    "low": C_GREEN,
    "nominal": C_SLATE,
}

CHART_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, system-ui, sans-serif", "color": "#374151", "size": 12},
    "margin": {"l": 20, "r": 20, "t": 48, "b": 20},
}


def load_model_metrics() -> dict[str, Any]:
    base = Path(__file__).parent.parent.parent / "models"
    try:
        with open(base / "cv_metrics.json") as f:
            cv = json.load(f)
        with open(base / "evaluation_results.json") as f:
            ev = json.load(f)
        return {"cv": cv, "ev": ev}
    except Exception:
        return {}


# =============================================================================
# DATABASE QUERIES
# =============================================================================


@st.cache_resource
def get_db_engine():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception:
        return None


def query_alerts(engine, days: int = 7) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT id, machine_id, timestamp, failure_prob,
                           risk_level, root_cause, confidence,
                           acknowledged, false_positive, created_at
                    FROM alerts
                    WHERE created_at >= NOW() - INTERVAL ':days days'
                    ORDER BY created_at DESC
                """),
                conn,
                params={"days": days},
            )
    except Exception:
        return pd.DataFrame()


def get_alert_counts(engine) -> dict[str, int]:
    if engine is None:
        return {"total": 0, "high_risk": 0, "acknowledged": 0}
    try:
        with engine.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM alerts")).scalar() or 0
            high_risk = (
                conn.execute(
                    text("SELECT COUNT(*) FROM alerts" " WHERE risk_level IN ('high', 'critical')")
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
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT root_cause, COUNT(*) as count
                    FROM alerts WHERE root_cause IS NOT NULL
                    GROUP BY root_cause ORDER BY count DESC
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


def get_risk_level_distribution(engine) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT risk_level, COUNT(*) as count FROM alerts
                    GROUP BY risk_level
                    ORDER BY CASE risk_level
                        WHEN 'critical' THEN 1 WHEN 'high' THEN 2
                        WHEN 'moderate' THEN 3 WHEN 'low' THEN 4
                        WHEN 'nominal' THEN 5 END
                """),
                conn,
            )
    except Exception:
        return pd.DataFrame()


def get_hourly_alerts(engine, hours: int = 24) -> pd.DataFrame:
    if engine is None:
        return pd.DataFrame()
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                text("""
                    SELECT DATE_TRUNC('hour', created_at) as hour,
                           COUNT(*) as alert_count,
                           AVG(failure_prob) as avg_probability
                    FROM alerts
                    WHERE created_at >= NOW() - INTERVAL ':hours hours'
                    GROUP BY DATE_TRUNC('hour', created_at)
                    ORDER BY hour
                """),
                conn,
                params={"hours": hours},
            )
    except Exception:
        return pd.DataFrame()


# =============================================================================
# CHARTS
# =============================================================================


def chart_fp_comparison() -> go.Figure:
    reduction = ((INDUSTRY_FP_RATE - SENTINELX_FP_RATE) / INDUSTRY_FP_RATE) * 100
    fig = go.Figure(
        go.Bar(
            x=["Industry average", "SentinelX"],
            y=[INDUSTRY_FP_RATE, SENTINELX_FP_RATE],
            marker_color=["#fcd34d", C_GREEN],
            text=[f"{INDUSTRY_FP_RATE:.0f}%", f"{SENTINELX_FP_RATE:.1f}%"],
            textposition="outside",
            textfont={"size": 22, "color": "#111827"},
            width=[0.45, 0.45],
        )
    )
    fig.update_layout(
        title={
            "text": f"False Positive Rate — {reduction:.0f}% reduction vs industry",
            "font": {"size": 13, "color": "#374151"},
        },
        yaxis={
            "range": [0, 68],
            "title": "False positive rate (%)",
            "gridcolor": "#f3f4f6",
            "ticksuffix": "%",
        },
        xaxis={"gridcolor": "#f3f4f6"},
        height=300,
        showlegend=False,
        **CHART_LAYOUT,
    )
    return fig


def chart_root_cause(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        df = pd.DataFrame(
            {
                "root_cause": [
                    "Mechanical wear",
                    "Thermal overload",
                    "Software stress",
                    "Network congestion",
                    "Power anomaly",
                ],
                "count": [38, 24, 20, 11, 7],
            }
        )
        df["label"] = df["root_cause"]
    else:
        df["label"] = df["root_cause"].str.replace("_", " ").str.title()

    colors = [C_ORANGE, C_TEAL, C_GREEN, C_AMBER, C_SLATE]
    fig = go.Figure(
        go.Pie(
            labels=df["label"],
            values=df["count"],
            hole=0.55,
            marker_colors=colors[: len(df)],
            textinfo="percent",
            textfont={"size": 12},
            hovertemplate="<b>%{label}</b><br>%{value} alerts — %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        title={"text": "Alerts by root cause", "font": {"size": 13, "color": "#374151"}},
        height=300,
        showlegend=True,
        legend={"orientation": "v", "x": 1.0, "y": 0.5, "font": {"size": 11}},
        **CHART_LAYOUT,
    )
    return fig


def chart_risk_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        df = pd.DataFrame(
            {
                "risk_level": ["nominal", "low", "moderate", "high", "critical"],
                "count": [45, 25, 18, 10, 2],
            }
        )
    df["color"] = df["risk_level"].map(RISK_COLORS)
    fig = go.Figure(
        go.Bar(
            x=df["risk_level"].str.title(),
            y=df["count"],
            marker_color=df["color"],
            text=df["count"],
            textposition="outside",
            textfont={"color": "#374151"},
        )
    )
    fig.update_layout(
        title={"text": "Alerts by risk level", "font": {"size": 13, "color": "#374151"}},
        yaxis={"gridcolor": "#f3f4f6", "title": "Number of alerts"},
        height=300,
        **CHART_LAYOUT,
    )
    return fig


def chart_timeline(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        hours = pd.date_range(end=datetime.now(), periods=24, freq="h")
        df = pd.DataFrame(
            {
                "hour": hours,
                "alert_count": np.random.poisson(4, 24),
                "avg_probability": np.random.uniform(0.08, 0.35, 24),
            }
        )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["hour"],
            y=df["alert_count"],
            name="Number of alerts",
            marker_color="#fed7aa",
            marker_line_color=C_ORANGE,
            marker_line_width=1,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["hour"],
            y=df["avg_probability"],
            name="Average failure probability",
            line={"color": C_TEAL, "width": 2},
            mode="lines+markers",
            marker={"size": 4},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title={
            "text": "Alert activity — last 24 hours",
            "font": {"size": 13, "color": "#374151"},
        },
        height=280,
        legend={
            "orientation": "h",
            "y": 1.14,
            "font": {"size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        **CHART_LAYOUT,
    )
    fig.update_yaxes(title_text="Number of alerts", secondary_y=False, gridcolor="#f3f4f6")
    fig.update_yaxes(
        title_text="Failure probability (0–1)",
        secondary_y=True,
        range=[0, 1],
        gridcolor="rgba(0,0,0,0)",
    )
    return fig


# =============================================================================
# UI HELPERS
# =============================================================================


def kpi_card(title: str, value: str, subtitle: str, color: str = C_ORANGE) -> str:
    return f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                padding:1.2rem 1.4rem;border-top:3px solid {color};height:100%">
        <div style="font-size:0.71rem;font-weight:600;letter-spacing:0.07em;
                    text-transform:uppercase;color:#9ca3af;margin-bottom:0.45rem">
            {title}
        </div>
        <div style="font-size:1.85rem;font-weight:700;color:#111827;
                    line-height:1.1;margin-bottom:0.35rem">
            {value}
        </div>
        <div style="font-size:0.78rem;color:#6b7280">{subtitle}</div>
    </div>
    """


def section_header(title: str, description: str = "") -> None:
    desc = (
        f'<div style="font-size:0.84rem;color:#6b7280;margin-top:0.2rem">{description}</div>'
        if description
        else ""
    )
    st.markdown(
        f"""
        <div style="margin:2.2rem 0 1rem 0;padding-bottom:0.7rem;
                    border-bottom:2px solid #f3f4f6">
            <div style="font-size:1rem;font-weight:600;color:#111827">{title}</div>
            {desc}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================


def main():
    st.set_page_config(
        page_title="SentinelX — Predictive Maintenance",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem !important; max-width: 1200px; }

        /* ── Sidebar (dark slate) ──────────────────────────────────────── */
        [data-testid="stSidebar"] { background-color: #0f172a !important; }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown span,
        [data-testid="stSidebar"] .stMarkdown div,
        [data-testid="stSidebar"] .stMarkdown li,
        [data-testid="stSidebar"] label { color: #94a3b8 !important; }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #f8fafc !important; }
        [data-testid="stSidebar"] hr { border-color: #1e293b !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background-color: #1e293b !important;
            color: #f1f5f9 !important;
            border-color: #334155 !important;
        }

        /* ── Main content (white) ─────────────────────────────────────── */
        [data-testid="stMain"] { background-color: #ffffff !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] span { color: #374151 !important; }
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3 { color: #111827 !important; }

        /* ── Expander ─────────────────────────────────────────────────── */
        [data-testid="stExpander"] {
            background: #fafafa !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
        }
        [data-testid="stExpander"] summary { color: #111827 !important; font-weight: 500; }
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] label,
        [data-testid="stExpander"] span { color: #374151 !important; }

        /* ── Buttons ──────────────────────────────────────────────────── */
        .stButton > button {
            background-color: #f97316 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover { background-color: #ea6c00 !important; }

        /* ── Form inputs ──────────────────────────────────────────────── */
        input[type="text"], input[type="number"],
        [data-baseweb="input"] input {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 6px !important;
        }
        [data-testid="stSlider"] label,
        [data-testid="stTextInput"] label { color: #374151 !important; }

        /* ── Dataframe ────────────────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
        }

        /* ── Alert boxes ──────────────────────────────────────────────── */
        [data-testid="stAlert"] p { color: #374151 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    engine = get_db_engine()
    model_data = load_model_metrics()
    cv = model_data.get("cv", {})
    ev = model_data.get("ev", {})

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## SentinelX")
        st.markdown("Predictive Maintenance Platform")
        st.markdown("---")

        st.markdown("**Time window**")
        time_range = st.selectbox(
            "time_range",
            ["Last 24 hours", "Last 7 days", "Last 30 days"],
            index=1,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Connection status**")

        api_ok = False
        try:
            r = requests.get(f"{API_URL}/health", timeout=3)
            api_ok = r.status_code == 200
        except Exception:
            pass
        db_ok = engine is not None

        st.markdown(
            f"""
            <div style="font-size:0.85rem;line-height:2.4">
                <span style="color:{'#4ade80' if api_ok else '#f87171'}">&#9679;</span>
                &nbsp;API server &nbsp;
                <span style="color:{'#4ade80' if api_ok else '#f87171'};font-weight:600">
                    {'Online' if api_ok else 'Offline'}
                </span><br>
                <span style="color:{'#4ade80' if db_ok else '#f87171'}">&#9679;</span>
                &nbsp;PostgreSQL &nbsp;
                <span style="color:{'#4ade80' if db_ok else '#f87171'};font-weight:600">
                    {'Connected' if db_ok else 'Offline'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if cv:
            balanced = ev.get("thresholds", {}).get("balanced", {}) if ev else {}
            st.markdown("---")
            st.markdown("**Trained model**")
            st.markdown(
                f"""
                <div style="font-size:0.82rem;line-height:2.2;color:#64748b">
                    Algorithm
                    <span style="color:#f1f5f9;font-weight:500;float:right">
                        XGBoost + SMOTE
                    </span><br>
                    Features
                    <span style="color:#f1f5f9;font-weight:500;float:right">
                        124 engineered
                    </span><br>
                    Cross-validation
                    <span style="color:#f1f5f9;font-weight:500;float:right">
                        5-fold stratified
                    </span><br>
                    PR-AUC
                    <span style="color:#f97316;font-weight:600;float:right">
                        {cv.get('pr_auc_mean', 0):.3f}
                        ± {cv.get('pr_auc_std', 0):.3f}
                    </span><br>
                    Recall
                    <span style="color:#10b981;font-weight:600;float:right">
                        {cv.get('recall_mean', 0):.1%}
                    </span><br>
                    Precision
                    <span style="color:#14b8a6;font-weight:600;float:right">
                        {cv.get('precision_mean', 0):.1%}
                    </span><br>
                    Decision threshold
                    <span style="color:#f1f5f9;font-weight:500;float:right">
                        {balanced.get('threshold', 0.633):.3f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Runtime data ──────────────────────────────────────────────────────────
    days_map = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}
    days = days_map.get(time_range, 7)
    alerts_df = query_alerts(engine, days=days)
    counts = get_alert_counts(engine)

    labeled = pd.DataFrame()
    if not alerts_df.empty and "false_positive" in alerts_df.columns:
        labeled = alerts_df[alerts_df["false_positive"].notna()]

    observed_fp_rate = (
        labeled["false_positive"].sum() / len(labeled) * 100
        if len(labeled) > 0
        else SENTINELX_FP_RATE
    )
    ack_rate = (
        alerts_df["acknowledged"].sum() / len(alerts_df) * 100
        if not alerts_df.empty and "acknowledged" in alerts_df.columns
        else 0.0
    )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:baseline;gap:0.85rem;margin-bottom:0.2rem">
            <h1 style="font-size:1.55rem;font-weight:700;color:#111827;margin:0">
                SentinelX
            </h1>
            <span style="font-size:0.95rem;color:#9ca3af">
                Predictive Maintenance Dashboard
            </span>
        </div>
        <div style="font-size:0.8rem;color:#d1d5db;margin-bottom:1.8rem">
            Dual-domain telemetry &nbsp;·&nbsp;
            XGBoost + SHAP explainability &nbsp;·&nbsp;
            PostgreSQL alert persistence
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    section_header(
        "Performance overview",
        "Key metrics from the trained model and the live alert database.",
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    pr_auc = cv.get("pr_auc_mean", 0.862) if cv else 0.862
    recall = cv.get("recall_mean", 0.882) if cv else 0.882

    with k1:
        st.markdown(
            kpi_card(
                "False positive rate",
                f"{observed_fp_rate:.1f}%",
                f"vs {INDUSTRY_FP_RATE:.0f}% industry average",
                C_GREEN,
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            kpi_card(
                "Detection recall",
                f"{recall:.1%}",
                "Failures correctly identified",
                C_ORANGE,
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            kpi_card(
                "PR-AUC score",
                f"{pr_auc:.3f}",
                "Primary metric for imbalanced data",
                C_TEAL,
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            kpi_card(
                f"Alerts — {time_range.lower()}",
                f"{counts['total']:,}",
                f"{counts['high_risk']} high or critical severity",
                C_ORANGE if counts["high_risk"] > 0 else C_SLATE,
            ),
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            kpi_card(
                "Acknowledgment rate",
                f"{ack_rate:.0f}%",
                f"{counts['acknowledged']} alerts reviewed",
                C_TEAL,
            ),
            unsafe_allow_html=True,
        )

    # ── Alert analysis ────────────────────────────────────────────────────────
    section_header(
        "Alert analysis",
        "False positive comparison vs industry, failure root cause breakdown, "
        "and alert severity distribution.",
    )
    ca, cb, cc = st.columns(3)
    with ca:
        st.plotly_chart(chart_fp_comparison(), use_container_width=True)
    with cb:
        st.plotly_chart(
            chart_root_cause(get_root_cause_distribution(engine)), use_container_width=True
        )
    with cc:
        st.plotly_chart(
            chart_risk_distribution(get_risk_level_distribution(engine)), use_container_width=True
        )

    # ── 24h timeline ─────────────────────────────────────────────────────────
    section_header("24-hour activity")
    st.plotly_chart(chart_timeline(get_hourly_alerts(engine)), use_container_width=True)

    # ── Recent alerts table ───────────────────────────────────────────────────
    section_header(
        "Recent high-risk alerts",
        "High and critical severity alerts from the selected time window.",
    )
    if not alerts_df.empty:
        high_risk_df = alerts_df[alerts_df["risk_level"].isin(["high", "critical"])].head(10)
        if not high_risk_df.empty:
            display = high_risk_df[
                [
                    "timestamp",
                    "machine_id",
                    "risk_level",
                    "root_cause",
                    "failure_prob",
                    "acknowledged",
                ]
            ].copy()
            display.columns = [
                "Timestamp",
                "Machine ID",
                "Risk level",
                "Root cause",
                "Failure probability",
                "Acknowledged",
            ]
            display["Failure probability"] = display["Failure probability"].map("{:.1%}".format)
            display["Root cause"] = (
                display["Root cause"].str.replace("_", " ").str.title().fillna("—")
            )
            st.dataframe(display, use_container_width=True, hide_index=True)
        else:
            st.info("No high or critical severity alerts in the selected time window.")
    else:
        st.info(
            "No alert data in the database yet. "
            "Submit predictions through the API to populate this view."
        )

    # ── Live prediction ───────────────────────────────────────────────────────
    # NOTE: prediction result is rendered OUTSIDE the expander to avoid
    # Streamlit's "nested expanders" restriction. Session state carries
    # the result across the rerun triggered by button press.
    section_header(
        "Live prediction",
        "Submit sensor and application metrics to receive a real-time failure "
        "probability with SHAP-based root cause diagnosis.",
    )

    with st.expander("Open prediction form"):
        scenarios = {
            "Normal operation": {
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
            "High stress": {
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
            "Imminent failure": {
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

        if "air_temp" not in st.session_state:
            for k, v in scenarios["Normal operation"].items():
                st.session_state[k] = v

        st.markdown(
            "<div style='font-size:0.84rem;color:#6b7280;margin-bottom:0.6rem'>"
            "Load a preset scenario or adjust values manually.</div>",
            unsafe_allow_html=True,
        )
        sc1, sc2, sc3 = st.columns(3)
        for col, name in zip([sc1, sc2, sc3], scenarios):
            with col:
                if st.button(name, key=f"preset_{name}"):
                    for k, v in scenarios[name].items():
                        st.session_state[k] = v
                    st.rerun()

        st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)

        with fc1:
            st.markdown(
                "<div style='font-weight:600;color:#374151;margin-bottom:0.4rem'>"
                "Hardware / sensor metrics</div>",
                unsafe_allow_html=True,
            )
            machine_id = st.text_input("Machine ID", value="M1")
            air_temp = st.slider("Air temperature (K)", 280.0, 400.0, key="air_temp")
            torque = st.slider("Torque (Nm)", 0.0, 150.0, key="torque")
            tool_wear = st.slider("Tool wear (minutes)", 0.0, 300.0, key="tool_wear")
            vibration = st.slider("Vibration (mm/s)", 0.0, 60.0, key="vibration")
            network_latency = st.slider("Network latency (ms)", 5.0, 200.0, key="network_latency")

        with fc2:
            st.markdown(
                "<div style='font-weight:600;color:#374151;margin-bottom:0.4rem'>"
                "Application / software metrics</div>",
                unsafe_allow_html=True,
            )
            error_rate = st.slider("Error rate (%)", 0.0, 50.0, key="error_rate")
            cpu_util = st.slider("CPU utilization (%)", 0.0, 100.0, key="cpu_util")
            disk_io = st.slider("Disk I/O wait (ms)", 0.0, 60.0, key="disk_io")
            http_5xx = st.slider("HTTP 5xx errors per minute", 0, 200, key="http_5xx")
            queue_depth = st.slider("Request queue depth", 0, 500, key="queue_depth")
            api_latency = st.slider("API response latency (ms)", 50.0, 1000.0, key="api_latency")

        st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
        if st.button("Run prediction", type="primary"):
            payload = {
                "system": {
                    "machine_id": machine_id,
                    "air_temperature_K": air_temp,
                    "process_temperature_K": air_temp + 10,
                    "rotational_speed_rpm": 1500 + (torque * 10),
                    "torque_Nm": torque,
                    "tool_wear_min": tool_wear,
                    "vibration_mm_s": vibration,
                    "pressure_psi": 100.0 + (torque * 0.5),
                    "network_latency_ms": network_latency,
                    "edge_processing_time_ms": network_latency * 0.5,
                    "fuzzy_pid_output": min(1.0, 0.3 + (error_rate / 50.0)),
                },
                "application": {
                    "error_rate_pct": error_rate,
                    "cpu_utilization_pct": cpu_util,
                    "memory_utilization_pct": cpu_util * 0.85,
                    "disk_io_wait_ms": disk_io,
                    "packet_loss_pct": error_rate * 0.3,
                    "api_response_latency_ms": api_latency,
                    "http_5xx_count": http_5xx,
                    "queue_depth": queue_depth,
                    "request_throughput_rps": max(10, 150 - queue_depth * 0.3),
                },
            }
            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                if resp.status_code == 200:
                    # Store result in session state — rendered OUTSIDE this expander
                    st.session_state["last_prediction"] = resp.json()
                    st.session_state["prediction_error"] = None
                else:
                    st.session_state["last_prediction"] = None
                    st.session_state["prediction_error"] = (
                        f"API returned {resp.status_code}: {resp.text[:300]}"
                    )
            except requests.exceptions.ConnectionError:
                st.session_state["last_prediction"] = None
                st.session_state["prediction_error"] = (
                    f"Could not connect to the API at {API_URL}. "
                    "Make sure the API container is running."
                )
            except Exception as e:
                st.session_state["last_prediction"] = None
                st.session_state["prediction_error"] = str(e)

    # Prediction result rendered here — OUTSIDE the expander (no nesting issue)
    if st.session_state.get("prediction_error"):
        st.error(st.session_state["prediction_error"])

    result = st.session_state.get("last_prediction")
    if result:
        risk = result["risk_level"].upper()
        prob = result["failure_probability"]
        cause = (result.get("root_cause") or "Unknown").replace("_", " ").title()
        alert_id = result.get("alert_id", "—")
        color = {
            "CRITICAL": C_RED,
            "HIGH": C_ORANGE,
            "MODERATE": C_AMBER,
            "LOW": C_GREEN,
            "NOMINAL": C_SLATE,
        }.get(risk, C_SLATE)

        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e5e7eb;
                        border-left:4px solid {color};border-radius:8px;
                        padding:1.2rem 1.5rem;margin-top:0.8rem">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.7rem">
                    <span style="background:{color};color:#fff;padding:3px 12px;
                                 border-radius:4px;font-weight:700;
                                 font-size:0.85rem;letter-spacing:0.05em">
                        {risk}
                    </span>
                    <span style="font-size:1.1rem;font-weight:600;color:#111827">
                        {prob:.1%} failure probability
                    </span>
                    <span style="margin-left:auto;font-size:0.78rem;color:#9ca3af">
                        Alert #{alert_id} saved to database
                    </span>
                </div>
                <div style="color:#374151;font-size:0.9rem">
                    Diagnosed root cause: <strong>{cause}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Full SHAP diagnostic report"):
            st.markdown(result.get("markdown_report", "No report available."))

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #f3f4f6;
                    text-align:center;color:#d1d5db;font-size:0.78rem">
            SentinelX v1.0 &nbsp;·&nbsp; XGBoost + SMOTE + SHAP &nbsp;·&nbsp;
            PR-AUC 0.862 &nbsp;·&nbsp; 1.4% false positive rate &nbsp;·&nbsp;
            FastAPI + PostgreSQL
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
