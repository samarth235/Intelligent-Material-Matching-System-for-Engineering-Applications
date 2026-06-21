"""
Intelligent Material Selection & Alloy Recommendation System
Streamlit Web Application — Professional Dark UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from recommender import MaterialRecommender
import numpy as np
from datetime import datetime
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MatSelect — Intelligent Material Matching",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design System CSS ───────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── App Background ── */
.stApp {
    background: #080d1a !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,130,246,0.12) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 90% 80%, rgba(6,182,212,0.06) 0%, transparent 60%) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1525 !important;
    border-right: 1px solid rgba(59,130,246,0.15) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] .sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.25rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(59,130,246,0.15);
    margin-bottom: 1.2rem;
}
[data-testid="stSidebar"] .sidebar-brand span {
    font-size: 1.15rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
[data-testid="stSidebar"] .sidebar-section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569 !important;
    margin: 1.4rem 0 0.5rem 0;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(59,130,246,0.12) !important;
    margin: 1rem 0 !important;
}

/* ── Radio (mode selector) ── */
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 8px;
    padding: 0.45rem 0.7rem !important;
    margin-bottom: 2px !important;
    transition: background 0.15s;
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.1);
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] span:first-child {
    display: none;
}

/* ── Sliders ── */
[data-testid="stSlider"] > label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] {
    display: none;
}
[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background: #3b82f6 !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.2) !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > label,
[data-testid="stMultiSelect"] > label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background: #111827 !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div:hover,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:hover {
    border-color: rgba(59,130,246,0.6) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563eb, #0891b2) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.5) !important;
    background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}
[data-testid="stDownloadButton"] > button {
    background: rgba(16,185,129,0.15) !important;
    color: #10b981 !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(16,185,129,0.25) !important;
    border-color: rgba(16,185,129,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Text Input ── */
[data-testid="stTextInput"] > label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stTextInput"] input {
    background: #111827 !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Number Input ── */
[data-testid="stNumberInput"] > label {
    color: #cbd5e1 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stNumberInput"] input {
    background: #111827 !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0d1525 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid rgba(59,130,246,0.12) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    color: #64748b !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
    border: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #94a3b8 !important;
    background: rgba(59,130,246,0.08) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #2563eb, #0891b2) !important;
    color: #fff !important;
    box-shadow: 0 2px 10px rgba(37,99,235,0.35) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid rgba(59,130,246,0.12) !important;
    border-radius: 10px !important;
    padding: 0.9rem 1rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(59,130,246,0.3) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #10b981 !important;
    font-size: 0.78rem !important;
}

/* ── Dataframe / Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(59,130,246,0.12) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] iframe {
    border-radius: 10px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {
    border-color: #3b82f6 transparent transparent transparent !important;
}

/* ── Alerts / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Headings in main content ── */
.main h1, .main h2, .main h3 {
    color: #f1f5f9 !important;
}

/* ── Markdown text ── */
.main p, .main li, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #94a3b8 !important;
}

/* ── Horizontal rule ── */
hr {
    border-color: rgba(59,130,246,0.1) !important;
    margin: 1.5rem 0 !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* ── Custom component classes ── */
.hero-container {
    text-align: center;
    padding: 3rem 2rem 2rem;
    margin-bottom: 0.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #60a5fa;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    font-weight: 800;
    line-height: 1.15;
    background: linear-gradient(135deg, #f1f5f9 0%, #60a5fa 40%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.02em;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 400;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius: 2px;
    margin: 1.5rem auto 0;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(59,130,246,0.1);
}
.section-header-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #2563eb22, #0891b222);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.section-header-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.card {
    background: #111827;
    border: 1px solid rgba(59,130,246,0.12);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.card:hover {
    border-color: rgba(59,130,246,0.28);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
}
.card-gold {
    border-color: rgba(251,191,36,0.35) !important;
    background: linear-gradient(160deg, #1a1505 0%, #111827 100%) !important;
}
.card-gold:hover { border-color: rgba(251,191,36,0.6) !important; }
.card-silver {
    border-color: rgba(148,163,184,0.35) !important;
    background: linear-gradient(160deg, #111827 0%, #0f1a2e 100%) !important;
}
.card-silver:hover { border-color: rgba(148,163,184,0.6) !important; }
.card-bronze {
    border-color: rgba(180,120,60,0.35) !important;
    background: linear-gradient(160deg, #160e05 0%, #111827 100%) !important;
}
.card-bronze:hover { border-color: rgba(180,120,60,0.6) !important; }
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    border-radius: 50%;
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #d97706); color: #fff; box-shadow: 0 2px 8px rgba(245,158,11,0.4); }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: #fff; box-shadow: 0 2px 8px rgba(100,116,139,0.4); }
.rank-3 { background: linear-gradient(135deg, #b4783c, #92612d); color: #fff; box-shadow: 0 2px 8px rgba(180,120,60,0.4); }
.material-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.3rem 0;
    line-height: 1.3;
}
.material-type-chip {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 0.7rem;
    font-weight: 600;
    color: #60a5fa;
    margin-bottom: 1rem;
    letter-spacing: 0.04em;
}
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(6,182,212,0.18));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 8px;
    padding: 6px 14px;
    margin-bottom: 1rem;
    width: 100%;
    justify-content: center;
}
.score-label { font-size: 0.72rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
.score-value { font-size: 1.2rem; font-weight: 800; color: #60a5fa; }
.prop-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.prop-item {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
}
.prop-label { font-size: 0.68rem; font-weight: 600; color: #475569; text-transform: uppercase; letter-spacing: 0.06em; }
.prop-value { font-size: 0.92rem; font-weight: 700; color: #cbd5e1; margin-top: 1px; }
.reason-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 100px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #34d399;
    margin: 3px 4px 3px 0;
}
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #60a5fa;
}
.footer {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    border-top: 1px solid rgba(59,130,246,0.08);
    margin-top: 3rem;
}
.footer-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.25rem;
}
.footer-sub {
    font-size: 0.75rem;
    color: #1e293b;
}
.subsection-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 0.75rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── Plotly Dark Template ─────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(17,24,39,0.6)',
    font=dict(color='#94a3b8', family='Inter, sans-serif', size=12),
    title_font=dict(color='#e2e8f0', size=14, family='Inter, sans-serif'),
    xaxis=dict(
        gridcolor='rgba(59,130,246,0.07)',
        linecolor='rgba(59,130,246,0.12)',
        tickfont=dict(color='#64748b', size=11),
        title_font=dict(color='#64748b'),
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor='rgba(59,130,246,0.07)',
        linecolor='rgba(59,130,246,0.12)',
        tickfont=dict(color='#64748b', size=11),
        title_font=dict(color='#64748b'),
        zeroline=False,
    ),
    legend=dict(
        bgcolor='rgba(13,21,37,0.8)',
        bordercolor='rgba(59,130,246,0.15)',
        borderwidth=1,
        font=dict(color='#94a3b8', size=11),
    ),
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=['#3b82f6', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'],
)

MEDAL_COLORS = ['rgba(245,158,11,0.7)', 'rgba(148,163,184,0.7)', 'rgba(180,120,60,0.7)']
CARD_CLASSES = ['card-gold', 'card-silver', 'card-bronze']
RANK_CLASSES = ['rank-1', 'rank-2', 'rank-3']
RANK_LABELS  = ['#1', '#2', '#3']

# ─── Session State ────────────────────────────────────────────────────────────
if 'recommender' not in st.session_state:
    db_path = 'materials_large.csv' if os.path.exists('materials_large.csv') else 'materials.csv'
    st.session_state.recommender = MaterialRecommender(db_path)

recommender = st.session_state.recommender

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span>⚙️ MatSelect</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Navigation</p>', unsafe_allow_html=True)

    mode = st.radio(
        "Mode",
        ["🔮  Quick Recommend", "🔍  Advanced Search", "📚  Material Database", "⚖️  Compare"],
        label_visibility="collapsed"
    )
    # Normalise to short key
    mode_key = mode.split("  ")[1].strip()

    st.markdown("---")

    if mode_key == "Quick Recommend":
        st.markdown('<p class="sidebar-section-label">Priority Weights (1 – 10)</p>', unsafe_allow_html=True)

        strength = st.slider("💪 Strength", 1, 10, 7)
        weight   = st.slider("🪶 Low Weight", 1, 10, 7)
        corrosion = st.slider("🛡️ Corrosion Resistance", 1, 10, 5)
        cost     = st.slider("💰 Low Cost", 1, 10, 3)
        thermal  = st.slider("🌡️ Thermal Conductivity", 1, 10, 3)

        st.markdown("---")
        st.markdown('<p class="sidebar-section-label">Application Filter</p>', unsafe_allow_html=True)
        application = st.selectbox(
            "Domain",
            ["All"] + recommender.get_available_applications(),
            label_visibility="collapsed"
        )
        recommend_button = st.button("Get Recommendations →", use_container_width=True)

# ─── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚙️ Engineering Intelligence</div>
    <h1 class="hero-title">Intelligent Material<br>Matching System</h1>
    <p class="hero-subtitle">Precision material recommendations powered by weighted scoring and machine learning for engineering applications.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ─── Mode: Quick Recommend ────────────────────────────────────────────────────
if mode_key == "Quick Recommend":
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🔮</div>
        <p class="section-header-text">Material Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    if recommend_button:
        with st.spinner("Analysing database…"):
            app_filter = None if application == "All" else application
            recommendations = recommender.get_weighted_recommendations(
                strength_weight=strength,
                weight_weight=weight,
                corrosion_weight=corrosion,
                cost_weight=cost,
                thermal_weight=thermal,
                top_n=5,
                application_filter=app_filter
            )

        if len(recommendations) == 0:
            st.error(f"No materials found for the **{application}** application. Try a different filter.")
        else:
            top3 = recommendations.head(3)

            # ── Top 3 Cards ──────────────────────────────────────────────
            st.markdown('<p class="subsection-title">🏆 Top Recommendations</p>', unsafe_allow_html=True)
            cols = st.columns(3, gap="medium")

            for idx, (_, row) in enumerate(top3.iterrows()):
                card_cls  = CARD_CLASSES[idx]
                rank_cls  = RANK_CLASSES[idx]
                rank_lbl  = RANK_LABELS[idx]
                alloy_type = str(row.get('Alloy_Type', '')).strip()

                with cols[idx]:
                    st.markdown(f"""
                    <div class="card {card_cls}">
                        <div class="rank-badge {rank_cls}">{rank_lbl}</div>
                        <div class="material-name">{row['Material']}</div>
                        {'<div class="material-type-chip">' + alloy_type + '</div>' if alloy_type else ''}
                        <div class="score-badge">
                            <span class="score-label">Score</span>
                            <span class="score-value">{row['Recommendation_Score']:.1f}</span>
                        </div>
                        <div class="prop-grid">
                            <div class="prop-item">
                                <div class="prop-label">Tensile Str.</div>
                                <div class="prop-value">{row['Tensile_Strength_MPa']:.0f} <small style="font-size:0.6rem;color:#475569">MPa</small></div>
                            </div>
                            <div class="prop-item">
                                <div class="prop-label">Density</div>
                                <div class="prop-value">{row['Density_g_cm3']:.2f} <small style="font-size:0.6rem;color:#475569">g/cm³</small></div>
                            </div>
                            <div class="prop-item">
                                <div class="prop-label">Corrosion</div>
                                <div class="prop-value">{row['Corrosion_Resistance_0_10']:.1f}<small style="font-size:0.6rem;color:#475569">/10</small></div>
                            </div>
                            <div class="prop-item">
                                <div class="prop-label">Cost Index</div>
                                <div class="prop-value">{row['Cost_Index_1_10']:.1f}<small style="font-size:0.6rem;color:#475569">/10</small></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Detail Tabs ───────────────────────────────────────────────
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊  Charts", "🔬  Material Detail", "📋  Properties Table", "💡  Selection Reasoning"
            ])

            # ── Tab 1: Charts ─────────────────────────────────────────────
            with tab1:
                st.markdown('<p class="subsection-title">Radar — Normalised Property Profile</p>', unsafe_allow_html=True)
                categories = [
                    'Strength', 'Low Density', 'Corrosion', 'Low Cost', 'Thermal', 'Hardness'
                ]

                radar_fig = go.Figure()
                for i, (_, mat) in enumerate(top3.iterrows()):
                    vals = [
                        mat['Tensile_Strength_MPa_norm'],
                        10 - mat['Density_g_cm3_norm'],
                        mat['Corrosion_Resistance_0_10_norm'],
                        10 - mat['Cost_Index_1_10_norm'],
                        mat['Thermal_Conductivity_W_mK_norm'],
                        mat['Hardness_HB_norm'],
                    ]
                    color = ['#3b82f6', '#06b6d4', '#8b5cf6'][i]
                    radar_fig.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=mat['Material'],
                        line=dict(color=color, width=2),
                        fillcolor=color.replace(')', ', 0.1)').replace('rgb', 'rgba') if 'rgb' in color else color + '1a',
                        marker=dict(color=color, size=6),
                    ))

                radar_fig.update_layout(
                    **PLOTLY_LAYOUT,
                    polar=dict(
                        bgcolor='rgba(17,24,39,0.4)',
                        radialaxis=dict(
                            visible=True, range=[0, 10],
                            gridcolor='rgba(59,130,246,0.1)',
                            linecolor='rgba(59,130,246,0.1)',
                            tickfont=dict(color='#475569', size=9),
                        ),
                        angularaxis=dict(
                            gridcolor='rgba(59,130,246,0.1)',
                            linecolor='rgba(59,130,246,0.1)',
                            tickfont=dict(color='#94a3b8', size=11),
                        ),
                    ),
                    margin=dict(l=60, r=60, t=30, b=30),
                    height=400,
                )
                st.plotly_chart(radar_fig, use_container_width=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown('<p class="subsection-title">Property Comparison</p>', unsafe_allow_html=True)

                mat_names = top3['Material'].tolist()
                c1, c2 = st.columns(2)

                def make_bar(df, col, title, color_scale):
                    fig = px.bar(df, x='Material', y=col, title=title,
                                 color=col, color_continuous_scale=color_scale)
                    fig.update_layout(**PLOTLY_LAYOUT, title_text=title,
                                      showlegend=False, height=280,
                                      coloraxis_showscale=False)
                    fig.update_traces(marker_line_width=0, width=0.45)
                    return fig

                with c1:
                    st.plotly_chart(
                        make_bar(top3, 'Tensile_Strength_MPa', 'Tensile Strength (MPa)', 'Blues'),
                        use_container_width=True)
                with c2:
                    st.plotly_chart(
                        make_bar(top3, 'Density_g_cm3', 'Density — lower is better (g/cm³)', 'Reds_r'),
                        use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(
                        make_bar(top3, 'Corrosion_Resistance_0_10', 'Corrosion Resistance (/10)', 'Greens'),
                        use_container_width=True)
                with c4:
                    st.plotly_chart(
                        make_bar(top3, 'Cost_Index_1_10', 'Cost Index — lower is better (/10)', 'Oranges_r'),
                        use_container_width=True)

            # ── Tab 2: Material Detail ────────────────────────────────────
            with tab2:
                selected_material = st.selectbox(
                    "Select material to inspect:",
                    top3['Material'].tolist()
                )
                explanation = recommender.get_material_explanation(selected_material)
                # Render in a card-like container
                st.markdown(f"""
                <div class="card" style="margin-top:0.5rem;">
                """, unsafe_allow_html=True)
                st.markdown(explanation)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Tab 3: Properties Table ───────────────────────────────────
            with tab3:
                display_cols = [
                    'Material', 'Alloy_Type', 'Density_g_cm3',
                    'Yield_Strength_MPa', 'Tensile_Strength_MPa',
                    'Hardness_HB', 'Corrosion_Resistance_0_10',
                    'Thermal_Conductivity_W_mK', 'Cost_Index_1_10',
                    'Melting_Point_C', 'Machinability',
                ]
                avail_cols = [c for c in display_cols if c in top3.columns]
                st.dataframe(
                    top3[avail_cols].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

            # ── Tab 4: Selection Reasoning ────────────────────────────────
            with tab4:
                for idx, (_, mat) in enumerate(top3.iterrows()):
                    rank_cls = RANK_CLASSES[idx]
                    rank_lbl = RANK_LABELS[idx]

                    reasons = []
                    if strength >= 7 and mat['Tensile_Strength_MPa'] > 600:
                        reasons.append("High tensile strength")
                    elif strength >= 5 and mat['Tensile_Strength_MPa'] > 400:
                        reasons.append("Adequate strength")
                    if weight >= 7 and mat['Density_g_cm3'] < 5:
                        reasons.append("Lightweight")
                    if corrosion >= 7 and mat['Corrosion_Resistance_0_10'] >= 9:
                        reasons.append("Excellent corrosion resistance")
                    elif corrosion >= 5 and mat['Corrosion_Resistance_0_10'] >= 7:
                        reasons.append("Good corrosion resistance")
                    if cost <= 3 and mat['Cost_Index_1_10'] <= 4:
                        reasons.append("Cost effective")
                    if thermal >= 7 and mat['Thermal_Conductivity_W_mK'] > 100:
                        reasons.append("High thermal conductivity")
                    elif thermal >= 5 and mat['Thermal_Conductivity_W_mK'] > 30:
                        reasons.append("Good thermal performance")
                    if mat['Machinability'] >= 8:
                        reasons.append("Excellent machinability")

                    chips_html = "".join(
                        f'<span class="reason-chip">✓ {r}</span>' for r in reasons
                    ) if reasons else '<span class="reason-chip">✓ Specialised properties</span>'

                    st.markdown(f"""
                    <div class="card" style="margin-bottom:0.75rem;">
                        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">
                            <div class="rank-badge {rank_cls}" style="margin-bottom:0">{rank_lbl}</div>
                            <div class="material-name" style="margin:0">{mat['Material']}</div>
                        </div>
                        <div>{chips_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Empty state prompt
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem;margin-top:1rem;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;">⚗️</div>
            <div style="font-size:1rem;font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;">
                Configure your priorities
            </div>
            <div style="font-size:0.88rem;color:#475569;">
                Adjust the sliders in the sidebar and click <strong style="color:#60a5fa;">Get Recommendations →</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Mode: Advanced Search ────────────────────────────────────────────────────
elif mode_key == "Advanced Search":
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🔍</div>
        <p class="section-header-text">Advanced Material Search</p>
    </div>
    """, unsafe_allow_html=True)

    search_column_map = {
        "Tensile Strength (MPa)": "Tensile_Strength_MPa",
        "Density (g/cm³)":        "Density_g_cm3",
        "Corrosion Resistance":   "Corrosion_Resistance_0_10",
        "Cost Index":             "Cost_Index_1_10",
        "Thermal Conductivity":   "Thermal_Conductivity_W_mK",
    }

    c1, c2, c3 = st.columns([2, 1, 1], gap="medium")
    with c1:
        search_type = st.selectbox("Filter Property", list(search_column_map.keys()))
    with c2:
        min_val = st.number_input("Min Value", value=0.0, step=1.0)
    with c3:
        max_val = st.number_input("Max Value", value=1000.0, step=1.0)

    search_col = search_column_map[search_type]
    filtered = recommender.materials_df[
        (recommender.materials_df[search_col] >= min_val) &
        (recommender.materials_df[search_col] <= max_val)
    ].sort_values(search_col, ascending=False)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-chip">📦 {len(filtered)} materials found</div>
        <div class="stat-chip">📏 Range: {min_val} – {max_val}</div>
    </div>
    """, unsafe_allow_html=True)

    avail = [c for c in ['Material', 'Alloy_Type', search_col, 'Primary_Application'] if c in filtered.columns]
    st.dataframe(
        filtered[avail].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    # Distribution chart
    if len(filtered) > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        hist_fig = px.histogram(
            filtered, x=search_col,
            nbins=30,
            title=f"Distribution — {search_type}",
            color_discrete_sequence=['#3b82f6'],
        )
        hist_fig.update_layout(**PLOTLY_LAYOUT, height=280)
        hist_fig.update_traces(marker_line_width=0)
        st.plotly_chart(hist_fig, use_container_width=True)


# ─── Mode: Material Database ──────────────────────────────────────────────────
elif mode_key == "Material Database":
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📚</div>
        <p class="section-header-text">Complete Material Database</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([3, 1], gap="medium")
    with c1:
        search_term = st.text_input("🔎  Search by name…", placeholder="e.g. Aluminium, Ti-6Al-4V, Steel…")
    with c2:
        app_filter = st.selectbox(
            "Application",
            ["All"] + recommender.get_available_applications()
        )

    filtered_db = recommender.materials_df.copy()
    if search_term:
        filtered_db = filtered_db[
            filtered_db['Material'].str.contains(search_term, case=False, na=False)
        ]
    if app_filter != "All":
        filtered_db = filtered_db[
            filtered_db['Primary_Application'].str.contains(app_filter, case=False, na=False)
        ]

    total_all = len(recommender.materials_df)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-chip">📦 {len(filtered_db)} of {total_all} materials</div>
        {'<div class="stat-chip">🔎 ' + search_term + '</div>' if search_term else ''}
        {'<div class="stat-chip">🏭 ' + app_filter + '</div>' if app_filter != "All" else ''}
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(filtered_db.reset_index(drop=True), use_container_width=True, hide_index=True)

    csv = filtered_db.to_csv(index=False)
    st.download_button(
        label="📥  Download CSV",
        data=csv,
        file_name=f"materials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ─── Mode: Compare ────────────────────────────────────────────────────────────
elif mode_key == "Compare":
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">⚖️</div>
        <p class="section-header-text">Material Comparison</p>
    </div>
    """, unsafe_allow_html=True)

    all_mats = recommender.get_material_names()

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        m1 = st.selectbox("Material A", all_mats, key="m1")
    with c2:
        m2 = st.selectbox("Material B", all_mats, index=min(1, len(all_mats)-1), key="m2")
    with c3:
        enable_m3 = st.checkbox("Add a third material")
        m3 = None
        if enable_m3:
            m3 = st.selectbox("Material C", all_mats, index=min(2, len(all_mats)-1), key="m3")

    materials_to_compare = [m for m in [m1, m2, m3] if m]
    comparison = recommender.compare_materials(materials_to_compare)

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar for comparison
    comp_props = ['Tensile_Strength_MPa', 'Density_g_cm3', 'Corrosion_Resistance_0_10',
                  'Cost_Index_1_10', 'Thermal_Conductivity_W_mK', 'Hardness_HB']
    comp_labels = ['Strength (MPa)', 'Density (g/cm³)', 'Corrosion', 'Cost', 'Thermal', 'Hardness']

    normalized_comp = comparison.copy()
    for col in comp_props:
        if col in normalized_comp.columns:
            col_min = recommender.materials_df[col].min()
            col_max = recommender.materials_df[col].max()
            if col_max - col_min != 0:
                normalized_comp[col + '_n'] = (normalized_comp[col] - col_min) / (col_max - col_min) * 10
            else:
                normalized_comp[col + '_n'] = 5

    comp_radar = go.Figure()
    colors = ['#3b82f6', '#06b6d4', '#8b5cf6']
    for i, (_, row) in enumerate(normalized_comp.iterrows()):
        vals = [row.get(p + '_n', 0) for p in comp_props]
        c = colors[i % len(colors)]
        comp_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=comp_labels + [comp_labels[0]],
            fill='toself',
            name=row['Material'],
            line=dict(color=c, width=2),
            marker=dict(color=c, size=6),
        ))

    comp_radar.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(17,24,39,0.4)',
            radialaxis=dict(visible=True, range=[0, 10],
                gridcolor='rgba(59,130,246,0.1)', linecolor='rgba(59,130,246,0.1)',
                tickfont=dict(color='#475569', size=9)),
            angularaxis=dict(
                gridcolor='rgba(59,130,246,0.1)', linecolor='rgba(59,130,246,0.1)',
                tickfont=dict(color='#94a3b8', size=11)),
        ),
        height=420, margin=dict(l=60, r=60, t=30, b=30),
    )
    st.plotly_chart(comp_radar, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="subsection-title">Side-by-Side Properties</p>', unsafe_allow_html=True)

    def cmp_bar(col, title):
        fig = px.bar(comparison, x='Material', y=col, title=title,
                     color='Material', color_discrete_sequence=['#3b82f6', '#06b6d4', '#8b5cf6'])
        fig.update_layout(**PLOTLY_LAYOUT, height=270, showlegend=False)
        fig.update_traces(marker_line_width=0, width=0.45)
        return fig

    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(cmp_bar('Tensile_Strength_MPa', 'Tensile Strength (MPa)'), use_container_width=True)
    with c2: st.plotly_chart(cmp_bar('Density_g_cm3', 'Density (g/cm³)'), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(cmp_bar('Corrosion_Resistance_0_10', 'Corrosion Resistance (/10)'), use_container_width=True)
    with c4: st.plotly_chart(cmp_bar('Cost_Index_1_10', 'Cost Index (/10)'), use_container_width=True)

    st.markdown('<p class="subsection-title">Full Comparison Table</p>', unsafe_allow_html=True)
    st.dataframe(comparison.reset_index(drop=True), use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-title">⚙️ Intelligent Material Matching System</div>
    <div class="footer-sub">Engineering Material Selection · RVCE · Data sourced from material databases and engineering handbooks</div>
</div>
""", unsafe_allow_html=True)
