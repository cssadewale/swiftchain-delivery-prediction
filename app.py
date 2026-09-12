"""
SwiftChain Analytics — Delivery Delay Prediction
Streamlit Web Application

Author  : Adewale Samson Adeagbo
Email   : buildingmyictcareer@gmail.com
Phone   : +2348100866322
GitHub  : github.com/cssadewale
LinkedIn: linkedin.com/in/adewalesamsonadeagbo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwiftChain Delivery Predictor",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --primary:   #00c896;
        --danger:    #e74c3c;
        --warning:   #f39c12;
        --dark:      #0d1117;
        --card:      #161b22;
        --border:    #30363d;
        --text:      #e6edf3;
        --subtext:   #8b949e;
    }

    /* Background */
    .stApp { background-color: var(--dark); color: var(--text); }
    .stApp > header { background-color: transparent; }

    /* Main font */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* Metric cards */
    .metric-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
        line-height: 1;
    }
    .metric-label {
        font-size: 0.78rem;
        color: var(--subtext);
        margin-top: 6px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    /* Risk result box */
    .result-late {
        background: linear-gradient(135deg, #2d1515 0%, #1a0f0f 100%);
        border: 1px solid var(--danger);
        border-left: 4px solid var(--danger);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
    }
    .result-ontime {
        background: linear-gradient(135deg, #1a1a0f 0%, #131310 100%);
        border: 1px solid var(--warning);
        border-left: 4px solid var(--warning);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
    }
    .result-early {
        background: linear-gradient(135deg, #0d2018 0%, #0a1812 100%);
        border: 1px solid var(--primary);
        border-left: 4px solid var(--primary);
        border-radius: 12px;
        padding: 24px 28px;
        margin: 16px 0;
    }
    .result-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-subtitle {
        color: var(--subtext);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Section headers */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary);
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    /* Stat bar */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
        font-size: 0.88rem;
    }
    .stat-label { color: var(--subtext); }
    .stat-val { font-family: 'Space Mono', monospace; color: var(--text); }

    /* Insight pill */
    .insight-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .pill-red   { background: rgba(231,76,60,0.18);  color: #e74c3c; border: 1px solid rgba(231,76,60,0.35); }
    .pill-green { background: rgba(0,200,150,0.15);  color: #00c896; border: 1px solid rgba(0,200,150,0.3); }
    .pill-amber { background: rgba(243,156,18,0.15); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }

    /* Streamlit overrides */
    .stSelectbox > div > div, .stNumberInput > div > div > input,
    .stSlider > div { background: var(--card) !important; }
    div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }
    .stSidebar { background-color: #0d1117; border-right: 1px solid var(--border); }
    .stButton > button {
        background: linear-gradient(135deg, #00c896, #00a87a);
        color: #0d1117;
        font-weight: 700;
        font-family: 'Space Mono', monospace;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 0.95rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e6ad, #00c896);
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,200,150,0.35);
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model_artifacts():
    """
    Load the trained GradientBoostingClassifier and fitted StandardScaler.
    Both files must exist in the same directory as app.py.
    """
    base_dir = Path(__file__).resolve().parent
    model_path  = base_dir / "swiftchain_delay_predictor.pkl"
    scaler_path = base_dir / "swiftchain_scaler.pkl"

    if not os.path.exists(model_path):
        return None, None, f"Model file not found: '{model_path}'"
    if not os.path.exists(scaler_path):
        return None, None, f"Scaler file not found: '{scaler_path}'"

    try:
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Fail early with a useful message if someone uploads mismatched
        # artifacts. The model was trained on 309 columns and the scaler on
        # the 15 numeric columns listed in scaler.feature_names_in_.
        if not hasattr(model, "feature_names_in_"):
            raise ValueError("The model artifact has no feature_names_in_ metadata")
        if not hasattr(scaler, "feature_names_in_"):
            raise ValueError("The scaler artifact has no feature_names_in_ metadata")
        missing_numeric = [
            col for col in scaler.feature_names_in_
            if col not in model.feature_names_in_
        ]
        if missing_numeric:
            raise ValueError(
                "Scaler columns missing from model schema: "
                + ", ".join(missing_numeric)
            )
        if len(model.feature_names_in_) != 309:
            raise ValueError(
                f"Unexpected model schema: expected 309 features, "
                f"found {len(model.feature_names_in_)}"
            )
        return model, scaler, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

model, scaler, load_error = load_model_artifacts()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px 0;'>
        <div style='font-size:2.2rem; margin-bottom:6px;'>🚚</div>
        <div style='font-family:"Space Mono",monospace; font-size:0.85rem;
                    color:#00c896; letter-spacing:0.08em;'>SWIFTCHAIN</div>
        <div style='font-size:0.7rem; color:#8b949e; margin-top:2px;'>
            Delivery Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Model Performance</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.84rem; line-height:2;'>
        <div class="stat-row">
            <span class="stat-label">Algorithm</span>
            <span class="stat-val" style='font-size:0.75rem;'>Gradient Boosting</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Test Accuracy</span>
            <span class="stat-val" style='color:#00c896;'>62.0%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Weighted F1</span>
            <span class="stat-val" style='color:#00c896;'>0.5791</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">CV F1 (5-fold)</span>
            <span class="stat-val">0.5768 ± 0.0091</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Late Recall</span>
            <span class="stat-val" style='color:#e74c3c;'>68.1%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Early Recall</span>
            <span class="stat-val" style='color:#00c896;'>78.0%</span>
        </div>
        <div class="stat-row" style='border:none;'>
            <span class="stat-label">Training Records</span>
            <span class="stat-val">15,549</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Top Features</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; line-height:1.9; color:#8b949e;'>
        <div>🔴 <b style='color:#e6edf3;'>Shipping Mode</b> — 67.2%</div>
        <div>🟡 <b style='color:#e6edf3;'>Dispatch Lag</b> — 17.9%</div>
        <div>⬜ <b style='color:#e6edf3;'>Other 307 features</b> — 14.9%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Built By</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.82rem; color:#8b949e; line-height:2;'>
        <div><b style='color:#e6edf3;'>Adewale Samson Adeagbo</b></div>
        <div>Lead Data Scientist</div>
        <div><a href='https://linkedin.com/in/adewalesamsonadeagbo'
               style='color:#00c896; text-decoration:none;'>LinkedIn ↗</a></div>
        <div><a href='https://github.com/cssadewale'
               style='color:#00c896; text-decoration:none;'>GitHub ↗</a></div>
    </div>
    """, unsafe_allow_html=True)


# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 28px 0;'>
    <div style='font-family:"Space Mono",monospace; font-size:1.65rem;
                font-weight:700; color:#e6edf3; line-height:1.2;'>
        SwiftChain Delivery Delay Predictor
    </div>
    <div style='color:#8b949e; font-size:0.92rem; margin-top:8px; max-width:560px;'>
        Enter order and shipping details below. The model will assess the delivery
        risk based on 309 features learned from 15,549 global logistics orders
        (2015 – 2018).
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top metric strip ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">Outcome Classes</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">67%</div>
        <div class="metric-label">Top Feature Weight</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">62%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">5</div>
        <div class="metric-label">Global Markets</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Order & Shipping Details</div>',
            unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("**Shipping**")
    shipping_mode = st.selectbox(
        "Shipping Mode ⭐",
        ["Standard Class", "Second Class", "First Class", "Same Day"],
        help="Most important predictor — accounts for 67.2% of model importance."
    )
    shipping_duration = st.number_input(
        "Dispatch Lag (days) ⭐",
        min_value=0, max_value=365, value=7,
        help="Days between order placement and actual dispatch. "
             "Second most important feature (17.9% importance). "
             "Recommended SLA: ≤ 3 days for Standard Class."
    )
    market = st.selectbox(
        "Market",
        ["Europe", "LATAM", "USCA", "Pacific Asia", "Africa"],
        help="Operating market. All five markets perform within ±1.7 pp of the "
             "22.8% global late rate — market alone is not a strong predictor."
    )

with col_right:
    st.markdown("**Order Profile**")
    customer_segment = st.selectbox(
        "Customer Segment",
        ["Consumer", "Corporate", "Home Office"],
        help="Consumer is the largest segment (~53.6% of orders)."
    )
    order_item_quantity = st.slider(
        "Order Item Quantity",
        min_value=1, max_value=5, value=2,
        help="Number of units in this order line."
    )
    profit_per_order = st.number_input(
        "Profit Per Order ($)",
        min_value=-500.0, max_value=500.0, value=25.0, step=5.0,
        help="Financial features have near-zero model importance, "
             "but are part of the full feature vector."
    )

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮  Predict Delivery Outcome", use_container_width=False)


# ── Prediction & Risk Output ──────────────────────────────────────────────────
def build_model_input(shipping_mode, shipping_duration, market,
                      customer_segment, order_item_quantity, profit_per_order):
    """Build the exact 309-column schema used by the saved model.

    The public UI intentionally exposes only six operational inputs. Other
    numeric fields are imputed with the training scaler means and other
    categorical fields use their training reference category. This is a
    transparent partial-information prediction, not a replacement for a
    complete order-level feature record.
    """
    if model is None or scaler is None:
        raise RuntimeError(load_error or "Model artifacts are unavailable")

    feature_names = list(model.feature_names_in_)
    numeric_names = list(scaler.feature_names_in_)
    row = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Start numeric features at their training means. After scaling, these
    # become zero (the neutral value expected by the trained estimator).
    for name, mean in zip(numeric_names, scaler.mean_):
        if name in row.columns:
            row.at[0, name] = float(mean)

    supplied_numeric = {
        "shipping_duration": shipping_duration,
        "order_item_quantity": order_item_quantity,
        "profit_per_order": profit_per_order,
    }
    for name, value in supplied_numeric.items():
        if name in row.columns:
            row.at[0, name] = float(value)

    # One-hot columns use the same drop-first convention as training. The
    # omitted categories are the reference levels (First Class, Africa,
    # Consumer), so they correctly remain all-zero.
    categorical_values = {
        f"shipping_mode_{shipping_mode}": 1.0,
        f"market_{market}": 1.0,
        f"customer_segment_{customer_segment}": 1.0,
    }
    for name, value in categorical_values.items():
        if name in row.columns:
            row.at[0, name] = value

    # Scale exactly the 15 numeric features and preserve the fitted order.
    row[numeric_names] = scaler.transform(row[numeric_names])
    return row


if predict_clicked:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">ML Delivery Assessment</div>',
                unsafe_allow_html=True)

    if load_error or model is None or scaler is None:
        st.error(f"The trained delivery model could not be loaded: {load_error}")
    else:
        try:
            model_input = build_model_input(
                shipping_mode, shipping_duration, market,
                customer_segment, order_item_quantity, profit_per_order
            )
            predicted_class = int(model.predict(model_input)[0])
            probabilities = model.predict_proba(model_input)[0]
            class_probabilities = dict(zip(model.classes_, probabilities))
            label_map = {-1: "Late", 0: "On-Time", 1: "Early"}
            outcome_label = label_map.get(predicted_class, str(predicted_class))
            confidence = float(class_probabilities[predicted_class])

            if predicted_class == -1:
                st.error(f"🔴 **Predicted outcome: {outcome_label}**")
            elif predicted_class == 0:
                st.warning(f"🟡 **Predicted outcome: {outcome_label}**")
            else:
                st.success(f"🟢 **Predicted outcome: {outcome_label}**")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted outcome", outcome_label)
            c2.metric("Model confidence", f"{confidence:.1%}")
            c3.metric("Model features", str(len(model.feature_names_in_)))

            probability_table = pd.DataFrame({
                "Outcome": [label_map.get(int(c), str(c)) for c in model.classes_],
                "Probability": [float(p) for p in probabilities],
            })
            st.dataframe(
                probability_table.style.format({"Probability": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.info(
                "This prediction uses the six fields shown above. Uncollected "
                "order, customer, product and location fields are imputed at "
                "their training reference values. For production use, pass the "
                "complete 41-field order record through a persisted preprocessing pipeline."
            )
        except Exception as exc:
            st.error(f"Prediction failed because the input schema does not match the artifact: {exc}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #30363d; padding-top:20px;
            display:flex; justify-content:space-between; align-items:center;
            flex-wrap:wrap; gap:8px;'>
    <div style='font-size:0.78rem; color:#8b949e;'>
        Built by <b style='color:#e6edf3;'>Adewale Samson Adeagbo</b>
        &nbsp;·&nbsp; Lead Data Scientist / ML Engineer &nbsp;·&nbsp; Lagos, Nigeria
    </div>
    <div style='font-size:0.78rem;'>
        <a href='https://linkedin.com/in/adewalesamsonadeagbo'
           style='color:#00c896; text-decoration:none; margin-right:16px;'>LinkedIn ↗</a>
        <a href='https://github.com/cssadewale'
           style='color:#00c896; text-decoration:none;'>GitHub ↗</a>
    </div>
</div>
""", unsafe_allow_html=True)
