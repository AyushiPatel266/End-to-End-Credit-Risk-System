import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

if dark:
    bg_primary      = "#0a0f1e"
    bg_card         = "#0d1526"
    text_primary    = "#f0f4ff"
    text_secondary  = "#a0aec0"
    text_muted      = "#8892aa"
    border_color    = "rgba(255,255,255,0.06)"
    accent          = "#63b3ed"
    accent_dim      = "rgba(99,179,237,0.1)"
    accent_border   = "rgba(99,179,237,0.3)"
    input_bg        = "#0d1526"
    input_border    = "rgba(255,255,255,0.1)"
    header_bg       = "linear-gradient(135deg, #0d1b3e 0%, #0a0f1e 50%, #0d2137 100%)"
    header_border   = "rgba(99,179,237,0.15)"
    plot_bg         = "#0d1526"
    plot_text       = "#e8eaf0"
    divider         = "rgba(255,255,255,0.06)"
    toggle_icon     = "☀️"
    toggle_label    = "Light mode"
    spine_color     = "#2d3748"
    high_card_bg    = "linear-gradient(135deg, #2d0f0f 0%, #1a0a0a 100%)"
    high_card_br    = "rgba(245,101,101,0.4)"
    low_card_bg     = "linear-gradient(135deg, #0d2d1a 0%, #0a1a0f 100%)"
    low_card_br     = "rgba(72,187,120,0.4)"
    high_text       = "#fc8181"
    low_text        = "#68d391"
    insight_high_bg = "rgba(245,101,101,0.08)"
    insight_low_bg  = "rgba(72,187,120,0.08)"
    insight_high_cl = "#fbd5d5"
    insight_low_cl  = "#c6f6d5"
    toggle_btn_bg     = "rgba(99,179,237,0.1)"
    toggle_btn_border = "rgba(99,179,237,0.3)"
    toggle_btn_color  = "#63b3ed"
    toggle_btn_hover  = "rgba(99,179,237,0.18)"
else:
    bg_primary      = "#f7f9fc"
    bg_card         = "#ffffff"
    text_primary    = "#1a202c"
    text_secondary  = "#4a5568"
    text_muted      = "#718096"
    border_color    = "rgba(0,0,0,0.08)"
    accent          = "#2b6cb0"
    accent_dim      = "rgba(43,108,176,0.08)"
    accent_border   = "rgba(43,108,176,0.25)"
    input_bg        = "#ffffff"
    input_border    = "rgba(0,0,0,0.12)"
    header_bg       = "linear-gradient(135deg, #ebf4ff 0%, #f7f9fc 50%, #e8f0fe 100%)"
    header_border   = "rgba(43,108,176,0.2)"
    plot_bg         = "#ffffff"
    plot_text       = "#1a202c"
    divider         = "rgba(0,0,0,0.07)"
    toggle_icon     = "🌙"
    toggle_label    = "Dark mode"
    spine_color     = "#e2e8f0"
    high_card_bg    = "linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%)"
    high_card_br    = "rgba(229,62,62,0.3)"
    low_card_bg     = "linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%)"
    low_card_br     = "rgba(56,161,105,0.3)"
    high_text       = "#c53030"
    low_text        = "#276749"
    insight_high_bg = "rgba(254,215,215,0.6)"
    insight_low_bg  = "rgba(198,246,213,0.6)"
    insight_high_cl = "#742a2a"
    insight_low_cl  = "#1c4532"
    toggle_btn_bg     = "rgba(43,108,176,0.08)"
    toggle_btn_border = "rgba(43,108,176,0.25)"
    toggle_btn_color  = "#2b6cb0"
    toggle_btn_hover  = "rgba(43,108,176,0.15)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.stApp {{ background: {bg_primary}; color: {text_primary}; }}

.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    width: 100% !important;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

p, label, .stMarkdown,
div[data-testid="stMarkdownContainer"] p {{
    color: {text_secondary} !important;
}}

input[type="number"] {{
    background: {input_bg} !important;
    color: {text_primary} !important;
    border: 1px solid {input_border} !important;
    border-radius: 8px !important;
}}
div[data-testid="stNumberInput"] > div {{
    background: {input_bg} !important;
    border: 1px solid {input_border} !important;
    border-radius: 8px !important;
}}
div[data-testid="stNumberInput"] button {{
    background: {bg_card} !important;
    color: {text_secondary} !important;
    border: none !important;
}}
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {{
    color: {text_secondary} !important;
    font-size: 0.875rem !important;
}}

/* Both buttons — same style as info banner */
.stButton > button {{
    background: {toggle_btn_bg} !important;
    border: 1px solid {toggle_btn_border} !important;
    color: {toggle_btn_color} !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    font-family: 'DM Sans', sans-serif !important;
    width: 100% !important;
    min-height: 46px !important;
    transition: all 0.2s ease !important;
    transform: none !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}}
.stButton > button:hover {{
    background: {toggle_btn_hover} !important;
    transform: none !important;
    box-shadow: none !important;
}}

hr {{ border-color: {divider} !important; margin: 1rem 0 !important; }}

.main-header {{
    background: {header_bg};
    border: 1px solid {header_border};
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}}
.main-header::before {{
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 350px; height: 350px;
    background: radial-gradient(circle, {accent_dim} 0%, transparent 70%);
    pointer-events: none;
}}
.header-badge {{
    display: inline-block;
    background: {accent_dim};
    border: 1px solid {accent_border};
    color: {accent};
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.7rem;
}}
.header-title {{
    font-size: 2rem;
    font-weight: 600;
    color: {text_primary};
    margin: 0.3rem 0;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}
.header-subtitle {{
    color: {text_muted};
    font-size: 0.9rem;
    font-weight: 300;
    margin-top: 0.2rem;
}}

.section-label {{
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {accent};
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.8rem;
    margin-top: 0.3rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {accent_border};
}}

.info-box {{
    background: {accent_dim};
    border: 1px solid {accent_border};
    border-radius: 8px;
    padding: 0.7rem 1.1rem;
    font-size: 0.82rem;
    color: {accent} !important;
    margin-bottom: 1rem;
    line-height: 1.5;
}}

.result-card-high {{
    background: {high_card_bg};
    border: 1px solid {high_card_br};
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}}
.result-card-low {{
    background: {low_card_bg};
    border: 1px solid {low_card_br};
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}}
.result-label {{
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 0.4rem;
}}
.result-decision {{
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}}
.prob-display {{
    font-size: 3rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    letter-spacing: -0.03em;
    margin: 0.3rem 0;
}}
.prob-sub {{ color: {text_muted}; font-size: 0.8rem; }}

.metric-card {{
    background: {bg_card};
    border: 1px solid {border_color};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}}
.metric-value {{
    font-size: 1.5rem;
    font-weight: 600;
    font-family: 'DM Mono', monospace;
    color: {text_primary};
}}
.metric-label {{
    font-size: 0.72rem;
    color: {text_muted};
    margin-top: 0.2rem;
    letter-spacing: 0.02em;
}}

.insight-box {{
    border-radius: 8px;
    padding: 1rem 1.3rem;
    margin-top: 0.8rem;
    font-size: 0.875rem;
    line-height: 1.6;
}}
.insight-high {{
    background: {insight_high_bg};
    border-left: 3px solid {high_text};
    color: {insight_high_cl} !important;
}}
.insight-low {{
    background: {insight_low_bg};
    border-left: 3px solid {low_text};
    color: {insight_low_cl} !important;
}}

.stCaption {{ color: {text_muted} !important; font-size: 0.78rem !important; }}

.footer {{
    text-align: center;
    color: {text_muted};
    font-size: 0.75rem;
    padding: 1.5rem 0 0.5rem;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.03em;
    border-top: 1px solid {divider};
    margin-top: 1.5rem;
}}
</style>
""", unsafe_allow_html=True)


# Load artifacts
@st.cache_resource
def load_artifacts():
    model         = joblib.load('models/model.pkl')
    scaler        = joblib.load('models/scaler.pkl')
    threshold     = joblib.load('models/threshold.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    try:
        explainer = joblib.load('models/shap_explainer.pkl')
    except Exception:
        explainer = None
    return model, scaler, threshold, feature_names, explainer

model, scaler, threshold, feature_names, explainer = load_artifacts()


# Header
st.markdown(f"""
<div class="main-header">
    <div class="header-badge">AI-Powered · XGBoost + SHAP</div>
    <div class="header-title">Credit Risk Assessment</div>
    <div class="header-subtitle">
        Predict default probability with full explainability
    </div>
</div>
""", unsafe_allow_html=True)

# Info banner
info_col, toggle_col = st.columns([5, 1], gap="small")

with info_col:
    st.markdown(f"""
    <div class="info-box">
        💡 Trained on 150,000 real loan records · Handles class imbalance via SMOTE ·
        Optimized decision threshold · SHAP explainability for regulatory compliance
    </div>
    """, unsafe_allow_html=True)

with toggle_col:
    
    if st.button(
        f"{toggle_icon} {toggle_label}",
        key="theme_toggle",
        use_container_width=True
    ):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
        
# Inputs
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-label">👤 Personal Profile</div>',
                unsafe_allow_html=True)
    age = st.slider("Age", 18, 100, 35,
                    help="Applicant age in years")
    monthly_income = st.number_input(
        "Monthly Income ($)", min_value=0, max_value=50000,
        value=5000, step=100, help="Gross monthly income in USD")
    number_of_dependents = st.slider("Number of Dependents", 0, 10, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🏦 Debt Profile</div>',
                unsafe_allow_html=True)
    revolving_utilization = st.slider(
        "Revolving Credit Utilization", 0.0, 1.0, 0.3, 0.01,
        help="0 = none used · 1 = fully maxed out")
    debt_ratio = st.slider(
        "Debt Ratio", 0.0, 10.0, 0.35, 0.01,
        help="Monthly debt payments / monthly gross income")

with col_right:
    st.markdown('<div class="section-label">💳 Credit Lines</div>',
                unsafe_allow_html=True)
    number_of_open_credit = st.slider("Open Credit Lines & Loans", 0, 30, 5)
    number_real_estate    = st.slider("Real Estate Loans", 0, 10, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚠️ Late Payment History</div>',
                unsafe_allow_html=True)
    st.caption("Number of times the applicant was past due on payments")
    times_30_59 = st.slider("30–59 Days Late", 0, 20, 0)
    times_60_89 = st.slider("60–89 Days Late", 0, 20, 0)
    times_90    = st.slider("90+ Days Late",   0, 20, 0)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button(
    "Run Credit Risk Assessment",
    use_container_width=True
)


# Results
if predict_btn:
    input_data = pd.DataFrame([[
        revolving_utilization, age, times_30_59, debt_ratio,
        monthly_income, number_of_open_credit, times_90,
        number_real_estate, times_60_89, number_of_dependents
    ]], columns=feature_names)

    prob    = model.predict_proba(input_data)[0][1]
    is_high = prob >= threshold
    clr     = high_text if is_high else low_text

    st.markdown("---")
    st.markdown("### Assessment Result")

    res1, res2, res3 = st.columns([2, 1, 1], gap="large")

    with res1:
        card = "result-card-high" if is_high else "result-card-low"
        icon = "🔴" if is_high else "🟢"
        word = "Reject Application" if is_high else "Approve Application"
        st.markdown(f"""
        <div class="{card}">
            <div class="result-label" style="color:{clr}">Decision {icon}</div>
            <div class="result-decision" style="color:{clr}">{word}</div>
            <div class="prob-display" style="color:{clr}">{prob*100:.1f}%</div>
            <div class="prob-sub">probability of default</div>
        </div>""", unsafe_allow_html=True)

    with res2:
        risk_level = (
            "Critical" if prob > 0.7 else
            "High"     if prob > threshold else
            "Moderate" if prob > 0.3 else "Low"
        )
        risk_color = (
            "#fc8181" if prob > 0.7 else
            "#f6ad55" if prob > threshold else
            "#fbd38d" if prob > 0.3 else "#68d391"
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{risk_color}">{risk_level}</div>
            <div class="metric-label">Risk Level</div>
        </div>""", unsafe_allow_html=True)

    with res3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.3rem">{threshold*100:.1f}%</div>
            <div class="metric-label">Approval Threshold</div>
        </div>""", unsafe_allow_html=True)

    # Business insight
    if is_high:
        st.markdown(f"""
        <div class="insight-box insight-high">
            <strong>Risk Alert:</strong> {prob*100:.1f}% default probability exceeds the
            {threshold*100:.1f}% approval threshold. Approving this loan exposes the
            institution to direct financial loss.<br>
            <strong>Recommended:</strong> Reject or request additional collateral.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="insight-box insight-low">
            <strong>Low Risk:</strong> {prob*100:.1f}% default probability is well below
            the {threshold*100:.1f}% threshold. This is a creditworthy applicant.<br>
            <strong>Recommended:</strong> Approve with standard loan terms.
        </div>""", unsafe_allow_html=True)

    # Applicant summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Applicant Summary")
    late_total = times_30_59 + times_60_89 + times_90
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, val, lbl in zip(
        [s1, s2, s3, s4, s5],
        [str(age), f"${monthly_income:,}",
         f"{revolving_utilization:.0%}", f"{debt_ratio:.2f}", str(late_total)],
        ["Age", "Monthly Income", "Credit Util.", "Debt Ratio", "Late Payments"]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.2rem">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # SHAP
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="section-label">🔍 Model Explainability - SHAP Analysis</div>',
        unsafe_allow_html=True)
    st.caption("Red = pushed risk up · Blue = pushed risk down · Longer bar = bigger impact")

    try:
        shap_values_single = explainer(input_data, check_additivity=False)

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(plot_bg)
        ax.set_facecolor(plot_bg)

        shap.plots.waterfall(shap_values_single[0], max_display=10, show=False)

        for text_obj in fig.findobj(matplotlib.text.Text):
            text_obj.set_color(plot_text)
            if text_obj.get_fontsize() < 9:
                text_obj.set_fontsize(9)

        ax.tick_params(colors=plot_text, labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        for line in ax.get_lines():
            col_val = line.get_color()
            if col_val in ['black', '#000000'] or col_val == (0, 0, 0, 1):
                line.set_color(plot_text)

        ax.xaxis.label.set_color(plot_text)
        ax.yaxis.label.set_color(plot_text)

        plt.title('What drove this risk score?',
                  color=plot_text, fontsize=12,
                  pad=12, loc='left', fontweight='bold')
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    except Exception:
        st.info("SHAP explanation could not be generated for this input.")


# Footer
st.markdown(f"""
<div class="footer">
    Built by Ayushi &nbsp;·&nbsp; End-to-End Credit Risk ML System &nbsp;·&nbsp;
    XGBoost + SHAP + Streamlit &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)