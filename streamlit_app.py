import io
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.append(str(PROJECT_SRC))

from optimizer import run_optimizer  # noqa: E402


st.set_page_config(
    page_title="Hackathon 2025 - Pricing Optimizer",
    layout="wide",
    page_icon="static/logo.png",
    initial_sidebar_state="expanded",
)

BRAND_COLORS = {
    "black": "#000000",
    "primary": "#f5e003",
    "white": "#ffffff",
    "accent": "#e5b611",
    "secondary": "#d1a33c",
    "muted": "#c7c7c7",
        }

st.markdown(
    f"""
    <style>
        :root {{
            --brand-black: {BRAND_COLORS["black"]};
            --brand-primary: {BRAND_COLORS["primary"]};
            --brand-accent: {BRAND_COLORS["accent"]};
        }}
        header[data-testid="stHeader"] {{
            display: none;
        }}
        .stApp {{
            background: linear-gradient(180deg, #ffffff 0%, #fffdf4 72%);
            color: var(--brand-black);
            font-family: 'Outfit', 'Segoe UI', sans-serif;
        }}
        .block-container {{
            padding-top: 0.75rem;
            padding-bottom: 1.0rem;
            max-width: 1200px;
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #161616 0%, #111111 45%, #080808 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
            color: #f5f5f5;
            padding: 2.2rem 1.6rem 2.4rem 1.6rem;
        }}
        [data-testid="stSidebar"] * {{
            color: #f5f5f5 !important;
        }}
        [data-testid="stSidebar"] img {{
            display: block;
            margin: 0 auto 1.2rem auto;
        }}
        [data-testid="stSidebar"] div[data-baseweb="slider"] > div {{
            padding-top: 0.4rem;
            padding-bottom: 0.8rem;
        }}
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {{
            background: rgba(255, 255, 255, 0.22);
            height: 6px;
            border-radius: 999px;
        }}
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] > div {{
            background: linear-gradient(90deg, #f5e003 0%, #f0bc00 100%);
            border-radius: 999px;
            height: 6px;
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"] > div {{
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: #1f1f1f;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.45);
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"]:hover > div,
        [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within > div {{
            border-color: rgba(245, 224, 3, 0.7);
            box-shadow: 0 0 0 2px rgba(245, 224, 3, 0.22);
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"] input {{
            color: #f5f5f5 !important;
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"] button {{
            background: #2a2a2a;
            color: {BRAND_COLORS["primary"]};
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.16);
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"] button:hover {{
            background: #353535;
        }}
        [data-testid="stSidebar"] div[data-baseweb="input"] button svg {{
            fill: {BRAND_COLORS["primary"]};
        }}
        [data-testid="stSidebar"] .stButton button {{
            background: linear-gradient(120deg, #f6da2e 0%, #f5e003 45%, #f0bc00 100%);
            color: var(--brand-black);
            border-radius: 32px;
            border: none;
            font-weight: 700;
            padding: 0.6rem 1.6rem;
            box-shadow: 0 14px 28px rgba(245, 224, 3, 0.35);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }}
        [data-testid="stSidebar"] .stButton button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 18px 30px rgba(224, 182, 0, 0.45);
        }}
        [data-testid="stSidebar"] .stButton button:active {{
            transform: translateY(0);
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
        }}
        .headline {{
            font-size: 2.6rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin-bottom: 0.35rem;
            color: var(--brand-black);
        }}
        .headline .accent {{
            color: var(--brand-primary);
        }}
        .subheadline {{
            color: #595959;
            font-size: 1.05rem;
            max-width: 900px;
        }}
        .section-label {{
            color: {BRAND_COLORS["secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            font-weight: 700;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.1rem;
            margin: 0.25rem 0 0.55rem 0;
        }}
        .kpi-card {{
            background: #ffffff;
            border-radius: 18px;
            border: 1px solid #ebebeb;
            box-shadow: 0 14px 24px rgba(0, 0, 0, 0.05);
            padding: 12px 16px;
        }}
        .kpi-card h4 {{
            margin: 0;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #6b6b6b;
        }}
        .kpi-card .value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--brand-black);
            margin-top: 0.2rem;
        }}
        .kpi-card .delta {{
            font-size: 0.88rem;
            color: #606060;
            margin-top: 0.15rem;
        }}
        .kpi-card .delta span {{
            font-weight: 600;
        }}
        .stMarkdown h2, .stMarkdown h3 {{
            color: var(--brand-black) !important;
        }}
        h2 span.accent {{
            color: var(--brand-primary);
        }}
        .stDataFrame, .stTable {{
            border-radius: 18px;
            border: 1px solid #ededed;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.05);
        }}
        .stDownloadButton button {{
            background: {BRAND_COLORS["secondary"]};
            color: var(--brand-black);
            border-radius: 22px;
            border: none;
            font-weight: 600;
            padding: 0.45rem 1.25rem;
            box-shadow: 0 10px 18px rgba(0, 0, 0, 0.12);
        }}
        .stDownloadButton button:hover {{
            background: {BRAND_COLORS["accent"]};
        }}
        .stTabs [role="tablist"] {{
            gap: 0.5rem;
        }}
        .stTabs [role="tab"] {{
            background: #f8f8f8;
            border-radius: 18px;
            padding: 0.45rem 1.3rem;
            font-weight: 600;
            border: 1px solid transparent;
            color: #5c5c5c;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(120deg, #fdf1a1 0%, #f5e003 90%);
            color: var(--brand-black);
            border-color: rgba(229, 182, 17, 0.4);
            box-shadow: 0 6px 18px rgba(229, 182, 17, 0.35);
        }}
        .section-header {{
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--brand-black);
            display: inline-flex;
            align-items: center;
            gap: 0.75rem;
            margin-top: 1.2rem;
            margin-bottom: 0.7rem;
        }}
        .section-header span {{
            background: linear-gradient(90deg, {BRAND_COLORS["primary"]} 0%, {BRAND_COLORS["accent"]} 100%);
            height: 6px;
            width: 60px;
            border-radius: 999px;
            display: inline-block;
        }}
        .filter-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #7a7a7a;
            margin-bottom: 0.35rem;
        }}
        div[data-baseweb="select"] > div {{
            border-radius: 18px;
            border: 1px solid #d9d9d9;
            min-height: 50px;
            padding: 0 12px;
            color: var(--brand-black);
            background-color: #ffffff;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
        }}
        div[data-baseweb="select"]:hover > div {{
            border-color: rgba(229, 182, 17, 0.6);
            box-shadow: 0 0 0 2px rgba(245,224,3,0.2);
        }}
        div[data-baseweb="select"] span[data-baseweb="tag"] {{
            background: rgba(245,224,3,0.2);
            color: var(--brand-black);
        }}
        div[data-testid="stPopover"] button {{
            width: 100%;
            justify-content: space-between;
            background: #fafafa !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 14px !important;
            color: #2d2d2d !important;
            font-weight: 600;
            padding: 0.55rem 0.9rem;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
        }}
        div[data-testid="stPopover"] button:hover {{
            border-color: rgba(229, 182, 17, 0.65) !important;
            box-shadow: 0 0 0 2px rgba(245,224,3,0.18);
        }}
        div[data-testid="stPopover"] button span {{
            color: #2d2d2d !important;
        }}
        div[data-baseweb="popover"] [data-testid="stCheckbox"] label {{
            color: var(--brand-black) !important;
        }}
        .hero-card {{
            background: linear-gradient(135deg, #141414 0%, #1d1d1d 60%, #2c2c2c 100%);
            border-radius: 28px;
            padding: 0.9rem 1.6rem;
            color: #ffffff;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.24);
            margin-bottom: 0.65rem;
            position: relative;
            overflow: hidden;
        }}
        .hero-card::after {{
            content: "";
            position: absolute;
            right: -55px;
            top: -55px;
            width: 180px;
            height: 180px;
            background: radial-gradient(circle at center, rgba(245,224,3,0.45), rgba(245,224,3,0));
        }}
        .hero-eyebrow {{
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.72rem;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 0.5rem;
        }}
        .hero-title {{
            font-size: 2.05rem;
            font-weight: 700;
            letter-spacing: -0.015em;
            margin: 0;
        }}
        .hero-title span {{
            color: {BRAND_COLORS["primary"]};
        }}
        .hero-subtitle {{
            margin-top: 0.6rem;
            max-width: 650px;
            font-size: 0.95rem;
            line-height: 1.38;
            color: rgba(255, 255, 255, 0.78);
        }}
        .hero-accent {{
            margin-top: 0.9rem;
            width: 120px;
            height: 3px;
            background: linear-gradient(90deg, {BRAND_COLORS["primary"]} 0%, {BRAND_COLORS["accent"]} 100%);
            border-radius: 999px;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Lightweight sidebar toggle (when header is hidden)
if "force_sidebar_open" not in st.session_state:
    st.session_state["force_sidebar_open"] = False

toggle_cols = st.columns([0.06, 0.94])
with toggle_cols[0]:
    _icon = "|||" if not st.session_state["force_sidebar_open"] else "×"
    if st.button(_icon, key="sidebar_toggle_button"):
        st.session_state["force_sidebar_open"] = not st.session_state["force_sidebar_open"]

if st.session_state["force_sidebar_open"]:
    st.markdown(
        """
        <style>
          /* Force-show Streamlit sidebar even when collapsed */
          [data-testid="stSidebar"] {
            transform: none !important;
            visibility: visible !important;
            width: 360px !important;
            min-width: 360px !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_insights_assistant():
    # Lightweight Insights popover (no external APIs)
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I'm your pricing insights assistant. This is a placeholder chat — "
                    "no GenAI is called yet. Ask about PINC, MACO, volume, or guardrails."
                ),
            }
        ]

    st.markdown(
        """
        <style>
          .assistant-card { background:#fff; border:1px solid #ebebeb; border-radius:16px; padding:10px 12px; box-shadow:0 10px 18px rgba(0,0,0,.04); margin-bottom:8px; }
          .assistant-user { color:#202020; }
          .assistant-bot { color:#333333; }
          .assistant-meta { font-size:12px; color:#6b6b6b; margin-bottom:6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([6, 1])
    with cols[1]:
        with st.popover("Insights"):
            md = globals().get("metadata", {}) or {}
            pf = globals().get("portfolio_df", pd.DataFrame())
            pinc_actual = float(md.get("pinc_actual", 0.0))
            try:
                maco_row = pf.loc[pf["metric"] == "MACO"].iloc[0]
                nr_row = pf.loc[pf["metric"] == "Net Revenue"].iloc[0]
                maco_delta = float(maco_row.get("new", 0.0) - maco_row.get("base", 0.0))
                nr_delta = float(nr_row.get("new", 0.0) - nr_row.get("base", 0.0))
            except Exception:
                maco_delta = 0.0
                nr_delta = 0.0
            st.markdown(
                f"<div class='assistant-meta'>PINC {pinc_actual:.2%} | MACO delta {billions(maco_delta)} | NR delta {billions(nr_delta)}</div>",
                unsafe_allow_html=True,
            )

            for msg in st.session_state["assistant_messages"]:
                role_cls = "assistant-user" if msg["role"] == "user" else "assistant-bot"
                st.markdown(
                    f"<div class='assistant-card {role_cls}'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )

            with st.form("assistant_form", clear_on_submit=True):
                q = st.text_input(
                    "Ask for insights",
                    placeholder="e.g., How can I improve MACO while keeping volume within +5%?",
                )
                submitted = st.form_submit_button("Send")
                if submitted and q:
                    st.session_state["assistant_messages"].append({"role": "user", "content": q})
                    st.session_state["assistant_messages"].append({
                        "role": "assistant",
                        "content": (
                            "Placeholder: I'd look at reallocating PINC toward inelastic SKUs, "
                            "try alternative 50-COP step mixes, and verify segment/size ladders."
                        ),
                    })
def render_ui_drawers():
    # Right: assistant drawer
    if "assistant_open" not in st.session_state:
        st.session_state["assistant_open"] = False
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {"role": "assistant", "content": "Hi! I'm your pricing insights assistant. Placeholder chat — no GenAI calls."}
        ]
    right_open_css = "0" if st.session_state["assistant_open"] else "-420px"
    # Left: inputs drawer
    if "inputs_open" not in st.session_state:
        st.session_state["inputs_open"] = False
    left_open_css = "0" if st.session_state["inputs_open"] else "-420px"

    st.markdown(
        f"""
        <style>
          #assistant-drawer {{ position: fixed; top: 90px; right: {right_open_css}; width: 380px; height: calc(100% - 110px);
            background:#fff; border-left:1px solid #ebebeb; box-shadow:-10px 0 24px rgba(0,0,0,0.06); border-top-left-radius:12px; border-bottom-left-radius:12px; padding:12px; z-index:1000; transition:right 180ms ease-in-out; }}
          #assistant-toggle {{ position: fixed; top: 96px; right: 8px; z-index: 1001; }}
          .assistant-card {{ background:#fff; border:1px solid #ebebeb; border-radius:12px; padding:8px 10px; margin-bottom:8px; }}
          .assistant-user {{ color:#202020; }} .assistant-bot {{ color:#333333; }} .assistant-meta {{ font-size:12px; color:#6b6b6b; margin-bottom:6px; }}
          #inputs-drawer {{ position: fixed; top: 90px; left: {left_open_css}; width: 360px; height: calc(100% - 110px);
            background:#111; color:#f5f5f5; border-right:1px solid rgba(255,255,255,0.12); border-top-right-radius:12px; border-bottom-right-radius:12px; padding:14px; z-index:1000; transition:left 180ms ease-in-out; }}
          #inputs-toggle {{ position: fixed; top: 96px; left: 8px; z-index: 1001; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Floating toggles
    st.markdown("<div id='assistant-toggle'></div>", unsafe_allow_html=True)
    if st.button(("Close Assistant" if st.session_state["assistant_open"] else "Open Assistant"), key="assistant_toggle_btn"):
        st.session_state["assistant_open"] = not st.session_state["assistant_open"]
    st.markdown("<div id='inputs-toggle'></div>", unsafe_allow_html=True)
    if st.button(("Close Inputs" if st.session_state["inputs_open"] else "Open Inputs"), key="inputs_toggle_btn"):
        st.session_state["inputs_open"] = not st.session_state["inputs_open"]

    # Assistant drawer content
    st.markdown("<div id='assistant-drawer'>", unsafe_allow_html=True)
    md = globals().get("metadata", {}) or {}
    pf = globals().get("portfolio_df", pd.DataFrame())
    pinc_actual = float(md.get("pinc_actual", 0.0))
    try:
        maco_row = pf.loc[pf["metric"] == "MACO"].iloc[0]
        nr_row = pf.loc[pf["metric"] == "Net Revenue"].iloc[0]
        maco_delta = float(maco_row.get("new", 0.0) - maco_row.get("base", 0.0))
        nr_delta = float(nr_row.get("new", 0.0) - nr_row.get("base", 0.0))
    except Exception:
        maco_delta = 0.0
        nr_delta = 0.0
    st.markdown(
        f"<div class='assistant-meta'>PINC {pinc_actual:.2%} | MACO Delta {billions(maco_delta)} | NR Delta {billions(nr_delta)}</div>",
        unsafe_allow_html=True,
    )
    for msg in st.session_state["assistant_messages"]:
        cls = "assistant-user" if msg["role"] == "user" else "assistant-bot"
        st.markdown(f"<div class='assistant-card {cls}'>{msg['content']}</div>", unsafe_allow_html=True)
    with st.form("assistant_form", clear_on_submit=True):
        q = st.text_input("Ask for insights", placeholder="e.g., How can I improve MACO while keeping volume within +5%?")
        submitted = st.form_submit_button("Send")
        if submitted and q:
            st.session_state["assistant_messages"].append({"role": "user", "content": q})
            st.session_state["assistant_messages"].append({"role": "assistant", "content": "Placeholder: Reallocate PINC to inelastic SKUs, try 50-COP step mixes, and preserve ladders."})
    st.markdown("</div>", unsafe_allow_html=True)

    # Inputs drawer content
    st.markdown("<div id='inputs-drawer'>", unsafe_allow_html=True)
    st.markdown("<div style='font-weight:700;margin-bottom:8px;'>Optimization Inputs</div>", unsafe_allow_html=True)
    dp = st.session_state.get("drawer_pinc", 0.02)
    dfloor = st.session_state.get("drawer_floor", -300.0)
    dceil = st.session_state.get("drawer_ceil", 500.0)
    dstep = st.session_state.get("drawer_step", 50.0)
    dp = st.slider("National PINC", min_value=0.0, max_value=0.06, value=float(dp), step=0.005, format="%.3f", key="drawer_pinc")
    dfloor = st.number_input("Price floor delta", value=float(dfloor), step=50.0, key="drawer_floor")
    dceil = st.number_input("Price ceiling delta", value=float(dceil), step=50.0, key="drawer_ceil")
    dstep = st.number_input("Price step (multiples of)", value=float(dstep), min_value=1.0, step=1.0, key="drawer_step")
    if st.button("Run (Drawer)", key="drawer_run"):
        with st.spinner("Running optimizer..."):
            st.session_state["current_result"] = cached_optimize(dp, dfloor, dceil, dstep)
    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def cached_optimize(pinc_target: float, floor_delta: float, ceiling_delta: float, price_step: float):
    return run_optimizer(
        pinc_target=pinc_target,
        price_floor_delta=floor_delta,
        price_ceiling_delta=ceiling_delta,
        price_step=price_step,
    )


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    with io.StringIO() as buffer:
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")


def billions(value: float) -> str:
    # Display all currency values in millions (M)
    try:
        v = float(value)
    except Exception:
        return str(value)
    return f"{v/1e6:,.2f} M"


def growth_badge(delta: float) -> str:
    color = '#1b7a1b' if delta >= 0 else '#c0392b'
    symbol = '+ ' if delta >= 0 else '- '
    return f"<span style='color:{color};'>{symbol}{abs(delta):.2%}</span>"


def sorted_unique(series: pd.Series) -> List[str]:
    values: List[str] = []
    for val in series.dropna():
        text = str(val).strip()
        if text and text.lower() != "nan":
            values.append(text)
    return sorted(set(values))


def dropdown_multiselect(label: str, options: List[str]) -> List[str]:
    safe_label = label.lower().replace(" ", "_")
    selection_key = f"{safe_label}_selection"
    select_all_key = f"{selection_key}_all"

    if selection_key not in st.session_state:
        st.session_state[selection_key] = options.copy()
    current = st.session_state[selection_key]

    if not current or len(current) == len(options):
        summary = "All selected"
    elif len(current) == 1:
        summary = current[0]
    elif len(current) == 2:
        summary = ", ".join(current)
    else:
        summary = ", ".join(current[:2]) + f" +{len(current) - 2}"
    button_label = f"{label}: {summary}"

    with st.popover(button_label):
        select_all_default = len(current) == len(options)
        select_all = st.checkbox("Select all", key=select_all_key, value=select_all_default)

        if select_all:
            st.session_state[selection_key] = options.copy()
            for opt in options:
                st.session_state[f"{selection_key}_{opt}"] = True
        new_selection: List[str] = []
        for opt in options:
            opt_key = f"{selection_key}_{opt}"
            st.session_state.setdefault(opt_key, opt in current or select_all)
            checked = st.checkbox(opt, key=opt_key)
            if checked:
                new_selection.append(opt)
        st.session_state[selection_key] = new_selection or options.copy()

    return st.session_state[selection_key]


def price_delta_color(value: float) -> str:
    if value > 0:
        return "background-color: #e6f5e6; color: #1b7a1b; font-weight:600;"
    if value < 0:
        return "background-color: #ffeaea; color:#c0392b; font-weight:600;"
    return ""


def render_scenario_card(name: str, portfolio: pd.DataFrame, metadata: Dict) -> None:
    st.markdown(f"**Scenario:** {name}")
    cols = st.columns(3)
    maco_delta = portfolio.loc[portfolio["metric"] == "MACO", "delta"].iloc[0]
    nr_delta = portfolio.loc[portfolio["metric"] == "Net Revenue", "delta"].iloc[0]
    pinc_value = metadata.get("pinc_actual", 0)
    cols[0].metric("MACO delta", f"{maco_delta:,.0f}")
    cols[1].metric("NR delta", f"{nr_delta:,.0f}")
    cols[2].metric("PINC", f"{pinc_value:.3%}")


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Revenue Growth Management</div>
        <h1 class="hero-title">Dynamic Pricing <span>Studio</span></h1>
        <p class="hero-subtitle">
            Allocate national PINC targets across ABI's portfolio, simulate trade-offs instantly, and export
            recommendations ready for BrewVision hand-off.
        </p>
        <div class="hero-accent"></div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("")

with st.sidebar:
    st.image("static/logo.png", width=160)
    st.markdown(
        "<div style='font-size:1.2rem;font-weight:700;margin-top:1rem;margin-bottom:0.25rem;'>Optimization Inputs</div>",
        unsafe_allow_html=True,
    )
    st.caption("Adjust national stretch targets and list-price guardrails before running the optimizer.")

    pinc = st.slider("National PINC", min_value=0.0, max_value=0.06, value=0.02, step=0.005, format="%.3f")
    floor_delta = st.number_input("Price floor delta", value=-300.0, step=50.0, help="Lower bound vs. current PTC")
    ceiling_delta = st.number_input("Price ceiling delta", value=500.0, step=50.0, help="Upper bound vs. current PTC")
    price_step = st.number_input(
        "Price step (multiples of)", value=50.0, min_value=1.0, step=1.0, help="Rounded PTC increments"
    )
    run_clicked = st.button("Run Optimization")

if "current_result" not in st.session_state or run_clicked:
    with st.spinner("Running optimizer..."):
        st.session_state["current_result"] = cached_optimize(pinc, floor_delta, ceiling_delta, price_step)

result = st.session_state["current_result"]
summary_df: pd.DataFrame = result["summary"].copy()
portfolio_df: pd.DataFrame = result["portfolio"].copy()
architecture_df: pd.DataFrame = result["architecture"].copy()
metadata = result["metadata"]
constraints = result.get("constraints", {})

render_insights_assistant()

if not metadata.get("success", False):
    st.warning(f"Optimizer returned a baseline scenario. Reason: {metadata.get('status', 'infeasible constraints')}")  # noqa: E501

st.markdown(
    f"<h2 style='color:{BRAND_COLORS['black']};margin-bottom:0.25rem;'>Portfolio <span class='accent'>Impact</span></h2>",
    unsafe_allow_html=True,
)

portfolio_display = portfolio_df.copy()
if {"base", "new"}.issubset(portfolio_display.columns):
    portfolio_display["delta"] = portfolio_display["new"] - portfolio_display["base"]
    # Present MACO/NR in millions for readability
    mask_m = portfolio_display["metric"].isin(["MACO", "Net Revenue"]) if "metric" in portfolio_display.columns else []
    try:
        portfolio_display.loc[mask_m, ["base", "new", "delta"]] = (
            portfolio_display.loc[mask_m, ["base", "new", "delta"]] / 1e6
        )
    except Exception:
        pass

maco_new = portfolio_df.loc[portfolio_df["metric"] == "MACO", "new"].iloc[0]
nr_new = portfolio_df.loc[portfolio_df["metric"] == "Net Revenue", "new"].iloc[0]
maco_base = portfolio_df.loc[portfolio_df["metric"] == "MACO", "base"].iloc[0]
nr_base = portfolio_df.loc[portfolio_df["metric"] == "Net Revenue", "base"].iloc[0]
pinc_actual = metadata.get("pinc_actual", 0)
iterations = metadata.get("iterations")

kpi_cards = f"""
<div class="kpi-row">
    <div class="kpi-card">
        <h4>MACO</h4>
        <div class="value">{billions(maco_new)}</div>
        <div class="delta">Baseline {billions(maco_base)} &nbsp; {growth_badge((maco_new - maco_base) / max(maco_base, 1e-6))}</div>
    </div>
    <div class="kpi-card">
        <h4>Net Revenue</h4>
        <div class="value">{billions(nr_new)}</div>
        <div class="delta">Baseline {billions(nr_base)} &nbsp; {growth_badge((nr_new - nr_base) / max(nr_base, 1e-6))}</div>
    </div>
    <div class="kpi-card">
        <h4>Portfolio PINC</h4>
        <div class="value">{pinc_actual:.3%}</div>
        <div class="delta">Target {pinc:.3%}</div>
    </div>
    <div class="kpi-card">
        <h4>Solver Iterations</h4>
        <div class="value">{iterations if iterations is not None else '-'}</div>
        <div class="delta">Post-constraint scaling</div>
    </div>
</div>
"""
st.markdown(kpi_cards, unsafe_allow_html=True)

st.dataframe(portfolio_display, width="stretch", hide_index=True)

tab_recos, tab_mapping = st.tabs(["SKU Recommendations", "Sell-in / Sell-out Mapping"])

with tab_recos:
    st.markdown(
        f"<div class='section-header'>SKU Recommendations<span></span></div>",
        unsafe_allow_html=True,
    )

    summary_df["brand"] = summary_df["brand"].astype(str)
    summary_df["pack"] = summary_df["pack"].astype(str)
    summary_df["segment"] = summary_df["segment"].astype(str)
    # Ensure Arrow compatibility for text-like identifiers
    if "size" in summary_df.columns:
        summary_df["size"] = summary_df["size"].astype(str)
    if "size_group" in summary_df.columns:
        summary_df["size_group"] = summary_df["size_group"].astype(str)

    brands = sorted_unique(summary_df["brand"])
    packs = sorted_unique(summary_df["pack"])
    segments = sorted_unique(summary_df["segment"])

    filter_cols = st.columns(3)
    with filter_cols[0]:
        st.markdown("<div class='filter-label'>Select brand(s)</div>", unsafe_allow_html=True)
        selected_brands = dropdown_multiselect("Brand", brands)
    with filter_cols[1]:
        st.markdown("<div class='filter-label'>Select pack / SKU</div>", unsafe_allow_html=True)
        selected_packs = dropdown_multiselect("Pack", packs)
    with filter_cols[2]:
        st.markdown("<div class='filter-label'>Select segment</div>", unsafe_allow_html=True)
        selected_segments = dropdown_multiselect("Segment", segments)

    filtered = summary_df[
        summary_df["brand"].isin(selected_brands)
        & summary_df["pack"].isin(selected_packs)
        & summary_df["segment"].isin(selected_segments)
    ].copy()
    filtered["volume_delta"] = filtered["volume_new"] - filtered["volume_base"]
    filtered["nr_delta"] = filtered["nr_new"] - filtered["nr_base"]
    filtered["maco_delta"] = filtered["maco_new"] - filtered["maco_base"]
    filtered["price_delta"] = filtered["price_new"] - filtered["price_base"]

    styled = filtered.style.applymap(price_delta_color, subset=["price_delta"])
    st.dataframe(styled, width="stretch", hide_index=True)

    download_cols = st.columns(3)
    download_cols[0].download_button(
        "Download SKU Plan",
        data=df_to_csv_bytes(summary_df),
        file_name="optimized_prices.csv",
        mime="text/csv",
    )
    download_cols[1].download_button(
        "Download Portfolio Summary",
        data=df_to_csv_bytes(portfolio_df),
        file_name="portfolio_summary.csv",
        mime="text/csv",
    )
    download_cols[2].download_button(
        "Download Price Architecture",
        data=df_to_csv_bytes(architecture_df),
        file_name="price_architecture.csv",
        mime="text/csv",
    )

    st.markdown("### Price Architecture View")
    architecture_long = architecture_df.melt(
        id_vars=["brand", "pack", "size", "segment", "size_group"],
        value_vars=["price_base", "price_new"],
        var_name="price_type",
        value_name="price",
    )
    architecture_long["price_type"] = architecture_long["price_type"].map(
        {"price_base": "Current PTC", "price_new": "Recommended PTC"}
    )
    fig = px.bar(
        architecture_long,
        x="brand",
        y="price",
        color="price_type",
        barmode="group",
        hover_data=["pack", "size"],
        title="SKU Price Architecture",
    )
    fig.update_layout(xaxis_title="Brand", yaxis_title="PTC")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<h3 style='color:{BRAND_COLORS['black']};margin-top:2.5rem;'>Guardrails</h3>",
        unsafe_allow_html=True,
    )

    maco_delta = float(constraints.get("maco_delta", 0.0))
    volume_ratio = float(constraints.get("volume_ratio", 1.0))
    industry_ratio = float(constraints.get("industry_ratio", 1.0))
    market_share_actual = float(constraints.get("market_share", metadata.get("base_market_share", 0.0)))
    share_drop = float(constraints.get("share_drop", 0.0))
    pinc_actual_guard = float(constraints.get("pinc_actual", metadata.get("pinc_actual", 0.0)))
    base_share = float(metadata.get("base_market_share", 0.0))

    guardrail_cards = f"""
    <div class="kpi-row" style="margin-top:0.5rem;">
        <div class="kpi-card">
            <h4>MACO Delta</h4>
            <div class="value">{billions(maco_delta)}</div>
            <div class="delta">Must be non-negative</div>
        </div>
        <div class="kpi-card">
            <h4>ABI Volume</h4>
            <div class="value">{volume_ratio:.2%}</div>
            <div class="delta">Guardrail: -1% to +5%</div>
        </div>
        <div class="kpi-card">
            <h4>Industry Volume</h4>
            <div class="value">{industry_ratio:.2%}</div>
            <div class="delta">Guardrail: â‰¥ 99%</div>
        </div>
        <div class="kpi-card">
            <h4>Market Share</h4>
            <div class="value">{market_share_actual:.2%}</div>
            <div class="delta">Baseline {base_share:.2%}</div>
        </div>
    </div>
    """
    st.markdown(guardrail_cards, unsafe_allow_html=True)

    secondary_guardrail_cards = f"""
    <div class="kpi-row" style="margin-top:0.5rem;">
        <div class="kpi-card">
            <h4>Share Drop vs. Base</h4>
            <div class="value">{share_drop:.2%}</div>
            <div class="delta">Guardrail: â‰¤ 0.5%</div>
        </div>
        <div class="kpi-card">
            <h4>Portfolio PINC</h4>
            <div class="value">{pinc_actual_guard:.3%}</div>
            <div class="delta">Target {pinc:.3%}</div>
        </div>
    </div>
    """
    st.markdown(secondary_guardrail_cards, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.caption("NR/HL by Segment")
        segment_df = pd.DataFrame([constraints.get("segment_nr_hl", {})]).T.reset_index()
        segment_df.columns = ["segment", "nr_per_hl"]
        st.dataframe(segment_df, hide_index=True, width="stretch")
    with right:
        st.caption("NR/HL by Size Group")
        size_df = pd.DataFrame([constraints.get("size_group_nr_hl", {})]).T.reset_index()
        size_df.columns = ["size_group", "nr_per_hl"]
        st.dataframe(size_df, hide_index=True, width="stretch")
        st.caption("Hierarchy check: Value < Core < Premium and Small > Regular > Large.")

    st.markdown("### Scenario Workspace")
    scenario_name = st.text_input("Scenario name")
    if "scenarios" not in st.session_state:
        st.session_state["scenarios"] = []

    if st.button("Save Scenario", key="save_scenario") and scenario_name:
        st.session_state["scenarios"].append(
            {
                "name": scenario_name,
                "portfolio": portfolio_display.copy(),
                "metadata": metadata.copy(),
            }
        )
        st.success(f"Scenario '{scenario_name}' saved.")

    if st.session_state["scenarios"]:
        for scenario in st.session_state["scenarios"]:
            render_scenario_card(scenario["name"], scenario["portfolio"], scenario["metadata"])

with tab_mapping:
    st.markdown(
        f"<div class='section-header'>Sell-in / Sell-out Mapping<span></span></div>",
        unsafe_allow_html=True,
    )
    mapping_path = Path("outputs/round2/sellout_sellin_mapping.xlsx")
    if mapping_path.exists():
        mapping_df = pd.read_excel(mapping_path)
        # Ensure Arrow compatibility: cast object columns to string
        for c in mapping_df.select_dtypes(include=["object"]).columns:
            mapping_df[c] = mapping_df[c].astype(str)
        st.caption(
            "Slugified brand, pack, and size tokens bridge Sell-out SKUs to Sell-in keys. Sample below (first 500 rows)."
        )
        st.dataframe(mapping_df.head(500), width="stretch", hide_index=True)
        st.download_button(
            "Download full mapping",
            data=df_to_csv_bytes(mapping_df),
            file_name="sellout_sellin_mapping.csv",
            mime="text/csv",
        )
    else:
        st.info("Mapping file not found. Run the optimizer to regenerate `sellout_sellin_mapping.xlsx`.")


st.markdown("----")
st.markdown(
    f"<p style='color:{BRAND_COLORS['black']};opacity:0.8;'>Tip: expose this optimizer via the FastAPI service "
    "<code>uvicorn src.api:app --reload</code> or run the Streamlit app locally with "
    "<code>streamlit run streamlit_app.py</code>.</p>",
    unsafe_allow_html=True,
)






