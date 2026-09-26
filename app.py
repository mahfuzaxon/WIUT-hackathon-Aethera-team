"""
Team Aethera — Fraud Alert Intelligence Platform
WIUT FinTech Hackathon 2026

Structure (per section, all in one file to keep Streamlit Cloud deploys
simple — see the accompanying explanation for why):
  1. Config & styling
  2. Data loading (real data -> demo fallback, clearly labeled)
  3. Risk scoring (transparent demo function OR a real model if present)
  4. Section renderers: Command Center, Alert Investigation,
     Behavior Analytics, ML Features, Model Insights, Pipeline & Data,
     About the Solution
  5. Sidebar navigation + routing
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. CONFIG & STYLING
# ============================================================
st.set_page_config(
    page_title="Team Aethera — Fraud Alert Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Known from the full offline pipeline (7M-row transaction log). This
# lightweight deployment only ships the aggregated per-signal features,
# so we surface this as a fixed reference stat rather than recomputing it.
RAW_TRANSACTIONS_KNOWN = 6_987_663

CUSTOM_CSS = """
<style>
:root{
  --ink:#10151A; --panel:#1A2128; --panel2:#151B21; --line:#2A333B;
  --text:#E9EEF1; --muted:#93A3AC; --safe:#3FA98A; --risk:#E15241; --gold:#E4A93C;
}
.block-container{padding-top:2rem;}
[data-testid="stMetricValue"]{font-weight:700;}
.aeth-card{
  background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin-bottom:12px;
}
.aeth-card h4{margin:0 0 6px 0; font-size:15px;}
.aeth-card p{margin:0; color:var(--muted); font-size:13.5px;}
.badge{
  display:inline-block; padding:3px 12px; border-radius:20px; font-size:12.5px;
  font-weight:600; letter-spacing:.01em;
}
.badge-low{background:rgba(63,169,138,.18); color:#6FCBAF;}
.badge-medium{background:rgba(228,169,60,.18); color:#E4A93C;}
.badge-high{background:rgba(225,82,65,.18); color:#F08A7D;}
.badge-critical{background:#E15241; color:#fff;}
.pipeline-flow{display:flex; flex-direction:column; gap:0; align-items:center;}
.pipeline-node{
  background:var(--panel); border:1px solid var(--line); border-radius:10px;
  padding:12px 22px; width:100%; max-width:420px; text-align:center; font-weight:600;
}
.pipeline-arrow{color:var(--muted); font-size:20px; margin:2px 0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# 2. DATA LOADING (real -> demo fallback)
# ============================================================
FEATURE_COLUMNS = [
    "total_tx_count", "total_amount", "avg_amount", "max_amount",
    "burst_ratio_3d_30d", "amount_ratio_3d_30d",
    "dir_ratio_chiqim", "dir_ratio_kirim",
    "type_ratio_xalqaro", "type_ratio_karta", "type_ratio_bank_otkazmasi", "type_ratio_naqd",
    "night_tx_ratio",
]


@st.cache_data
def load_data():
    """Loads the real precomputed feature file if present; otherwise
    generates a clearly-labeled synthetic dataset with the same schema
    and a realistic class separation, so the demo still tells the right
    story with no real data attached."""
    for path, reader in [
        ("master_unified_dataset.csv", lambda p: pd.read_csv(p, parse_dates=["signal_sanasi"])),
        ("train_features.parquet", pd.read_parquet),
    ]:
        try:
            df = reader(path)
            return df, False
        except Exception:
            continue
    return _generate_demo_data(), True


def _generate_demo_data(n: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y = rng.choice([0, 1], size=n, p=[0.8282, 0.1718])

    def split(low0, high0, low1, high1):
        return np.where(y == 1, rng.uniform(low1, high1, n), rng.uniform(low0, high0, n))

    df = pd.DataFrame({
        "signal_id": [f"SG_{i:06d}" for i in range(n)],
        "signal_sanasi": pd.date_range("2026-01-01", periods=n, freq="h"),
        "eskalatsiya": y,
        "total_tx_count": rng.integers(10, 500, n),
        "total_amount": rng.uniform(10, 300, n),
        "avg_amount": rng.uniform(0.01, 5.0, n),
        "max_amount": split(1.0, 5.0, 4.0, 20.0),
        "burst_ratio_3d_30d": split(0.03, 0.20, 0.35, 0.85),
        "amount_ratio_3d_30d": split(0.03, 0.20, 0.30, 0.80),
        "dir_ratio_chiqim": split(0.20, 0.55, 0.55, 0.95),
        "type_ratio_xalqaro": split(0.00, 0.10, 0.15, 0.50),
        "type_ratio_karta": rng.uniform(0.10, 0.60, n),
        "type_ratio_bank_otkazmasi": rng.uniform(0.05, 0.40, n),
        "type_ratio_naqd": rng.uniform(0.00, 0.30, n),
        "night_tx_ratio": split(0.02, 0.15, 0.15, 0.45),
    })
    df["dir_ratio_kirim"] = 1 - df["dir_ratio_chiqim"]
    return df


# ============================================================
# 3. RISK SCORING — transparent demo function, real model optional
# ============================================================
RISK_WEIGHTS = {
    "burst_ratio_3d_30d": 0.30,
    "night_tx_ratio": 0.20,
    "dir_ratio_chiqim": 0.20,
    "type_ratio_xalqaro": 0.20,
    "amount_ratio_3d_30d": 0.10,
}


@st.cache_resource
def load_trained_model():
    """If a real trained model artifact exists in the repo, use it.
    Otherwise the app falls back to the transparent demo scorer below."""
    try:
        import joblib
        return joblib.load("lgbm_model.pkl")
    except Exception:
        return None


def demo_risk_score(values: dict) -> float:
    """
    DEMO SCORING — NOT a trained ML model.
    A weighted, normalized blend of behavioral ratios (each already 0-1),
    scaled to 0-100. This exists so the UI/UX can be demoed end-to-end
    before a real model is wired in. To connect the real model, replace
    the call site below with `model.predict_proba([[...]])[0][1] * 100`.
    """
    score, total_w = 0.0, 0.0
    for feat, w in RISK_WEIGHTS.items():
        v = values.get(feat)
        if v is not None and pd.notna(v):
            score += w * min(max(float(v), 0.0), 1.0)
            total_w += w
    return round((score / total_w) * 100, 1) if total_w else 0.0


def risk_level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def risk_badge(level: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Critical": "badge-critical"}[level]
    return f'<span class="badge {cls}">{level.upper()}</span>'


# ============================================================
# SECTION RENDERERS
# ============================================================
def render_command_center(df, demo_mode, model_connected):
    st.markdown("## AI-Powered Fraud Alert Intelligence")
    st.caption("Turning millions of raw transaction logs into a single, explainable escalation decision per alert.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Security Signals", f"{len(df):,}")
    c2.metric("Escalation Rate", f"{df['eskalatsiya'].mean():.1%}")
    c3.metric("High-Risk Alerts", f"{(df['risk_level'].isin(['High', 'Critical'])).sum():,}")
    c4.metric("Transactions Analyzed", f"{RAW_TRANSACTIONS_KNOWN:,}" if not demo_mode else "N/A (demo)")
    c5.metric("Features Engineered", f"{sum(c in df.columns for c in FEATURE_COLUMNS)}")

    st.markdown("#### Risk overview")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        counts = df["eskalatsiya"].value_counts().rename({0: "Dismissed", 1: "Escalated"})
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color=counts.index, color_discrete_map={"Dismissed": "#3FA98A", "Escalated": "#E15241"})
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        risk_counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig2 = px.bar(x=risk_counts.index, y=risk_counts.values,
                       color=risk_counts.index,
                       color_discrete_map={"Low": "#3FA98A", "Medium": "#E4A93C", "High": "#E8735F", "Critical": "#E15241"})
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Alerts",
                            margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Recent alerts")
    recent = df.sort_values("signal_sanasi", ascending=False).head(6)
    show_cols = [c for c in ["signal_id", "signal_sanasi", "risk_level", "eskalatsiya"] if c in recent.columns]
    st.dataframe(recent[show_cols], use_container_width=True, hide_index=True)

    if st.button("Investigate an Alert →", type="primary"):
        st.session_state.nav = "Alert Investigation"
        st.rerun()


def render_alert_investigation(df, model, model_connected):
    st.markdown("## Alert Investigation")
    st.caption("Select a real signal or enter values manually to see how the escalation decision is reached.")

    mode = st.radio("Source", ["Pick an existing signal", "Enter values manually"], horizontal=True)

    if mode == "Pick an existing signal":
        sample_ids = df["signal_id"].sample(min(50, len(df)), random_state=1).tolist()
        chosen = st.selectbox("signal_id", sorted(sample_ids))
        row = df.loc[df["signal_id"] == chosen].iloc[0]
        values = {k: row.get(k) for k in RISK_WEIGHTS}
        extra = {k: row.get(k) for k in ["total_tx_count", "total_amount", "avg_amount", "max_amount"]}
    else:
        c1, c2 = st.columns(2)
        with c1:
            tx_count = st.number_input("Transaction count", 1, 2000, 120)
            total_amt = st.number_input("Total amount", 0.0, 1000.0, 80.0)
            avg_amt = st.number_input("Average amount", 0.0, 50.0, 0.6)
            max_amt = st.number_input("Maximum amount", 0.0, 50.0, 4.0)
        with c2:
            burst = st.slider("Burst activity ratio (3d/30d)", 0.0, 1.0, 0.30)
            outgoing = st.slider("Outgoing transaction ratio", 0.0, 1.0, 0.45)
            intl = st.slider("International transaction ratio", 0.0, 1.0, 0.10)
            night = st.slider("Night transaction ratio (00:00–05:00)", 0.0, 1.0, 0.10)
        values = {
            "burst_ratio_3d_30d": burst, "night_tx_ratio": night,
            "dir_ratio_chiqim": outgoing, "type_ratio_xalqaro": intl,
            "amount_ratio_3d_30d": burst,  # reuse burst as a stand-in spend-velocity proxy in manual mode
        }
        extra = {"total_tx_count": tx_count, "total_amount": total_amt, "avg_amount": avg_amt, "max_amount": max_amt}

    if model_connected:
        # Real model path — wire your feature vector in the exact training order.
        feat_vec = [[values.get(f, 0) for f in RISK_WEIGHTS]]
        score = round(float(model.predict_proba(feat_vec)[0][1]) * 100, 1)
        score_source = "Live model (LightGBM)"
    else:
        score = demo_risk_score(values)
        score_source = "Demo scoring (no trained model artifact found — see note below)"

    level = risk_level(score)

    st.markdown("#### Risk assessment")
    c1, c2 = st.columns([1, 2])
    with c1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#E15241" if score >= 50 else "#3FA98A"},
                   "steps": [
                       {"range": [0, 25], "color": "#1E2A26"},
                       {"range": [25, 50], "color": "#2A2A1E"},
                       {"range": [50, 75], "color": "#2E211D"},
                       {"range": [75, 100], "color": "#3A1E1A"},
                   ]},
        ))
        gauge.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"**Risk level:** {risk_badge(level)}", unsafe_allow_html=True)
        st.caption(score_source)

    with c2:
        st.markdown("**Why this score**")
        median_baseline = df.loc[df["eskalatsiya"] == 0, list(RISK_WEIGHTS)].median()
        reasons = []
        labels = {
            "burst_ratio_3d_30d": "Unusual burst activity vs. a typical account",
            "night_tx_ratio": "Elevated off-peak (night) transaction behavior",
            "dir_ratio_chiqim": "High share of outgoing transfers",
            "type_ratio_xalqaro": "Notable international transaction exposure",
            "amount_ratio_3d_30d": "Spending concentrated in a short recent window",
        }
        for feat, label in labels.items():
            v = values.get(feat)
            base = median_baseline.get(feat)
            if v is not None and base is not None and v > base * 1.5:
                reasons.append(f"- **{label}** ({v:.0%} vs. typical {base:.0%})")
        if reasons:
            st.markdown("\n".join(reasons))
        else:
            st.markdown("- No individual signal is far above baseline; risk is driven by their combination.")
        st.dataframe(pd.DataFrame([extra]), use_container_width=True, hide_index=True)


def render_behavior_analytics(df):
    st.markdown("## Behavior Analytics")
    st.caption("Filter the signal population and see how behavior diverges between dismissed and escalated alerts.")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            esc_filter = st.multiselect("Escalation status", ["Dismissed", "Escalated"], default=["Dismissed", "Escalated"])
        with f2:
            risk_filter = st.multiselect("Risk level", ["Low", "Medium", "High", "Critical"],
                                          default=["Low", "Medium", "High", "Critical"])
        with f3:
            intl_min = st.slider("Minimum international ratio", 0.0, 1.0, 0.0)

        if "signal_sanasi" in df.columns:
            dmin, dmax = df["signal_sanasi"].min(), df["signal_sanasi"].max()
            date_range = st.slider("Date range", min_value=dmin.to_pydatetime(), max_value=dmax.to_pydatetime(),
                                    value=(dmin.to_pydatetime(), dmax.to_pydatetime()))
        else:
            date_range = None

    status_map = {"Dismissed": 0, "Escalated": 1}
    mask = df["eskalatsiya"].isin([status_map[s] for s in esc_filter]) & df["risk_level"].isin(risk_filter)
    if "type_ratio_xalqaro" in df.columns:
        mask &= df["type_ratio_xalqaro"] >= intl_min
    if date_range:
        mask &= df["signal_sanasi"].between(date_range[0], date_range[1])
    fdf = df.loc[mask]

    st.caption(f"{len(fdf):,} of {len(df):,} signals match the current filters.")
    if fdf.empty:
        st.warning("No signals match these filters.")
        return

    fdf = fdf.assign(status=fdf["eskalatsiya"].map({0: "Dismissed", 1: "Escalated"}))
    color_map = {"Dismissed": "#3FA98A", "Escalated": "#E15241"}

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = px.bar(fdf["status"].value_counts().reset_index(), x="status", y="count",
                      color="status", color_discrete_map=color_map, title="Escalated vs. dismissed")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    with r1c2:
        fig = px.box(fdf, x="status", y="max_amount", color="status", color_discrete_map=color_map,
                      title="Transaction size (max amount) by class")
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig = px.scatter(fdf, x="burst_ratio_3d_30d", y="night_tx_ratio", color="status",
                          color_discrete_map=color_map, opacity=0.7,
                          title="Burst velocity vs. night activity")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with r2c2:
        chan_means = fdf.groupby("status")["type_ratio_xalqaro"].mean().reset_index()
        fig = px.bar(chan_means, x="status", y="type_ratio_xalqaro", color="status",
                      color_discrete_map=color_map, title="International transaction risk (mean share)")
        fig.update_layout(showlegend=False, height=320, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(fdf, x="dir_ratio_chiqim", color="status", color_discrete_map=color_map,
                        barmode="overlay", opacity=0.6, title="Outgoing vs. incoming transaction behavior")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)


def render_ml_features(df):
    st.markdown("## ML Features")
    st.caption("26 behavioral features, grouped by what they measure.")

    groups = {
        "Behavior": {
            "total_tx_count": "Total historical transactions on the account. High-volume accounts need relative — not absolute — thresholds.",
            "avg_amount": "Mean transaction size, the account's normal spending baseline.",
            "max_amount": "Largest single transaction — one outsized transfer is itself a signal.",
            "burst_ratio_3d_30d": "Share of a month's activity that happened in just the last 3 days. The single strongest fraud signal we found.",
        },
        "Transaction Direction": {
            "dir_ratio_chiqim": "Share of transfers that are outgoing. Fraud drains balances outward.",
            "dir_ratio_kirim": "Share of transfers that are incoming. A healthy account stays balanced between the two.",
        },
        "Channel": {
            "type_ratio_xalqaro": "Share of transfers on the international channel — rare normally, common in cash-out fraud.",
            "type_ratio_karta": "Share of card-based transfers.",
            "type_ratio_bank_otkazmasi": "Share of standard bank wire transfers.",
            "type_ratio_naqd": "Share of cash-based operations.",
        },
        "Time": {
            "night_tx_ratio": "Share of activity between 00:00–05:00 — off-peak hours a genuine user rarely transacts in.",
            "amount_ratio_3d_30d": "Same idea as burst ratio, but for spending volume instead of transaction count.",
        },
    }

    for group_name, feats in groups.items():
        st.markdown(f"#### {group_name}")
        cols = st.columns(len(feats))
        for col, (feat, desc) in zip(cols, feats.items()):
            with col:
                example = f"{df[feat].median():.2f}" if feat in df.columns else "n/a"
                st.markdown(
                    f'<div class="aeth-card"><h4>{feat}</h4><p>{desc}</p>'
                    f'<p style="margin-top:8px;color:var(--gold);font-weight:600;">Example: {example}</p></div>',
                    unsafe_allow_html=True,
                )


def render_model_insights(df, model, model_connected):
    st.markdown("## Model Insights")

    st.markdown("#### Class imbalance")
    counts = df["eskalatsiya"].value_counts().rename({0: "Dismissed", 1: "Escalated"})
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                  color_discrete_map={"Dismissed": "#3FA98A", "Escalated": "#E15241"})
    fig.update_layout(showlegend=False, height=260, xaxis_title="", yaxis_title="Signals")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature importance")
    available = [f for f in RISK_WEIGHTS if f in df.columns]
    if model_connected and hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_[:len(available)], index=available)
        title = "Feature importance (trained LightGBM model)"
    else:
        importances = df[available].corrwith(df["eskalatsiya"]).abs().sort_values(ascending=False)
        title = "Correlation with escalation (proxy — no trained model artifact found)"
    fig = px.bar(importances.sort_values(), orientation="h", title=title)
    fig.update_layout(showlegend=False, height=320, xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model performance")
    if model_connected:
        st.info("A trained model artifact is loaded. Wire in your real cross-validation metrics here once computed.")
    else:
        st.warning(
            "No trained model artifact (`lgbm_model.pkl`) or metrics file found in this deployment — "
            "performance numbers are intentionally **not shown** rather than fabricated. "
            "See `Aethera_WIUT_Hackathon.ipynb` for the full 5-fold cross-validated results."
        )


def render_pipeline_data(df, demo_mode):
    st.markdown("## Pipeline & Data")
    st.caption("How 6.98M raw transaction rows become a single risk score per alert.")

    steps = ["Raw Transactions", "Data Cleaning", "Aggregation", "Behavioral Features",
             "Feature Engineering (26 features)", "ML Model", "Risk Score", "Alert Escalation"]
    st.markdown('<div class="pipeline-flow">', unsafe_allow_html=True)
    for i, step in enumerate(steps):
        st.markdown(f'<div class="pipeline-node">{step}</div>', unsafe_allow_html=True)
        if i < len(steps) - 1:
            st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### Dataset statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Signals in this dataset", f"{len(df):,}")
    c2.metric("Escalation rate", f"{df['eskalatsiya'].mean():.2%}")
    c3.metric("Raw transactions (full pipeline)",
              f"{RAW_TRANSACTIONS_KNOWN:,}" if not demo_mode else "N/A — demo dataset in this deployment")
    st.caption(
        "This deployment loads the precomputed per-signal feature file, not the raw 7M-row transaction log — "
        "recomputing 26 aggregate features from the full log on every app boot isn't necessary once the "
        "features are known, and the raw file is too large to ship in the repo."
    )


def render_about():
    st.markdown("## About the Solution")
    st.markdown("""
**Problem.** Automated banking fraud monitors generate far more alerts than analysts can review by hand — over 80% turn out to be false alarms, delaying attention on the real ones.

**Solution.** Team Aethera compresses each account's pre-alert transaction history into 26 behavioral features — velocity, direction, channel, and timing — and scores every alert for escalation risk.

**Innovation.** Rather than fixed thresholds, every feature is a *ratio relative to the account's own recent history*, so the same model generalizes across very different account sizes and habits.

**Technical approach.** Transaction logs are joined to alerts on `signal_id`, filtered to a strict pre-alert window to avoid leakage, aggregated into behavioral features, and classified with a 5-fold stratified LightGBM ensemble.

**Expected impact.** Cutting manual review volume by surfacing the alerts most likely to be real fraud first, without discarding the long tail entirely.

**Future development.** Wire in the trained model artifact for live scoring, add case-level analyst feedback to retrain the model over time, and extend the feature set with merchant- and device-level signals.
""")


# ============================================================
# 5. LOAD, SCORE, ROUTE
# ============================================================
master_df, demo_mode = load_data()
model = load_trained_model()
model_connected = model is not None

master_df["risk_score"] = master_df.apply(
    lambda r: (demo_risk_score({k: r.get(k) for k in RISK_WEIGHTS})), axis=1
)
master_df["risk_level"] = master_df["risk_score"].apply(risk_level)

NAV_OPTIONS = [
    "Command Center", "Alert Investigation", "Behavior Analytics",
    "ML Features", "Model Insights", "Pipeline & Data", "About the Solution",
]
if "nav" not in st.session_state:
    st.session_state.nav = NAV_OPTIONS[0]

with st.sidebar:
    st.markdown("### 🛡 Team Aethera")
    st.caption("Fraud Alert Intelligence Platform")
    st.session_state.nav = st.radio("Navigate", NAV_OPTIONS,
                                     index=NAV_OPTIONS.index(st.session_state.nav),
                                     label_visibility="collapsed")
    st.divider()
    st.caption("🟡 Demo data" if demo_mode else "🟢 Live dataset")
    st.caption("🟢 Model connected" if model_connected else "🟡 Demo scoring (no model artifact)")

nav = st.session_state.nav
if nav == "Command Center":
    render_command_center(master_df, demo_mode, model_connected)
elif nav == "Alert Investigation":
    render_alert_investigation(master_df, model, model_connected)
elif nav == "Behavior Analytics":
    render_behavior_analytics(master_df)
elif nav == "ML Features":
    render_ml_features(master_df)
elif nav == "Model Insights":
    render_model_insights(master_df, model, model_connected)
elif nav == "Pipeline & Data":
    render_pipeline_data(master_df, demo_mode)
elif nav == "About the Solution":
    render_about()
