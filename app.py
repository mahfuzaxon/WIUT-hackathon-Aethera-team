import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Team Aethera - WIUT FinTech Hackathon",
    page_icon="🛡️",
    layout="wide"
)

# --- TITLE & HEADER ---
st.title("🛡️ Team Aethera: Fraud Signal Escalation Dashboard")
st.caption("WIUT FinTech Hackathon 2026 | Exploratory Data Analysis & Predictive Pipeline")

# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_data():
    # Load sample/master dataset if available, or generate dataset for display
    try:
        df = pd.read_csv("master_unified_dataset.csv")
    except:
        # Fallback to loading raw files if available locally
        signals = pd.read_csv("train_signals.csv")
        df = signals.copy()
    return df

try:
    master_df = load_data()
except Exception as e:
    st.error(f"Error loading master dataset: {e}")
    master_df = None

# --- TAB NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 1. Executive Summary", 
    "🗂️ 2. Dataset Overview", 
    "📊 3. Behavioral EDA Insights", 
    "⚙️ 4. Modeling Strategy", 
    "🎯 5. Conclusion"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab1:
    st.header("Executive Summary & Problem Approach")
    st.markdown("""
    **Team Aethera** developed an end-to-end Machine Learning pipeline to predict whether automated banking security alerts (`signal_id`) 
    will be **Escalated (1 - True Fraud Risk)** or **Dismissed (0 - False Alarm)**.
    
    By linking over **6.9 million historical transaction records** to 14,000 security signals, we engineered 26 behavioral features 
    capturing **Burst Velocity** (short-term activity spikes), **Off-Peak Night Activity**, and **Transfer Channel Ratios**.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Training Signals", "14,000", help="Total security alerts evaluated")
    col2.metric("Escalation Rate (Fraud)", "17.18%", help="Proportion of true fraud alerts (Class Imbalance)")
    col3.metric("Total Transactions Analyzed", "6,987,663", help="Raw historical transaction rows aggregated")

# ==========================================
# TAB 2: DATASET OVERVIEW
# ==========================================
with tab2:
    st.header("Dataset Overview & Structure")
    st.write("""
    The dataset consists of two relational sources linked together via `signal_id`. To avoid data leakage, transaction time windows 
    were strictly filtered relative to each signal's alert date (`signal_sanasi`).
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("1. `train_signals.csv` (Alert Metadata)")
        st.markdown("""
        * **`signal_id`**: Unique alert identifier (Primary Key).
        * **`signal_sanasi`**: Date when the alert triggered.
        * **`eskalatsiya`**: Target label ($0 = \\text{Dismissed}$, $1 = \\text{Escalated}$).
        """)
    
    with col_b:
        st.subheader("2. `train_transactions.parquet` (Transaction Logs)")
        st.markdown("""
        * **`signal_id`**: Foreign key linking transaction to alert.
        * **`tranzaksiya_vaqti`**: Exact timestamp of transfer.
        * **`kirim_chiqim`**: Direction (`kirim` / `chiqim`).
        * **`tranzaksiya_turi`**: Payment method (`karta`, `bank_otkazmasi`, `naqd`, `xalqaro`).
        * **`miqdor_indeksi`**: Standardized transaction size index.
        """)

# ==========================================
# TAB 3: BEHAVIORAL EDA INSIGHTS (CHARTS)
# ==========================================
with tab3:
    st.header("Exploratory Data Analysis: Key Behavioral Patterns")
    st.write("Comparing feature distributions between **Dismissed (0)** and **Escalated (1)** signals reveals clear risk indicators.")
    
    if master_df is not None and 'eskalatsiya' in master_df.columns:
        
        # CHART 1: Target Class Distribution
        st.subheader("1. Target Class Distribution (Imbalance Analysis)")
        target_counts = master_df['eskalatsiya'].value_counts().reset_index()
        target_counts.columns = ['Status', 'Count']
        target_counts['Status_Name'] = target_counts['Status'].map({0: 'Dismissed (0)', 1: 'Escalated (1)'})
        
        fig_target = px.pie(
            target_counts, 
            values='Count', 
            names='Status_Name', 
            color='Status_Name',
            color_discrete_map={'Dismissed (0)': '#2E86C1', 'Escalated (1)': '#E74C3C'},
            hole=0.4,
            title="Distribution of Target Escalations (82.8% Dismissed vs 17.2% Escalated)"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        st.info("💡 **Insight:** Only 17.18% of alerts represent true fraud. This class imbalance motivated our use of 5-Fold Stratified Cross-Validation.")

        st.divider()

        # CHART 2: Burst Velocity Comparison (3d vs 30d Ratio)
        st.subheader("2. Burst Velocity Ratio (`burst_ratio_3d_30d`) Distribution")
        if 'burst_ratio_3d_30d' in master_df.columns:
            fig_burst = px.box(
                master_df, 
                x='eskalatsiya', 
                y='burst_ratio_3d_30d',
                color='eskalatsiya',
                labels={'eskalatsiya': 'Eskalatsiya (0=Dismissed, 1=Escalated)', 'burst_ratio_3d_30d': '3-Day to 30-Day Activity Ratio'},
                color_discrete_map={0: '#2E86C1', 1: '#E74C3C'},
                title="Burst Velocity Spike Comparison"
            )
            st.plotly_chart(fig_burst, use_container_width=True)
            st.info("💡 **Insight:** Escalated signals show significantly higher short-term activity surges (`burst_ratio_3d_30d`), proving that fraudsters rapidly execute transactions right before an account gets blocked.")

        st.divider()

        # CHART 3: Night Transaction Ratio
        st.subheader("3. Off-Peak Night Transactions (`night_tx_ratio`)")
        if 'night_tx_ratio' in master_df.columns:
            fig_night = px.histogram(
                master_df, 
                x='night_tx_ratio', 
                color='eskalatsiya', 
                barmode='overlay',
                nbins=30,
                color_discrete_map={0: '#2E86C1', 1: '#E74C3C'},
                title="Proportion of Nighttime Activity (00:00 - 05:00 AM)"
            )
            st.plotly_chart(fig_night, use_container_width=True)
            st.info("💡 **Insight:** High concentration of off-peak nighttime transfers strongly correlates with true fraud alerts.")

    else:
        st.warning("Master dataset not loaded or missing columns for rendering interactive charts.")

# ==========================================
# TAB 4: MODELING STRATEGY
# ==========================================
with tab4:
    st.header("EDA-Driven Modeling Strategy & Feature Engineering")
    st.markdown("""
    Our Machine Learning pipeline directly translates the insights discovered during EDA into model inputs:
    
    1. **Feature Engineering (26 Features Created):**
       * **Burst Ratios:** Created `burst_ratio_3d_30d` and `amount_ratio_3d_30d` to quantify velocity spikes.
       * **Channel Breakdown:** Extracted percentage shares of `xalqaro` (international) and `chiqim` (outgoing) transactions.
       * **Temporal Metrics:** Computed `night_tx_ratio` and time elapsed since the last transaction (`last_tx_days`).
    
    2. **Model Choice — LightGBM Classifier:**
       * Selected **LightGBM** for its superior performance on tabular datasets and ability to handle non-linear feature interactions (e.g., high burst ratio combined with international transfers).
    
    3. **Cross-Validation Setup:**
       * Implemented **5-Fold Stratified Cross-Validation** to ensure every fold maintained the ground-truth 17.18% escalation ratio, preventing class imbalance bias.
    """)

# ==========================================
# TAB 5: CONCLUSION
# ==========================================
with tab5:
    st.header("Conclusion & Key Findings")
    st.success("""
    ### Summary of Findings:
    * **Short-term velocity spikes (`burst_ratio_3d_30d`) and nocturnal transfers (`night_tx_ratio`)** are the strongest behavioral predictors of fraud escalation.
    * Transforming nearly 7 million raw transaction rows into 26 normalized, signal-level features enabled our LightGBM model to effectively isolate true fraud risks ($1$) from high-volume false alarms ($0$).
    * All predictions were successfully generated and formatted in `team_Aethera.csv` for submission.
    """)