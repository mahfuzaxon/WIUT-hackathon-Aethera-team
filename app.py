import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Fintech Risk Alert Analysis", layout="wide")

st.title("🚨 Fraud & Alert Escalation EDA Dashboard")
st.markdown("Exploratory Data Analysis for automated financial monitoring alerts.")

# Dynamically set base directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    # Try finding files in current directory first, fallback to local path if running on Mac
    signals_file = "train_signals.csv"
    if not os.path.exists(os.path.join(BASE_DIR, signals_file)):
        signals_path = "/Users/Mahfuzaxon/Desktop/fintech_data/train_signals.csv"
    else:
        signals_path = os.path.join(BASE_DIR, signals_file)
        
    train_signals = pd.read_csv(signals_path)
    
    # Check if parquet file is available
    tx_file = "train_transactions.parquet"
    tx_path = os.path.join(BASE_DIR, tx_file)
    if not os.path.exists(tx_path):
        tx_path = "/Users/Mahfuzaxon/Desktop/fintech_data/train_transactions.parquet"
        
    train_tx = None
    if os.path.exists(tx_path):
        train_tx = pd.read_parquet(tx_path)
        
    return train_signals, train_tx

try:
    train_signals, train_tx = load_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Training Signals", f"{len(train_signals):,}")
    col2.metric("Escalation Rate", f"{train_signals['eskalatsiya'].mean():.2%}")
    if train_tx is not None:
        col3.metric("Total Historical Transactions", f"{len(train_tx):,}")
    else:
        col3.metric("Total Test Signals", "6,000")

    st.divider()

    st.subheader("1. Target Class Distribution")
    fig_target = px.pie(
        train_signals, 
        names='eskalatsiya', 
        title="Escalated (1) vs Dismissed (0) Alerts",
        color_discrete_sequence=['#2ecc71', '#e74c3c']
    )
    st.plotly_chart(fig_target, use_container_width=True)

    if train_tx is not None:
        st.subheader("2. Transaction Types and Directions")
        col_a, col_b = st.columns(2)

        fig_type = px.histogram(
            train_tx, 
            x="tranzaksiya_turi", 
            color="kirim_chiqim", 
            barmode="group", 
            title="Transaction Types by Direction"
        )
        col_a.plotly_chart(fig_type, use_container_width=True)

        fig_dist = px.box(
            train_tx.sample(min(5000, len(train_tx))), 
            x="tranzaksiya_turi", 
            y="miqdor_indeksi", 
            title="Transaction Size Distribution"
        )
        col_b.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("2. Key Modeling Insights")
    st.markdown("""
    - **Burst Velocity:** Rapid transaction surges within 3 to 7 days prior to an alert strongly correlate with escalation.
    - **International Transfers (`xalqaro`):** Accounts with higher proportions of international transfers carry significantly higher escalation probabilities.
    - **Night Activity:** A surge in transaction activity during off-peak hours (00:00 - 05:00) serves as an effective risk indicator.
    """)

except Exception as e:
    st.error(f"Error loading datasets: {e}")