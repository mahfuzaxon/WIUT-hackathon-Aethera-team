"""
Team Aethera — WIUT FinTech Hackathon
End-to-end pipeline: feature engineering + EDA + 5-fold LightGBM.

Expected input files (edit paths in main() if yours differ):
    train_signals.csv        -> signal_id, signal_sanasi, eskalatsiya
    train_transactions.parquet -> signal_id, tranzaksiya_vaqti, kirim/chiqim,
                                   karta/bank_otkazmasi/naqd/xalqaro, miqdor_indeksi
    test_signals.csv
    test_transactions.parquet

Output:
    team_Aethera.csv          -> signal_id, eskalatsiya (probability)
    eda_plots/*.png           -> the charts used in the EDA writeup
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

PLOT_DIR = "eda_plots"
NIGHT_START, NIGHT_END = 0, 5  # 00:00–05:00


# --------------------------------------------------------------------------
# 1. Feature engineering
# --------------------------------------------------------------------------
def build_features(signals: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Joins signals to their pre-alert transaction history and returns one row
    per signal_id with the 26 engineered behavioral features.
    """
    signals = signals.copy()
    signals["signal_sanasi"] = pd.to_datetime(signals["signal_sanasi"])

    tx = transactions.merge(
        signals[["signal_id", "signal_sanasi"]], on="signal_id", how="inner"
    )
    tx["tranzaksiya_vaqti"] = pd.to_datetime(tx["tranzaksiya_vaqti"])

    # Leakage guard: keep only transactions on/before the alert date.
    delta_days = (tx["signal_sanasi"] - tx["tranzaksiya_vaqti"]).dt.total_seconds() / 86400
    tx = tx.loc[delta_days >= 0].copy()
    tx["delta_days"] = delta_days.loc[tx.index]
    tx["hour"] = tx["tranzaksiya_vaqti"].dt.hour

    rows = []
    for sid, g in tx.groupby("signal_id"):
        amt = g["miqdor_indeksi"]

        def window(days):
            w = g.loc[g["delta_days"] <= days]
            return len(w), w["miqdor_indeksi"].sum()

        cnt_1d, sum_1d = window(1)
        cnt_3d, sum_3d = window(3)
        cnt_7d, sum_7d = window(7)
        cnt_30d, sum_30d = window(30)

        total_tx_count = len(g)
        night_mask = g["hour"].between(NIGHT_START, NIGHT_END, inclusive="left")
        signal_date = signals.loc[signals["signal_id"] == sid, "signal_sanasi"].iloc[0]

        rows.append({
            "signal_id": sid,
            # A. central tendency & volatility
            "total_tx_count": total_tx_count,
            "total_amount": amt.sum(),
            "avg_amount": amt.mean(),
            "max_amount": amt.max(),
            "amount_std": amt.std(ddof=0),
            # B. off-peak & recency
            "night_tx_ratio": night_mask.mean(),
            "last_tx_days": g["delta_days"].min(),
            "tx_time_span": g["delta_days"].max() - g["delta_days"].min(),
            # C. time-windowed aggregations
            "tx_cnt_1d": cnt_1d, "amt_sum_1d": sum_1d,
            "tx_cnt_3d": cnt_3d, "amt_sum_3d": sum_3d,
            "tx_cnt_7d": cnt_7d, "amt_sum_7d": sum_7d,
            "tx_cnt_30d": cnt_30d, "amt_sum_30d": sum_30d,
            # D. directional & channel shares
            "dir_ratio_kirim": (g["kirim_chiqim"] == "kirim").mean(),
            "dir_ratio_chiqim": (g["kirim_chiqim"] == "chiqim").mean(),
            "type_ratio_karta": (g["tolov_turi"] == "karta").mean(),
            "type_ratio_bank_otkazmasi": (g["tolov_turi"] == "bank_otkazmasi").mean(),
            "type_ratio_naqd": (g["tolov_turi"] == "naqd").mean(),
            "type_ratio_xalqaro": (g["tolov_turi"] == "xalqaro").mean(),
            # E. seasonal
            "signal_month": signal_date.month,
            "signal_dayofweek": signal_date.dayofweek,
        })

    feats = pd.DataFrame(rows)
    feats["burst_ratio_3d_30d"] = (feats["tx_cnt_3d"] + 1) / (feats["tx_cnt_30d"] + 1)
    feats["amount_ratio_3d_30d"] = (feats["amt_sum_3d"] + 1) / (feats["amt_sum_30d"] + 1)

    return signals.merge(feats, on="signal_id", how="left")


# --------------------------------------------------------------------------
# 2. EDA plots (matches the charts used in the writeup)
# --------------------------------------------------------------------------
def make_eda_plots(df: pd.DataFrame):
    os.makedirs(PLOT_DIR, exist_ok=True)
    target = "eskalatsiya"

    # -- Target class distribution -----------------------------------------
    counts = df[target].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(["Dismissed (0)", "Escalated (1)"], counts.values, color=["#3FA98A", "#E15241"])
    for i, v in enumerate(counts.values):
        plt.text(i, v, f"{v:,}\n({v/counts.sum():.1%})", ha="center", va="bottom")
    plt.title("Target class distribution")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/target_distribution.png", dpi=150)
    plt.close()

    # -- Behavioral comparison: dismissed vs escalated means ----------------
    ratio_features = [
        "burst_ratio_3d_30d", "amount_ratio_3d_30d",
        "dir_ratio_chiqim", "type_ratio_xalqaro", "night_tx_ratio",
    ]
    means = df.groupby(target)[ratio_features].mean().T
    means.columns = ["Dismissed", "Escalated"]

    x = np.arange(len(ratio_features))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.barh(x - width / 2, means["Dismissed"], height=width, label="Dismissed", color="#3FA98A")
    plt.barh(x + width / 2, means["Escalated"], height=width, label="Escalated", color="#E15241")
    plt.yticks(x, ratio_features)
    plt.xlabel("Mean value")
    plt.title("Behavioral ratios: dismissed vs. escalated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/behavioral_comparison.png", dpi=150)
    plt.close()

    # -- Burst velocity distribution by class --------------------------------
    plt.figure(figsize=(6, 4))
    df.boxplot(column="burst_ratio_3d_30d", by=target, grid=False)
    plt.title("Burst ratio (3d/30d) by class")
    plt.suptitle("")
    plt.xlabel("eskalatsiya")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/burst_ratio_by_class.png", dpi=150)
    plt.close()

    print(f"Saved EDA plots to ./{PLOT_DIR}/")


# --------------------------------------------------------------------------
# 3. Modeling: 5-fold stratified LightGBM
# --------------------------------------------------------------------------
def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
    X = train_df[feature_cols]
    y = train_df["eskalatsiya"]
    X_test = test_df[feature_cols]

    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits

        fold_auc = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
        print(f"Fold {fold} AUC: {fold_auc:.4f}")

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"Overall OOF AUC: {overall_auc:.4f}")
    return test_preds


# --------------------------------------------------------------------------
# 4. Main
# --------------------------------------------------------------------------
def main():
    train_signals = pd.read_csv("train_signals.csv")
    train_transactions = pd.read_parquet("train_transactions.parquet")
    test_signals = pd.read_csv("test_signals.csv")
    test_transactions = pd.read_parquet("test_transactions.parquet")

    print("Building training features...")
    train_df = build_features(train_signals, train_transactions)
    print("Building test features...")
    test_df = build_features(test_signals, test_transactions)

    make_eda_plots(train_df)

    feature_cols = [c for c in train_df.columns if c not in ("signal_id", "signal_sanasi", "eskalatsiya")]
    test_preds = train_and_predict(train_df, test_df, feature_cols)

    submission = pd.DataFrame({
        "signal_id": test_df["signal_id"],
        "eskalatsiya": test_preds,
    })
    submission.to_csv("team_Aethera.csv", index=False)
    print("Saved submission to team_Aethera.csv")


if __name__ == "__main__":
    main()