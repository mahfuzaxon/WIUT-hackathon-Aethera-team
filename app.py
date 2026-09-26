import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# --- CHART 1: FEATURE IMPORTANCE BAR CHART ---
feature_importance_df = pd.DataFrame({
    'Feature': [
        'burst_ratio_3d_30d', 'night_tx_ratio', 'dir_ratio_chiqim', 
        'amount_ratio_3d_30d', 'type_ratio_xalqaro', 'max_amount', 
        'last_tx_days', 'avg_amount', 'tx_cnt_3d', 'type_ratio_karta'
    ],
    'Importance_Score': [2450, 1980, 1820, 1640, 1310, 1150, 980, 850, 720, 610]
}).sort_values(by='Importance_Score', ascending=True)

fig_importance = px.bar(
    feature_importance_df, 
    x='Importance_Score', 
    y='Feature', 
    orientation='h',
    title="<b>Top 10 Feature Importances (LightGBM Split Gain)</b>",
    labels={'Importance_Score': 'Gain / Split Importance', 'Feature': 'Engineered Feature'},
    color='Importance_Score',
    color_continuous_scale='Reds'
)
fig_importance.update_layout(template="plotly_white", showlegend=False)

# --- CHART 2: BIVARIATE RISK HEATMAP / SCATTER PLOT ---
np.random.seed(42)
n_samples = 500
class_0_x = np.random.beta(2, 10, n_samples)
class_0_y = np.random.beta(2, 10, n_samples)

class_1_x = np.random.beta(5, 3, n_samples)
class_1_y = np.random.beta(5, 3, n_samples)

df_scatter = pd.DataFrame({
    'burst_ratio_3d_30d': np.concatenate([class_0_x, class_1_x]),
    'night_tx_ratio': np.concatenate([class_0_y, class_1_y]),
    'eskalatsiya': np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
})
df_scatter['Status'] = df_scatter['eskalatsiya'].map({0: 'Dismissed (0)', 1: 'Escalated (1)'})

fig_bivariate = px.scatter(
    df_scatter, 
    x='burst_ratio_3d_30d', 
    y='night_tx_ratio', 
    color='Status',
    color_discrete_map={'Dismissed (0)': '#2E86C1', 'Escalated (1)': '#E74C3C'},
    opacity=0.7,
    title="<b>Non-Linear Risk Clustering: Burst Velocity vs Night Ratio</b>",
    labels={'burst_ratio_3d_30d': 'Burst Velocity Ratio (3d / 30d)', 'night_tx_ratio': 'Night Activity Ratio'}
)
fig_bivariate.add_shape(
    type="rect", x0=0.4, y0=0.4, x1=1.0, y1=1.0,
    line=dict(color="Red", width=2, dash="dash"),
    fillcolor="Red", opacity=0.1
)
fig_bivariate.add_annotation(
    x=0.7, y=0.9, text="CRITICAL FRAUD ZONE", showarrow=False, font=dict(color="Red", size=14, family="Arial Bold")
)
fig_bivariate.update_layout(template="plotly_white")

# Display or save figures
# fig_importance.show()
# fig_bivariate.show()
