import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="RUL Dashboard", layout="wide")

# ===============================
# WHITE PREMIUM UI
# ===============================
st.markdown("""
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: #0f172a;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
}
h1, h2, h3 {
    color: #1e293b;
}
section[data-testid="stSidebar"] {
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>⚡ Intelligent RUL Monitoring System</h1>
<p style='text-align: center; color: gray;'>Predictive Maintenance Dashboard</p>
""", unsafe_allow_html=True)

# ===============================
# FIX PICKLE
# ===============================
class RULScaler:
    def transform(self, arr):
        return arr / 125.0
    def inverse_transform(self, arr):
        return arr * 125.0

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    with open("session_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    df = pd.DataFrame({
        "Actual_RUL": cache["rul_test"],
        "Predicted_RUL": cache["y_pred_ensemble"],
        "LSTM": cache["y_pred_lstm"],
        "GRU": cache["y_pred_gru"],
        "Alert": cache["alerts"]
    })

    return df

df = load_data()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Controls")

status_filter = st.sidebar.selectbox(
    "Filter Status",
    ["All", "NORMAL", "WARNING", "CRITICAL"]
)

sample_id = st.sidebar.slider("Select Sample", 0, len(df)-1, 0)

filtered_df = df if status_filter == "All" else df[df["Alert"] == status_filter]
row = df.iloc[sample_id]

# ===============================
# KPI CARDS
# ===============================
normal = (df["Alert"] == "NORMAL").sum()
warning = (df["Alert"] == "WARNING").sum()
critical = (df["Alert"] == "CRITICAL").sum()

col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><h3>Healthy</h3><h1>{normal}</h1></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><h3>Warning</h3><h1>{warning}</h1></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><h3>Critical</h3><h1>{critical}</h1></div>", unsafe_allow_html=True)

st.markdown("---")

# ===============================
# MODEL PERFORMANCE (RECTANGLE BARS)
# ===============================
from sklearn.metrics import mean_squared_error

rmse_lstm = np.sqrt(mean_squared_error(df["Actual_RUL"], df["LSTM"]))
rmse_gru = np.sqrt(mean_squared_error(df["Actual_RUL"], df["GRU"]))
rmse_ens = np.sqrt(mean_squared_error(df["Actual_RUL"], df["Predicted_RUL"]))

perf_df = pd.DataFrame({
    "Model": ["LSTM", "GRU", "Ensemble"],
    "RMSE": [rmse_lstm, rmse_gru, rmse_ens]
})

fig = px.bar(
    perf_df,
    x="Model",
    y="RMSE",
    color="Model",
    color_discrete_sequence=["#3b82f6", "#6366f1", "#22c55e"]
)

fig.update_traces(width=0.4)

fig.update_layout(
    height=350,
    bargap=0.5,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#0f172a"),
    title="Model RMSE Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# ===============================
# ALERT DISTRIBUTION
# ===============================
fig = px.pie(
    df,
    names="Alert",
    color_discrete_sequence=["#22c55e","#facc15","#ef4444"]
)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ENGINE DETAILS
# ===============================
st.subheader("Selected Sample")

col1, col2 = st.columns(2)

with col1:
    st.metric("Actual RUL", round(row["Actual_RUL"], 2))
    st.metric("Predicted RUL", round(row["Predicted_RUL"], 2))

    if row["Alert"] == "NORMAL":
        st.success("NORMAL")
    elif row["Alert"] == "WARNING":
        st.warning("WARNING")
    else:
        st.error("CRITICAL")

with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=row["Predicted_RUL"],
        gauge={
            'axis': {'range': [0, 130]},
            'steps': [
                {'range': [0, 20], 'color': "#ef4444"},
                {'range': [20, 50], 'color': "#facc15"},
                {'range': [50, 130], 'color': "#22c55e"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# RUL TREND
# ===============================
fig = go.Figure()
fig.add_trace(go.Scatter(y=df["Actual_RUL"][:500], name="Actual"))
fig.add_trace(go.Scatter(y=df["Predicted_RUL"][:500], name="Ensemble"))
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ALERT SCATTER
# ===============================
color_map = {"NORMAL":"#22c55e","WARNING":"#facc15","CRITICAL":"#ef4444"}

fig = go.Figure()
fig.add_trace(go.Scatter(
    y=df["Predicted_RUL"][:2000],
    mode='markers',
    marker=dict(color=df["Alert"].map(color_map), size=4)
))

fig.add_hline(y=50, line_dash="dash", line_color="#facc15")
fig.add_hline(y=20, line_dash="dash", line_color="#ef4444")

st.plotly_chart(fig, use_container_width=True)

# ===============================
# DATA TABLE
# ===============================
st.subheader("Data Table")

def color_alert(val):
    if val == "NORMAL":
        return "background-color: #22c55e; color: white"
    elif val == "WARNING":
        return "background-color: #facc15; color: black"
    else:
        return "background-color: #ef4444; color: white"

rows = st.slider("Rows to display", 100, 1000, 500)

df_display = filtered_df.head(rows)
styled_df = df_display.style.map(color_alert, subset=["Alert"])

st.dataframe(styled_df, use_container_width=True)