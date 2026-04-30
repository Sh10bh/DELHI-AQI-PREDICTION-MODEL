import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Delhi AQI Predictor",
    page_icon="🌫️",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    model  = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

def get_aqi_category(aqi):
    if aqi <= 50:   return "Good",                    "#00e400", "🟢"
    if aqi <= 100:  return "Moderate",                "#ffff00", "🟡"
    if aqi <= 150:  return "Unhealthy for Sensitive", "#ff7e00", "🟠"
    if aqi <= 200:  return "Unhealthy",               "#ff0000", "🔴"
    if aqi <= 300:  return "Very Unhealthy",          "#8f3f97", "🟣"
    return             "Hazardous",                   "#7e0023", "⚫"

st.title("🌫️ Delhi AQI Predictor")
st.caption("XGBoost · GridSearchCV tuned · R² 0.93+ · 9 features")
st.divider()

col_input, col_output = st.columns([1, 1.6], gap="large")

with col_input:
    st.subheader("📅 Date & Context")
    month          = st.slider("Month",               1,  12, datetime.now().month)
    year           = st.slider("Year",             2015, 2025, datetime.now().year)
    holidays_count = st.slider("Holidays in month",   0,   5, 1)

    st.subheader("🏭 Pollutant Levels (μg/m³)")
    pm25  = st.slider("PM2.5",  0.0, 500.0,  95.0, step=0.5)
    pm10  = st.slider("PM10",   0.0, 600.0, 180.0, step=1.0)
    no2   = st.slider("NO2",    0.0, 200.0,  50.0, step=0.5)
    so2   = st.slider("SO2",    0.0, 100.0,  18.0, step=0.5)
    co    = st.slider("CO",     0.0,  50.0,   1.8, step=0.1)
    ozone = st.slider("Ozone",  0.0, 200.0,  38.0, step=0.5)

with col_output:
    input_df = pd.DataFrame([{
        'Month':          month,
        'Year':           year,
        'Holidays_Count': holidays_count,
        'PM2.5':          pm25,
        'PM10':           pm10,
        'NO2':            no2,
        'SO2':            so2,
        'CO':             co,
        'Ozone':          ozone
    }])

    input_scaled = scaler.transform(input_df)
    prediction   = model.predict(input_scaled)[0]
    category, color, emoji = get_aqi_category(prediction)

    st.subheader("📊 Prediction Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted AQI",  f"{prediction:.1f}")
    m2.metric("Category",       category)
    m3.metric("Model R²",       "0.93+")

    st.markdown(
        f"<div style='background:{color}18; border-left:4px solid {color};"
        f"padding:14px 18px; border-radius:8px; margin:12px 0'>"
        f"<span style='font-size:22px'>{emoji}</span> "
        f"<strong style='color:{color}; font-size:16px'>{category}</strong>"
        f"<span style='color:#aaa; margin-left:10px'>AQI {prediction:.1f}</span></div>",
        unsafe_allow_html=True
    )

    st.subheader("🔍 Feature Importance")
    feat_names  = input_df.columns.tolist()
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        orientation="h",
        marker_color=["#5643bd" if feat_names[i] in ["PM2.5","PM10","Ozone"]
                      else "#9b8ed4" for i in sorted_idx],
        text=[f"{importances[i]:.3f}" for i in sorted_idx],
        textposition="outside"
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=50, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12)),
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Input values sent to model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Built by **Shubh Gupta** · VIT Bhopal CSE · "
           "[GitHub](https://github.com/Sh10bh/DELHI-AQI-PREDICTION-MODEL)")

