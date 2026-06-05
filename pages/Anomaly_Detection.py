import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Real-Time Anomaly Detection Dashboard")

st.markdown("""
Detect abnormal equipment behavior using:

- Rolling Statistics
- Z-Score Analysis
- Temperature Monitoring
- Vibration Monitoring
- Failure Alert Generation
""")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Sensor Dataset",
    type=["csv"]
)

if uploaded_file:

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------

    df = pd.read_csv(uploaded_file)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    # ---------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------

    st.sidebar.header("⚙️ Monitoring Controls")

    threshold = st.sidebar.slider(
        "Z-Score Threshold",
        min_value=2.0,
        max_value=4.0,
        value=3.0,
        step=0.1
    )

    equipment = st.sidebar.selectbox(
        "Select Equipment",
        df["device"].unique()
    )

    # ---------------------------------------------------
    # FILTER DEVICE
    # ---------------------------------------------------

    device_df = df[
        df["device"] == equipment
    ].copy()

    # ---------------------------------------------------
    # SENSOR MAPPING
    # ---------------------------------------------------

    # Temperature
    device_df["Temperature"] = device_df["metric1"]

    # Vibration
    device_df["Vibration"] = device_df["metric7"]

    # Voltage
    device_df["Voltage"] = device_df["metric9"]

    # ---------------------------------------------------
    # ROLLING MEAN
    # ---------------------------------------------------

    device_df["Temp_Rolling_Mean"] = (
        device_df["Temperature"]
        .rolling(window=10)
        .mean()
    )

    device_df["Temp_Rolling_STD"] = (
        device_df["Temperature"]
        .rolling(window=10)
        .std()
    )

    device_df["Vib_Rolling_Mean"] = (
        device_df["Vibration"]
        .rolling(window=10)
        .mean()
    )

    device_df["Vib_Rolling_STD"] = (
        device_df["Vibration"]
        .rolling(window=10)
        .std()
    )

    # ---------------------------------------------------
    # Z-SCORE CALCULATION
    # ---------------------------------------------------

    device_df["Temp_Z"] = (
        (
            device_df["Temperature"]
            - device_df["Temperature"].mean()
        )
        /
        device_df["Temperature"].std()
    )

    device_df["Vib_Z"] = (
        (
            device_df["Vibration"]
            - device_df["Vibration"].mean()
        )
        /
        device_df["Vibration"].std()
    )

    # ---------------------------------------------------
    # ANOMALY FLAG
    # ---------------------------------------------------

    device_df["Anomaly_Alert"] = np.where(

        (abs(device_df["Temp_Z"]) > threshold)

        |

        (abs(device_df["Vib_Z"]) > threshold),

        1,

        0
    )

    # ---------------------------------------------------
    # KPI CARDS
    # ---------------------------------------------------

    total_hours = len(device_df)

    current_temp = round(
        device_df["Temperature"].iloc[-1],
        2
    )

    active_alerts = int(
        device_df["Anomaly_Alert"].sum()
    )

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Operating Hours",
            total_hours
        )

    with col2:
        st.metric(
            "Current Temperature",
            current_temp
        )

    with col3:
        st.metric(
            "Critical Alerts",
            active_alerts
        )

    st.divider()

    # ---------------------------------------------------
    # TEMPERATURE DRIFT
    # ---------------------------------------------------

    st.subheader(
        "🌡️ Temperature Drift Analysis"
    )

    failure_limit_temp = (
        device_df["Temperature"].mean()
        +
        threshold *
        device_df["Temperature"].std()
    )

    fig_temp = go.Figure()

    fig_temp.add_trace(

        go.Scatter(
            x=device_df["date"],
            y=device_df["Temperature"],
            mode="lines",
            name="Temperature"
        )
    )

    fig_temp.add_hline(

        y=failure_limit_temp,

        line_color="red",

        annotation_text="Failure Ceiling"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

    # ---------------------------------------------------
    # VIBRATION DRIFT
    # ---------------------------------------------------

    st.subheader(
        "📈 Vibration Drift Analysis"
    )

    failure_limit_vib = (

        device_df["Vibration"].mean()

        +

        threshold *

        device_df["Vibration"].std()
    )

    fig_vib = go.Figure()

    fig_vib.add_trace(

        go.Scatter(
            x=device_df["date"],
            y=device_df["Vibration"],
            mode="lines",
            name="Vibration"
        )
    )

    fig_vib.add_hline(

        y=failure_limit_vib,

        line_color="red",

        annotation_text="Failure Ceiling"
    )

    st.plotly_chart(
        fig_vib,
        use_container_width=True
    )

    # ---------------------------------------------------
    # ALERT TABLE
    # ---------------------------------------------------

    st.subheader(
        "🚨 Active Failure Alerts"
    )

    alerts = device_df[
        device_df["Anomaly_Alert"] == 1
    ]

    st.dataframe(
        alerts[
            [
                "date",
                "device",
                "Temperature",
                "Vibration",
                "Voltage",
                "Anomaly_Alert"
            ]
        ],
        use_container_width=True
    )

    # ---------------------------------------------------
    # DOWNLOAD ALERTS
    # ---------------------------------------------------

    csv = alerts.to_csv(index=False)

    st.download_button(
        label="📥 Download Alert Report",
        data=csv,
        file_name="anomaly_alerts.csv",
        mime="text/csv"
    )
