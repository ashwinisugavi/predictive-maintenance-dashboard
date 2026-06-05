import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Predictive Maintenance Analytics Dashboard")

uploaded_file = st.file_uploader(
    "Upload Predictive Maintenance Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df["date"] = pd.to_datetime(df["date"])

    # -------------------------------------
    # Sidebar
    # -------------------------------------

    st.sidebar.header("Filters")

    selected_device = st.sidebar.selectbox(
        "Select Equipment",
        ["All"] + list(df["device"].unique())
    )

    if selected_device != "All":
        df = df[df["device"] == selected_device]

    # -------------------------------------
    # KPI CARDS
    # -------------------------------------

    total_records = len(df)

    total_devices = df["device"].nunique()

    total_failures = df["failure"].sum()

    failure_rate = round(
        (total_failures / total_records) * 100,
        2
    )

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Records",
            total_records
        )

    with col2:
        st.metric(
            "Devices",
            total_devices
        )

    with col3:
        st.metric(
            "Failures",
            int(total_failures)
        )

    with col4:
        st.metric(
            "Failure Rate %",
            failure_rate
        )

    st.divider()

    # -------------------------------------
    # DATA PREVIEW
    # -------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # -------------------------------------
    # FAILURE TREND
    # -------------------------------------

    st.subheader("Failure Trend")

    failure_daily = (
        df.groupby("date")["failure"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        failure_daily,
        x="date",
        y="failure",
        title="Failure Trend Over Time"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------------------
    # SENSOR TRENDS
    # -------------------------------------

    st.subheader("Sensor Trends")

    sensor = st.selectbox(
        "Select Sensor Metric",
        [
            "metric1",
            "metric2",
            "metric3",
            "metric4",
            "metric5",
            "metric6",
            "metric7",
            "metric8",
            "metric9"
        ]
    )

    fig2 = px.line(
        df,
        x="date",
        y=sensor,
        color="device",
        title=f"{sensor} Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # -------------------------------------
    # DEVICE FAILURE ANALYSIS
    # -------------------------------------

    st.subheader("Device Failure Analysis")

    device_failure = (
        df.groupby("device")["failure"]
        .sum()
        .reset_index()
        .sort_values(
            "failure",
            ascending=False
        )
    )

    fig3 = px.bar(
        device_failure,
        x="device",
        y="failure",
        title="Failures by Device"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    # -------------------------------------
    # FAILURE RECORDS
    # -------------------------------------

    st.subheader("Failure Records")

    failure_df = df[
        df["failure"] == 1
    ]

    st.dataframe(
        failure_df
    )
