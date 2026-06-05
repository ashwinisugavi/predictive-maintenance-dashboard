import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Prediction Center",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Predictive Maintenance Prediction Center")

st.markdown("""
Predict equipment failure risk using the trained machine learning model.

Enter the latest sensor readings and click **Predict Failure Risk**.
""")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

try:

    model = joblib.load(
        "models/best_model.pkl"
    )

    st.success(
        "Model Loaded Successfully"
    )

except:

    st.error(
        "best_model.pkl not found. Train and save a model first."
    )

    st.stop()

# --------------------------------------------------
# SENSOR INPUTS
# --------------------------------------------------

st.subheader("Sensor Inputs")

col1,col2,col3 = st.columns(3)

with col1:

    metric1 = st.number_input(
        "Metric 1 (Temperature)",
        value=50.0
    )

    metric2 = st.number_input(
        "Metric 2",
        value=10.0
    )

    metric3 = st.number_input(
        "Metric 3",
        value=5.0
    )

with col2:

    metric4 = st.number_input(
        "Metric 4",
        value=20.0
    )

    metric5 = st.number_input(
        "Metric 5",
        value=15.0
    )

    metric6 = st.number_input(
        "Metric 6",
        value=25.0
    )

with col3:

    metric7 = st.number_input(
        "Metric 7 (Vibration)",
        value=30.0
    )

    metric8 = st.number_input(
        "Metric 8",
        value=12.0
    )

    metric9 = st.number_input(
        "Metric 9 (Voltage)",
        value=220.0
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Failure Risk"):

    input_data = pd.DataFrame(
        [[
            metric1,
            metric2,
            metric3,
            metric4,
            metric5,
            metric6,
            metric7,
            metric8,
            metric9
        ]],

        columns=[
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

    prediction = model.predict(
        input_data
    )[0]

    # ------------------------------------
    # CONVERT TO RISK %
    # ------------------------------------

    risk_score = float(prediction)

    risk_percent = max(
        0,
        min(
            round(risk_score * 100,2),
            100
        )
    )

    st.divider()

    st.subheader(
        "Prediction Results"
    )

    col1,col2,col3 = st.columns(3)

    with col1:

        st.metric(
            "Failure Risk %",
            f"{risk_percent}%"
        )

    with col2:

        if risk_percent > 70:

            status = "Critical"

        elif risk_percent > 40:

            status = "Warning"

        else:

            status = "Healthy"

        st.metric(
            "Equipment Status",
            status
        )

    with col3:

        st.metric(
            "Prediction Time",
            datetime.now().strftime("%H:%M:%S")
        )

    # ------------------------------------
    # GAUGE CHART
    # ------------------------------------

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk_percent,

            title={
                "text":
                "Failure Risk Score"
            },

            gauge={
                "axis":{
                    "range":[0,100]
                },

                "steps":[

                    {
                        "range":[0,40],
                        "color":"green"
                    },

                    {
                        "range":[40,70],
                        "color":"yellow"
                    },

                    {
                        "range":[70,100],
                        "color":"red"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------
    # MAINTENANCE ADVICE
    # ------------------------------------

    st.subheader(
        "Maintenance Recommendation"
    )

    if risk_percent > 70:

        st.error("""
        Immediate inspection required.

        • Check vibration levels

        • Inspect motor bearings

        • Verify power supply

        • Schedule maintenance
        """)

    elif risk_percent > 40:

        st.warning("""
        Early warning detected.

        • Monitor equipment

        • Review operating conditions

        • Increase inspection frequency
        """)

    else:

        st.success("""
        Equipment operating normally.

        • Continue routine monitoring

        • No immediate action required
        """)

    # ------------------------------------
    # DOWNLOAD REPORT
    # ------------------------------------

    report = pd.DataFrame({

        "Metric1":[metric1],
        "Metric2":[metric2],
        "Metric3":[metric3],
        "Metric4":[metric4],
        "Metric5":[metric5],
        "Metric6":[metric6],
        "Metric7":[metric7],
        "Metric8":[metric8],
        "Metric9":[metric9],

        "Failure_Risk_%":[risk_percent],

        "Status":[status]
    })

    csv = report.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="prediction_report.csv",
        mime="text/csv"
    )
