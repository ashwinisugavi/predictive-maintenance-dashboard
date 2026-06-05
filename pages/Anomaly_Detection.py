st.title("⚠️ Real-Time Anomaly Detection Dashboard")

st.markdown("""
### Detect abnormal equipment behavior using:

✅ Rolling Statistics

✅ Z-Score Analysis

✅ Temperature Monitoring

✅ Vibration Monitoring

✅ Failure Alert Generation

---

### Upload Sensor Dataset

Upload a CSV file containing:

- Timestamp (date)
- Equipment ID (device)
- Sensor Metrics (metric1 ... metric9)
- Failure Status (failure)

The system will:

1. Calculate rolling mean and rolling standard deviation
2. Compute Z-Scores for temperature and vibration
3. Detect anomalies based on the selected threshold
4. Generate critical alerts
5. Visualize sensor drift and failure ceilings
""")

uploaded_file = st.file_uploader(
    "📂 Upload Sensor Dataset",
    type=["csv"]
)
