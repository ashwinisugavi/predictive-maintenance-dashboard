import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

st.title("🔍 Root Cause Analysis")

file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if file:

    df = pd.read_csv(file)

    st.subheader(
        "Failure Distribution"
    )

    st.bar_chart(
        df["failure"].value_counts()
    )

    st.subheader(
        "Correlation Heatmap"
    )

    corr = df.corr(
        numeric_only=True
    )

    fig,ax = plt.subplots(
        figsize=(12,8)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    st.subheader(
        "Failure Correlation"
    )

    failure_corr = corr[
        "failure"
    ].sort_values(
        ascending=False
    )

    st.dataframe(
        failure_corr
    )

    X = df[
        [
            'metric1',
            'metric2',
            'metric3',
            'metric4',
            'metric5',
            'metric6',
            'metric7',
            'metric8',
            'metric9'
        ]
    ]

    y = df["failure"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X,y)

    importance = pd.DataFrame({

        "Feature":
        X.columns,

        "Importance":
        model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.subheader(
        "Feature Importance"
    )

    st.dataframe(
        importance
    )

    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )
