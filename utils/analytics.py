import pandas as pd


def get_failure_rate(df):

    failures = df["failure"].sum()

    total = len(df)

    return round((failures / total) * 100, 2)


def device_failure_summary(df):

    return (
        df.groupby("device")["failure"]
        .sum()
        .reset_index()
        .sort_values(
            "failure",
            ascending=False
        )
    )


def correlation_with_failure(df):

    corr = df.corr(
        numeric_only=True
    )

    return (
        corr["failure"]
        .sort_values(
            ascending=False
        )
    )
