import numpy as np


def calculate_zscore(series):

    return (
        series - series.mean()
    ) / series.std()


def detect_anomaly(df, threshold=3):

    temp_z = calculate_zscore(
        df["metric1"]
    )

    vib_z = calculate_zscore(
        df["metric7"]
    )

    df["Anomaly_Alert"] = np.where(

        (abs(temp_z) > threshold)

        |

        (abs(vib_z) > threshold),

        1,

        0
    )

    return df


def rolling_statistics(df):

    df["temp_mean"] = (
        df["metric1"]
        .rolling(10)
        .mean()
    )

    df["temp_std"] = (
        df["metric1"]
        .rolling(10)
        .std()
    )

    df["vib_mean"] = (
        df["metric7"]
        .rolling(10)
        .mean()
    )

    df["vib_std"] = (
        df["metric7"]
        .rolling(10)
        .std()
    )

    return df
