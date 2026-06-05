import joblib
import pandas as pd


def save_model(
        model,
        path
):

    joblib.dump(
        model,
        path
    )


def load_model(path):

    return joblib.load(path)


def create_prediction_dataframe(
        values
):

    columns = [

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

    return pd.DataFrame(
        [values],
        columns=columns
    )
