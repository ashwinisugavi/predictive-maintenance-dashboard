import pandas as pd

def load_data(file):

    df = pd.read_csv(file)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df
