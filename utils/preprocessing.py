from sklearn.model_selection import train_test_split


def prepare_features(df):

    feature_cols = [
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

    X = df[feature_cols]

    y = df["failure"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
