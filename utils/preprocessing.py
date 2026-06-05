from sklearn.model_selection import train_test_split

def split_data(df):

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

    y = df['failure']

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
