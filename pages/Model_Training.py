import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

st.title("🤖 Model Training & Optimization")

uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    features = [
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

    X = df[features]
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression":
        LinearRegression(),

        "KNN":
        KNeighborsRegressor(),

        "Decision Tree":
        DecisionTreeRegressor(),

        "Random Forest":
        RandomForestRegressor(
            random_state=42
        )
    }

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, pred)

        cv = np.mean(
            cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="r2"
            )
        )

        results.append([
            name,
            mse,
            rmse,
            r2,
            cv
        ])

    results_df = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "R2",
            "CV Score"
        ]
    )

    st.subheader("Model Performance")

    st.dataframe(results_df)

    st.divider()

    st.subheader("Hyperparameter Tuning")

    if st.button("Optimize Random Forest"):

        param_grid = {

            "n_estimators":
            [100,200,300],

            "max_depth":
            [5,10,15,None],

            "min_samples_split":
            [2,5,10],

            "min_samples_leaf":
            [1,2,4]
        }

        rf = RandomForestRegressor(
            random_state=42
        )

        grid = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(
            X_train,
            y_train
        )

        best_model = grid.best_estimator_

        pred = best_model.predict(
            X_test
        )

        mse = mean_squared_error(
            y_test,
            pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            pred
        )

        st.success(
            "Optimization Complete"
        )

        st.write(
            "Best Parameters"
        )

        st.json(
            grid.best_params_
        )

        st.metric(
            "Optimized RMSE",
            round(rmse,4)
        )

        st.metric(
            "Optimized R2",
            round(r2,4)
        )

        joblib.dump(
            best_model,
            "models/best_model.pkl"
        )

        st.success(
            "Model Saved"
        )
