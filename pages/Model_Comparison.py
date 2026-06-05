import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
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

import plotly.express as px

st.title("📈 Model Comparison Dashboard")

file = st.file_uploader(
    "Upload Dataset",
    type=["csv"]
)

if file:

    df = pd.read_csv(file)

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

    X_train,X_test,y_train,y_test = train_test_split(
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
        RandomForestRegressor()
    }

    results=[]

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred=model.predict(X_test)

        mse=mean_squared_error(y_test,pred)

        rmse=np.sqrt(mse)

        r2=r2_score(y_test,pred)

        cv=np.mean(
            cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="r2"
            )
        )

        results.append(
            [
                name,
                mse,
                rmse,
                r2,
                cv
            ]
        )

    result_df=pd.DataFrame(
        results,
        columns=[
            "Model",
            "MSE",
            "RMSE",
            "R2",
            "CV"
        ]
    )

    st.dataframe(result_df)

    fig1=px.bar(
        result_df,
        x="Model",
        y="RMSE",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2=px.bar(
        result_df,
        x="Model",
        y="R2",
        title="R2 Comparison"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3=px.bar(
        result_df,
        x="Model",
        y="CV",
        title="Cross Validation Comparison"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )
