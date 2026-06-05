from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def get_models():

    return {

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
