from data_module import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only
#This for now is a testing archive for me to learn how to use scikit-learn for the implementation of linear models
direct_data = fetch_california_housing()
data=pd.DataFrame(direct_data.data, columns=direct_data.feature_names)
dataframe=DataModule()
dataframe.direct_data_load(data)
X = dataframe.get_features()
y = dataframe.get_target()
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Coeficiente:", model.coef_)
print("Intercepto:", model.intercept_)
print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R²):", r2)

if __name__ == "__main__":
    pass #This space in the future will include a simple test for the linear regression model, but for now it's not necessary