from data_module import DataModule
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only
from typing import Optional
from matplotlib import pyplot as plt, figure as PltObject
#This model is to be highly integrated with the gui.
#I don't think I need to add memorization or saving/loading capabilities here,
#so this will simply be a model object that the gui can interact with to create and evaluate linear regression models.
#As stated, all comments and docstrings will be in English for consistency with the rest of the codebase.
class LinearRegressionModel:
    def __init__(self):
        self.model = SklearnLinearRegression()
        self.initialized = False
        self.feature_names = None
        self.metrics_train = {'r2': None, 'mse': None}
        self.metrics_test = {'r2': None, 'mse': None}
        self.coef_ = None
        self.intercept_ = None

    def fit_and_evaluate(self, X_train: Optional[pd.DataFrame], y_train: pd.Series, X_test: Optional[pd.DataFrame], y_test: pd.Series): #In the gui you already can split data, so no need to do it here
        """Trains the model and evaluates both data splits"""                                                                           #X_train and X_test can be also pd.Series in case of single feature chosen
        if isinstance(X_train, pd.Series) != isinstance(X_test, pd.Series):
            raise ValueError("X_train y X_test deben ser del mismo tipo (DataFrame o Series)")
    
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            raise ValueError("Número de muestras inconsistente entre X e y")
        
        if not all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All features must be valid numeric types")
        if X_train.isnull().any().any() or y_train.isnull().any(): #We should not get here if gui is used properly
            raise ValueError("Data contains NaN")
        
        #--We save feature names for formula visual representation
        self.feature_names = X_train.columns.tolist()
        
        #--Fit and predictions
        self.model.fit(X_train, y_train)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        #--Train metrics and we store them
        y_train_pred = self.model.predict(X_train)
        self.metrics_train['mse'] = mean_squared_error(y_train, y_train_pred)
        self.metrics_train['r2'] = r2_score(y_train, y_train_pred)

        #--Test metrics and we store them(DOD requests both)
        y_test_pred = self.model.predict(X_test)
        self.metrics_test['mse'] = mean_squared_error(y_test, y_test_pred)
        self.metrics_test['r2'] = r2_score(y_test, y_test_pred)
        
        self.initialized = True
        return self.metrics_train, self.metrics_test

    def formula_string(self) -> str:  # DOD requests formula representation
        
        if not self.initialized:
            return "Model not initialized"

        terms = []
        for name, coef in zip(self.feature_names, self.coef_):
            terms.append(f"{coef:.3f}*{name}")
        
        return f"y = {' + '.join(terms)} + {self.intercept_:.3f}"

    def can_plot(self):
        """Checks if the model can be plotted (1 feature)"""
        return self.initialized and len(self.feature_names) == 1

    def get_plot_item(self, X_train, y_train, X_test, y_test) -> PltObject:
        """Generates a plot of the model predictions against actual data"""
        if not self.can_plot():
            raise ValueError("Only models with 1 feature can be plotted(two dimensions only)")
        elif not self.initialized:
            raise ValueError("Model not initialized") #We could also return None or an empty plot, but this informs the user better

        x_line = np.linspace(min(X_train.iloc[:,0]), max(X_train.iloc[:,0]), 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        plt.figure(figsize=(12, 8))
        plt.scatter(X_train, y_train, color='blue', label='Train Data', alpha=0.5, s=20)  # alpha being transparency, s being point size
        plt.scatter(X_test, y_test, color='green', label='Test Data', alpha=0.5, s=20)
        plt.plot(x_line, y_line, color='red', label='Prediction Line', linewidth=3)
        plt.xlabel(self.feature_names[0])#DOD requests feature name in plot
        plt.ylabel("Target")
        plt.title("Linear Regression")
        plt.legend()
        return plt #Returning the plt object for further manipulation or display

    def predict(self, X):
        """Makes predictions on new data"""
        if not self.initialized:
            raise ValueError("Model not initialized")
        return self.model.predict(X)        
    
    @property
    def is_initialized(self): #in case we need to check externally if the model is created
        return self.initialized
    
    @property
    def get_R2(self): #This two are to be called by gui after evaluation, to show results
        if self.initialized:
            return self.metrics_test['r2']#DOD requests test metrics only
        return "NO DATA" #If not initialized
    
    @property
    def get_MSE(self):
        """Returns MSE on test data or high value if uninitialized"""
        if self.initialized:
            return self.metrics_test['mse']#DOD requests test metrics only
        return "NO DATA" #If not initialized

    @property
    def metrics(self):
        """Returns all metrics in a clear format"""
        if not self.initialized:
            return "NO DATA"
        return {
            'train': self.metrics_train,
            'test': self.metrics_test
        }


if __name__ == "__main__":
    Ans=input("Run a test example of LinearRegressionModel? (y/n): ")
    if Ans.lower() == "y":
        from sklearn.model_selection import train_test_split
        
        model = LinearRegressionModel()
        example_data = fetch_california_housing(as_frame=True)
        columns=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        feature=input(f"Select feature from {columns}(0-{len(columns)-1}): ")
        while not feature in [str(i) for i in range(len(columns))]:
            feature=input(f"Invalid selection. Select feature from {columns}(0-{len(columns)-1}): ")
        X = example_data.data[[columns[int(feature)]]]  # Using only one feature 
        y = example_data.target
        
        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model.fit_and_evaluate(X_train, y_train, X_test, y_test)
        print("Formula:", model.formula_string())
        print("Metrics:", model.metrics)
        plt = model.get_plot_item(X_train, y_train, X_test, y_test)
        plt.show()
    else:
        print("Ok, no test for you then.")