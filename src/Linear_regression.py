from data_module import DataModule
import pandas as pd
from sklearn.linear_model import LinearRegression as Sklearn_LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only

#This model is to be highly integrated with the gui.
#I don't think I need to add memorization or saving/loading capabilities here,
#so this will simply be a model object that the gui can interact with to create and evaluate linear regression models.
class LinearRegressionModel(Sklearn_LinearRegression):
    def __init__(self, data=None, target=None, initialized=False, R2=None, MSE=None):
        super().__init__()
        self.initialized = False#to easily check if the model has been created
        self.R2 = None
        self.MSE = None
    
    @property
    def is_initialized(self): #in case we need to check externally if the model is created
        return self.initialized
    
    @property
    def get_R2(self): #This two are to be called by gui after evaluation, to show results
        if self.initialized:
            return self.R2
        return 9999#arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong
    
    @property
    def get_MSE(self):
        if self.initialized:
            return self.MSE
        return 9999#arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong

    def create_linear_model(self, training_data:pd.DataFrame, training_target:pd.Series):#columns selected in the gui as features and target
        self._data = training_data
        self._target = training_target
        self.fit(self._data, self._target)
        return self
    
    def evaluate_model(self, test_data:DataModule, test_target):
        self.initialized = True
        predictions = self.predict(test_data)
        self.MSE = mean_squared_error(test_target, predictions)
        self.R2 = r2_score(test_target, predictions)
        return self.MSE, self.R2   


if __name__ == "__main__":
    model = LinearRegressionModel()
    example_data = fetch_california_housing(as_frame=True)
    model.create_linear_model(example_data.data, example_data.target)
