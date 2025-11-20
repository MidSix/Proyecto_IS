from data_module import DataModule
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only
from typing import Optional
from matplotlib import pyplot as plt, figure as PltObject
import unittest  #For automatic testing purposes
from sklearn.model_selection import train_test_split #Cool sklearn function to split data, so I don't have to import the data_split module here
from sklearn.datasets import make_regression#This one generates regression datasets for testing, we should aim for deterministic data for unit tests


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
        plt.grid(True, linestyle='--', alpha=0.8)
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
class TestLinearRegressionModel(unittest.TestCase):#Some function names got a bit long, but my naming skills go so far
    def setUp(self):
        X, y = make_regression(n_samples=300, n_features=3, noise=5.0, random_state=0)#As mentioned, deterministic data for unit tests
        self.X = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
        self.y = pd.Series(y, name='target')

        self.X1 = self.X[['f1']]  # single-feature split for plotting tests
        self.X1_train, self.X1_test, self.y1_train, self.y1_test = train_test_split(
            self.X1, self.y, test_size=0.2, random_state=42
        )

        # multi-feature split, why only test single feature when we can test both?
        self.Xm_train, self.Xm_test, self.ym_train, self.ym_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.model = LinearRegressionModel()

    def test_fit_and_evaluate_single_feature(self): #With all the plt stuff
        train_metrics, test_metrics = self.model.fit_and_evaluate(
            self.X1_train, self.y1_train, self.X1_test, self.y1_test
        )
        self.assertIsInstance(train_metrics, dict)#I simply love this function, you'll see it a lot(although it is kinda silly)
        self.assertIsInstance(test_metrics, dict)
        self.assertIsInstance(train_metrics['mse'], float)
        self.assertIsInstance(test_metrics['r2'], float) #would it be fine if we also tested train r2 and test mse?Yes, but this is enough to check both are calculated because if one fails the others will too
        self.assertTrue(self.model.can_plot())

    def test_get_plot_item_returns_figure(self):
        self.model.fit_and_evaluate(self.X1_train, self.y1_train, self.X1_test, self.y1_test)
        fig = self.model.get_plot_item(self.X1_train, self.y1_train, self.X1_test, self.y1_test)
        self.assertIsInstance(fig, PltObject)#We need to get sure the returned object is a plt object, bc the gui will use it directly

    def test_fit_and_evaluate_multi_feature(self): #Now this one is the real stuff
        self.model.fit_and_evaluate(self.Xm_train, self.ym_train, self.Xm_test, self.ym_test)
        self.assertFalse(self.model.can_plot())#Multi-feature models can't plot, so this should be false. Smart test huh?, caffeine running through my veins does wonders(Also raises blood pressure, but whatever)
        metrics = self.model.metrics
        self.assertIn('train', metrics)#We just check that the metrics dictionary has both train and test keys, just to be sure
        self.assertIn('test', metrics)

    def test_predict_consistency_dataframe_vs_ndarray(self):#This one is tricky, it ensures predict works the same with both input types. You never know when the gui will send one or the other(Or at leats I don't know)
        self.model.fit_and_evaluate(self.Xm_train, self.ym_train, self.Xm_test, self.ym_test)
        pred_df = self.model.predict(self.Xm_test)
        pred_nd = self.model.predict(self.Xm_test.to_numpy())
        self.assertEqual(pred_df.shape[0], len(self.Xm_test)) #Ensures output length matches input length, why the f would this happen? I don't know, but the model might get freaky
        self.assertTrue(np.allclose(pred_df, pred_nd, atol=1e-6)) #Ensures both outputs are nearly identical, within a small tolerance in case of floating point differences(It gave me errors before)

    def test_error_on_mismatched_shapes(self):#This one ensures that the model raises an error when input shapes don't match, that is why we have the with self.assertRaises block
        with self.assertRaises(ValueError):
            self.model.fit_and_evaluate(self.X1_train.iloc[:-1], self.y1_train, self.X1_test, self.y1_test)



if __name__ == "__main__":#I always love to put something fancy here, but the DOD had to be boring and ask for CI/CD compatibility(no fun allowed)
    unittest.main()