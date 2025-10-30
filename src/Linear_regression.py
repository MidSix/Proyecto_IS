from data_module import DataModule
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing#this dataset is for testing purposes only
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import argparse

#This model is to be highly integrated with the gui.
#I don't think I need to add memorization or saving/loading capabilities here,
#so this will simply be a model object that the gui can interact with to create and evaluate linear regression models.
#As stated, all comments and docstrings will be in English for consistency with the rest of the codebase.
class LinearRegressionModel:
    def __init__(self, train_df = None):
        self.model = LinearRegression()
        self.initialized = False
        self.feature_names = None
        self.metrics_train = {'r2': None, 'mse': None}
        self.metrics_test = {'r2': None, 'mse': None}
        self.coef_ = None
        self.intercept_ = None

    def fit_and_evaluate(self, X_train:pd.DataFrame,
                         y_train:pd.Series, X_test:pd.DataFrame,
                           y_test:pd.Series):
        #In the gui you already can split data, so no need to do it here
        """Trains the model and evaluates both data splits"""
        if not all(X_train.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            raise ValueError("All features must be valid numeric types")
        if X_train.isnull().any() or y_train.isnull().any():
            #We should not get here if gui is used properly
            raise ValueError("Data contains NaN")

        # We save feature names for formula visual representation
        self.feature_names = X_train.columns.tolist()

        # Fit and predictions
        # We only train the model, which is basically find the parameters
        # of our regression line, only with the train set. That's why we only
        # pass to fit the feats and target of the train model.
        self.model.fit(X_train, y_train)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

        # Train metrics and we store them
        y_train_pred = self.model.predict(X_train)
        self.metrics_train['mse'] = mean_squared_error(y_train, y_train_pred)
        self.metrics_train['r2'] = r2_score(y_train, y_train_pred)

        # Test metrics and we store them(DOD requests both)
        y_test_pred = self.model.predict(X_test)
        self.metrics_test['mse'] = mean_squared_error(y_test, y_test_pred)
        self.metrics_test['r2'] = r2_score(y_test, y_test_pred)

        self.initialized = True
        return self.metrics_train, self.metrics_test

    def formula_string(self):  # DOD requests formula representation
        if not self.initialized:
            return "Model not initialized"
        terms = []
        for name, coef in zip(self.feature_names, np.atleast_1d(self.coef_)):
            terms.append(f"{coef:.3f}*{name}")
        # Intercept with explicit sign
        b0 = float(self.intercept_)
        sign = '+' if b0 >= 0 else '-'
        return f"y = {' + '.join(terms)} {sign} {abs(b0):.3f}"

    def can_plot(self):
        """Checks if the model can be plotted (1 feature)"""
        return self.initialized and len(self.feature_names) == 1

    def get_plot_data(self, X_train, X_test):
        """Returns data for plotting if the model has 1 feature"""
        if not self.can_plot():
            raise ValueError("Only models with 1 feature can be plotted")
        x_train_col = np.asarray(X_train.iloc[:, 0], dtype=float)
        x_min, x_max = float(np.min(x_train_col)), float(np.max(x_train_col))
        x_line = np.linspace(x_min, x_max, 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        return {
            'x_train': X_train.iloc[:, 0],
            'x_test': X_test.iloc[:, 0],
            'y_line': y_line,
            'x_line': x_line
        }

    def get_plot_figure(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Figure:
        """Return a Matplotlib Figure for simple linear regression (1 feature).
            This figure is the object that is gonna be showed in the """
        if not self.can_plot():
            return
        # Prepare data
        plot_data = self.get_plot_data(X_train, X_test if X_test is not None else X_train)
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        # Scatter train
        ax.scatter(X_train.iloc[:, 0], y_train, label='Train', alpha=0.75)
        # Optionally scatter test if provided
        if X_test is not None and y_test is not None:
            ax.scatter(X_test.iloc[:, 0], y_test, label='Test', alpha=0.75, marker='x')
        # Regression line
        ax.plot(plot_data['x_line'], plot_data['y_line'], label='Fit', linewidth=2)
        ax.set_xlabel(self.feature_names[0])
        ax.set_ylabel('y')
        ax.set_title('Linear Regression (simple)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        fig.tight_layout()
        return fig

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
        return 9999 #arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong

    @property
    def get_MSE(self):
        if self.initialized:
            return self.metrics_test['mse']#DOD requests test metrics only
        return 9999 #arbitrary high value to indicate uninitialized model, like if u get this you know something's wrong


if __name__ == "__main__":
    model = LinearRegressionModel()
    example_data = fetch_california_housing(as_frame=True)
    model.fit_and_evaluate(example_data.data, example_data.target, example_data.data, example_data.target)