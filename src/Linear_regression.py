from data_module import DataModule
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing  # this dataset is for testing purposes only
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional


# This model is to be highly integrated with the gui.
# I don't think I need to add memorization or saving/loading capabilities here,
# so this will simply be a model object that the gui can interact with to create and evaluate linear regression models.
# As stated, all comments and docstrings will be in English for consistency with the rest of the codebase.
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.initialized = False
        self.feature_names = None
        self.metrics_train = {'r2': None, 'mse': None}
        self.metrics_test = {'r2': None, 'mse': None}
        self.coef_ = None
        self.intercept_ = None
        self.regression_line = None
        self.summary = None

    def set_df(self, train_df=None, test_df=None):
        try:
            if not train_df.empty and not test_df.empty:
                self.x_train = train_df.iloc[:, 1:]
                self.y_train = train_df.iloc[:, 0]
                self.x_test = test_df.iloc[:, 1:]
                self.y_test = test_df.iloc[:, 0]
        except Exception as e:
            print(f"dataframe empty: {e}")

    def fit_and_evaluate(self):
        try:
            # In the gui you already can split data, so no need to do it here
            """Trains the model and evaluates both data splits"""
            # previous handler only works with series, but because we can do
            # multiple regression we can't use it.
            if isinstance(self.x_train, pd.Series):
                self.x_train = self.x_train.to_frame()
                self.x_test = self.x_test.to_frame()
            self.y_train = self.y_train.to_frame()
            self.y_test = self.y_test.to_frame()

            if not all(pd.api.types.is_numeric_dtype(self.x_train[col]) for col in self.x_train.columns):
                raise ValueError("All features must be valid numeric types")
            if not all(pd.api.types.is_numeric_dtype(self.y_train[col]) for col in self.y_train.columns):
                raise ValueError("The target must be valid numeric type")
            if self.x_train.isnull().any().any() or self.y_train.isnull().any().any():
                # We should not get here if gui is used properly
                raise ValueError("Data contains NaN")
        except Exception as e:
            # This will be used to make a Qmessage box in the future.
            return self.metrics_train, self.metrics_test, self.summary, e

        # We save feature names for formula visual representation
        self.feature_names = self.x_train.columns.tolist()

        # Fit and predictions
        self.model.fit(self.x_train, self.y_train)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_

        # Train metrics
        y_train_pred = self.model.predict(self.x_train)
        self.metrics_train['mse'] = mean_squared_error(self.y_train, y_train_pred)
        self.metrics_train['r2'] = r2_score(self.y_train, y_train_pred)

        # Test metrics
        y_test_pred = self.model.predict(self.x_test)
        self.metrics_test['mse'] = mean_squared_error(self.y_test, y_test_pred)
        self.metrics_test['r2'] = r2_score(self.y_test, y_test_pred)

        self.initialized = True
        # returns a tuple with two dictionaries
        self.formula_string()
        self.summary = (f'Regression Line:\n {self.regression_line}\n\n'
                        'Train metrics:\n'
                        f'MSE : {self.metrics_train["mse"]}\n'
                        f'R2  : {self.metrics_train["r2"]}\n\n'
                        'Test metrics\n'
                        f'MSE : {self.metrics_test["mse"]}\n'
                        f'R2  : {self.metrics_test["r2"]}')
        return self.metrics_train, self.metrics_test, self.summary, None

    def formula_string(self) -> str:  # DOD requests formula representation
        if not self.initialized:
            return "Model not initialized"
        terms = []
        # sklearn returns a 2D array with the coefficient per target.
        # We will never have multi-output, so flatten:
        for name, coef in zip(self.feature_names, np.atleast_1d(np.ravel(self.coef_))):
            terms.append(f"{coef:.3f}*{name}")
        # Intercept with explicit sign
        b0 = float(self.intercept_)
        sign = '+' if b0 >= 0 else '-'
        self.regression_line = f"{self.y_train.columns.tolist()[0]} = {' + '.join(terms)} {sign} {abs(b0):.3f}"
        return self.regression_line

    def can_plot(self):
        """Checks if the model can be plotted (1 feature)"""
        return self.initialized and len(self.feature_names) == 1

    def get_plot_item(self):
        # function used for testing purposes only
        """Generates a plot of the model predictions against actual data"""
        if not self.can_plot():
            raise ValueError("Only models with 1 feature can be plotted(two dimensions only)")
        elif not self.initialized:
            raise ValueError("Model not initialized")  # We could also return None or an empty plot, but this informs the user better

        x_line = np.linspace(min(self.x_train.iloc[:, 0]), max(self.x_train.iloc[:, 0]), 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        plt.figure(figsize=(12, 8))
        plt.scatter(self.x_train, self.y_train, color='blue', label='Train Data', alpha=0.5, s=20)  # alpha being transparency, s being point size
        plt.scatter(self.x_test, self.y_test, color='green', label='Test Data', alpha=0.5, s=20)
        plt.plot(x_line, y_line, color='red', label='Prediction Line', linewidth=3)
        plt.xlabel(self.feature_names[0])  # DOD requests feature name in plot
        plt.ylabel("Target")
        plt.title("Linear Regression")
        plt.legend()
        return plt  # Returning the plt object for further manipulation or display

    def get_plot_figure(self) -> Figure:
        """Return a Matplotlib Figure for simple linear regression (1 feature)."""
        if not self.can_plot():
            return None

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        # Line data
        x_line = np.linspace(
            self.x_train.iloc[:, 0].min(),
            self.x_train.iloc[:, 0].max(),
            100
        )
        y_line = self.model.predict(x_line.reshape(-1, 1))

        # Points + line
        ax.scatter(self.x_train.iloc[:, 0], self.y_train, color='blue', label='Train', alpha=0.5)
        ax.scatter(self.x_test.iloc[:, 0], self.y_test, color='green', label='Test', alpha=0.5)
        ax.plot(x_line, y_line, color='red', linewidth=2)

        # Labels and styling
        ax.set_xlabel(f"{self.feature_names[0]}\n(feature)")
        ax.set_ylabel(f"{self.y_train.columns.tolist()[0]}\n(target)")
        ax.set_title("Linear Regression (Simple)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        fig.tight_layout()
        return fig

    def get_y_vs_yhat_figure(self, split: str = "test") -> Figure:
        """
        Returns a Figure with the regression between y (true) and ŷ (predicted).
        Axes: x = y (true), y = ŷ (predicted).
        Draws: scatter, ideal line ŷ = y, and fitted line ŷ = a*y + b.
        Works with single or multiple linear regression (no feature limit).
        """
        if not self.initialized:
            return None
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        # Select split
        X = self.x_train if split == "train" else self.x_test
        y_true = self.y_train if split == "train" else self.y_test

        # Vectors
        y_true_vec = np.ravel(y_true.values if hasattr(y_true, "values") else y_true)
        y_pred_vec = np.ravel(self.model.predict(X))

        # Fit ŷ ≈ a*y + b
        reg = LinearRegression()
        reg.fit(y_true_vec.reshape(-1, 1), y_pred_vec)
        a = float(reg.coef_[0])
        b = float(reg.intercept_)

        # Common range (square plot to visualize ŷ = y at 45°)
        lo = float(min(y_true_vec.min(), y_pred_vec.min()))
        hi = float(max(y_true_vec.max(), y_pred_vec.max()))
        x_line = np.linspace(lo, hi, 200)
        y_line_ideal = x_line
        y_line_fit = a * x_line + b

        # Figure
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(y_true_vec, y_pred_vec, alpha=0.5, label=f"{split.capitalize()} points")
        ax.plot(x_line, y_line_ideal, linewidth=2, label="Ideal  ŷ = y")
        ax.plot(x_line, y_line_fit, linewidth=2, linestyle="--",
                label=f"Fit  ŷ = {a:.3f}·y {'+' if b >= 0 else '-'} {abs(b):.3f}")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{y_true.columns.tolist()[0] if hasattr(y_true, 'columns') else 'y'} (true)")
        ax.set_ylabel("ŷ (predicted)")
        ax.set_title(f"Parity plot y vs ŷ ({split})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig

    def predict(self, X):
        """Makes predictions on new data"""
        if not self.initialized:
            raise ValueError("Model not initialized")
        return self.model.predict(X)

    @property
    def is_initialized(self):  # in case we need to check externally if the model is created
        return self.initialized

    @property
    def get_R2(self):  # This two are to be called by gui after evaluation, to show results
        if self.initialized:
            return self.metrics_test['r2']  # DOD requests test metrics only
        return "NO DATA"  # If not initialized

    @property
    def get_MSE(self):
        """Returns MSE on test data or high value if uninitialized"""
        if self.initialized:
            return self.metrics_test['mse']  # DOD requests test metrics only
        return "NO DATA"  # If not initialized

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
    Ans = input("Run a test example of LinearRegressionModel? (y/n): ")
    if Ans.lower() == "y":
        from sklearn.model_selection import train_test_split

        model = LinearRegressionModel()
        example_data = fetch_california_housing(as_frame=True)
        columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                   'Population', 'AveOccup', 'Latitude', 'Longitude']
        feature = input(f"Select feature from {columns}(0-{len(columns) - 1}): ")
        while not feature in [str(i) for i in range(len(columns))]:
            feature = input(f"Invalid selection. Select feature from {columns}(0-{len(columns) - 1}): ")
        X = example_data.data[[columns[int(feature)]]]  # Using only one feature
        y = example_data.target

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.x_train, model.x_test, model.y_train, model.y_test = X_train, X_test, y_train, y_test
        model.fit_and_evaluate()
        print("Formula:", model.formula_string())
        print("Metrics:", model.metrics)

        # Original plot (simple regression only)
        if model.can_plot():
            plt = model.get_plot_item()
            plt.show()

        # NEW: parity plot (works for simple or multiple)
        fig = model.get_y_vs_yhat_figure(split="test")
        if fig is not None:
            fig.show()
    else:
        print("Ok, no test for you then.")
