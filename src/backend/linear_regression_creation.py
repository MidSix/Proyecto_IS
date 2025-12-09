import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 # This dataset is for testing purposes only
from matplotlib.figure import Figure


# This model is to be highly integrated with the gui.
# I don't think I need to add memorization or
# saving/loading capabilities here, so this will simply be a model
# object that the gui can interact with to create and
# evaluate linear regression models. As stated, all comments and
# docstrings will be in English for consistency with the rest of the
# codebase.
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
            # In the gui you already can split data,
            # so no need to do it here
            """Trains the model and evaluates both data splits"""
            # previous handler only works with series,
            # but because we can do
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
        self.target_name = self.y_train.columns.tolist()[0]
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
        self.summary = (f'Regression Line:\n{self.regression_line}\n\n'
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

        # We separate the sign and magnitude of each coefficient
        coefs = np.atleast_1d(np.ravel(self.coef_))
        signs = ['+' if c >= 0 else '-' for c in coefs]
        terms = [f"{abs(c):.3f}*{name}"
                 for c, name in zip(coefs, self.feature_names)]

        lines = []
        n = len(terms)

        # We go through in blocks of 2 terms
        for idx, start in enumerate(range(0, n, 2)):
            group = range(start, min(start + 2, n))

            if idx == 0:
                # First line: target = ...
                line = f"{self.target_name} = "
            else:
                # Following lines: indented
                line = " " * 10

            for j in group:
                if idx == 0 and j == 0:
                    # First term of them all: without '+' if it's positive
                    if signs[j] == '+':
                        line += terms[j]
                    else:
                        line += f"- {terms[j]}"
                else:
                    line += f" {signs[j]} {terms[j]}"

            lines.append(line)

        # We intercept with sign at the end
        b0 = float(np.ravel(self.intercept_)[0])
        sign0 = '+' if b0 >= 0 else '-'
        if lines:
            lines[-1] = f"{lines[-1]} {sign0} {abs(b0):.3f}"
        else:
            lines = [f"{self.target_name} = {sign0} {abs(b0):.3f}"]

        self.regression_line = "\n".join(lines)
        return self.regression_line

    def can_plot(self):
        """Checks if the model can be plotted (1 feature)"""
        return self.initialized and len(self.feature_names) == 1

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
        x_line = pd.DataFrame({self.feature_names[0]: x_line})
        y_line = self.model.predict(x_line)

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

        # Common range (square plot to visualize ŷ = y at 45°)
        lo = float(min(y_true_vec.min(), y_pred_vec.min()))
        hi = float(max(y_true_vec.max(), y_pred_vec.max()))
        x_line = np.linspace(lo, hi, 200)
        y_line_ideal = x_line

        # Figure
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(y_true_vec, y_pred_vec, alpha=0.5, label=f"{split.capitalize()} points")
        ax.plot(x_line, y_line_ideal, linewidth=2, label="Ideal  ŷ = y", color="red")

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{y_true.columns.tolist()[0] if hasattr(y_true, 'columns') else 'y'} (true)")
        ax.set_ylabel("ŷ n(predicted)")
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
    def is_initialized(self):  # Just in case we need to check externally if the model is created
        return self.initialized

    @property
    def get_test_R2(self):
        if self.initialized:
            return self.metrics_test['r2']  # DOD requests test metrics only
        return None  # If not initialized

    @property
    def get_train_R2(self):
        if self.initialized:
            return self.metrics_train['r2']  # DOD requests test metrics only
        return None  # If not initialized

    @property
    def get_test_MSE(self):
        """Returns MSE on test data or high value if uninitialized"""
        if self.initialized:
            return self.metrics_test['mse']  # DOD requests test metrics only
        return None  # If not initialized

    @property
    def get_train_MSE(self):
        """Returns MSE on test data or high value if uninitialized"""
        if self.initialized:
            return self.metrics_train['mse']  # DOD requests test metrics only
        return None  # If not initialized

    @property
    def metrics(self):
        """Returns all metrics in a clear format"""
        if not self.initialized:
            return None
        return {
            'train': self.metrics_train,
            'test': self.metrics_test
        }