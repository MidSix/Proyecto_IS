import pandas as pd
import numpy as np
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 # this dataset is for testing purposes only
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional


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
        terms = []
        # sklearn returns a 2D array with the coefficient per target.
        # We will never have multi-output, so flatten:
        for name, coef in zip(self.feature_names, np.atleast_1d(np.ravel(self.coef_))):
            terms.append(f"{coef:.3f}*{name}")
        
        # --- Changed here: We access index 0 to extract the scalar ---
        b0 = float(self.intercept_[0]) 
        # --------------------------------------------------------------------
        
        sign = '+' if b0 >= 0 else '-'
        self.regression_line = f"{self.target_name} = {' + '.join(terms)} {sign} {abs(b0):.3f}"
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
    def is_initialized(self):  # in case we need to check externally if the model is created
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


# ------ Added Unit Tests below ------

# ------ Added Unit Tests below (Following the style of data_loader.py) ------

class TestLinearRegressionModel(unittest.TestCase):
    def setUp(self):
        # Create a deterministic dataset: y = 2*x + 1
        data_train = {
            "target":   [3, 5, 7, 9],    # COLUMNA 0 (Target)
            "feature1": [1, 2, 3, 4]     # COLUMNA 1 (Feature)
        }
        data_test = {
            "target":   [11, 13],
            "feature1": [5, 6]
        }
        # IMPORTANTE: Forzamos el orden de columnas para que coincida con set_df
        # set_df asume que la columna 0 es el target y las siguientes son features.
        self.df_train = pd.DataFrame(data_train)[["target", "feature1"]]
        self.df_test = pd.DataFrame(data_test)[["target", "feature1"]]
        
        self.model = LinearRegressionModel()

    def test_initialization_state(self):
        # Verify initial state is clean
        self.assertFalse(self.model.initialized)
        self.assertIsNone(self.model.feature_names)
        self.assertIsNone(self.model.get_train_R2)

    def test_set_df_success(self):
        self.model.set_df(self.df_train, self.df_test)
        # Check if data was split into X and y internally
        self.assertEqual(self.model.x_train.shape, (4, 1)) # 4 rows, 1 col
        self.assertEqual(self.model.y_train.shape, (4,))   # 4 rows

    def test_fit_and_evaluate_success(self):
        # Full flow test
        self.model.set_df(self.df_train, self.df_test)
        
        # fit_and_evaluate returns (metrics_train, metrics_test, summary, error)
        m_train, m_test, summary, error = self.model.fit_and_evaluate()
        
        # 1. Check Error is None
        self.assertIsNone(error)
        
        # 2. Check Initialization
        self.assertTrue(self.model.initialized)
        
        # 3. Check Maths (y = 2x + 1)
        # Slope (coef) should be 2. 
        # CORRECCIÓN: Accedemos a [0][0] porque al ser DataFrame, coef_ es [[2.0]]
        self.assertAlmostEqual(self.model.coef_[0][0], 2.0, places=5)
        # Intercept should be 1
        self.assertAlmostEqual(self.model.intercept_[0], 1.0, places=5)
        # R2 should be 1.0 (perfect fit)
        self.assertAlmostEqual(m_train['r2'], 1.0, places=5)

    def test_fit_invalid_data_nan(self):
        # Create dirty data
        df_dirty = self.df_train.copy()
        df_dirty.iloc[0, 0] = np.nan # Introduce NaN
        
        self.model.set_df(df_dirty, self.df_test)
        _, _, _, error = self.model.fit_and_evaluate()
        
        # Must return an error (Exception object)
        self.assertIsNotNone(error)
        self.assertIn("NaN", str(error))

    def test_formula_string(self):
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        
        formula = self.model.formula_string()
        # Should look like "target = 2.000*feature1 + 1.000"
        self.assertIn("target", formula)
        self.assertIn("feature1", formula)
        self.assertIn("2.000", formula)

    def test_predict_success(self):
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        
        # Predict for input 10. Expected: 2*10 + 1 = 21
        # El nombre de la columna debe coincidir con el usado en fit (feature1)
        input_data = pd.DataFrame({"feature1": [10]})
        prediction = self.model.predict(input_data)
        self.assertAlmostEqual(prediction[0][0], 21.0, places=5)

    def test_predict_uninitialized_error(self):
        # Trying to predict before fitting
        with self.assertRaises(ValueError):
            self.model.predict(pd.DataFrame({"a":[1]}))

    def test_can_plot_logic(self):
        # Case 1: 1 feature -> Can plot
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        self.assertTrue(self.model.can_plot())
        
        # Case 2: 2 features -> Cannot plot (in 2D simple view)
        # Create df with Target + 2 features
        df_multi = pd.DataFrame({
            "target": [1,2,3],
            "f1": [1,2,3], 
            "f2": [1,2,3]
        })
        self.model.set_df(df_multi, df_multi)
        self.model.fit_and_evaluate()
        self.assertFalse(self.model.can_plot())

    def test_get_plot_figure_returns_object(self):
        # Ensure it returns a Figure object and doesn't crash
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        fig = self.model.get_plot_figure()
        self.assertIsInstance(fig, Figure)


# The next condition checks if the script is running directly (standalone) and not as an
# imported module. If so, it triggers unittest.main(), which is responsible for:
# 1. Scanning the entire file for classes inheriting from unittest.TestCase.
# 2. Automatically discovering all methods starting with the "test_" prefix.
# 3. Running those tests and reporting success/failure results to the console.
# This block is fundamental for GitHub Actions (CI/CD) to validate the code automatically,
# ensuring that future changes do not break the model creation logic.

if __name__ == "__main__":
    unittest.main()
