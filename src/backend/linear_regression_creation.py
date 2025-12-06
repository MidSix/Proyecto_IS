import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
 # this dataset is for testing purposes only
from matplotlib.figure import Figure


# This model is highly integrated with the GUI. It is an in-memory
# object that the GUI uses to create and evaluate linear regression
# models. Comments and docstrings are in English for consistency
# with the rest of the codebase.
class LinearRegressionModel:
    """Encapsulates a scikit-learn linear regression model with metrics.

    This class manages the creation, training, and evaluation of linear
    regression models. It stores training and test data, computes
    metrics, generates formula representations, and can plot results.

    Attributes
    ----------
    model : LinearRegression
        Scikit-learn LinearRegression instance.
    initialized : bool
        True if the model has been fitted.
    feature_names : list or None
        Names of features from training data.
    metrics_train : dict
        Dictionary with keys 'r2' and 'mse' for training metrics.
    metrics_test : dict
        Dictionary with keys 'r2' and 'mse' for test metrics.
    coef : array or None
        Fitted coefficients from the model.
    intercept : float or None
        Fitted intercept from the model.
    regression_line : str or None
        Formatted string representation of the regression formula.
    summary : str or None
        Formatted summary of metrics and formula.

    Methods
    -------
    set_df(train_df, test_df)
        Assign training and test DataFrames.
    fit_and_evaluate()
        Train model and compute metrics.
    formula_string()
        Generate regression formula as a formatted string.
    can_plot()
        Check if simple regression plot is possible.
    get_plot_figure()
        Return a matplotlib Figure for simple regression.
    get_y_vs_yhat_figure(split)
        Return a parity plot (y vs ŷ) for any regression.
    predict(X)
        Make predictions on new data.
    """
    def __init__(self) -> None:
        """Initialize the LinearRegressionModel."""
        self.model = LinearRegression()
        self.initialized = False
        self.feature_names = None
        self.metrics_train = {'r2': None, 'mse': None}
        self.metrics_test = {'r2': None, 'mse': None}
        self.coef_ = None
        self.intercept_ = None
        self.regression_line = None
        self.summary = None

    def set_df(self, train_df: pd.DataFrame = None,
               test_df: pd.DataFrame = None) -> None:
        """Assign training and test DataFrames to the model.

        Parameters
        ----------
        train_df : pd.DataFrame, optional
            Training data with first column as target, rest as features.
            Default is None.
        test_df : pd.DataFrame, optional
            Test data with same structure as train_df. Default is None.

        Returns
        -------
        None
        """
        # The GUI already performs the data split, so this method only
        # assigns training and test splits.
        try:
            if not train_df.empty and not test_df.empty:
                self.x_train = train_df.iloc[:, 1:]
                self.y_train = train_df.iloc[:, 0]
                self.x_test = test_df.iloc[:, 1:]
                self.y_test = test_df.iloc[:, 0]
        except Exception as e:
            print(f"dataframe empty: {e}")

    def fit_and_evaluate(self):
        """Train the model and compute metrics on both splits.

        This method validates input data, fits the model on training
        data, and computes Mean Squared Error (MSE) and R² score on
        both training and test sets. Returns metrics and a formatted
        summary string.

        Returns
        -------
        tuple
            A 4-tuple: (metrics_train, metrics_test, summary, error).
            If an error occurs, the error is non-None and metrics/summary
            may be incomplete.
        """
        try:
            # The previous handler only worked with Series,
            # but for multiple regression we must handle DataFrames,
            # so we convert Series to DataFrames when needed.
            """Trains the model and evaluates both data splits"""
            # previous handler only works with series,
            # but because we can do
            # multiple regression we can't use it.
            if isinstance(self.x_train, pd.Series):
                self.x_train = self.x_train.to_frame()
                self.x_test = self.x_test.to_frame()
            self.y_train = self.y_train.to_frame()
            self.y_test = self.y_test.to_frame()

            if not all(
                pd.api.types.is_numeric_dtype(self.x_train[col])
                for col in self.x_train.columns
            ):
                raise ValueError("All features must be valid numeric types")
            if not all(
                pd.api.types.is_numeric_dtype(self.y_train[col])
                for col in self.y_train.columns
            ):
                raise ValueError("The target must be valid numeric type")
            if (
                self.x_train.isnull().any().any()
                or self.y_train.isnull().any().any()
            ):
                # We should not get here if GUI is used properly
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
        self.metrics_train['mse'] = mean_squared_error(
            self.y_train,
            y_train_pred
            )
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

    # DOD requests formula representation
    def formula_string(self) -> str:
        """Generate a formatted string of the regression formula.

        Produces a multi-line representation of the fitted regression
        equation in the form: target = c1*feat1 + c2*feat2 + ... + b0

        Returns
        -------
        str
            Multi-line formatted regression formula, or
            'Model not initialized' if the model has not been fitted.
        """
        if not self.initialized:
            return "Model not initialized"

        # Separamos signo y magnitud de cada coeficiente
        coefs = np.atleast_1d(np.ravel(self.coef_))
        signs = ['+' if c >= 0 else '-' for c in coefs]
        terms = [f"{abs(c):.3f}*{name}"
                 for c, name in zip(coefs, self.feature_names)]

        lines = []
        n = len(terms)

        # Recorremos en bloques de 2 términos
        for idx, start in enumerate(range(0, n, 2)):
            group = range(start, min(start + 2, n))

            if idx == 0:
                # Primera línea: target = ...
                line = f"{self.target_name} = "
            else:
                # Líneas siguientes: indentadas
                line = " " * 10

            for j in group:
                if idx == 0 and j == 0:
                    # Primer término de todos: sin '+' si es positivo
                    if signs[j] == '+':
                        line += terms[j]
                    else:
                        line += f"- {terms[j]}"
                else:
                    line += f" {signs[j]} {terms[j]}"

            lines.append(line)

        # Intercepto con signo al final
        b0 = float(np.ravel(self.intercept_)[0])
        sign0 = '+' if b0 >= 0 else '-'
        if lines:
            lines[-1] = f"{lines[-1]} {sign0} {abs(b0):.3f}"
        else:
            lines = [f"{self.target_name} = {sign0} {abs(b0):.3f}"]

        self.regression_line = "\n".join(lines)
        return self.regression_line

    def can_plot(self) -> bool:
        """Check if the model can be plotted (requires 1 feature).

        Returns
        -------
        bool
            True if model is initialized with exactly one feature.
        """
        return self.initialized and len(self.feature_names) == 1

    def get_plot_figure(self) -> Figure:
        """Return a matplotlib Figure for simple linear regression.

        Plots training and test points, and the fitted regression line.
        Only works when model has exactly one feature.

        Returns
        -------
        Figure or None
            A matplotlib Figure with the plot, or None if plot is not
            possible (not initialized or multiple features).
        """
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
        ax.scatter(
            self.x_train.iloc[:, 0],
            self.y_train, color='blue',
            label='Train', alpha=0.5
        )
        ax.scatter(
            self.x_test.iloc[:, 0],
            self.y_test, color='green',
            label='Test', alpha=0.5
            )

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
        """Generate a parity plot (y_true vs y_predicted).

        Creates a scatter plot of true vs predicted values with an
        ideal fit line (ŷ = y) for visual assessment of model quality.
        Works for any regression (simple or multiple).

        Parameters
        ----------
        split : str, optional
            Which data split to plot. Must be 'train' or 'test'.
            Default is 'test'.

        Returns
        -------
        Figure or None
            A matplotlib Figure with the parity plot, or None if model
            is not initialized.

        Raises
        ------
        ValueError
            If split is not 'train' or 'test'.
        """
        if not self.initialized:
            return None
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        # Select split
        X = self.x_train if split == "train" else self.x_test
        y_true = self.y_train if split == "train" else self.y_test

        # Vectors
        y_true_vec = np.ravel(
            y_true.values if hasattr(y_true, "values") else y_true
            )
        y_pred_vec = np.ravel(self.model.predict(X))

        # Common range (square plot to visualize ŷ = y at 45°)
        lo = float(min(y_true_vec.min(), y_pred_vec.min()))
        hi = float(max(y_true_vec.max(), y_pred_vec.max()))
        x_line = np.linspace(lo, hi, 200)
        y_line_ideal = x_line

        # Figure
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.scatter(
            y_true_vec,
            y_pred_vec,
            alpha=0.5,
            label=f"{split.capitalize()} points"
            )
        ax.plot(
            x_line,
            y_line_ideal,
            linewidth=2,
            label="Ideal  ŷ = y",
            color="red"
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        target_name = y_true.columns.tolist()[0]
        ax.set_xlabel(
            f"{target_name if hasattr(y_true, 'columns') else 'y'} (true)"
            )
        ax.set_ylabel("ŷ n(predicted)")
        ax.set_title(f"Parity plot y vs ŷ ({split})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data for prediction.

        Returns
        -------
        np.ndarray
            Predicted target values.

        Raises
        ------
        ValueError
            If the model has not been initialized (fitted).
        """
        if not self.initialized:
            raise ValueError("Model not initialized")
        return self.model.predict(X)

    @property
    def is_initialized(self) -> bool:
        """Check if the model has been fitted.

        Returns
        -------
        bool
            True if model has been trained via fit_and_evaluate().
        """
        # in case we need to check externally if the model is created
        return self.initialized

    @property
    def get_test_R2(self) -> float:
        """Return R² score on test data.

        Returns
        -------
        float or None
            R² score if model is initialized, None otherwise.
        """
        if self.initialized:
            # DOD requests test metrics only
            return self.metrics_test['r2']
        return None  # If not initialized

    @property
    def get_train_R2(self) -> float:
        """Return R² score on training data.

        Returns
        -------
        float or None
            R² score if model is initialized, None otherwise.
        """
        if self.initialized:
            # DOD requests test metrics only
            return self.metrics_train['r2']
        return None  # If not initialized

    @property
    def get_test_MSE(self) -> float:
        """Return Mean Squared Error on test data.

        Returns
        -------
        float or None
            MSE if model is initialized, None otherwise.
        """
        if self.initialized:
            # DOD requests test metrics only
            return self.metrics_test['mse']
        return None  # If not initialized

    @property
    def get_train_MSE(self) -> float:
        """Return Mean Squared Error on training data.

        Returns
        -------
        float or None
            MSE if model is initialized, None otherwise.
        """
        if self.initialized:
            # DOD requests test metrics only
            return self.metrics_train['mse']
        return None  # If not initialized

    @property
    def metrics(self) -> dict:
        """Return all metrics from both training and test splits.

        Returns
        -------
        dict or None
            Dictionary with keys 'train' and 'test', each containing
            'r2' and 'mse' keys, or None if model not initialized.
        """
        if not self.initialized:
            return None
        return {
            'train': self.metrics_train,
            'test': self.metrics_test
        }