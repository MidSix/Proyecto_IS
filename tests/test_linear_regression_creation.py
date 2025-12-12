"""Unit tests for LinearRegressionModel backend component.

Tests model initialization, data setting, fitting, evaluation, prediction,
and visualization capabilities. Verifies error handling for invalid data
and uninitialized models.
"""
import unittest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from src.backend.linear_regression_creation import LinearRegressionModel

class TestLinearRegressionModel(unittest.TestCase):
    """Test suite for LinearRegressionModel class.

    Tests model lifecycle from initialization through fitting, evaluation,
    prediction, and visualization across simple and multiple regression.

    Methods
    -------
    setUp()
        Create deterministic training and test data.
    test_initialization_state()
        Verify model starts in uninitialized state.
    test_set_df_success()
        Test successful DataFrame assignment.
    test_fit_and_evaluate_success()
        Test complete fit workflow and coefficient validation.
    test_fit_invalid_data_nan()
        Test error handling for data with NaN values.
    test_formula_string()
        Test regression formula generation.
    test_predict_success()
        Test prediction on new data.
    test_predict_uninitialized_error()
        Test error when predicting before fitting.
    test_can_plot_logic()
        Test plotting capability detection.
    test_get_plot_figure_returns_object()
        Test plot figure generation.
    """
    def setUp(self) -> None:
        """Create deterministic training and test data.

        Generates linear data (y = 2*x + 1) with known coefficients
        for validation.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create a deterministic dataset: y = 2*x + 1
        # This ensures R2 is 1.0 and we know the coefficients exactly.
        data_train = {
            "target":   [3, 5, 7, 9],
            "feature1": [1, 2, 3, 4]
        }
        data_test = {
            "target":   [11, 13],
            "feature1": [5, 6]
        }
        self.df_train = pd.DataFrame(data_train)
        self.df_test = pd.DataFrame(data_test)

        self.model = LinearRegressionModel()

    def test_initialization_state(self) -> None:
        """Verify model starts in uninitialized state.

        Checks that new model instance has no coefficients, feature
        names, or metrics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Verify initial state is clean
        self.assertFalse(self.model.initialized)
        self.assertIsNone(self.model.feature_names)
        self.assertIsNone(self.model.get_train_R2)

    def test_set_df_success(self) -> None:
        """Test successful DataFrame assignment and X/y splitting.

        Verifies set_df() correctly separates target and features.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model.set_df(self.df_train, self.df_test)
        # Check if data was split into X and y internally
        self.assertEqual(self.model.x_train.shape, (4, 1)) # 4 rows,
                                                                # 1 col
        # 4 rows (Series) or (4,1) depending on pandas version/handling
        self.assertEqual(self.model.y_train.shape[0], 4)

    def test_fit_and_evaluate_success(self) -> None:
        """Test complete fit workflow and coefficient validation.

        Verifies model fits correctly, no errors returned, coefficients
        match expected values, and R2 score is perfect (1.0).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Full flow test
        self.model.set_df(self.df_train, self.df_test)

        # fit_and_evaluate returns
        # (metrics_train, metrics_test, summary, error)
        m_train, m_test, summary, error = self.model.fit_and_evaluate()

        # 1. Check Error is None
        self.assertIsNone(error)

        # 2. Check Initialization
        self.assertTrue(self.model.initialized)

        # 3. Check Maths (y = 2x + 1)
        # Slope (coef) should be 2
        self.assertAlmostEqual(self.model.coef_[0][0], 2.0, places=5)
        # Intercept should be 1
        self.assertAlmostEqual(float(np.ravel(self.model.intercept_)[0]),
                               1.0, places=5)
        # R2 should be 1.0 (perfect fit)
        self.assertAlmostEqual(m_train['r2'], 1.0, places=5)

    def test_fit_invalid_data_nan(self) -> None:
        """Test error handling for data with NaN values.

        Verifies fit_and_evaluate() returns error when data contains
        NaN values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create dirty data
        df_dirty = self.df_train.copy()
        df_dirty.iloc[0, 0] = np.nan  # Introduce NaN in target

        self.model.set_df(df_dirty, self.df_test)
        _, _, _, error = self.model.fit_and_evaluate()

        # Must return an error (Exception object)
        self.assertIsNotNone(error)
        self.assertIn("NaN", str(error))

    def test_formula_string(self) -> None:
        """Test regression formula generation.

        Verifies formula_string() produces readable formula with correct
        target, feature names, and coefficient values.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()

        formula = self.model.formula_string()
        # Should look like "target = 2.000*feature1 + 1.000"
        self.assertIn("target", formula)
        self.assertIn("feature1", formula)
        self.assertIn("2.000", formula)

    def test_predict_success(self) -> None:
        """Test prediction on new data.

        Verifies predict() returns correct value for known input.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()

        # Predict for input 10. Expected: 2*10 + 1 = 21
        input_data = pd.DataFrame({"feature1": [10]})
        prediction = self.model.predict(input_data)
        self.assertAlmostEqual(prediction[0][0], 21.0, places=5)

    def test_predict_uninitialized_error(self) -> None:
        """Test error when predicting before fitting.

        Verifies ValueError raised when predict() called on uninitialized
        model.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Trying to predict before fitting
        with self.assertRaises(ValueError):
            self.model.predict(pd.DataFrame({"a": [1]}))

    def test_can_plot_logic(self) -> None:
        """Test plotting capability detection.

        Verifies can_plot() returns True for single feature and False
        for multiple features.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Case 1: 1 feature -> Can plot
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        self.assertTrue(self.model.can_plot())

        # Case 2: 2 features -> Cannot plot (in 2D simple view)
        df_multi = pd.DataFrame(
            {"target": [1, 2, 3],
             "f1": [1, 2, 3],
             "f2": [1, 2, 3]}
        )
        self.model.set_df(df_multi, df_multi)
        self.model.fit_and_evaluate()
        self.assertFalse(self.model.can_plot())

    def test_get_plot_figure_returns_object(self) -> None:
        """Test plot figure generation returns matplotlib Figure.

        Verifies get_plot_figure() returns valid Figure object.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Ensure it returns a Figure object and doesn't crash
        self.model.set_df(self.df_train, self.df_test)
        self.model.fit_and_evaluate()
        fig = self.model.get_plot_figure()
        self.assertIsInstance(fig, Figure)

# The next condition checks if the script is
# running directly (standalone) and not as an imported module.
# If so, it triggers unittest.main(), which is responsible for:
# 1. Scanning the entire file for classes
# inheriting from unittest.TestCase.
# 2. Automatically discovering all methods
# starting with the "test_" prefix.
# 3. Running those tests and reporting
# success/failure results to the console. This block is fundamental
# for GitHub Actions (CI/CD) to validate the code automatically,
# ensuring that future changes do not break the model creation logic.
if __name__ == "__main__":
    unittest.main()