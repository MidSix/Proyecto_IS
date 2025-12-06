"""Unit tests for linear regression model I/O functionality.

Tests saving and loading of trained linear regression models in .joblib
format, including model metadata and metrics preservation.
"""
import unittest
import tempfile
import os
import joblib

from src.backend.linear_regression_io import (
    load_model_data,
    save_model_data
)

class MockLinearModel:
    """Mock object simulating LinearRegressionModel for I/O testing.

    Provides attributes matching LinearRegressionModel structure
    (regression_line, feature_names, metrics) without dependencies
    on scikit-learn or complex model logic. Enables isolated testing
    of file I/O without involving model creation logic.

    Attributes
    ----------
    regression_line : str
        Regression equation as string.
    feature_names : list
        List of input feature names.
    target_name : str
        Output target column name.
    get_train_R2 : float
        R2 score on training data.
    get_train_MSE : float
        Mean squared error on training data.
    get_test_R2 : float
        R2 score on test data.
    get_test_MSE : float
        Mean squared error on test data.
    """
    def __init__(self) -> None:
        """Initialize mock model with sample data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.regression_line = "y = 2.5 * x + 10"
        self.feature_names = ["Feature_A"]
        self.target_name = "Target_B"
        # Simulamos los getters como valores directos
        self.get_train_R2 = 0.95
        self.get_train_MSE = 0.05
        self.get_test_R2 = 0.92
        self.get_test_MSE = 0.08

# This class serves as a "Mock Object" designed
# to simulate the internal structure and behavior
# of the actual LinearRegressionModel class
# found in 'linear_regression_creation.py'.
# Its sole purpose is to provide the 'save_model_data'
# function with a valid object containing
# the necessary attributes
# (regression_line, feature_names, getters) without relying on the
# real model's complex logic or external dependencies like Scikit-Learn.
# By using this mock, we isolate the
# Input/Output testing environment, ensuring that the tests
# verify only the file handling logic, keeping them
# fast, deterministic, and independent.

class TestLinearRegressionIO(unittest.TestCase):
    """Test suite for model I/O (save/load) functionality.

    Tests saving models to .joblib files, loading them back, handling
    path extensions, error cases, and metadata preservation.

    Methods
    -------
    setUp()
        Initialize mock model and sample data.
    test_save_model_data_success()
        Test successful model saving to .joblib file.
    test_save_adds_extension()
        Test automatic .joblib extension addition.
    test_save_empty_path()
        Test error on empty file path.
    test_load_model_data_success()
        Test successful model loading and summary generation.
    test_load_handles_missing_keys()
        Test robustness with incomplete model data.
    """
    def setUp(self) -> None:
        """Initialize mock model and sample data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.mock_model = MockLinearModel()
        # Estructura de diccionario imitando lo que se guarda en el .joblib
        self.sample_model_data = {
            "formula": "y = 2.5 * x + 10",
            "metrics": {
                "train": {"R2": 0.95, "MSE": 0.05},
                "test": {"R2": 0.92, "MSE": 0.08},
            },
            "description": "Test Description",
        }

    def test_save_model_data_success(self) -> None:
        """Test successful model saving and content verification.

        Saves mock model to temporary .joblib file, verifies file
        exists and contains correct metadata when reloaded.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".joblib") as tmp:
            tmp_path = tmp.name

        try:
            # Ejecutar función de guardado con el Mock
            save_model_data(tmp_path, self.mock_model, "Test Description")

            # Verificar que el archivo existe y tiene contenido
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)

            # Verificar que el contenido es
            # correcto al cargarlo de vuelta
            loaded_data = joblib.load(tmp_path)
            self.assertEqual(loaded_data["formula"],
                             self.mock_model.regression_line)
            self.assertEqual(loaded_data["description"], "Test Description")
            self.assertEqual(loaded_data["metrics"]["train"]["R2"], 0.95)

        finally:
            # Limpieza del archivo temporal
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_adds_extension(self) -> None:
        """Test automatic .joblib extension addition.

        Verifies save_model_data() adds .joblib extension if not
        already present in file path.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Test para ver si añade .joblib si falta
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            base_path = tmp.name  # Sin extensión
            expected_path = base_path + ".joblib"

        try:
            save_model_data(base_path, self.mock_model, "Desc")
            self.assertTrue(os.path.exists(expected_path))
        finally:
            if os.path.exists(base_path):
                try:
                    os.remove(base_path)
                except Exception:
                    pass

            if os.path.exists(expected_path):
                try:
                    os.remove(expected_path)
                except Exception:
                    pass

    def test_save_empty_path(self) -> None:
        """Test error raised on empty file path.

        Verifies ValueError raised when save_model_data() called with
        empty path string.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Debe dar error si la ruta está vacía
        with self.assertRaises(ValueError):
            save_model_data("", self.mock_model, "Desc")

    def test_load_model_data_success(self) -> None:
        """Test successful model loading and summary generation.

        Loads sample model data, generates summary, and verifies all
        expected information present in output.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Test de carga y resumen
        summary, desc = load_model_data(self.sample_model_data)

        self.assertEqual(desc, "Test Description")
        self.assertIsInstance(summary, list)

        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("y = 2.5 * x + 10", full_text)
        self.assertIn("R2: 0.95", full_text)

    def test_load_handles_missing_keys(self) -> None:
        """Test robustness with incomplete model data.

        Verifies load_model_data() handles missing keys gracefully
        by substituting N/A for unavailable metrics.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Test de robustez con datos incompletos
        incomplete_data = {"formula": "y=x"}
        summary, desc = load_model_data(incomplete_data)

        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("MSE: N/A", full_text)


if __name__ == "__main__":
    unittest.main()
