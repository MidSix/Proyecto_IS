import unittest
import tempfile
import os
import joblib

from src.backend.linear_regression_io import (
    load_model_data,
    save_model_data
)

class MockLinearModel:
    """
    A dummy class (Mock) to simulate the behavior of LinearRegressionModel.
    Required for the save test to function independently of the creation file.
    """
    def __init__(self):
        self.regression_line = "y = 2.5 * x + 10"
        self.feature_names = ["Feature_A"]
        self.target_name = "Target_B"
        # We simulate tne getters as direct values
        self.get_train_R2 = 0.95
        self.get_train_MSE = 0.05
        self.get_test_R2 = 0.92
        self.get_test_MSE = 0.08
# This class serves as a "Mock Object" designed to simulate the internal structure and behavior
# of the actual LinearRegressionModel class found in 'linear_regression_creation.py'.
# Its sole purpose is to provide the 'save_model_data' function with a valid object containing
# the necessary attributes (regression_line, feature_names, getters) without relying on the
# real model's complex logic or external dependencies like Scikit-Learn.
# By using this mock, we isolate the Input/Output testing environment, ensuring that the tests
# verify only the file handling logic, keeping them fast, deterministic, and independent.

class TestLinearRegressionIO(unittest.TestCase):
    def setUp(self):
        self.mock_model = MockLinearModel()
        # Dictionary structure mimicking what is stored in the .joblib
        self.sample_model_data = {
            "formula": "y = 2.5 * x + 10",
            "input_columns": ["Feature_A"],
            "output_column": "Target_B",
            "metrics": {
                "train": {"R2": 0.95, "MSE": 0.05},
                "test": {"R2": 0.92, "MSE": 0.08},
            },
            "description": "Test Description",
        }

    def test_save_model_data_success(self):
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp_path = tmp.name

        try:
            # Run save function with the Mock
            save_model_data(tmp_path, self.mock_model, "Test Description")

            # Verify that the file exists and has any content
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)

            # Verify that the content is correct when uploading it back
            loaded_data = joblib.load(tmp_path)
            self.assertEqual(loaded_data["formula"],
                             self.mock_model.regression_line)
            self.assertEqual(loaded_data["description"], "Test Description")
            self.assertEqual(loaded_data["metrics"]["train"]["R2"], 0.95)

        finally:
            # Temp file cleansing
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_adds_extension(self):
        # Testing to see if it puts .joblib if it's missing
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            base_path = tmp.name  # No extension
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

    def test_save_empty_path(self):
        # Should give an error if path is empty
        with self.assertRaises(ValueError):
            save_model_data("", self.mock_model, "Desc")

    def test_load_model_data_success(self):
        # Load and summary test
        summary, desc = load_model_data(self.sample_model_data)

        self.assertEqual(desc, "Test Description")
        self.assertIsInstance(summary, list)

        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("y = 2.5 * x + 10", full_text)
        self.assertIn("R2: 0.95", full_text)

    def test_load_handles_missing_keys(self):
        # Robustness test with missing data
        incomplete_data = {"formula": "y=x"}
        summary, desc = load_model_data(incomplete_data)

        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("MSE: N/A", full_text)


if __name__ == "__main__":
    unittest.main()
