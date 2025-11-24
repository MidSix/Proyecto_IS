import joblib
import unittest 
import tempfile 
import os
# Don't need to use a class. We don't need to store states.
# I mean, you could store states but they are meaningless here.
# A class is only useful when you need to store meaningful
# states(attributes) and use methods associated to those states.
# We don't need that here.
def load_model_data(model_data: dict):
    """
    load model data from a dictionary and
    returns a summary and description.
    """
    try:

        formula = model_data.get("formula", "")
        input_cols = model_data.get("input_columns", [])
        output_col = model_data.get("output_column", "")
        metrics = model_data.get("metrics", {})
        description = model_data.get("description", "")

        # Construir texto de resumen atractivo
        train_metrics = metrics.get("train", {})
        test_metrics = metrics.get("test", {})

        summary_lines = [
            f"Regression Line:",
            "",
            f"{formula}",
            "",
            f"Inputs: {', '.join(input_cols)}",
            f"Output: {output_col}",
            "",
            "Train metrics:",
            f"MSE: {train_metrics.get('MSE', 'N/A')}",
            f"R2: {train_metrics.get('R2', 'N/A')}",
            "",
            "Test metrics:",
            f"MSE: {test_metrics.get('MSE', 'N/A')}",
            f"R2: {test_metrics.get('R2', 'N/A')}"
        ]

        # f'Regression Line:\n{self.regression_line}\n\n'
        # 'Train metrics:\n'
        # f'MSE : {self.metrics_train["mse"]}\n'
        # f'R2  : {self.metrics_train["r2"]}\n\n'
        # 'Test metrics\n'
        # f'MSE : {self.metrics_test["mse"]}\n'
        # f'R2  : {self.metrics_test["r2"]}'
        return summary_lines, description
    except Exception as e:
        raise e

def save_model_data(file_path: str, model: dict, model_description: str):
    try:
        if not file_path:
            raise ValueError("File path is empty.")
        if not file_path.endswith(".joblib"):
                file_path += ".joblib"

        # Structure to be saved
        model_data = {
                "formula": model.regression_line,
                "input_columns": model.feature_names,
                "output_column": model.target_name,
                "metrics": {
                    "train": {"R2": model.get_train_R2,
                              "MSE": model.get_train_MSE},
                    "test": {"R2": model.get_test_R2,
                             "MSE": model.get_test_MSE},
                },
                "description": model_description
            }

        joblib.dump(model_data, file_path)
    except Exception as e:
            raise e

class MockLinearModel:
    """
    Clase falsa (Mock) para simular el comportamiento de LinearRegressionModel.
    Necesaria para que el test de guardado funcione sin depender del archivo de creación.
    """
    def __init__(self):
        self.regression_line = "y = 2.5 * x + 10"
        self.feature_names = ["Feature_A"]
        self.target_name = "Target_B"
        # Simulamos los getters como valores directos
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
        # Estructura de diccionario imitando lo que se guarda en el .joblib
        self.sample_model_data = {
            "formula": "y = 2.5 * x + 10",
            "input_columns": ["Feature_A"],
            "output_column": "Target_B",
            "metrics": {
                "train": {"R2": 0.95, "MSE": 0.05},
                "test": {"R2": 0.92, "MSE": 0.08},
            },
            "description": "Test Description"
        }



    def test_save_model_data_success(self):
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
            tmp_path = tmp.name
        
        try:
            # Ejecutar función de guardado con el Mock
            save_model_data(tmp_path, self.mock_model, "Test Description")
            
            # Verificar que el archivo existe y tiene contenido
            self.assertTrue(os.path.exists(tmp_path))
            self.assertTrue(os.path.getsize(tmp_path) > 0)
            
            # Verificar que el contenido es correcto al cargarlo de vuelta
            loaded_data = joblib.load(tmp_path)
            self.assertEqual(loaded_data["formula"], self.mock_model.regression_line)
            self.assertEqual(loaded_data["description"], "Test Description")
            self.assertEqual(loaded_data["metrics"]["train"]["R2"], 0.95)
            
        finally:
            # Limpieza del archivo temporal
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    


    def test_save_adds_extension(self):
        # Test para ver si añade .joblib si falta
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            base_path = tmp.name # Sin extensión
            expected_path = base_path + ".joblib"
            
        try:
            save_model_data(base_path, self.mock_model, "Desc")
            self.assertTrue(os.path.exists(expected_path))
        finally:
            if os.path.exists(base_path):
                try: os.remove(base_path)
                except: pass
            if os.path.exists(expected_path):
                try: os.remove(expected_path)
                except: pass



    def test_save_empty_path(self):
        # Debe dar error si la ruta está vacía
        with self.assertRaises(ValueError):
            save_model_data("", self.mock_model, "Desc")



    def test_load_model_data_success(self):
        # Test de carga y resumen
        summary, desc = load_model_data(self.sample_model_data)
        
        self.assertEqual(desc, "Test Description")
        self.assertIsInstance(summary, list)
        
        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("y = 2.5 * x + 10", full_text)
        self.assertIn("R2: 0.95", full_text)



    def test_load_handles_missing_keys(self):
        # Test de robustez con datos incompletos
        incomplete_data = {"formula": "y=x"} 
        summary, desc = load_model_data(incomplete_data)
        
        full_text = "\n".join(summary)
        self.assertIn("Regression Line:", full_text)
        self.assertIn("MSE: N/A", full_text)


        

if __name__ == "__main__":
    unittest.main()
# This condition checks if the script is running directly (standalone) and not as an
# imported module. If so, it triggers unittest.main(), which is responsible for:
# 1. Scanning the entire file for classes inheriting from unittest.TestCase.
# 2. Automatically discovering all methods starting with the "test_" prefix.
# 3. Running those tests and reporting success/failure results to the console.
# This block is fundamental for GitHub Actions (CI/CD) to validate the code automatically,
# ensuring that future changes do not break the saving/loading functionality.