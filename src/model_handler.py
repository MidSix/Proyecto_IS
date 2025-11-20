import joblib

def load_model_data(model_data: dict):
    """
    load model data from a dictionary and returns a summary and description.
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
            f"Formula: {formula}",
            f"Inputs: {', '.join(input_cols)}",
            f"Output: {output_col}",
            "",
            "Train metrics:",
            f"  R2: {train_metrics.get('R2', 'N/A')}, MSE: {train_metrics.get('MSE', 'N/A')}",
            "",
            "Test metrics:",
            f"  R2: {test_metrics.get('R2', 'N/A')}, MSE: {test_metrics.get('MSE', 'N/A')}",
        ]
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
                    "train": {"R2": model.get_train_R2, "MSE": model.get_train_MSE},
                    "test": {"R2": model.get_test_R2, "MSE": model.get_test_MSE},
                },
                "description": model_description
            }

        joblib.dump(model_data, file_path)
    except Exception as e:
            raise e