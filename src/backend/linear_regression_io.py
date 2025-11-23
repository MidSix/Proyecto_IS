import joblib
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