import joblib
from backend.linear_regression_creation import LinearRegressionModel
# We do not need a class here because we do not store meaningful
# state; simple functions are enough.
def load_model_data(model_data: dict) -> tuple:
    """Load model data from dictionary and return summary and description.

    Extracts formula, metrics, and description from a model data
    dictionary and returns a formatted summary as a list of strings.

    Parameters
    ----------
    model_data : dict
        Dictionary containing keys 'formula', 'metrics', and
        'description'.

    Returns
    -------
    tuple
        A 2-tuple of (summary_lines, description) where summary_lines
        is a list of formatted strings and description is a string.

    Raises
    ------
    Exception
        If an error occurs during data extraction.
    """
    try:
        formula = model_data.get("formula", "")
        metrics = model_data.get("metrics", {})
        description = model_data.get("description", "")

        # Construir texto de resumen atractivo
        train_metrics = metrics.get("train", {})
        test_metrics = metrics.get("test", {})

        summary_lines = [
            "Regression Line:",
            "",
            f"{formula}",
            "",
            "Train metrics:",
            f"MSE: {train_metrics.get('MSE', 'N/A')}",
            f"R2: {train_metrics.get('R2', 'N/A')}",
            "",
            "Test metrics:",
            f"MSE: {test_metrics.get('MSE', 'N/A')}",
            f"R2: {test_metrics.get('R2', 'N/A')}"
        ]

        return summary_lines, description
    except Exception as e:
        raise e

def save_model_data(
        file_path: str,
        model: LinearRegressionModel,
        model_description: str
) -> None:
    """Save a trained linear regression model to a joblib file.

    Serializes the model object along with formula, feature names,
    metrics, and description into a joblib file. Appends '.joblib'
    extension if not already present.

    Parameters
    ----------
    file_path : str
        Path where the model file will be saved. Must not be empty.
        Extension '.joblib' is added if missing.
    model : LinearRegressionModel
        Fitted model instance to save.
    model_description : str
        User-provided description of the model.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If file_path is empty.
    Exception
        If an error occurs during file saving.
    """
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
            "description": model_description,
            "model": model
        }

        joblib.dump(model_data, file_path)
    except Exception as e:
            raise e