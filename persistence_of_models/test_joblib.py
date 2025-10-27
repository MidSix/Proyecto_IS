from joblib import dump, load
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
modelo = LinearRegression()
modelo.fit(X, y)

# Save model to file
dump(modelo, "modelo_joblib.joblib")
print("Modelo guardado con joblib.")

# Load model from file
modelo_cargado = load("modelo_joblib.joblib")

# Check that it is still working
prediccion = modelo_cargado.predict([[6]])
print("Predicción del modelo cargado:", prediccion)
