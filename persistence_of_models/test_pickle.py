import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
modelo = LinearRegression()
modelo.fit(X, y)

# Save model to file
with open("modelo_pickle.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Modelo guardado con pickle.")

# Load model from file
with open("modelo_pickle.pkl", "rb") as f:
    modelo_cargado = pickle.load(f)

# Check that it is still working
prediccion = modelo_cargado.predict([[6]])
print("Predicción del modelo cargado:", prediccion)
