import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sys
import os

# ---------------------------------------------------------------
# 📦 Dataset Externo: Historical Product Demand (Kaggle)
# Variables: Product_Code, Warehouse, Product_Category, Date, Order_Demand
# ---------------------------------------------------------------

THRESHOLD = 1e8  # umbral alto porque los valores de demanda son grandes

print("--- Debug: Iniciando validación del modelo con dataset externo ---")

# --- Cargar dataset ---
data_path = "Historical Product Demand.csv"
if not os.path.exists(data_path):
    print(f"--- ERROR: No se encontró el archivo '{data_path}'. Asegúrate de subirlo al repositorio. ---")
    sys.exit(1)

data = pd.read_csv(data_path)
data = data.dropna()

# Convertir fecha y codificar variables categóricas
data["Date"] = pd.to_datetime(data["Date"])
data["Days_Since_Start"] = (data["Date"] - data["Date"].min()).dt.days
for col in ["Product_Code", "Warehouse", "Product_Category"]:
    data[col] = data[col].astype("category").cat.codes

# Variables predictoras y objetivo
X = data[["Product_Code", "Warehouse", "Product_Category", "Days_Since_Start"]]
y = data["Order_Demand"].astype(float)

# División igual que en entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---")

# --- Cargar modelo ---
model_filename = "model.pkl"
if not os.path.exists(model_filename):
    print(f"--- ERROR: No se encontró el modelo '{model_filename}'. Ejecuta train.py primero. ---")
    print(f"--- Archivos disponibles en el directorio actual: {os.listdir(os.getcwd())} ---")
    sys.exit(1)

print(f"--- Debug: Cargando modelo desde {model_filename} ---")
model = joblib.load(model_filename)

# --- Predicción ---
try:
    y_pred = model.predict(X_test)
except ValueError as e:
    print(f"--- ERROR durante la predicción: {e} ---")
    print(f"El modelo esperaba {model.n_features_in_} características, pero se encontró {X_test.shape[1]}.")
    sys.exit(1)

# --- Métrica de evaluación ---
mse = mean_squared_error(y_test, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# --- Validación final ---
if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad y pasa la validación.")
    sys.exit(0)
else:
    print("❌ El modelo no cumple el umbral esperado. Deteniendo pipeline.")
    sys.exit(1)
