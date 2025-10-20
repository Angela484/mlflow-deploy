import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from mlflow.models import infer_signature
import sys
import traceback

# ---------------------------------------------------------------
# 📦 Dataset Externo:
# Historical Product Demand – Fuente: Kaggle
# Variables: Product_Code, Warehouse, Product_Category, Date, Order_Demand
# ---------------------------------------------------------------

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Definir Paths ---
workspace_dir = os.getcwd()
mlruns_dir = os.path.join(workspace_dir, "mlruns")
tracking_uri = "file://" + os.path.abspath(mlruns_dir)
artifact_location = "file://" + os.path.abspath(mlruns_dir)

# --- Asegurar existencia del directorio mlruns ---
os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o usar experimento ---
experiment_name = "CI-CD-Lab2"
experiment_id = None
try:
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location
    )
except mlflow.exceptions.MlflowException:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

# ---------------------------------------------------------------
# 🧠 Cargar Dataset Externo
# ---------------------------------------------------------------
data = pd.read_csv("Historical Product Demand.csv")

# Limpieza y preprocesamiento
data = data.dropna()

# Convertir fecha a valor numérico (días desde la fecha mínima)
data["Date"] = pd.to_datetime(data["Date"])
data["Days_Since_Start"] = (data["Date"] - data["Date"].min()).dt.days

# Codificar variables categóricas con códigos numéricos
for col in ["Product_Code", "Warehouse", "Product_Category"]:
    data[col] = data[col].astype("category").cat.codes

# Variables predictoras y objetivo
X = data[["Product_Code", "Warehouse", "Product_Category", "Days_Since_Start"]]
# --- Limpiar columna Order_Demand ---
# Algunos valores están entre paréntesis y representan números negativos, por ejemplo '(100)'
data["Order_Demand"] = (
    data["Order_Demand"]
    .astype(str)
    .str.replace("(", "-", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace(",", "", regex=False)
)

# Convertir a float
y = data["Order_Demand"].astype(float)


# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)

# ---------------------------------------------------------------
# 🧾 Registrar modelo en MLflow
# ---------------------------------------------------------------
try:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        print(f"✅ Modelo registrado correctamente. MSE: {mse:.4f}")

        # Guardar localmente
        import joblib
        joblib.dump(model, "model.pkl")
        print("✅ Modelo guardado como model.pkl")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    sys.exit(1)

