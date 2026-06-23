import sys
import os
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_ingestion import load_data
from features import clean_data, filter_training_data, prepare_features

# --- CONFIGURACIÓN MLFLOW PROTEGIDA ---
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("SecureCarX_Insurance_Claims")
    print(f"Intento de conexión a MLflow en {MLFLOW_URI} configurado.")
except Exception as e:
    print(f"Aviso: Error configurando MLflow ({e}). Se usará solo modo local.")

def run_training(input_path, model_dir):
    """Orquestador del pipeline de entrenamiento con registro opcional en MLflow."""
    
    # Intentamos iniciar un run, pero no bloqueamos si falla
    active_run = None
    try:
        active_run = mlflow.start_run()
        print("Iniciada sesión en MLflow.")
    except Exception as e:
        print(f"No se pudo iniciar sesión en MLflow (Error: {e}). El proceso continuará en local.")

    # 1. Ingestión
    print(f"Cargando datos desde: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe.")
        return

    raw_df = load_data(input_path)
    
    # 2. Limpieza y Filtrado
    print("Limpiando y filtrando datos...")
    cleaned_df = clean_data(raw_df)
    training_df = filter_training_data(cleaned_df)
    
    # 3. Ingeniería de Atributos
    print("Preparando características...")
    model_df = prepare_features(training_df)
    
    # 4. Separación X e y
    X = model_df.drop(columns=['CLAIM_PAID_LOG'], errors='ignore')
    y = model_df['CLAIM_PAID_LOG']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Entrenamiento (Hiperparámetros del Hito 3)
    n_estimators = 150
    print(f"Entrenando modelo con {len(X_train)} muestras...")
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=7,               # Evita que el árbol memorice de más
        min_samples_leaf=10,       # Exige al menos 10 coches por hoja
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 6. Evaluación rápida
    y_pred_log = model.predict(X_test)
    y_pred_euros = np.expm1(y_pred_log)
    y_test_euros = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test_euros, y_pred_euros)
    r2 = r2_score(y_test, y_pred_log)
    
    print(f"Modelo entrenado. MAE: {mae:.2f} €")
    
    # 7. Registro en MLflow (Solo si la conexión tuvo éxito)
    if active_run:
        try:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("input_data_path", input_path)
            mlflow.log_metric("mae_euros", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(model, "model", registered_model_name="SecureCarX_RF")
            print("Métricas y modelo registrados en MLflow.")
            mlflow.end_run()
        except Exception as e:
            print(f"Error al registrar datos en MLflow: {e}")

    # 8. PERSISTENCIA 
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_model.joblib")
    columns_path = os.path.join(model_dir, "model_columns.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    
    print(f"MODELO GUARDADO LOCALMENTE EN: {model_path}")

if __name__ == "__main__":
    # Calculamos de forma dinámica la raíz de 'backend' (un nivel arriba de 'src')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "securecarx_dataset.csv")
    DEFAULT_OUTPUT = os.path.join(BASE_DIR, "models")

    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento SecureCarX")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)

    args = parser.parse_args()
    run_training(input_path=args.input, model_dir=args.output_dir)