"""Monitor de observabilidad: calcula métricas, detecta drift y dispara el
reentrenamiento automático (feedback loop)."""
import os
import time
import warnings

import mlflow
import pandas as pd
import requests

warnings.filterwarnings("ignore")

from config import (
    MLFLOW_URI,
    MODELS_DIR,
    PREDICTION_LOG_PATH,
    RAW_DATA_PATH,
    TRAINING_DATASET_PATH,
)
from drift import (
    DRIFT_THRESHOLD_PCT,
    MIN_PROD_SAMPLES,
    REF_SAMPLE_SIZE,
    analizar_drift,
    calcular_metricas_operativas,
    enviar_alerta,
)
from feedback import build_training_dataset
from train import run_training

API_RELOAD_URL = os.getenv("API_RELOAD_URL", "http://api:8000/reload")


def recargar_modelo_api():
    try:
        response = requests.post(API_RELOAD_URL, timeout=10)
        print(f"Recarga del modelo en la API: {response.status_code} {response.json()}")
    except Exception as e:
        print(f"No se pudo recargar el modelo en la API: {e}")


def run_monitoring():
    print("\n--- Iniciando análisis de Drift e Inferencia de Métricas ---")
    if not os.path.exists(PREDICTION_LOG_PATH) or not os.path.exists(RAW_DATA_PATH):
        print("Esperando archivos de datos...")
        return

    try:
        ref_data = pd.read_csv(RAW_DATA_PATH)
        if len(ref_data) > REF_SAMPLE_SIZE:
            ref_data = ref_data.sample(n=REF_SAMPLE_SIZE, random_state=42).reset_index(drop=True)

        prod_logs = pd.read_csv(PREDICTION_LOG_PATH)
        if prod_logs.empty:
            print("Logs vacíos.")
            return

        if len(prod_logs) < MIN_PROD_SAMPLES:
            print(f"Volumen insuficiente en producción ({len(prod_logs)}/{MIN_PROD_SAMPLES} peticiones). Esperando más datos para analizar Drift.")
            calcular_metricas_operativas(prod_logs)
            return

        calcular_metricas_operativas(prod_logs)
        porcentaje_drift = analizar_drift(ref_data, prod_logs)
        print(f"Análisis de Modelo: {porcentaje_drift:.2f}% de las variables presentan Data Drift.")

        if porcentaje_drift >= DRIFT_THRESHOLD_PCT:
            enviar_alerta("MODELO (Data Drift)", f"Se ha detectado deriva masiva de datos ({porcentaje_drift:.2f}%).")
            mlflow.set_tracking_uri(MLFLOW_URI)
            print("Disparando FEEDBACK LOOP: Iniciando Reentrenamiento Automático...")
            training_path = build_training_dataset(TRAINING_DATASET_PATH)
            print(f"Dataset de reentrenamiento (referencia + feedback): {training_path}")
            run_training(input_path=training_path, model_dir=MODELS_DIR, n_trials=10)
            recargar_modelo_api()
            print("FEEDBACK LOOP COMPLETADO: modelo reentrenado y recargado en la API.")
    except Exception as e:
        print(f"ERROR en el proceso: {e}")


if __name__ == "__main__":
    print("Monitor iniciado (Modo Polling - 10s)")
    last_size = 0
    while True:
        if os.path.exists(PREDICTION_LOG_PATH):
            current_size = os.path.getsize(PREDICTION_LOG_PATH)
            if current_size != last_size:
                run_monitoring()
                last_size = current_size
        time.sleep(10)
