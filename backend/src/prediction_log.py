"""Registro persistente de las predicciones servidas por la API."""
import csv
import fcntl
import os
import threading
from datetime import datetime

from config import PREDICTION_LOG_PATH

FIELDNAMES = ["timestamp", "input_data", "predicted_amount", "latency_ms", "status"]
_LOCK = threading.Lock()


def log_prediction(input_data, prediction, latency):
    """Guarda la petición completa para poder detectar Data Drift después.

    Usa un lock en memoria (hilos) y un lock de fichero (procesos) para que
    varios workers de la API puedan escribir sin corromper el CSV.
    """
    os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(PREDICTION_LOG_PATH)

    with _LOCK:
        with open(PREDICTION_LOG_PATH, mode="a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "input_data": str(input_data),
                    "predicted_amount": round(prediction, 2),
                    "latency_ms": round(latency * 1000, 2),
                    "status": "success",
                })
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
