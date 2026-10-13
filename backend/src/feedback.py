"""Almacén de feedback etiquetado y construcción del dataset de reentrenamiento.

El feedback loop necesita la etiqueta real (CLAIM_PAID) de las peticiones de
producción. La API la recibe por el endpoint /feedback y aquí se acumula para
reentrenar combinando el dataset de referencia con los casos reales.
"""
import csv
import fcntl
import os
import threading
from datetime import datetime

import pandas as pd

from config import FEEDBACK_LOG_PATH, RAW_DATA_PATH

CAR_FIELDS = [
    "OBJECT_ID", "INSR_BEGIN", "INSR_END", "INSR_TYPE", "PROD_YEAR", "PREMIUM",
    "INSURED_VALUE", "SEX", "MAKE", "USAGE", "TYPE_VEHICLE", "SEATS_NUM",
    "CARRYING_CAPACITY", "CCM_TON", "EFFECTIVE_YR",
]
FIELDNAMES = ["timestamp"] + CAR_FIELDS + ["CLAIM_PAID"]
_LOCK = threading.Lock()


def log_feedback(record):
    """Añade un caso etiquetado al CSV de feedback (seguro ante concurrencia)."""
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    file_exists = os.path.isfile(FEEDBACK_LOG_PATH)

    row = {"timestamp": datetime.now().isoformat()}
    row.update({k: record.get(k) for k in CAR_FIELDS})
    row["CLAIM_PAID"] = record["CLAIM_PAID"]

    with _LOCK:
        with open(FEEDBACK_LOG_PATH, mode="a", newline="") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_feedback():
    """Devuelve el feedback acumulado como DataFrame (sin la columna timestamp)."""
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return pd.DataFrame()
    fb = pd.read_csv(FEEDBACK_LOG_PATH)
    return fb.drop(columns=["timestamp"], errors="ignore")


def build_training_dataset(output_path):
    """Combina el dataset de referencia etiquetado con el feedback acumulado."""
    frames = [pd.read_csv(RAW_DATA_PATH)]
    fb = load_feedback()
    if not fb.empty:
        frames.append(fb)
    combined = pd.concat(frames, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    return output_path
