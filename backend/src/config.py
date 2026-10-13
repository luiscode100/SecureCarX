"""Configuración centralizada de rutas y MLflow.

Única fuente de verdad para saber dónde viven los datos, los modelos y los
experimentos, tanto en local como dentro de los contenedores Docker.
Se puede sobreescribir con las variables de entorno DATA_DIR, MODELS_DIR,
MLRUNS_DIR y MLFLOW_TRACKING_URI.
"""
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
IN_CONTAINER = os.path.exists("/app/src")

if IN_CONTAINER:
    PROJECT_ROOT = "/app"
    _DATA_DIR = "/app/data"
    _MODELS_DIR = "/app/models"
    _MLRUNS_DIR = "/app/mlruns"
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))
    _DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    _MODELS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
    _MLRUNS_DIR = os.path.join(PROJECT_ROOT, "artifacts", "mlruns")

DATA_DIR = os.getenv("DATA_DIR", _DATA_DIR)
MODELS_DIR = os.getenv("MODELS_DIR", _MODELS_DIR)
MLRUNS_DIR = os.getenv("MLRUNS_DIR", _MLRUNS_DIR)

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "securecarx_dataset.csv")
PREDICTION_LOG_PATH = os.path.join(DATA_DIR, "operational", "prediction_logs.csv")
FEEDBACK_LOG_PATH = os.path.join(DATA_DIR, "operational", "feedback_logs.csv")
TRAINING_DATASET_PATH = os.path.join(DATA_DIR, "processed", "training_with_feedback.csv")
DRIFT_REPORT_PATH = os.path.join(DATA_DIR, "reports", "drift_report.html")

MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
MODEL_COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.joblib")
MODEL_ENCODINGS_PATH = os.path.join(MODELS_DIR, "model_encodings.joblib")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///" + os.path.join(MLRUNS_DIR, "mlflow.db"))
MLFLOW_EXPERIMENT = "SecureCarX_Insurance_Claims"
REGISTERED_MODEL_NAME = "SecureCarX_RF"
