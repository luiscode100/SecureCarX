import sys
import os
import argparse
import time
import csv
from datetime import datetime

# Aseguramos que Python encuentre los módulos en la carpeta src
sys.path.append(os.path.join(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from infer import ModelInference

# --- CONFIGURACIÓN DE LOGS PARA HITO 5 ---
LOG_FILE = "data/prediction_logs.csv"

def log_prediction(input_data, prediction, latency):
    """Guarda la petición completa para detectar Data Drift después."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='') as f:
        # CAMBIO AQUÍ: Añadimos "input_data" a la lista de columnas permitidas
        fieldnames = ["timestamp", "input_data", "predicted_amount", "latency_ms", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "input_data": str(input_data), # Lo guardamos como texto/string
            "predicted_amount": round(prediction, 2),
            "latency_ms": round(latency * 1000, 2),
            "status": "success"
        })
        
# 1. Estructura de datos de entrada
class CarData(BaseModel):
    OBJECT_ID: int
    INSR_BEGIN: str
    INSR_END: str
    INSR_TYPE: str
    PROD_YEAR: int
    PREMIUM: float
    INSURED_VALUE: float
    SEX: str
    MAKE: str
    USAGE: str
    TYPE_VEHICLE: str
    SEATS_NUM: int
    CARRYING_CAPACITY: float
    CCM_TON: float
    EFFECTIVE_YR: int

# 2. Inicialización
app = FastAPI(title="SecureCarX - Predicción de Siniestros")
inference_engine = ModelInference()

@app.get("/")
def home():
    return {"message": "API de SecureCarX activa. Usa /predict para estimar costes."}

@app.get("/health")
def health_check():
    if inference_engine.model is not None:
        return {"status": "healthy", "model_loaded": True, "version": "1.0.0"}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.post("/predict")
def predict(data: CarData):
    start_time = time.time() # HITO 5: Inicio cronómetro
    try:
        input_dict = data.dict()
        prediction = inference_engine.predict_claim(input_dict)
        
        if isinstance(prediction, str):
            raise HTTPException(status_code=400, detail=prediction)
        
        # HITO 5: Cálculo de latencia y guardado de log
        latency = time.time() - start_time
        log_prediction(input_dict, prediction, latency)
            
        return {
            "status": "success",
            "predicted_claim_amount": round(prediction, 2),
            "currency": "EUR",
            "latency_ms": round(latency * 1000, 2)
        }
    except Exception as e:
        # También podrías loguear los fallos aquí si quisieras
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Servidor API SecureCarX")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)