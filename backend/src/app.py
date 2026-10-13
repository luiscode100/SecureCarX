import argparse
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from feedback import log_feedback
from infer import ModelInference
from prediction_log import log_prediction

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


class FeedbackData(CarData):
    CLAIM_PAID: float


ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

app = FastAPI(title="SecureCarX - Predicción de Siniestros")
inference_engine = ModelInference()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOWED_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

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
    start_time = time.time()
    try:
        input_dict = data.model_dump()
        prediction = inference_engine.predict_claim(input_dict)

        if isinstance(prediction, str):
            raise HTTPException(status_code=400, detail=prediction)

        latency = time.time() - start_time
        log_prediction(input_dict, prediction, latency)

        return {
            "status": "success",
            "predicted_claim_amount": round(prediction, 2),
            "currency": "EUR",
            "latency_ms": round(latency * 1000, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
def feedback(data: FeedbackData):
    """Recibe la etiqueta real (coste del siniestro) para el feedback loop."""
    try:
        log_feedback(data.model_dump())
        return {"status": "recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model():
    """Recarga el modelo desde disco tras un reentrenamiento (feedback loop)."""
    try:
        inference_engine.reload()
        return {"status": "reloaded", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recargando el modelo: {e}")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Servidor API SecureCarX")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host del servidor")
    parser.add_argument("--port", type=int, default=8000, help="Puerto del servidor")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
