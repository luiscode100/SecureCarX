import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

# Añade la carpeta 'src' al camino de búsqueda
sys.path.append(os.path.join(os.path.dirname(__file__)))
from features import clean_data, prepare_features

# Configuración de MLflow desde variables de entorno
#MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
#MLFLOW_URI = "file:///app/mlruns"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

class ModelInference:
    def __init__(self, model_path="models/rf_model.joblib", columns_path="models/model_columns.joblib", use_mlflow=False):
        """
        Inicia el motor de inferencia. 
        Puede cargar el modelo localmente o desde el Model Registry de MLflow.
        """
        self.use_mlflow = use_mlflow
        self.columns_path = columns_path

        if self.use_mlflow:
            # Opción Bonus: Carga desde el registro de modelos (Hito 4 Bonus / Hito 5)
            print(f"Cargando modelo desde MLflow Registry: {MLFLOW_URI}")
            model_name = "SecureCarX_RF"
            model_version = "latest" # O "Production" para el feedback loop del Hito 5
            self.model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
        else:
            # Opción Estándar: Carga local (Hito 4)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el modelo local en: {model_path}")
            self.model = joblib.load(model_path)

        # Las columnas siempre se cargan para asegurar la estructura de entrada
        if not os.path.exists(self.columns_path):
            raise FileNotFoundError(f"No se encontró el archivo de columnas en: {self.columns_path}")
        self.model_columns = joblib.load(self.columns_path)

    def predict_claim(self, input_data: dict):
        df_input = pd.DataFrame([input_data])
        
        # Procesamiento (Hito 4: Ingeniería de características [cite: 7])
        df_clean = clean_data(df_input)
        if df_clean.empty:
            return "Error: Los datos de entrada no cumplen las reglas de validación."
            
        df_prepared = prepare_features(df_clean)
        
        # Alinear columnas con el entrenamiento
        df_final = df_prepared.reindex(columns=self.model_columns, fill_value=0)
        
        pred_log = self.model.predict(df_final)
        return float(np.expm1(pred_log)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Inferencia SecureCarX")
    parser.add_argument("--model", type=str, default="models/rf_model.joblib", help="Ruta local del modelo")
    parser.add_argument("--columns", type=str, default="models/model_columns.joblib", help="Ruta de columnas")
    parser.add_argument("--mlflow", action="store_true", help="Usar MLflow para cargar el modelo")
    
    args = parser.parse_args()

    try:
        inference = ModelInference(model_path=args.model, columns_path=args.columns, use_mlflow=args.mlflow)
        
        sample = {
            'OBJECT_ID': 1, 'INSR_BEGIN': '2024-01-01', 'INSR_END': '2025-01-01',
            'INSR_TYPE': '1202', 'PROD_YEAR': 2020, 'PREMIUM': 1200, 
            'INSURED_VALUE': 35000, 'SEX': '1', 'MAKE': 'TOYOTA', 
            'USAGE': 'Private', 'TYPE_VEHICLE': 'Car', 'SEATS_NUM': 5, 
            'CARRYING_CAPACITY': 0, 'CCM_TON': 1600, 'EFFECTIVE_YR': 2024
        }
        
        resultado = inference.predict_claim(sample)
        print(f"Predicción ({'MLflow' if args.mlflow else 'Local'}): {resultado:.2f} €")
        
    except Exception as e:
        print(f"Error: {e}")