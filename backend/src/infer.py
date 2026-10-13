import os
import argparse
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from features import (
    clean_data,
    add_target,
    select_features,
    apply_target_encoding,
    one_hot_encode,
)
from config import (
    MLFLOW_URI,
    REGISTERED_MODEL_NAME,
    MODEL_PATH,
    MODEL_COLUMNS_PATH,
    MODEL_ENCODINGS_PATH,
)

mlflow.set_tracking_uri(MLFLOW_URI)

DEFAULT_MODEL = MODEL_PATH
DEFAULT_COLUMNS = MODEL_COLUMNS_PATH
DEFAULT_ENCODINGS = MODEL_ENCODINGS_PATH

class ModelInference:
    def __init__(self, model_path=DEFAULT_MODEL, columns_path=DEFAULT_COLUMNS,
                 encodings_path=DEFAULT_ENCODINGS, use_mlflow=False):
        """
        Inicia el motor de inferencia.
        Puede cargar el modelo localmente o desde el Model Registry de MLflow.
        """
        self.use_mlflow = use_mlflow
        self.model_path = model_path
        self.columns_path = columns_path
        self.encodings_path = encodings_path
        self.model = None
        self.model_columns = None
        self.encodings = None
        self._load()

    def _load(self):
        """Carga (o recarga) el modelo, las columnas y el target encoding."""
        if self.use_mlflow:
            # Opción Bonus: Carga desde el registro de modelos (Hito 4 Bonus / Hito 5)
            print(f"Cargando modelo desde MLflow Registry: {MLFLOW_URI}")
            model_version = "latest"  # O "Production" para el feedback loop del Hito 5
            self.model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_version}")
        else:
            # Opción Estándar: Carga local (Hito 4)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"No se encontró el modelo local en: {self.model_path}")
            self.model = joblib.load(self.model_path)

        # Las columnas siempre se cargan para asegurar la estructura de entrada
        if not os.path.exists(self.columns_path):
            raise FileNotFoundError(f"No se encontró el archivo de columnas en: {self.columns_path}")
        self.model_columns = joblib.load(self.columns_path)

        # El target encoding es opcional (modelos antiguos no lo tienen)
        self.encodings = joblib.load(self.encodings_path) if os.path.exists(self.encodings_path) else None

    def reload(self):
        """Vuelve a leer el modelo desde disco (usado por el feedback loop)."""
        self._load()
        return True

    def predict_claim(self, input_data: dict):
        df_input = pd.DataFrame([input_data])
        
        # Procesamiento (Hito 4: Ingeniería de características [cite: 7])
        df_clean = clean_data(df_input)
        if df_clean.empty:
            return "Error: Los datos de entrada no cumplen las reglas de validación."
            
        df_feat = select_features(add_target(df_clean))
        df_feat = apply_target_encoding(df_feat, self.encodings)
        skip = self.encodings['cols'] if self.encodings else None
        df_prepared = one_hot_encode(df_feat, skip_cols=skip)

        # Alinear columnas con el entrenamiento
        df_final = df_prepared.reindex(columns=self.model_columns, fill_value=0)
        
        pred_log = self.model.predict(df_final)
        return float(np.expm1(pred_log)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Inferencia SecureCarX")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ruta local del modelo")
    parser.add_argument("--columns", type=str, default=DEFAULT_COLUMNS, help="Ruta de columnas")
    parser.add_argument("--encodings", type=str, default=DEFAULT_ENCODINGS, help="Ruta del target encoding")
    parser.add_argument("--mlflow", action="store_true", help="Usar MLflow para cargar el modelo")

    args = parser.parse_args()

    try:
        inference = ModelInference(
            model_path=args.model, columns_path=args.columns,
            encodings_path=args.encodings, use_mlflow=args.mlflow,
        )
        
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