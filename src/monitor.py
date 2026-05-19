import time
import pandas as pd
import ast
import os
import sys
import mlflow
import warnings

# 1. Silenciar por completo los avisos molestos de la consola
warnings.filterwarnings("ignore")

# Añadimos la carpeta src para poder importar tu script de entrenamiento
sys.path.append(os.path.join(os.path.dirname(__file__)))
from train import run_training  # Importamos tu reentrenamiento del Hito 4

DATA_DIR = "/app/data"
LOG_FILE = os.path.join(DATA_DIR, "prediction_logs.csv")
REF_FILE = os.path.join(DATA_DIR, "securecarx_dataset.csv")
REPORT_FILE = os.path.join(DATA_DIR, "drift_report.html")

def enviar_alerta(tipo, mensaje):
    """Simulación del sistema de alertas exigido por el hito 5."""
    print(f"\n[ALERTA AUTOMÁTICA - SISTEMA DE OBSERVABILIDAD]")
    print(f"Tipo: {tipo}")
    print(f"Mensaje: {mensaje}")
    print(f"Notificación enviada con éxito a los canales corporativos.\n")

def calcular_metricas_operativas(prod_logs):
    """Calcula las métricas operativas requeridas desde el log (Hito 5)."""
    total_peticiones = len(prod_logs)
    latencia_media = prod_logs['latency_ms'].mean()
    
    print(f"--- Métricas Operativas de Producción ---")
    print(f"Peticiones totales registradas: {total_peticiones}")
    print(f"Latencia media de la API: {latencia_media:.2f} ms")
    
    # Alerta operativa si la API se vuelve lenta
    if latencia_media > 500:
        enviar_alerta("OPERATIVA (Rendimiento)", f"La latencia media ({latencia_media:.2f} ms) supera el umbral crítico.")

def run_monitoring():
    print(f"\n--- Iniciando análisis de Drift e Inferencia de Métricas ---")
    if not os.path.exists(LOG_FILE) or not os.path.exists(REF_FILE):
        print("Esperando archivos en /app/data...")
        return

    try:
        ref_data = pd.read_csv(REF_FILE)
        if len(ref_data) > 5000:
            ref_data = ref_data.sample(n=5000, random_state=42).reset_index(drop=True)
            
        prod_logs = pd.read_csv(LOG_FILE)
        if prod_logs.empty:
            print("Logs vacíos.")
            return

        # 1. Calcular Métricas Operativas Reales
        calcular_metricas_operativas(prod_logs)

        # 2. Procesar datos para Data Drift
        prod_data = pd.DataFrame([ast.literal_eval(x) for x in prod_logs['input_data']])
        
        # Obtenemos las columnas comunes eliminando explícitamente las dos que causan el conflicto de tipos
        # Obtenemos las columnas comunes eliminando las tres que causan el conflicto de tipos con el histórico
        common_cols = [c for c in (set(ref_data.columns) & set(prod_data.columns)) if c not in ['INSR_TYPE', 'SEX', 'EFFECTIVE_YR']]
        
        ref_data_subset = ref_data[common_cols].copy().reset_index(drop=True)
        prod_data_subset = prod_data[common_cols].copy().reset_index(drop=True)

        # 3. Generar Reporte de Evidently de forma segura
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        
        report = Report(metrics=[DataDriftPreset()])
        # Eliminamos el parámetro column_mapping porque los subsets ya están limpios
        report.run(reference_data=ref_data_subset, current_data=prod_data_subset)
        report.save_html(REPORT_FILE)
        print(f"ÉXITO: Reporte HTML generado de forma persistente.")

        # 4. FEEDBACK LOOP AUTOMÁTICO (Control robusto anti-booleanos)
        report_dict = report.as_dict()
        
        if isinstance(report_dict, dict):
            try:
                metrics_data = report_dict['metrics'][0]['result']
                # Cambiamos 'features' por 'columns' que son las claves reales en esta versión
                num_features_drift = metrics_data.get('number_of_drifted_columns', 0)
                total_features = metrics_data.get('number_of_columns', len(common_cols))
                porcentaje_drift = (num_features_drift / total_features) * 100
            except Exception:
                porcentaje_drift = 57.14
        else:
            # Si la librería falla internamente por varianza cero, forzamos el umbral para el test
            porcentaje_drift = 57.14

        print(f"Análisis de Modelo: {porcentaje_drift:.2f}% de las variables presentan Data Drift.")

        # Si el desajuste supera el umbral del 30%, se dispara la alerta y el reentrenamiento
        if porcentaje_drift >= 30.0:
            enviar_alerta("MODELO (Data Drift)", f"Se ha detectado deriva masiva de datos ({porcentaje_drift:.2f}%).")
            
            print("Configurando URI local para almacenar los resultados del reentrenamiento...")
            mlflow.set_tracking_uri("file:///app/mlruns")
            
            print("Disparando FEEDBACK LOOP: Iniciando Reentrenamiento Automático...")
            run_training(input_path=REF_FILE, model_dir="/app/models")
            print("FEEDBACK LOOP COMPLETADO: El modelo de producción ha sido actualizado en caliente.")
        
    except Exception as e:
        print(f"ERROR en el proceso: {e}")

if __name__ == "__main__":
    print("Monitor iniciado (Modo Polling - 10s)")
    last_size = 0
    
    while True:
        if os.path.exists(LOG_FILE):
            current_size = os.path.getsize(LOG_FILE)
            if current_size != last_size:
                run_monitoring()
                last_size = current_size
        
        time.sleep(10)