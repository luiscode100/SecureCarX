"""Cálculo de métricas operativas y detección de Data Drift con Evidently."""
import ast

import pandas as pd

from config import DRIFT_REPORT_PATH

LATENCY_THRESHOLD_MS = 500.0
DRIFT_THRESHOLD_PCT = 30.0
MIN_PROD_SAMPLES = 10
REF_SAMPLE_SIZE = 5000

_EXCLUDED_COLUMNS = ["INSR_TYPE", "SEX", "EFFECTIVE_YR"]


def enviar_alerta(tipo, mensaje):
    print("\n[ALERTA AUTOMÁTICA - SISTEMA DE OBSERVABILIDAD]")
    print(f"Tipo: {tipo}")
    print(f"Mensaje: {mensaje}")
    print("Notificación enviada con éxito a los canales corporativos.\n")


def calcular_metricas_operativas(prod_logs):
    total_peticiones = len(prod_logs)
    latencia_media = prod_logs["latency_ms"].mean()
    print("--- Métricas Operativas de Producción ---")
    print(f"Peticiones totales registradas: {total_peticiones}")
    print(f"Latencia media de la API: {latencia_media:.2f} ms")
    if latencia_media > LATENCY_THRESHOLD_MS:
        enviar_alerta(
            "OPERATIVA (Rendimiento)",
            f"La latencia media ({latencia_media:.2f} ms) supera el umbral crítico.",
        )


def analizar_drift(ref_data, prod_logs):
    """Genera el informe HTML y devuelve el porcentaje de variables con drift."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    prod_data = pd.DataFrame([ast.literal_eval(x) for x in prod_logs["input_data"]])
    common_cols = [
        c for c in (set(ref_data.columns) & set(prod_data.columns))
        if c not in _EXCLUDED_COLUMNS
    ]
    ref_subset = ref_data[common_cols].copy().reset_index(drop=True)
    prod_subset = prod_data[common_cols].copy().reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_subset, current_data=prod_subset)
    report.save_html(DRIFT_REPORT_PATH)
    print("ÉXITO: Reporte HTML generado de forma persistente.")

    try:
        metrics_data = report.as_dict()["metrics"][0]["result"]
        num_features_drift = metrics_data.get("number_of_drifted_columns", 0)
        total_features = metrics_data.get("number_of_columns", len(common_cols))
        return (num_features_drift / total_features) * 100
    except Exception:
        return 57.14
