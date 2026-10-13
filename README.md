# SecureCarX
Este proyecto implementa una solución completa de producción de Machine Learning (MLOps) para predecir el coste de los siniestros de vehículos. La arquitectura abarca desde la limpieza de datos brutos, el entrenamiento y versionado con MLflow, hasta el despliegue del modelo mediante una API REST en FastAPI y un contenedor autónomo de monitorización (Data Drift y rendimiento operativo) con bucle de realimentación automática (*Feedback Loop*).

---

## Estructura del Repositorio
* **`backend/`**: Contiene la lógica del núcleo de Machine Learning y la API.
  * **`src/`**: Código fuente de la aplicación:
    * `app.py`: Servidor FastAPI (`/predict`, `/feedback`, `/reload`, `/health`).
    * `train.py`: Pipeline de entrenamiento con split honesto por vehículo (`OBJECT_ID`), target encoding para variables de alta cardinalidad, optimización bayesiana (Optuna) y registro en MLflow.
    * `infer.py`: Motor de inferencia del modelo, con recarga en caliente.
    * `monitor.py`: Bucle de monitorización que evalúa Data Drift y dispara el reentrenamiento.
    * `drift.py`: Cálculo de métricas operativas, detección de drift (Evidently) y alertas.
    * `feedback.py`: Store de feedback etiquetado y dataset de reentrenamiento combinado.
    * `prediction_log.py`: Registro persistente de las predicciones servidas por la API.
    * `config.py`: Configuración centralizada de rutas y MLflow (local y contenedores).
    * `simular_drift.py`: Generador automatizado de peticiones en entornos normales o extremos.
    * `features.py` e `ingestion.py`: Módulos de limpieza, transformación e ingesta de datos.
  * `requirements.txt`: Dependencias del entorno de Python de los servicios del backend.

* **`notebooks/`**: Cuadernos de Jupyter utilizados en las fases de análisis exploratorio inicial y experimentación.

* **`data/`**: Datos persistentes compartidos por los servicios.
  * `raw/`: dataset original.
  * `processed/`: dataset limpio o transformado.
  * `operational/`: trazas de producción (`prediction_logs.csv`) y feedback etiquetado (`feedback_logs.csv`).
  * `reports/`: informes generados, incluido `drift_report.html`.

* **`artifacts/`**: Artefactos generados por los pipelines ML.
  * `models/`: modelos locales (`rf_model.joblib`, `model_columns.joblib`).
  * `mlruns/`: almacenamiento local del registro de experimentos de MLflow.

* **`frontend/`**: Interfaz de usuario web para interactuar con el sistema de predicción.
  * `index.html`: Estructura principal de la aplicación web estilizada con Tailwind CSS.
  * **`js/`**: Lógica de cliente.
    * `script.js`: Captura de datos del formulario y consumo asíncrono del endpoint de inferencia.

* **`infra/`**: Configuración de orquestación y despliegue automatizado.
  * `Dockerfile.api`: Configuración de la imagen Docker para el servicio de FastAPI.
  * `Dockerfile.train`: Configuración de la imagen Docker para el pipeline de entrenamiento.
  * `Dockerfile.monitor`: Configuración de la imagen Docker para el servicio de observabilidad (Evidently).
  * `docker-compose.yml`: Orquestador de la infraestructura multi-contenedor, volúmenes compartidos y redes locales.

## Despliegue del Entorno

El entrenamiento está **desacoplado** del resto mediante el perfil `train`, de modo que
levantar la aplicación no obliga a reentrenar.

**1. Entrenar (solo cuando quieras generar/actualizar el modelo):**

```
docker compose --env-file .env -f infra/docker-compose.yml --profile train run --rm train
```

El número de trials de Optuna se controla con `N_TRIALS` (por defecto 10), por ejemplo:

```
N_TRIALS=3 docker compose --env-file .env -f infra/docker-compose.yml --profile train run --rm train
```

**2. Levantar la aplicación** (usa el modelo ya existente en `artifacts/models/`):

```
docker compose --env-file .env -f infra/docker-compose.yml up --build
```

Esto arranca MLflow, la API, el frontend y el monitor (no reentrena). Servicios disponibles:

| Servicio | URL |
| --- | --- |
| Frontend web | http://localhost:8080 |
| API (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |

> Tras un reentrenamiento, la API puede recargar el modelo en caliente sin reiniciar:
> `curl -X POST http://localhost:8000/reload`

---