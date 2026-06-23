# SecureCarX
Este proyecto implementa una solución completa de producción de Machine Learning (MLOps) para predecir el coste de los siniestros de vehículos. La arquitectura abarca desde la limpieza de datos brutos, el entrenamiento y versionado con MLflow, hasta el despliegue del modelo mediante una API REST en FastAPI y un contenedor autónomo de monitorización (Data Drift y rendimiento operativo) con bucle de realimentación automática (*Feedback Loop*).

---

## Estructura del Repositorio
* **`backend/`**: Contiene la lógica del núcleo de Machine Learning y la API.
  * **`data/`**: Carpeta compartida (volumen Docker) que contiene el dataset original, las trazas de producción (`prediction_logs.csv`) y los reportes generados.
  * **`models/`**: Almacenamiento local de los artefactos del modelo entrenado (`rf_model.joblib`, `model_columns.joblib`).
  * **`src/`**: Código fuente de la aplicación:
    * `app.py`: Servidor FastAPI y lógica del endpoint `/predict`.
    * `train.py`: Pipeline de entrenamiento, optimización y registro en MLflow.
    * `monitor.py`: Script del monitor que calcula latencias, evalúa Data Drift y dispara el reentrenamiento.
    * `simular_drift.py`: Generador automatizado de peticiones en entornos normales o extremos.
    * `features.py` y `data_ingestion.py`: Módulos de limpieza, transformación e ingesta de datos.
    * `infer.py`: Motor de inferencia del modelo.
  * **`mlruns/`**: Servidor y base de datos local del registro de experimentos de MLflow.
  * **`notebook/`**: Cuadernos de Jupyter utilizados en las fases de análisis exploratorio inicial y experimentación.
  * `requirements.txt`: Dependencias del entorno de Python de los servicios del backend.

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

Para levantar la infraestructura:

```bash
docker compose -f infra/docker-compose.yml up --build

---