# SecureCarX
Este proyecto implementa una solución completa de producción de Machine Learning (MLOps) para predecir el coste de los siniestros de vehículos. La arquitectura abarca desde la limpieza de datos brutos, el entrenamiento y versionado con MLflow, hasta el despliegue del modelo mediante una API REST en FastAPI y un contenedor autónomo de monitorización (Data Drift y rendimiento operativo) con bucle de realimentación automática (*Feedback Loop*).

---

## Estructura del Repositorio

* **`data/`**: Carpeta compartida (volumen Docker) que contiene el dataset original, las trazas de producción (`prediction_logs.csv`) y los reportes generados.
* **`models/`**: Almacenamiento del modelo (`rf_model.joblib`).
* **`src/`**: Código de la aplicación:
    * `app.py`: Servidor FastAPI y lógica del endpoint `/predict`.
    * `train.py`: Pipeline de entrenamiento, optimización y registro en MLflow.
    * `monitor.py`: Script del monitor que calcula latencias, evalúa Data Drift y dispara el reentrenamiento.
    * `simular_drift.py`: Generador automatizado de peticiones en entornos normales o extremos.
* **`mlruns/`**: Servidor y base de datos local del registro de experimentos de MLflow.
* **`notebook/`**: Cuadernos de Jupyter utilizados en las fases de análisis exploratorio inicial.
* **`Dockerfile.*`**: Archivos de configuración de imágenes Docker independientes para la API, el Entrenador y el Monitor.
* **`docker-compose.yml`**: Orquestador de la infraestructura de contenedores.

---