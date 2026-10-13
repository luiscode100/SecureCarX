import os
import argparse

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import optuna
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer

from ingestion import load_data
from features import (
    clean_data,
    filter_training_data,
    add_target,
    select_features,
    fit_target_encoding,
    apply_target_encoding,
    one_hot_encode,
)
from config import (
    MLFLOW_URI,
    MLFLOW_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    RAW_DATA_PATH,
    MODELS_DIR,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"MLflow tracking URI configurado en {MLFLOW_URI}.")
except Exception as e:
    print(f"Aviso: Error configurando MLflow ({e}).")


def run_training(input_path, model_dir, n_trials=30):
    active_run = None
    try:
        active_run = mlflow.start_run()
        print("Iniciada sesión en MLflow.")
    except Exception as e:
        print(f"No se pudo iniciar sesión en MLflow (Error: {e}). El proceso continuará en local.")

    print(f"Cargando datos desde: {input_path}")
    if not os.path.exists(input_path):
        print(f"Error: El archivo {input_path} no existe.")
        return

    raw_df = load_data(input_path)
    cleaned_df = clean_data(raw_df)
    training_df = filter_training_data(cleaned_df)

    target_df = add_target(training_df)
    feature_df = select_features(target_df)
    groups = target_df['OBJECT_ID'].values
    X = feature_df.drop(columns=['CLAIM_PAID_LOG'], errors='ignore')
    y = feature_df['CLAIM_PAID_LOG']

    # Split honesto: los vehículos (OBJECT_ID) de test no aparecen en train.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

    # Target encoding de alta cardinalidad, ajustado SOLO con train.
    encoding = fit_target_encoding(pd.concat([X_train_raw, y_train], axis=1))
    X_train = one_hot_encode(apply_target_encoding(X_train_raw, encoding), skip_cols=encoding['cols'])
    X_test = one_hot_encode(apply_target_encoding(X_test_raw, encoding), skip_cols=encoding['cols'])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    def mae_euros(y_true_log, y_pred_log):
        return mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

    euro_scorer = make_scorer(mae_euros, greater_is_better=False)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=25),
            'max_depth': trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20, 30]),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(
            candidate, X_train, y_train,
            cv=GroupKFold(n_splits=3), groups=groups_train,
            scoring=euro_scorer, n_jobs=1,
        )
        return -scores.mean()

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"Mejores hiperparámetros: {best_params}")
    print(f"Mejor MAE en CV (euros): {study.best_value:.2f} €")

    model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    y_pred_euros = np.expm1(y_pred_log)
    y_test_euros = np.expm1(y_test)
    mae = mean_absolute_error(y_test_euros, y_pred_euros)
    r2 = r2_score(y_test, y_pred_log)
    print(f"Modelo optimizado evaluado. MAE: {mae:.2f} € | R2: {r2:.4f}")

    if active_run:
        try:
            for k, v in best_params.items():
                mlflow.log_param(k, v)
            mlflow.log_param("input_data_path", input_path)
            mlflow.log_param("optuna_n_trials", n_trials)
            mlflow.log_param("target_encoded_cols", ",".join(encoding['cols']))
            mlflow.log_param("split_strategy", "GroupShuffleSplit(OBJECT_ID)")
            mlflow.log_metric("mae_euros", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("cv_best_mae_euros", study.best_value)
            for t in study.trials:
                if t.value is not None:
                    with mlflow.start_run(nested=True):
                        mlflow.log_params(t.params)
                        mlflow.log_metric("mae_euros_cv", t.value)
            mlflow.sklearn.log_model(model, "model", registered_model_name=REGISTERED_MODEL_NAME)
            mlflow.end_run()
        except Exception as e:
            print(f"Error al registrar datos en MLflow: {e}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_model.joblib")
    columns_path = os.path.join(model_dir, "model_columns.joblib")
    encodings_path = os.path.join(model_dir, "model_encodings.joblib")
    joblib.dump(model, model_path)
    joblib.dump(X_train.columns.tolist(), columns_path)
    joblib.dump(encoding, encodings_path)
    print(f"MODELO GUARDADO LOCALMENTE EN: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Entrenamiento SecureCarX")
    parser.add_argument("--input", type=str, default=RAW_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=MODELS_DIR)
    parser.add_argument("--n_trials", type=int, default=int(os.getenv("N_TRIALS", "30")))
    args = parser.parse_args()
    run_training(input_path=args.input, model_dir=args.output_dir, n_trials=args.n_trials)
