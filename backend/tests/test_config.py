import importlib
import os

import config


def test_default_paths_are_absolute():
    assert os.path.isabs(config.DATA_DIR)
    assert os.path.isabs(config.MODELS_DIR)
    assert config.MLFLOW_URI.startswith("sqlite:///")


def test_mlflow_uri_default_points_to_sqlite_db():
    assert config.MLFLOW_URI == "sqlite:///" + os.path.join(config.MLRUNS_DIR, "mlflow.db")


def test_derived_paths_live_under_data_dir():
    assert config.RAW_DATA_PATH == os.path.join(config.DATA_DIR, "raw", "securecarx_dataset.csv")
    assert config.PREDICTION_LOG_PATH == os.path.join(config.DATA_DIR, "operational", "prediction_logs.csv")
    assert config.FEEDBACK_LOG_PATH == os.path.join(config.DATA_DIR, "operational", "feedback_logs.csv")
    assert config.TRAINING_DATASET_PATH == os.path.join(config.DATA_DIR, "processed", "training_with_feedback.csv")


def test_data_dir_can_be_overridden_by_env(monkeypatch, tmp_path):
    custom = str(tmp_path / "custom_data")
    monkeypatch.setenv("DATA_DIR", custom)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DATA_DIR == custom
        assert reloaded.RAW_DATA_PATH == os.path.join(custom, "raw", "securecarx_dataset.csv")
    finally:
        monkeypatch.delenv("DATA_DIR", raising=False)
        importlib.reload(config)
