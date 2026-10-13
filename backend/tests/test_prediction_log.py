import csv

import prediction_log


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_log_prediction_creates_file_with_header(tmp_path, monkeypatch):
    log_file = tmp_path / "prediction_logs.csv"
    monkeypatch.setattr(prediction_log, "PREDICTION_LOG_PATH", str(log_file))

    prediction_log.log_prediction({"MAKE": "TOYOTA"}, 1234.567, 0.05)

    rows = _read_rows(log_file)
    assert len(rows) == 1
    assert rows[0]["predicted_amount"] == "1234.57"
    assert rows[0]["latency_ms"] == "50.0"
    assert rows[0]["status"] == "success"


def test_log_prediction_appends_rows(tmp_path, monkeypatch):
    log_file = tmp_path / "prediction_logs.csv"
    monkeypatch.setattr(prediction_log, "PREDICTION_LOG_PATH", str(log_file))

    prediction_log.log_prediction({"MAKE": "A"}, 1.0, 0.01)
    prediction_log.log_prediction({"MAKE": "B"}, 2.0, 0.01)

    assert len(_read_rows(log_file)) == 2
