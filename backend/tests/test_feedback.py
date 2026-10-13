import pandas as pd
import pytest

import feedback


@pytest.fixture
def feedback_env(tmp_path, monkeypatch):
    path = tmp_path / "feedback_logs.csv"
    monkeypatch.setattr(feedback, "FEEDBACK_LOG_PATH", str(path))
    return path


def test_log_feedback_creates_file_with_label(feedback_env, sample_raw_df):
    record = sample_raw_df.iloc[0].to_dict()
    record["CLAIM_PAID"] = 999.0

    feedback.log_feedback(record)

    df = pd.read_csv(feedback_env)
    assert len(df) == 1
    assert df.loc[0, "CLAIM_PAID"] == 999.0
    assert "timestamp" in df.columns


def test_log_feedback_appends_rows(feedback_env, sample_raw_df):
    record = sample_raw_df.iloc[0].to_dict()
    feedback.log_feedback({**record, "CLAIM_PAID": 100.0})
    feedback.log_feedback({**record, "CLAIM_PAID": 200.0})
    assert len(pd.read_csv(feedback_env)) == 2


def test_build_training_dataset_combines_reference_and_feedback(tmp_path, monkeypatch, feedback_env, sample_raw_df):
    raw = tmp_path / "raw.csv"
    sample_raw_df.to_csv(raw, index=False)
    monkeypatch.setattr(feedback, "RAW_DATA_PATH", str(raw))

    feedback.log_feedback({**sample_raw_df.iloc[0].to_dict(), "CLAIM_PAID": 123.0})

    out = feedback.build_training_dataset(str(tmp_path / "combined.csv"))
    combined = pd.read_csv(out)
    assert len(combined) == len(sample_raw_df) + 1


def test_build_training_dataset_without_feedback_returns_reference(tmp_path, monkeypatch, feedback_env, sample_raw_df):
    raw = tmp_path / "raw.csv"
    sample_raw_df.to_csv(raw, index=False)
    monkeypatch.setattr(feedback, "RAW_DATA_PATH", str(raw))

    out = feedback.build_training_dataset(str(tmp_path / "combined.csv"))
    assert len(pd.read_csv(out)) == len(sample_raw_df)
