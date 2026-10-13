import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from features import (
    clean_data,
    filter_training_data,
    add_target,
    select_features,
    fit_target_encoding,
    apply_target_encoding,
    one_hot_encode,
)
from infer import ModelInference


@pytest.fixture
def trained_artifacts(tmp_path, sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    df = select_features(add_target(claims))
    X = df.drop(columns=["CLAIM_PAID_LOG"])
    y = df["CLAIM_PAID_LOG"]

    encoding = fit_target_encoding(pd.concat([X, y], axis=1), cols=["MAKE"])
    X = one_hot_encode(apply_target_encoding(X, encoding), skip_cols=encoding["cols"])

    model = RandomForestRegressor(n_estimators=3, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "rf_model.joblib"
    columns_path = tmp_path / "model_columns.joblib"
    encodings_path = tmp_path / "model_encodings.joblib"
    joblib.dump(model, model_path)
    joblib.dump(X.columns.tolist(), columns_path)
    joblib.dump(encoding, encodings_path)
    return str(model_path), str(columns_path), str(encodings_path)


VALID_SAMPLE = {
    "SEX": "1", "INSR_BEGIN": "2024-01-01", "INSR_END": "2025-01-01",
    "EFFECTIVE_YR": 2024, "INSR_TYPE": "1202", "INSURED_VALUE": 35000.0,
    "PREMIUM": 1200.0, "OBJECT_ID": 1, "PROD_YEAR": 2020, "SEATS_NUM": 5,
    "CARRYING_CAPACITY": 0.0, "TYPE_VEHICLE": "Automobile", "CCM_TON": 1600.0,
    "MAKE": "TOYOTA", "USAGE": "Private",
}


def test_predict_claim_returns_positive_float(trained_artifacts):
    model_path, columns_path, encodings_path = trained_artifacts
    engine = ModelInference(model_path=model_path, columns_path=columns_path, encodings_path=encodings_path)

    result = engine.predict_claim(VALID_SAMPLE)

    assert isinstance(result, float)
    assert result >= 0


def test_predict_claim_handles_unseen_category(trained_artifacts):
    model_path, columns_path, encodings_path = trained_artifacts
    engine = ModelInference(model_path=model_path, columns_path=columns_path, encodings_path=encodings_path)

    result = engine.predict_claim({**VALID_SAMPLE, "MAKE": "MARCA_NUNCA_VISTA"})

    assert isinstance(result, float)


def test_reload_returns_true_and_keeps_model_usable(trained_artifacts):
    model_path, columns_path, encodings_path = trained_artifacts
    engine = ModelInference(model_path=model_path, columns_path=columns_path, encodings_path=encodings_path)

    assert engine.reload() is True
    assert isinstance(engine.predict_claim(VALID_SAMPLE), float)


def test_predict_claim_rejects_invalid_input(trained_artifacts):
    model_path, columns_path, encodings_path = trained_artifacts
    engine = ModelInference(model_path=model_path, columns_path=columns_path, encodings_path=encodings_path)

    invalid = dict(VALID_SAMPLE, INSR_BEGIN="2025-01-01", INSR_END="2024-01-01")
    result = engine.predict_claim(invalid)

    assert isinstance(result, str)
