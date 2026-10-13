import numpy as np
import pandas as pd

from features import (
    clean_data,
    filter_training_data,
    prepare_features,
    add_target,
    select_features,
    fit_target_encoding,
    apply_target_encoding,
    one_hot_encode,
)


def test_clean_data_parses_dates_and_computes_duration(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["INSR_BEGIN"])
    assert (cleaned["DURATION"] > 0).all()


def test_clean_data_drops_invalid_date_range(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "INSR_BEGIN"] = "2025-06-01"
    df.loc[0, "INSR_END"] = "2024-01-01"
    assert len(clean_data(df)) == len(sample_raw_df) - 1


def test_clean_data_drops_non_positive_premium(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "PREMIUM"] = 0
    assert len(clean_data(df)) == len(sample_raw_df) - 1


def test_clean_data_drops_null_make(sample_raw_df):
    df = sample_raw_df.copy()
    df.loc[0, "MAKE"] = None
    assert len(clean_data(df)) == len(sample_raw_df) - 1


def test_filter_training_data_keeps_only_claims(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    assert claims["CLAIM_PAID"].notnull().all()
    assert len(claims) <= len(sample_raw_df)


def test_prepare_features_creates_log_target_and_drops_noise(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    model_df = prepare_features(claims)
    assert "CLAIM_PAID_LOG" in model_df.columns
    for dropped in ["OBJECT_ID", "INSR_BEGIN", "INSR_END", "EFFECTIVE_YR", "CLAIM_PAID"]:
        assert dropped not in model_df.columns


def test_add_target_creates_log_column(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    result = add_target(claims)
    assert "CLAIM_PAID_LOG" in result.columns
    assert np.allclose(result["CLAIM_PAID_LOG"], np.log1p(result["CLAIM_PAID"]))


def test_select_features_drops_noise_columns(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    result = select_features(add_target(claims))
    for dropped in ["OBJECT_ID", "INSR_BEGIN", "INSR_END", "EFFECTIVE_YR", "CLAIM_PAID"]:
        assert dropped not in result.columns
    assert "MAKE" in result.columns


def test_target_encoding_maps_unseen_category_to_global_mean(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    df = select_features(add_target(claims))
    encoding = fit_target_encoding(df, cols=["MAKE"])

    assert "MAKE" in encoding["cols"]
    assert np.isclose(encoding["global_mean"], df["CLAIM_PAID_LOG"].mean())

    transformed = apply_target_encoding(df, encoding)
    assert pd.api.types.is_float_dtype(transformed["MAKE"])

    unseen = apply_target_encoding(pd.DataFrame({"MAKE": ["MARCA_INEXISTENTE"]}), encoding)
    assert np.isclose(unseen["MAKE"].iloc[0], encoding["global_mean"])


def test_one_hot_encode_skips_target_encoded_columns(sample_raw_df):
    claims = filter_training_data(clean_data(sample_raw_df))
    df = select_features(add_target(claims))
    encoding = fit_target_encoding(df, cols=["MAKE"])
    encoded = apply_target_encoding(df, encoding)

    result = one_hot_encode(encoded, skip_cols=encoding["cols"])
    assert "MAKE" in result.columns
    assert not any(c.startswith("MAKE_") for c in result.columns)
