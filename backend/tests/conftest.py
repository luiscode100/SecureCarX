import os
import sys

import pandas as pd
import pytest

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture
def sample_raw_df():
    """Pequeño dataset con el mismo esquema que data/raw/securecarx_dataset.csv."""
    return pd.DataFrame({
        "SEX": ["1", "2", "1", "2"],
        "INSR_BEGIN": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
        "INSR_END": ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"],
        "EFFECTIVE_YR": [2024, 2023, 2022, 2021],
        "INSR_TYPE": ["1202", "1203", "1202", "1203"],
        "INSURED_VALUE": [35000.0, 20000.0, 15000.0, 40000.0],
        "PREMIUM": [1200.0, 800.0, 600.0, 1500.0],
        "OBJECT_ID": [1, 2, 3, 4],
        "PROD_YEAR": [2020, 2018, 2015, 2022],
        "SEATS_NUM": [5, 4, 5, 2],
        "CARRYING_CAPACITY": [0.0, 0.0, 0.0, 0.0],
        "TYPE_VEHICLE": ["Automobile", "Automobile", "Automobile", "Motorcycle"],
        "CCM_TON": [1600.0, 1400.0, 1200.0, 800.0],
        "MAKE": ["TOYOTA", "FORD", "RENAULT", "SEAT"],
        "USAGE": ["Private", "Private", "Professional", "Private"],
        "CLAIM_PAID": [500.0, None, 1200.0, 300.0],
    })
