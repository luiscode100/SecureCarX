import pandas as pd
import os

from config import RAW_DATA_PATH

def load_data(file_path=RAW_DATA_PATH):
    """Loads the raw CSV data and removes duplicates."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    return df
