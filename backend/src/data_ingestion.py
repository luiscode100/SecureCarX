import pandas as pd
import os

def load_data(file_path):
    """Loads the raw CSV data and removes duplicates."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    return df

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "securecarx_dataset.csv")

    raw_data = load_data(DATA_PATH)
    print(f"Data loaded successfully. Rows: {raw_data.shape[0]}")