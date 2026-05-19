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
    # Quick test
    raw_data = load_data("data/securecarx_dataset.csv")
    print(f"Data loaded successfully. Rows: {raw_data.shape[0]}")