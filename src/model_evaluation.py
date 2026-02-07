import yaml
import pickle
import pandas as pd
import os
import json
from sklearn.metrics import mean_squared_error, r2_score


def read_data(filepath):
    """Read CSV with proper handling of semicolon delimiters and quotes"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    
    # Clean header: remove outer quotes and unescape inner quotes
    header = lines[0]
    if header.startswith('"') and header.endswith('"'):
        header = header[1:-1]  # Remove outer quotes
    header = header.replace('""', '"')  # Unescape doubled quotes
    
    # Recombine cleaned header with data rows
    cleaned_lines = [header] + lines[1:]
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Parse using io.StringIO
    from io import StringIO
    df = pd.read_csv(StringIO(cleaned_text), sep=";")
    
    return df


def main():
    df = read_data("data/processed/test_features.csv")

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with NaN and drop any entirely NaN columns
    df = df.dropna()
    df = df.dropna(axis=1, how='all')
    
    # 🔑 SAME NUMERIC FILTER
    df = df.select_dtypes(include=["number"])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)

    metrics = {
        "mse": float(mean_squared_error(y, preds)),
        "r2": float(r2_score(y, preds))
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f)

    print("✅ model_evaluation done")


if __name__ == "__main__":
    main()