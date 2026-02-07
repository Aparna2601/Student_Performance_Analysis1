import yaml
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


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
    import io
    from io import StringIO
    df = pd.read_csv(StringIO(cleaned_text), sep=";")
    
    return df


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    df = read_data("data/processed/train_features.csv")

    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows and columns with all NaN values
    df = df.dropna(axis=0, how='all')
    df = df.dropna(axis=1, how='all')
    
    # Only keep numeric columns
    df = df.select_dtypes(include=["number"])

    if df.shape[1] < 2:
        raise ValueError(f"Not enough numeric columns: {df.shape[1]}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestRegressor(
        n_estimators=params["model_building"]["n_estimators"],
        random_state=params["model_building"]["random_state"]
    )

    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ model_building done")


if __name__ == "__main__":
    main()