import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    test_size = params["data_ingestion"]["test_size"]

    input_path = "dataset/raw/data.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    os.makedirs("data/raw", exist_ok=True)

    train, test = train_test_split(df, test_size=test_size, random_state=42)

    train.to_csv("data/raw/train.csv", index=False)
    test.to_csv("data/raw/test.csv", index=False)

    print("✅ data_ingestion done")


if __name__ == "__main__":
    main()