import os
import pandas as pd


def main():
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train_features.csv", index=False)
    test.to_csv("data/processed/test_features.csv", index=False)

    print("✅ data_preprocessing done")


if __name__ == "__main__":
    main()