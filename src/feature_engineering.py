import pandas as pd
import os

def main():
    train = pd.read_csv("data/processed/train_features.csv")
    test = pd.read_csv("data/processed/test_features.csv")

    # (Optional feature logic can be added here)

    os.makedirs("data/processed", exist_ok=True)

    train.to_csv("data/processed/train_features.csv", index=False)
    test.to_csv("data/processed/test_features.csv", index=False)

    print("✅ Feature engineering completed")

if __name__ == "__main__":
    main()