import yaml
import pickle
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score


def main():
    df = pd.read_csv("data/processed/test_features.csv")

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

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)

    print("✅ model_evaluation done")


if __name__ == "__main__":
    main()