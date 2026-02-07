import yaml
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    df = pd.read_csv("data/processed/train_features.csv")

    # 🔑 KEEP ONLY NUMERIC COLUMNS
    df = df.select_dtypes(include=["number"])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestRegressor(
        n_estimators=params["model_building"]["n_estimators"],
        random_state=params["model_building"]["random_state"]
    )

    model.fit(X, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ model_building done")


if __name__ == "__main__":
    main()