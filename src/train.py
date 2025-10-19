# src/train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    # Load IRIS dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split dataset for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make sure model directory exists
    os.makedirs("models", exist_ok=True)
    # Save trained model
    joblib.dump(model, "models/iris_model.joblib")
    print("Model saved to models/iris_model.joblib")

    # Optionally, save test split for evaluation scripts
    import pandas as pd
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_test, columns=["target"]).to_csv("data/y_test.csv", index=False)
    print("Test data saved to data/X_test.csv and data/y_test.csv")

if __name__ == "__main__":
    main()
