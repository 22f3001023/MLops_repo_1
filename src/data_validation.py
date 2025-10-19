# src/data_validation.py
import pandas as pd
from sklearn.datasets import load_iris

def validate_iris_data(X_data):
    # Basic checks for Iris dataset
    if not isinstance(X_data, pd.DataFrame):
        X_data = pd.DataFrame(X_data)

    expected_columns = 4
    if X_data.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} columns, but got {X_data.shape[1]}")

    # Check for missing values
    if X_data.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in the dataset.")

    # Check data types (assuming numeric)
    if not all(X_data.apply(lambda col: pd.api.types.is_numeric_dtype(col))):
         raise ValueError("Non-numeric data types found in the dataset.")

    print("Data validation passed.")
    return True

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    try:
        validate_iris_data(X)
    except ValueError as e:
        print(f"Data validation failed: {e}")
        