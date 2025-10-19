# tests/test_data_validation.py
import pytest
import pandas as pd
from sklearn.datasets import load_iris
from src.data_validation import validate_iris_data

def test_validate_iris_data_success():
    iris = load_iris()
    X = iris.data
    assert validate_iris_data(X)

def test_validate_iris_data_missing_values():
    iris = load_iris()
    X_df = pd.DataFrame(iris.data)
    X_df.iloc[0, 0] = None  # Introduce a missing value
    with pytest.raises(ValueError, match="Missing values found"):
        validate_iris_data(X_df)

def test_validate_iris_data_wrong_columns():
    iris = load_iris()
    X_df = pd.DataFrame(iris.data)
    X_df['extra_col'] = 1 # Add an extra column
    with pytest.raises(ValueError, match="Expected 4 columns"):
        validate_iris_data(X_df)

def test_validate_iris_data_non_numeric():
    iris = load_iris()
    X_df = pd.DataFrame(iris.data)
    X_df.iloc[0, 0] = "text" # Introduce non-numeric data
    with pytest.raises(ValueError, match="Non-numeric data types found"):
        validate_iris_data(X_df)