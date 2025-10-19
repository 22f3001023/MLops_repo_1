from sklearn.datasets import load_iris
import pandas as pd
import os

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

X, y = load_iris(return_X_y=True)
pd.DataFrame(X).to_csv('data/iris_data.csv', index=False)
pd.DataFrame(y, columns=['target']).to_csv('data/iris_target.csv', index=False)

print("Iris data saved to data/iris_data.csv and data/iris_target.csv")
