import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.evaluation import evaluate_model

# Load the trained model
model = joblib.load('models/iris_model.joblib')

# Load the data
iris = load_iris()
X, y = iris.data, iris.target

# Split for evaluation/testing
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run evaluation and write metrics to file
evaluate_model(model, X_test, y_test, output_path='metrics.json')
