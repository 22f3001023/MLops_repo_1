# src/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def train_model(n_estimators=100, random_state=42):
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, path="models/iris_model.joblib"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="models/iris_model.joblib"):
    return joblib.load(path)

if __name__ == "__main__":
    model, _, _ = train_model()
    save_model(model)