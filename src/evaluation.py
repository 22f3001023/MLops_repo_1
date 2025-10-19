# src/evaluation.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

def evaluate_model(model, X_test, y_test, output_path="metrics.json"):
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    # For multi-class, use average='weighted' or 'macro'
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {output_path}")
    return metrics

if __name__ == "__main__":
    # This part would typically load a model and test data
    # For demonstration, let's simulate
    from src.model import train_model
    model, X_test, y_test = train_model() # Train a dummy model

    metrics = evaluate_model(model, X_test, y_test, output_path="metrics.json")
    print(metrics)