# tests/test_evaluation.py
import pytest
import os
import json
from sklearn.datasets import load_iris
from src.model import train_model
from src.evaluation import evaluate_model

@pytest.fixture(scope="module")
def trained_model_and_data():
    # Train a model once for all evaluation tests
    model, X_test, y_test = train_model()
    return model, X_test, y_test

def test_evaluate_model_creates_file(tmp_path, trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    output_path = tmp_path / "metrics.json"
    evaluate_model(model, X_test, y_test, output_path=str(output_path))
    assert output_path.exists()

def test_evaluate_model_metrics_content(trained_model_and_data):
    model, X_test, y_test = trained_model_and_data
    metrics = evaluate_model(model, X_test, y_test, output_path="temp_metrics.json")
    os.remove("temp_metrics.json") # Clean up temp file

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert 0 <= metrics["accuracy"] <= 1