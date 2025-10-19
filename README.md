# MLops_repo_1

Week 4 MLops Assignment
# 🌸 Iris DVC MLOps Pipeline

This project demonstrates an end-to-end MLOps pipeline using the classic Iris dataset. It integrates data/model versioning, unit testing, and cloud-based CI/CD with GitHub Actions, DVC, and Google Cloud Storage.

## 🚀 Features

- **Automated data validation and model evaluation with pytest.**
- **Continuous Integration:** CI pipelines for both `dev` and `main` branches.
- **Model & data versioning with DVC**, tracked in a Google Cloud Storage bucket.
- **Unit and integration tests** are always run before a merge.
- **Comprehensive Markdown report** from each CI run is downloadable as an artifact.
- **Branching model:** `dev` for development, `main` for production releases.

## 🗂️ Project Structure

.
├── .github/workflows/ # CI/CD YAMLs for branch-specific automation
├── data/ # DVC-tracked data (placeholder)
├── models/ # DVC-tracked models (placeholder)
├── src/
│ ├── model.py # Model training
│ ├── data_validation.py # Data validation logic
│ ├── evaluation.py # Model evaluation logic
│ └── run_eval.py # Standalone evaluation runner
├── tests/
│ ├── test_data_validation.py # Unit tests for validation
│ └── test_evaluation.py # Unit tests for evaluation
├── params.yaml # Training parameters
├── requirements.txt # Dependencies
└── README.md # This file


## 🛠️ Pipeline Breakdown

1. **Train the model:**  
   `python src/model.py`

2. **Data validation:**  
   Included in both scripts and tested by `tests/test_data_validation.py`.

3. **Evaluation:**  
   `python run_eval.py` writes `metrics.json`.

4. **CI:**  
   On every push or PR to `dev` or `main`, GitHub Actions:
   - Installs dependencies
   - Pulls the latest data/model from DVC+GCS
   - Runs tests and evaluation
   - Generates a detailed `report.md` summary (available for download)

5. **DVC + Cloud:**  
   Data and model files are versioned and stored in GCS for reproducibility.

## ⚙️ Getting Started

- Clone the repository and install requirements.
- Set up DVC with your own GCP credentials and bucket if running outside CI.
- Use `python src/model.py` to train, `python run_eval.py` to evaluate.
- To see CI in action, push any changes to `dev` and watch the Actions tab!

## 📊 Reports

- Detailed CI reports are available as downloadable artifacts from the Actions tab in GitHub for every commit and PR.

## 🤝 Contributions

PRs and suggestions are welcome. This project is intended for educational MLOps demonstration.

---

**Author:**  
[Sanjay] · [IIT Madras Data Science](https://github.com/22f3001023/MLops_repo_1) · October 2025
