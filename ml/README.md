# 🧠 ExoHunter — Machine Learning Module

This folder contains all **machine learning components** of the ExoHunter project.  
It provides the pipelines, feature definitions, evaluation outputs, and documentation required to train and validate models.

---

## 📂 Structure

```bash
ml/
├─ artifacts/ # Models, scalers, encoders saved from training
│ ├─ models/
│ ├─ scalers_encoders/
│
├─ eval/ # Evaluation results
│ ├─ confusion_plots/ # Confusion matrices, plots, metrics
│ │ └─ metrics.json
│
├─ notebooks/ # Jupyter notebooks for manual exploration
│
├─ pipelines/ # Scripts for dataset creation & training
│ ├─ make_dataset.py # Build feature sets from raw KOI data
│ └─ (train.py, eval.py) [to be added]
│
├─ feature_specs.yaml # Defines which features are used & how
├─ splits.yaml # Train/validation/test split rules
├─ data_card.md # Documentation of dataset (composition, use, risks)
├─ model_card.md # Documentation of trained models (metrics, limits)
└─ README.md # This file
```

---

## ⚙️ Pipelines

### 1. Make Dataset
Convert raw KOI data (`.csv` or `.parquet`) into a **feature set**.

```bash
python ml/pipelines/make_dataset.py \
  --input data/raw/cumulative.csv \
  --feature-version v1 \
  --output-dir feature_store/v1 \
  --feature-specs ml/feature_specs.yaml \
  --schema-map schemas/expected_columns.json \
  --engine auto \
  --albedo 0.3 \
  --log-level INFO
```
Outputs:

Feature Parquet in feature_store/<version>/

Manifest JSON with metadata (e.g., timestamp, features used)

2. Train Model (to be added)

Will train baseline classifiers (scikit-learn, PyTorch, etc.)

Save models in ml/artifacts/models/

Track metrics with MLflow (configs/mlflow.yaml)

3. Evaluate Model (to be added)

Will produce metrics in ml/eval/confusion_plots/metrics.json

Export confusion matrices and performance plots

📑 Documentation

Dataset details: data_card.md

Model details: model_card.md

Feature specs: feature_specs.yaml

Experiment tracking: configured via ../configs/mlflow.yaml

🔎 Notes

notebooks/ are manual exploration notebooks — not part of automated pipelines.

metrics.json inside eval/ is generated automatically after evaluation (do not edit manually).

feature_specs.yaml and splits.yaml can be updated to change feature engineering or dataset splitting without touching code.

🛠 Dependencies

Core ML requirements are listed in project-level requirements.txt, including:

pandas, pyarrow, scikit-learn, mlflow, pyspark

See main README.md
 for installation.

👥 Maintainers

ExoHunter Project Team
Contact: support@exohunter.dev