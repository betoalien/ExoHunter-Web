# ExoHunter – Web Application

**ExoHunter** is a Python + Flask + JavaScript project for analyzing exoplanet datasets (e.g., Kepler cumulative KOI files). It standardizes data, computes orbital/physical metrics, classifies candidates, and prepares feature sets for Machine Learning (ML). The app can be used both for scientific-style classification (*Confirmed, Candidate, False Positive*) and for unique anomaly detection categories (*Anomalous Signal*, *No Result/Invalid Data*).

---

## 🚀 Features

- Upload local CSV/Parquet files or load from cloud buckets.  
- Detect and normalize dataset columns to the KOI standard.  
- Automatically choose **Pandas** (small files) or **PySpark** (large files).  
- Compute derived metrics: semi-major axis, equilibrium temperature, transit depth, insolation, etc.  
- Classify each object into categories (Confirmed Planet, Candidate, Rocky, Gas Giant, Hot Jupiter, Habitable Zone Candidate, False Positive, Anomalous Signal, No Result).  
- Export processed results to CSV/JSON with classification, scores, and flags.  
- Manage labeling for ML training (human annotations, queues, and review).  
- Build and store feature vectors for anomaly detection and supervised ML.  
- Include Jupyter notebooks for exploration, baseline models, and supervised experiments.

---

## 📂 Project Structure

```bash 
web_app/
├─ app.py # Flask app entrypoint
├─ config.py # Global configuration
├─ docker-compose.yml
├─ Dockerfile
├─ .env / .env.example # Environment variables
│
├─ configs/ # YAML configs
│ ├─ thresholds.yaml # Classification thresholds
│ ├─ labeling.yaml # Labeling priorities
│ └─ mlflow.yaml # MLflow tracking
│
├─ data/ # Data lifecycle
│ ├─ raw/ # Uploaded raw files
│ ├─ interim/ # Normalized/intermediate data
│ ├─ labeled/ # Human-labeled datasets
│ ├─ uploads/ # User uploads
│ └─ outputs/ # Processed CSV/JSON reports
│
├─ feature_store/ # Versioned feature vectors
│ ├─ v1/
│ └─ v2/
│
├─ ml/ # Machine Learning components
│ ├─ artifacts/ # Models, scalers, encoders
│ ├─ eval/ # Evaluation results, metrics, plots
│ ├─ notebooks/ # Jupyter workflows
│ ├─ pipelines/ # Dataset creation & training scripts
│ ├─ feature_specs.yaml # Feature definitions
│ ├─ splits.yaml # Train/val/test split rules
│ ├─ data_card.md # Dataset documentation
│ ├─ model_card.md # Model documentation
│ └─ README.md # ML-specific guide
│
├─ routes/ # Flask routes (API endpoints)
│ ├─ upload.py # Upload files or bucket URLs
│ ├─ process.py # Run pipeline (compute + classify)
│ ├─ results.py # Fetch processed results
│ ├─ labels.py # Label management endpoints
│ ├─ features.py # Expose feature vectors
│ └─ health.py # Health check
│
├─ schemas/ # Data schemas and dictionaries
│ ├─ expected_columns.json
│ └─ data_dictionary.md
│
├─ services/ # Core logic modules
│ ├─ storage.py # File I/O (local/bucket)
│ ├─ schema_detect.py # Column normalization
│ ├─ engine_select.py # Pandas vs Spark choice
│ ├─ load_dataframe.py # Safe data loading
│ ├─ compute_metrics.py # Derived metrics (a, Teq, F, etc.)
│ ├─ feature_builder.py # Build ML feature vectors
│ ├─ classify.py # Classification rules + categories
│ ├─ export_report.py # Generate CSV/JSON outputs
│ ├─ labeling_queue.py # Manage items for labeling
│ └─ logging_utils.py # Logging helpers
│
├─ workers/ # Background processing
│ ├─ spark_session.py
│ └─ tasks.py
│
├─ templates/ # HTML templates
│ ├─ index.html
│ └─ results.html
│
├─ static/ # Frontend assets
│ ├─ css/styles.css
│ └─ js/app.js
│
├─ tests/ # Unit tests
│ ├─ test_schema_detect.py
│ ├─ test_classify.py
│ └─ test_compute_metrics.py
└─ ...
```

---

## 🧪 Categories of Classification

- **confirmed_planet**  
- **candidate**  
- **likely_false_positive**  
- **rocky_candidate**  
- **super_earth_or_mini_neptune**  
- **hot_jupiter_candidate**  
- **habitable_zone_candidate**  
- **anomalous_signal** 🔮 (unique category for unusual or inconsistent data, possible exotic structures)  
- **no_result** ❌ (invalid or insufficient data)  

---

## ⚙️ Workflow

1. **Upload dataset** via `/api/upload` or UI.  
2. **Schema detection** maps columns to KOI standard.  
3. **Engine selection**: Pandas (small) or Spark (large).  
4. **Metrics computed** if missing (a, Teq, Insolation).  
5. **Classification** into categories with scores and flags.  
6. **Export report** (CSV/JSON).  
7. **Optional labeling**: push/review anomalies for ML.  
8. **Feature building**: store vectors in `feature_store/`.  
9. **ML training** (notebooks, pipelines).  

---

## 🔮 Machine Learning Prep

- Feature vectors stored in `feature_store/`.  
- Labels managed in `data/labeled/` with sources (human, heuristic, NASA disposition).  
- Baseline anomaly detection (Isolation Forest, LOF, etc.) in `ml/notebooks/03_baselines.ipynb`.  
- Supervised classifiers (XGBoost, LightGBM) in `ml/notebooks/04_supervised.ipynb`.  
- Model artifacts tracked in `ml/artifacts/` and documented in `ml/model_card.md`.  

---

## 🛠️ Setup

```bash
# Clone and enter project
git clone https://github.com/betoalien/ExoHunter-Web.git
cd ExoHunter-Web

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt

# Run Flask app
flask run


Optional:

Use Docker with docker-compose up.

Configure .env for storage backend (local, S3, GCS).

📜 Notes

Input format: CSV (Kepler cumulative) or Parquet.

Use Parquet for large-scale data.

Classification thresholds adjustable in configs/thresholds.yaml.

Labeling priorities editable in configs/labeling.yaml.
```

------
## Testing
Test this project, visit https://exohunter-60bacde2a816.herokuapp.com/

------

## Contact me

For questions or contributions, open an issue or PR on GitHub.
Email: conect@albertocardenas.com
Website: https://www.albertocardenas.com
