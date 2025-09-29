# 🧠 Model Card — ExoHunter

## 1. Model Overview
- **Name:** ExoHunter Baseline Classifier  
- **Version:** v1.0  
- **Date Trained:** 2025-09-27  
- **Type:** Supervised multi-class classifier (scikit-learn baseline)  
- **Purpose:**  
  Classify exoplanet candidates (KOI entries) into astrophysical categories,  
  including *anomalous_signal* and *no_result* as ExoHunter-specific labels.  

---

## 2. Intended Use
- **Primary use case:**  
  - Automated triage of KOI datasets for researchers.  
  - Prioritize *candidates* and *anomalous_signal* objects for human review.  
  - Provide quick classification for visualization and exploration in ExoHunter web app.  

- **Not intended for:**  
  - Official astrophysical confirmation of exoplanets.  
  - Mission-critical scientific publications without human verification.  
  - Use outside the domain of astronomy/astrophysics.

---

## 3. Training Data
- **Dataset:** ExoHunter KOI Derived Dataset (see [`data_card.md`](./data_card.md))  
- **Features used:**  
  - Orbital period (`koi_period`), semi-major axis (`koi_sma`)  
  - Planet radius (`koi_prad`), equilibrium temperature (`koi_teq`), insolation (`koi_insol`)  
  - Stellar properties (`koi_smass`, `koi_srad`, `koi_slogg`)  
  - Derived metrics (`transit_depth`, `density_est`)  
  - Stellar type (categorical)  
- **Labels:**  
  - `confirmed_planet`, `candidate`, `rocky_candidate`, `super_earth_or_mini_neptune`,  
    `hot_jupiter_candidate`, `habitable_zone_candidate`, `likely_false_positive`,  
    `anomalous_signal`, `no_result`.  

---

## 4. Model Details
- **Framework:** scikit-learn  
- **Baseline algorithms tested:**  
  - Logistic Regression  
  - Random Forest Classifier  
  - Gradient Boosting  
- **Hyperparameters:** Default (baseline).  
- **Feature scaling:** Standardization (z-score) for continuous variables.  
- **Encoding:** One-hot for categorical (`stellar_type`).  

---

## 5. Evaluation
- **Dataset split:** 70% train / 15% validation / 15% test.  
- **Metrics:** (baseline placeholder, see `ml/eval/metrics.json`)  
  - Accuracy: 0.84  
  - Macro Precision: 0.81  
  - Macro Recall: 0.79  
  - Macro F1: 0.80  
- **Confusion matrix:** stored in [`ml/eval/confusion_plots/metrics.json`](./eval/confusion_plots/metrics.json).  

---

## 6. Limitations
- Labels depend on **KOI dataset quality** → noisy, missing values.  
- Derived metrics assume albedo = 0.3; biases habitability classification.  
- Rare categories (e.g., *anomalous_signal*) are underrepresented → possible class imbalance.  
- Model may overfit to Kepler field (F/G-type stars). Not guaranteed to generalize to other missions (TESS, JWST).  

---

## 7. Ethical Considerations
- **Transparency:** All thresholds and labeling rules are open in `configs/`.  
- **Accountability:** Results should always be validated by a human astrophysicist before publication.  
- **Bias:** Dataset covers limited stellar populations → risk of bias against rare systems.  
- **Fair Use:** For research/educational use; not for operational mission decisions.  

---

## 8. Future Work
- Improve handling of class imbalance with SMOTE/oversampling.  
- Incorporate uncertainty estimates (e.g., Bayesian models, dropout).  
- Test with alternative missions (TESS, JWST) for generalization.  
- Add deep learning models (PyTorch/TF) for comparison.  
- Integrate with MLflow (see [`mlflow.yaml`](../configs/mlflow.yaml)) for experiment tracking.  

---

## 9. References
- Mitchell et al. (2019). Model Cards for Model Reporting.  
- NASA Exoplanet Archive: [https://exoplanetarchive.ipac.caltech.edu/](https://exoplanetarchive.ipac.caltech.edu/)  
- ExoHunter Data Card: [`data_card.md`](./data_card.md)
