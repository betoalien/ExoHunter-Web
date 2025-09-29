# 📊 Data Card — ExoHunter

## 1. Dataset Overview
- **Name:** ExoHunter KOI Derived Dataset  
- **Source:** NASA Exoplanet Archive (Kepler Objects of Interest cumulative table)  
- **Version:** v1.0  
- **Date Generated:** 2025-09-27  
- **Purpose:**  
  Standardized dataset for exoplanet candidate classification and anomaly detection in ExoHunter.  
  Used to build features, train ML models, and flag anomalous signals for human review.

---

## 2. Composition
- **Instances:** ~8,000 KOI entries (varies by update).  
- **Features (columns):**
  - `koi_period` → Orbital period (days)  
  - `koi_prad` → Planet radius (R⊕)  
  - `koi_teq` → Equilibrium temperature (K)  
  - `koi_insol` → Insolation flux (F/F⊕)  
  - `koi_smass` → Stellar mass (M☉)  
  - `koi_srad` → Stellar radius (R☉)  
  - `koi_slogg` → Stellar surface gravity  
  - `disposition` → NASA disposition (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`)  
  - Derived features (ExoHunter): depth, semi-major axis, anomaly flags.  

- **Labels (ExoHunter classification):**
  - `confirmed_planet`
  - `candidate`
  - `rocky_candidate`
  - `super_earth_or_mini_neptune`
  - `hot_jupiter_candidate`
  - `habitable_zone_candidate`
  - `likely_false_positive`
  - `anomalous_signal`
  - `no_result`  

---

## 3. Collection Process
- **Source system:** NASA Exoplanet Archive → cumulative.csv.  
- **Transformation:**  
  - Standardized schema using `schemas/expected_columns.json`.  
  - Missing columns normalized (filled or flagged).  
  - Derived metrics computed (`a`, `Teq`, `Insol`, density).  
- **Update frequency:** NASA updates monthly; local refresh TBD.  
- **Storage:**  
  - Raw → `data/raw/`  
  - Interim → `data/interim/`  
  - Labeled → `data/labeled/`  
  - Outputs → `data/outputs/`  

---

## 4. Intended Use
- Training and evaluating ML models for:
  - Exoplanet candidate classification.  
  - Anomaly detection (*anomalous signals*).  
- Exploration and visualization for astrophysics research.  
- Supporting **human-in-the-loop labeling** with prioritization rules.

**Not intended for:**  
- Official astrophysical confirmation (use NASA’s confirmed tables).  
- Medical, financial, or unrelated domains.

---

## 5. Risks & Limitations
- **Bias:** Based on Kepler field → not representative of all stellar populations.  
- **Missing values:** Some stars/planets lack full parameters (`koi_prad`, `koi_teq`).  
- **Derived metrics assumptions:**  
  - Albedo assumed = 0.3 (Earth-like).  
  - Stellar properties may have uncertainties.  
- **Labels:** “Anomalous Signal” and “No Result” are internal ExoHunter categories, not NASA standards.  
- **Updates:** Dataset is a snapshot; may diverge from latest NASA updates.

---

## 6. Maintenance
- **Maintainer:** ExoHunter Project Team  
- **Contact:** <support@exohunter.dev>  
- **Update cadence:** Planned quarterly refresh aligned with NASA archive.  
- **Versioning:** Each refresh stored under `data/labeled/YYYYMMDD/`.

---

## 7. Ethics & Governance
- **Transparency:** Dataset structure, thresholds, and labeling rules are open in `configs/`.  
- **Accountability:** Models trained on this data must disclose limitations in `ml/model_card.md`.  
- **Fair Use:** Dataset is for research/educational use. NASA data is public domain, ExoHunter adds derived fields and categories.  

---

## 8. References
- NASA Exoplanet Archive: [https://exoplanetarchive.ipac.caltech.edu/](https://exoplanetarchive.ipac.caltech.edu/)  
- Google Data Cards: [https://research.google/pubs/data-cards/](https://research.google/pubs/data-cards/)  
- Partnership on AI: Data Documentation Initiative
