# 📖 ExoHunter Data Dictionary

This document describes the key columns used in the **ExoHunter** project.  
It covers original KOI fields (from the NASA Exoplanet Archive cumulative.csv),  
derived metrics (computed in `services/compute_metrics.py`), and ExoHunter-specific fields.

---

## 🔹 Core KOI Columns (from NASA Exoplanet Archive)

| Column        | Type     | Units      | Description |
|---------------|----------|------------|-------------|
| `object_id`   | string   | —          | Unique identifier for the object (KOI name or synthetic ID). |
| `koi_period`  | float    | days       | Orbital period of planet candidate. |
| `koi_prad`    | float    | R⊕         | Planetary radius relative to Earth. |
| `koi_sma`     | float    | AU         | Semi-major axis of orbit. |
| `koi_teq`     | float    | K          | Estimated equilibrium temperature of the planet. |
| `koi_insol`   | float    | F⊕         | Insolation flux relative to Earth. |
| `koi_smass`   | float    | M☉         | Stellar mass relative to the Sun. |
| `koi_srad`    | float    | R☉         | Stellar radius relative to the Sun. |
| `koi_slogg`   | float    | log g      | Stellar surface gravity. |
| `koi_steff`   | float    | K          | Stellar effective temperature. |
| `koi_disposition` | string | —        | NASA pipeline disposition: `CANDIDATE`, `FALSE POSITIVE`, `CONFIRMED`. |

---

## 🔹 Derived Columns (computed in ExoHunter)

| Column            | Type   | Units | Description |
|-------------------|--------|-------|-------------|
| `transit_depth`   | float  | —     | Fractional flux drop δ = (Rp/Rs)². |
| `density_est`     | float  | g/cm³ | Approximate density (if mass proxy available). |
| `habitability_flag` | int  | {0,1} | Flag if planet lies within conservative habitable zone (Teq ~ 200–300K). |

---

## 🔹 ExoHunter Classification Fields

| Column      | Type    | Values | Description |
|-------------|---------|--------|-------------|
| `category`  | string  | `confirmed_planet`, `candidate`, `rocky_candidate`, `super_earth_or_mini_neptune`, `hot_jupiter_candidate`, `habitable_zone_candidate`, `likely_false_positive`, `anomalous_signal`, `no_result` | Final classification assigned by ExoHunter pipeline. |
| `score`     | float   | 0–1    | Confidence score assigned by rules/model. |
| `flags`     | string  | —      | Diagnostic notes, e.g., `"low_snr"`, `"missing_columns"`. |

---

## 🔹 Metadata Fields

| Column      | Type    | Description |
|-------------|---------|-------------|
| `source_file` | string | Path to the raw input file the object originated from. |
| `ts_ingest`   | string | Timestamp when the row was processed by ExoHunter. |
| `label`       | string | Human-in-the-loop label (if applied through `/api/labels/apply`). |

---

## 📝 Notes

- Not all columns will be present in every dataset.  
- Normalization of column names is handled in `services/schema_detect.py`.  
- Derived metrics rely on assumptions (e.g., **albedo = 0.3**) defined in configs.  
- ExoHunter-specific categories include **`anomalous_signal`** and **`no_result`**, inspired by anomalous/paranormal signal detection.

