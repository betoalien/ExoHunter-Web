# services/classify.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_thresholds(thresholds_path: Optional[str]) -> Dict[str, Any]:
    """Load YAML thresholds with safe defaults."""
    defaults: Dict[str, Any] = {
        "planet_radius": {
            "rocky_max": 2.0,
            "super_earth_max": 6.0,
            "gas_giant_min": 6.0,
            "hot_jupiter_min": 8.0,
        },
        "orbital_period": {"hot_jupiter_max": 10},
        "temperature": {"hot_min": 1000, "habitable_min": 180, "habitable_max": 310},
        "insolation": {"habitable_min": 0.5, "habitable_max": 2.0},
        "anomalous": {
            "max_radius": 30,
            "max_depth": 0.05,
            "duration_ratio_low": 0.5,
            "duration_ratio_high": 1.8,
            "teq_low": 80,
        },
    }
    if not thresholds_path:
        return defaults
    try:
        import yaml  # type: ignore
        with open(thresholds_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        # shallow merge
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(defaults.get(k), dict):
                defaults[k].update(v)
            else:
                defaults[k] = v
        return defaults
    except Exception:
        return defaults


# -----------------------------
# Utilities
# -----------------------------
def _get(df_row: Dict[str, Any], name: str) -> Optional[float]:
    v = df_row.get(name)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except Exception:
        return None


def _append_flag(flags: List[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _normalize_disposition_to_category(disp_raw: Optional[str]) -> Optional[str]:
    """
    Normaliza disposiciones de NASA a nuestras categorías base para comparación simple.
      - CONFIRMED -> confirmed_planet
      - CANDIDATE -> candidate
      - FALSE POSITIVE -> likely_false_positive
    Cualquier otra cosa o vacío -> None
    """
    if not disp_raw:
        return None
    s = str(disp_raw).strip().upper()
    if s == "CONFIRMED":
        return "confirmed_planet"
    if s == "CANDIDATE":
        return "candidate"
    if s == "FALSE POSITIVE":
        return "likely_false_positive"
    return None


# -----------------------------
# Core rule evaluation
# -----------------------------
def _evaluate_rules(row: Dict[str, Any], th: Dict[str, Any]) -> Tuple[str, float, List[str]]:
    """
    Returns (category, score, flags) for one row.
    Score is a simple heuristic [0,1]; you can replace with ML later.
    """
    flags: List[str] = []
    category = "no_result"
    score = 0.0

    # Extract variables (optional/missing-safe)
    period = _get(row, "koi_period")
    prad = _get(row, "koi_prad")
    teq = _get(row, "koi_teq")
    insol = _get(row, "koi_insol")
    depth = _get(row, "transit_depth")
    duration_ratio = _get(row, "duration_ratio")  # if your compute_metrics injects it

    # Quick missing-data gate
    if prad is None and period is None and teq is None and insol is None:
        _append_flag(flags, "FLAG_NO_FEATURES")
        return ("no_result", 0.0, flags)

    # NASA disposition shortcut (if present)
    disp = str(row.get("koi_disposition") or "").upper().strip()
    if disp == "CONFIRMED":
        return ("confirmed_planet", 0.95, flags)
    if disp == "FALSE POSITIVE":
        return ("likely_false_positive", 0.9, flags)
    # If CANDIDATE, we still run rules and possibly refine.

    # --- Anomalous Signal checks (custom) ---
    an = th.get("anomalous", {})
    if prad is not None and prad > float(an.get("max_radius", 30)):
        _append_flag(flags, "FLAG_RADIUS_TOO_LARGE")
    if depth is not None and depth > float(an.get("max_depth", 0.05)):
        _append_flag(flags, "FLAG_ANOMALOUS_DEPTH")
    if duration_ratio is not None:
        if duration_ratio < float(an.get("duration_ratio_low", 0.5)):
            _append_flag(flags, "FLAG_DURATION_TOO_SHORT")
        if duration_ratio > float(an.get("duration_ratio_high", 1.8)):
            _append_flag(flags, "FLAG_DURATION_TOO_LONG")
    if teq is not None and depth is not None:
        if teq < float(an.get("teq_low", 80)) and depth > 0.01:
            _append_flag(flags, "FLAG_COLD_DEEP_TRANSIT")

    # If we flagged anomalies fuertes, promueve a anomalous_signal
    strong_anomaly_flags = {"FLAG_RADIUS_TOO_LARGE", "FLAG_ANOMALOUS_DEPTH", "FLAG_COLD_DEEP_TRANSIT"}
    strong_hits = sum(1 for fl in flags if fl in strong_anomaly_flags)
    if strong_hits >= 1 or ("FLAG_DURATION_TOO_LONG" in flags and "FLAG_ANOMALOUS_DEPTH" in flags):
        category = "anomalous_signal"
        score = 0.6 + 0.1 * strong_hits  # 0.6 to 0.8
        score = min(score, 0.9)

    # --- Habitability-ish check ---
    temp_cfg = th.get("temperature", {})
    hz_min = float(temp_cfg.get("habitable_min", 180))
    hz_max = float(temp_cfg.get("habitable_max", 310))
    ins_cfg = th.get("insolation", {})
    ins_min = float(ins_cfg.get("habitable_min", 0.5))
    ins_max = float(ins_cfg.get("habitable_max", 2.0))

    in_hab_temp = teq is not None and (hz_min <= teq <= hz_max)
    in_hab_ins = insol is not None and (ins_min <= insol <= ins_max)

    if in_hab_temp or in_hab_ins:
        if category != "anomalous_signal":
            category = "habitable_zone_candidate"
            score = max(score, 0.55)
            if in_hab_temp and in_hab_ins:
                score = max(score, 0.7)

    # --- Radius + Period based types ---
    r_cfg = th.get("planet_radius", {})
    p_cfg = th.get("orbital_period", {})

    rocky_max = float(r_cfg.get("rocky_max", 2.0))
    super_earth_max = float(r_cfg.get("super_earth_max", 6.0))
    hot_jup_min = float(r_cfg.get("hot_jupiter_min", 8.0))
    hot_jup_pmax = float(p_cfg.get("hot_jupiter_max", 10.0))
    hot_min_teq = float(temp_cfg.get("hot_min", 1000))

    if prad is not None:
        if prad < rocky_max:
            if category not in ("anomalous_signal", "confirmed_planet"):
                category = "rocky_candidate"
                score = max(score, 0.6)
        elif prad < super_earth_max:
            if category not in ("anomalous_signal", "confirmed_planet"):
                category = "super_earth_or_mini_neptune"
                score = max(score, 0.55)
        elif prad >= hot_jup_min and (
            (period is not None and period <= hot_jup_pmax) or (teq is not None and teq >= hot_min_teq)
        ):
            if category != "anomalous_signal":
                category = "hot_jupiter_candidate"
                score = max(score, 0.65)

    # --- Candidate fallback ---
    if category in ("no_result",) or (
        disp == "CANDIDATE"
        and category
        not in (
            "anomalous_signal",
            "habitable_zone_candidate",
            "rocky_candidate",
            "super_earth_or_mini_neptune",
            "hot_jupiter_candidate",
        )
    ):
        if any(v is not None for v in (period, prad, teq, insol)):
            category = "candidate"
            score = max(score, 0.5)

    # --- No result guard ---
    if category == "no_result":
        _append_flag(flags, "FLAG_INSUFFICIENT_DATA")

    # Clamp score
    score = float(max(0.0, min(1.0, score)))

    # Flags for missing important fields
    if prad is None:
        _append_flag(flags, "FLAG_MISSING_PRAD")
    if period is None:
        _append_flag(flags, "FLAG_MISSING_PERIOD")
    if teq is None:
        _append_flag(flags, "FLAG_MISSING_TEQ")

    return category, score, flags


# -----------------------------
# Public API
# -----------------------------
def classify_df(df, thresholds_path: Optional[str] = None):
    """
    Classify an input DataFrame and return a pandas DataFrame with:
      - object_id
      - category
      - score
      - flags (semicolon-separated)
      - koi_disposition (original, si venía; None si no)
      - disposition_compare: missing | match | mismatch
      - is_disposition_match: True | False | None
      - color_hint: green | black | red
      - koi_period, koi_prad, koi_teq, koi_insol (si están)

    Notes:
      - Works with a Pandas DataFrame; if Spark DF is passed, will attempt .toPandas().
      - thresholds_path points to configs/thresholds.yaml
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas is required for classify_df: {e}")

    # Convert Spark DF to Pandas if needed
    if not hasattr(df, "to_dict") and hasattr(df, "toPandas"):
        df = df.toPandas()

    if not hasattr(df, "to_dict"):
        raise TypeError("classify_df expects a Pandas DataFrame or Spark DataFrame.")

    th = _load_thresholds(thresholds_path)

    # Ensure object_id exists; if not, synthesize
    if "object_id" not in df.columns:
        df = df.copy()
        df["object_id"] = [f"OBJ-{i}" for i in range(len(df))]

    # Columns we may want to echo back
    keep_cols = [c for c in ("object_id", "koi_period", "koi_prad", "koi_teq", "koi_insol") if c in df.columns]

    # Iteración
    # Nota: incluimos koi_disposition si existe para eco y comparación
    access_cols = keep_cols + [c for c in ("koi_disposition", "transit_depth", "duration_ratio") if c in df.columns]
    # Evita KeyError si ninguna de las extras existe
    it = df[access_cols].to_dict(orient="records")

    records: List[Dict[str, Any]] = []
    for row in it:
        category, score, flags = _evaluate_rules(row, th)

        disp_raw = row.get("koi_disposition")
        disp_norm = _normalize_disposition_to_category(disp_raw)

        # Semáforo de comparación
        if disp_raw is None or (isinstance(disp_raw, float) and math.isnan(disp_raw)) or str(disp_raw).strip() == "":
            disposition_compare = "missing"
            is_disposition_match: Optional[bool] = None
            color_hint = "green"  # Falta en CSV → aportamos nosotros
        else:
            if disp_norm is not None and disp_norm == category:
                disposition_compare = "match"
                is_disposition_match = True
                color_hint = "black"  # Coincide
            else:
                disposition_compare = "mismatch"
                is_disposition_match = False
                color_hint = "red"  # No coincide (p.ej., CSV: candidate, nosotros: anomalous_signal)

        rec = {
            **{k: row.get(k) for k in keep_cols},
            "koi_disposition": disp_raw,
            "category": category,
            "score": score,
            "flags": ";".join(flags) if flags else "",
            "disposition_compare": disposition_compare,
            "is_disposition_match": is_disposition_match,
            "color_hint": color_hint,
        }
        records.append(rec)

    # Columnas de salida ordenadas
    out_cols = keep_cols + [
        "koi_disposition",
        "category",
        "score",
        "flags",
        "disposition_compare",
        "is_disposition_match",
        "color_hint",
    ]

    out = pd.DataFrame.from_records(records, columns=out_cols)
    return out


# Backwards-compat convenience
def classify(df):
    """Alias to classify_df without thresholds path (uses defaults)."""
    return classify_df(df, thresholds_path=None)
