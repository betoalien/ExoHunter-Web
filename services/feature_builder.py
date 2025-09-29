# services/feature_builder.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def _ensure_pandas(df):
    """Accept Spark DF by converting to Pandas if needed."""
    if hasattr(df, "toPandas") and not hasattr(df, "to_dict"):
        return df.toPandas()
    return df

# -----------------------------
# Specs loader
# -----------------------------
def _load_specs(specs_path: str | Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    p = Path(specs_path)
    if not p.exists():
        raise FileNotFoundError(f"Feature specs not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        specs = yaml.safe_load(f) or {}
    # Normalize keys
    for key in ("features", "derived", "categorical"):
        if key not in specs or specs[key] is None:
            specs[key] = []
    return specs

# -----------------------------
# Utilities
# -----------------------------
def _to_numeric_series(s, coerce: bool = True):
    import pandas as pd  # type: ignore
    if coerce:
        return pd.to_numeric(s, errors="coerce")
    return s

def _is_finite(x) -> bool:
    try:
        return (x is not None) and math.isfinite(float(x))
    except Exception:
        return False

def _impute_numeric(col, strategy: str = "median", fill_value: float | int = 0):
    import pandas as pd  # type: ignore
    s = _to_numeric_series(col, coerce=True)
    if strategy == "zero":
        return s.fillna(0)
    if strategy == "mean":
        m = s.mean(skipna=True)
        return s.fillna(m if _is_finite(m) else 0)
    if strategy == "median":
        m = s.median(skipna=True)
        return s.fillna(m if _is_finite(m) else 0)
    if strategy == "constant":
        return s.fillna(fill_value)
    # default
    return s.fillna(s.median(skipna=True))

def _impute_categorical(col, strategy: str = "most_frequent", fill_value: str = "UNK"):
    import pandas as pd  # type: ignore
    s = col.astype("object")
    if strategy == "most_frequent":
        if len(s) == 0:
            return s
        mode = s.mode(dropna=True)
        mv = mode.iloc[0] if not mode.empty else fill_value
        return s.fillna(mv)
    if strategy == "constant":
        return s.fillna(fill_value)
    # default
    return s.fillna(fill_value)

def _transform_numeric(s, transform: Optional[str]):
    if not transform or transform == "none":
        return s
    if transform == "log1p":
        # log(1+x), clamp negatives to NaN
        return s.apply(lambda v: math.log1p(v) if _is_finite(v) and v >= 0 else float("nan"))
    if transform == "standardize":
        m = s.mean(skipna=True)
        sd = s.std(skipna=True)
        sd = sd if _is_finite(sd) and sd != 0 else 1.0
        return s.apply(lambda v: (v - m) / sd if _is_finite(v) else float("nan"))
    # fallback
    return s

def _onehot(df, colname: str, prefix: Optional[str] = None) -> Tuple[Any, List[str]]:
    import pandas as pd  # type: ignore
    pref = prefix or colname
    dummies = pd.get_dummies(df[colname].astype("category"), prefix=pref, dummy_na=False)
    # Sort for determinism
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return dummies, list(dummies.columns)

# -----------------------------
# Main builder
# -----------------------------
def build(df_in, specs_path: str | Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Build ML features based on ml/feature_specs.yaml.

    Returns:
        features_df (pandas.DataFrame), meta: {"features": [...], "target": <name or None>}
    """
    import pandas as pd  # type: ignore

    specs = _load_specs(specs_path)
    df = _ensure_pandas(df_in).copy()

    # Keep an id if present for traceability (not part of features)
    id_col = None
    for cand in ("object_id", "rowid", "id"):
        if cand in df.columns:
            id_col = cand
            break

    # Target (do not include in features)
    target_name = None
    if isinstance(specs.get("target"), dict):
        target_name = specs["target"].get("name")

    feature_cols: List[str] = []
    out_df_parts: List[Any] = []

    # ---------- Numeric features ----------
    for feat in specs.get("features", []):
        name = feat.get("name")
        if not name:
            continue
        transform = (feat.get("transform") or "none").lower()
        impute = (feat.get("impute") or "median").lower()

        if name not in df.columns:
            # Create NaN column to be imputed
            df[name] = float("nan")

        series = _impute_numeric(df[name], strategy=impute)
        series = _transform_numeric(series, transform)

        col_out = name
        # For transformations we keep the same name (simpler pipelines)
        out_df_parts.append(series.rename(col_out))
        feature_cols.append(col_out)

    # ---------- Derived (already computed in compute_metrics) ----------
    for feat in specs.get("derived", []):
        name = feat.get("name")
        if not name:
            continue
        transform = (feat.get("transform") or "none").lower()
        impute = (feat.get("impute") or "zero").lower()

        if name not in df.columns:
            # Missing derived field → insert NaN and impute as specified
            df[name] = float("nan")

        series = _impute_numeric(df[name], strategy=impute)
        series = _transform_numeric(series, transform)

        out_df_parts.append(series.rename(name))
        feature_cols.append(name)

    # ---------- Categorical ----------
    categorical_generated: List[str] = []
    for cat in specs.get("categorical", []):
        name = cat.get("name")
        if not name:
            continue
        encoding = (cat.get("encoding") or "onehot").lower()
        impute = (cat.get("impute") or "most_frequent").lower()

        if name not in df.columns:
            df[name] = None  # will be imputed

        col = _impute_categorical(df[name], strategy=impute)

        if encoding == "onehot":
            dummies, cols = _onehot(pd.DataFrame({name: col}), name)
            out_df_parts.append(dummies)
            categorical_generated.extend(cols)
        else:
            # If another encoding is desired, keep raw (rare)
            out_df_parts.append(col.rename(name))
            categorical_generated.append(name)

    # ---------- Assemble features DF ----------
    if not out_df_parts:
        # No features defined → empty DF with id if available
        features_df = pd.DataFrame(index=df.index)
    else:
        features_df = pd.concat(out_df_parts, axis=1)

    # Ensure deterministic column order
    ordered_cols = feature_cols + categorical_generated
    features_df = features_df.reindex(columns=ordered_cols)

    # Attach id for traceability (not part of features list but useful downstream)
    if id_col:
        features_df.insert(0, id_col, df[id_col].values)

    meta = {
        "features": ordered_cols,
        "target": target_name,
        "row_count": int(features_df.shape[0]),
        "col_count": int(features_df.shape[1]),
    }
    return features_df, meta
