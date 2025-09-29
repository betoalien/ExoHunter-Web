# services/export_report.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _is_pandas_df(df: Any) -> bool:
    return hasattr(df, "to_dict") and hasattr(df, "to_csv")


def _to_pandas(df: Any):
    """Accept Spark or Pandas; return Pandas DataFrame."""
    if _is_pandas_df(df):
        return df
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError("export_report requires a pandas.DataFrame or a Spark DataFrame convertible via .toPandas().")


def _stringify_flags(v: Any) -> str:
    # Accept list/tuple/str/None → str
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ";".join(map(str, v))
    return str(v)


def _coerce_columns(df, preferred_order: Optional[Iterable[str]] = None):
    """
    - Convert 'flags' to a friendly string
    - Reorder columns with preferred_order (keeping any extras at the end)
    """
    pdf = _to_pandas(df).copy()

    if "flags" in pdf.columns:
        pdf["flags"] = pdf["flags"].apply(_stringify_flags)

    if preferred_order:
        cols = list(pdf.columns)
        front = [c for c in preferred_order if c in cols]
        rest = [c for c in cols if c not in front]
        pdf = pdf[front + rest]

    return pdf


def _write_csv(df, path: Path) -> None:
    df.to_csv(path, index=False)


def _rows_json_safe(pdf) -> List[Dict[str, Any]]:
    """
    Return a list of dicts with NaN/Inf replaced by None (→ null in JSON).
    """
    try:
        import pandas as pd  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        # Fallback: to_dict; json.dumps below will fail if NaN remains, but we surface the error
        return pdf.to_dict(orient="records")

    # Reemplaza infs por NaN primero
    pdf = pdf.replace([np.inf, -np.inf], np.nan)
    # Convierte todos los NaN a None (manteniendo tipos)
    pdf = pdf.where(pd.notnull(pdf), None)

    return pdf.to_dict(orient="records")


def _write_json_rows(rows: List[Dict[str, Any]], path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
    payload = {"rows": rows}
    if meta:
        payload.update(meta)
    # allow_nan=False asegura que no se cuele NaN en el JSON
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _latest_pointers(out_dir: Path, csv_path: Path, json_path: Path) -> None:
    latest_csv = out_dir / "report_latest.csv"
    latest_json = out_dir / "report_latest.json"
    try:
        latest_csv.write_bytes(csv_path.read_bytes())
        latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        # non-fatal
        pass


# -----------------------------
# Public API
# -----------------------------
def write_outputs(
    results_df: Any,
    out_dir: str,
    timestamp: Optional[str] = None,
    preferred_order: Optional[Iterable[str]] = (
        "object_id",
        # Disposición original + nuestra categoría y comparadores primero
        "koi_disposition",
        "category",
        "disposition_compare",
        "is_disposition_match",
        "color_hint",
        # Score/flags y métricas clave
        "score",
        "flags",
        "koi_period",
        "koi_prad",
        "koi_teq",
        "koi_insol",
    ),
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Write CSV and JSON outputs for the classified results.

    Args:
        results_df: Pandas DataFrame (or Spark DF) with at least:
                    object_id, category, score, flags (optional koi_* columns)
                    (y, de ser posible: koi_disposition, disposition_compare, is_disposition_match, color_hint)
        out_dir: destination directory (e.g., data/outputs)
        timestamp: explicit timestamp; if None, one will be generated
        preferred_order: columns to place at the beginning in the CSV
        extra_meta: additional metadata to include in JSON (e.g., {"generated_by": "api/process"})

    Returns:
        {"csv_path": "<...>", "json_path": "<...>"}
    """
    out_dir_path = Path(out_dir)
    _ensure_dir(out_dir_path)

    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir_path / f"report_{ts}.csv"
    json_path = out_dir_path / f"report_{ts}.json"

    # Normalize DF and order columns
    pdf = _coerce_columns(results_df, preferred_order=preferred_order)

    # CSV
    _write_csv(pdf, csv_path)

    # JSON (rows + meta), SANITIZANDO NaN/Inf → null
    meta = {"timestamp": ts, "row_count": int(getattr(pdf, "shape", [0, 0])[0])}
    if extra_meta:
        meta.update(extra_meta)

    rows = _rows_json_safe(pdf)
    _write_json_rows(rows, json_path, meta=meta)

    # Update "latest" pointers
    _latest_pointers(out_dir_path, csv_path, json_path)

    return {"csv_path": str(csv_path), "json_path": str(json_path)}
