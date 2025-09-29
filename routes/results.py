# routes/results.py
from __future__ import annotations

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, current_app, send_file

results_bp = Blueprint("results", __name__, url_prefix="/api/result")


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, code: int = 400, **extra):
    return jsonify({"error": message, **extra}), code


def _root() -> Path:
    base = current_app.config.get("BASE_DIR")
    return Path(base) if base else Path(__file__).resolve().parents[1]


def _outputs_dir() -> Path:
    return _root() / "data" / "outputs"


def _find_latest_report() -> Dict[str, Optional[Path]]:
    """
    Returns dict with latest csv/json Paths if present, plus 'ts' (timestamp str) if detectable.
    Also honors the convenience pointers 'report_latest.csv/json' if present.
    """
    out_dir = _outputs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer "latest" pointers if present
    latest_csv = out_dir / "report_latest.csv"
    latest_json = out_dir / "report_latest.json"
    if latest_csv.exists() or latest_json.exists():
        return {
            "csv": latest_csv if latest_csv.exists() else None,
            "json": latest_json if latest_json.exists() else None,
            "ts": "latest",
        }

    # Otherwise, try to pick newest by timestamp pattern
    candidates_csv = sorted(out_dir.glob("report_*.csv"), reverse=True)
    candidates_json = sorted(out_dir.glob("report_*.json"), reverse=True)

    def ts_from_name(p: Path) -> Optional[str]:
        m = re.search(r"report_(\d{8}_\d{6})\.", p.name)
        return m.group(1) if m else None

    best_csv = candidates_csv[0] if candidates_csv else None
    best_json = candidates_json[0] if candidates_json else None

    # Try to sync timestamps if possible (prefer pair with same ts)
    if best_csv and best_json:
        ts_csv = ts_from_name(best_csv)
        ts_json = ts_from_name(best_json)
        if ts_csv == ts_json:
            return {"csv": best_csv, "json": best_json, "ts": ts_csv}

    # If not paired, just return the newest available of each
    ts = ts_from_name(best_json or best_csv) or "unknown"
    return {"csv": best_csv, "json": best_json, "ts": ts}


def _load_rows_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """
    Expected JSON shape:
      { "rows": [ {...}, {...} ] }
    Fallback: if it's a bare list, accept it too.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    return []


def _csv_to_rows(csv_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas not available to read CSV: {e}")

    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    return df.to_dict(orient="records")


def _sanitize_scalar(v: Any) -> Any:
    # Replace NaN / ±Inf with None → null in JSON
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
    return v


def _sanitize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Walk rows and replace any NaN/Inf in scalars, list items, or nested dicts with None.
    """
    clean: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            # unexpected shape, keep as-is
            clean.append(r)  # type: ignore
            continue
        out: Dict[str, Any] = {}
        for k, v in r.items():
            if isinstance(v, dict):
                out[k] = {kk: _sanitize_scalar(vv) for kk, vv in v.items()}
            elif isinstance(v, list):
                out[k] = [_sanitize_scalar(x) for x in v]
            else:
                out[k] = _sanitize_scalar(v)
        clean.append(out)
    return clean


# -----------------------------
# Routes
# -----------------------------
@results_bp.get("/latest")
def result_latest():
    """
    Return the most recent results as JSON:
    {
      "csv_path": ".../data/outputs/report_YYYYMMDD_HHMMSS.csv",
      "json_path": ".../data/outputs/report_YYYYMMDD_HHMMSS.json",
      "rows": [ {...}, ... ],
      "timestamp": "<ts|latest>"
    }
    If JSON is missing, falls back to parsing the CSV.
    """
    try:
        found = _find_latest_report()
        csv_path = found.get("csv")
        json_path = found.get("json")

        rows: List[Dict[str, Any]] = []
        if json_path and json_path.exists():
            rows = _load_rows_from_json(json_path)
        elif csv_path and csv_path.exists():
            rows = _csv_to_rows(csv_path, limit=None)
        else:
            return _json_error("No report found in data/outputs/. Run /api/process first.", 404)

        rows = _sanitize_rows(rows)

        return jsonify(
            {
                "csv_path": str(csv_path) if csv_path else None,
                "json_path": str(json_path) if json_path else None,
                "rows": rows,
                "timestamp": found.get("ts"),
            }
        )
    except Exception as e:
        return _json_error(f"Failed to load latest results: {e}", 500)


@results_bp.get("/files")
def result_files():
    """
    List available report files in data/outputs/.
    {
      "items": [
        {"csv": ".../report_YYYYMMDD_HHMMSS.csv", "json": ".../report_YYYYMMDD_HHMMSS.json", "ts": "YYYYMMDD_HHMMSS"},
        ...
      ]
    }
    """
    try:
        out = _outputs_dir()
        out.mkdir(parents=True, exist_ok=True)

        csvs = sorted(out.glob("report_*.csv"))
        jsons = sorted(out.glob("report_*.json"))

        def ts(p: Path) -> Optional[str]:
            m = re.search(r"report_(\d{8}_\d{6})\.", p.name)
            return m.group(1) if m else None

        by_ts: Dict[str, Dict[str, Optional[str]]] = {}
        for p in csvs:
            t = ts(p)
            if not t:
                continue
            by_ts.setdefault(t, {})["csv"] = str(p)
        for p in jsons:
            t = ts(p)
            if not t:
                continue
            by_ts.setdefault(t, {})["json"] = str(p)

        items = [{"ts": t, "csv": v.get("csv"), "json": v.get("json")} for t, v in sorted(by_ts.items(), reverse=True)]
        return jsonify({"items": items})
    except Exception as e:
        return _json_error(f"Failed to list files: {e}", 500)


@results_bp.get("/download/csv")
def download_latest_csv():
    try:
        found = _find_latest_report()
        csv_path = found.get("csv")
        if not csv_path or not csv_path.exists():
            return _json_error("No CSV report found.", 404)
        return send_file(csv_path, as_attachment=True, download_name=csv_path.name, mimetype="text/csv")
    except Exception as e:
        return _json_error(f"Failed to download CSV: {e}", 500)


@results_bp.get("/download/json")
def download_latest_json():
    try:
        found = _find_latest_report()
        json_path = found.get("json")
        if not json_path or not json_path.exists():
            return _json_error("No JSON report found.", 404)
        return send_file(json_path, as_attachment=True, download_name=json_path.name, mimetype="application/json")
    except Exception as e:
        return _json_error(f"Failed to download JSON: {e}", 500)
