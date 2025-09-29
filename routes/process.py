# routes/process.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from flask import Blueprint, jsonify, request, current_app

process_bp = Blueprint("process", __name__, url_prefix="/api")


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, code: int = 400, **extra):
    return jsonify({"error": message, **extra}), code


def _root() -> Path:
    base = current_app.config.get("BASE_DIR")
    return Path(base) if base else Path(__file__).resolve().parents[1]


def _uploads_index_path() -> Path:
    return _root() / "data" / "uploads" / "uploads_index.json"


def _outputs_dir() -> Path:
    return _root() / "data" / "outputs"


def _configs_dir() -> Path:
    return _root() / "configs"


def _load_uploads_index() -> Dict[str, str]:
    idx = _uploads_index_path()
    if not idx.exists():
        return {}
    try:
        with open(idx, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _resolve_uploaded_file(file_id: str) -> Path:
    """
    Resolve a file_id to a local path. Uses data/uploads/uploads_index.json if present.
    Fallback: try to treat file_id as a direct path or a filename inside data/uploads/.
    """
    uploads = _load_uploads_index()

    # Direct mapping
    if file_id in uploads:
        p = Path(uploads[file_id])
        if p.exists():
            return p

    # If file_id looks like a path and exists
    p = Path(file_id)
    if p.exists():
        return p

    # Try common extensions inside data/uploads
    up_dir = _root() / "data" / "uploads"
    for ext in (".csv", ".parquet"):
        cand = up_dir / f"{file_id}{ext}"
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Could not resolve file_id '{file_id}'. "
        "Ensure it was uploaded and indexed in data/uploads/uploads_index.json."
    )


def _import_services():
    """
    Import required service modules with friendly error messages.
    Not all are strictly required; we degrade gracefully when missing.
    """
    services: Dict[str, Any] = {}
    missing = []

    def try_import(name: str):
        import importlib
        try:
            return importlib.import_module(name)
        except Exception:
            missing.append(name)
            return None

    services["engine_select"] = try_import("services.engine_select")
    services["load_dataframe"] = try_import("services.load_dataframe")
    services["schema_detect"] = try_import("services.schema_detect")
    services["compute_metrics"] = try_import("services.compute_metrics")
    services["classify"] = try_import("services.classify")
    services["export_report"] = try_import("services.export_report")

    return services, missing


def _select_engine(services: Dict[str, Any], path: Path, requested: str | None = None) -> str:
    if requested in ("pandas", "spark"):
        return requested
    mod = services.get("engine_select")
    if mod and hasattr(mod, "select_engine"):
        try:
            return mod.select_engine(str(path))
        except Exception:
            pass
    return "pandas"


def _load_df(services: Dict[str, Any], path: Path, engine: str):
    mod = services.get("load_dataframe")
    if not mod or not hasattr(mod, "load_df"):
        # Minimal fallback: try pandas directly
        import pandas as pd  # type: ignore
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        return df, {"rows": len(df), "cols": len(df.columns), "engine": "pandas"}
    return mod.load_df(str(path), engine=engine)


def _normalize_schema(services: Dict[str, Any], df):
    mod = services.get("schema_detect")
    if not mod or not hasattr(mod, "normalize_columns"):
        # No normalization available; return as-is
        return df, []
    # Optional: load expected map
    exp_map = _root() / "schemas" / "expected_columns.json"
    if exp_map.exists() and hasattr(mod, "load_expected_map"):
        try:
            mod.load_expected_map(str(exp_map))
        except Exception:
            pass
    df_norm, mapping = mod.normalize_columns(df)
    return df_norm, mapping


def _enrich_metrics(services: Dict[str, Any], df, assumptions: Dict[str, Any], options: Dict[str, Any]):
    mod = services.get("compute_metrics")
    if not mod or not hasattr(mod, "enrich"):
        return df
    try:
        return mod.enrich(df, assumptions=assumptions, options=options)
    except Exception:
        return df


def _classify(services: Dict[str, Any], df):
    mod = services.get("classify")
    if not mod:
        # Fallback: return df with minimal columns
        import pandas as pd  # type: ignore
        out_cols = ["object_id", "koi_period", "koi_prad", "koi_teq", "koi_insol"]
        tmp = {}
        for c in out_cols:
            tmp[c] = df[c] if c in df.columns else None
        fallback = pd.DataFrame(tmp)
        fallback["category"] = "no_result"
        fallback["score"] = 0.0
        fallback["flags"] = ""
        return fallback

    # Prefer classify.classify_df if available
    if hasattr(mod, "classify_df"):
        return mod.classify_df(df, thresholds_path=str(_configs_dir() / "thresholds.yaml"))

    # Otherwise, try a naive rule-based function if present
    if hasattr(mod, "classify"):
        return mod.classify(df)

    # Final fallback
    return _classify.__defaults__[0]  # type: ignore


def _write_outputs(services: Dict[str, Any], results_df, outputs_dir: Path) -> Tuple[str, str]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / f"report_{ts}.csv"
    json_path = outputs_dir / f"report_{ts}.json"

    # Try service exporter first
    mod = services.get("export_report")
    if mod and hasattr(mod, "write_outputs"):
        try:
            paths = mod.write_outputs(results_df, out_dir=str(outputs_dir), timestamp=ts)
            # Expecting {"csv_path": "...", "json_path": "..."} or tuple
            if isinstance(paths, dict):
                csv_path = Path(paths.get("csv_path", csv_path))
                json_path = Path(paths.get("json_path", json_path))
            elif isinstance(paths, (list, tuple)) and len(paths) >= 2:
                csv_path = Path(paths[0])
                json_path = Path(paths[1])
        except Exception:
            # Fall back to simple writer below
            pass

    # Simple writers
    try:
        results_df.to_csv(csv_path, index=False)
    except Exception:
        # If Spark DF, convert to pandas
        try:
            pdf = results_df.toPandas()
            pdf.to_csv(csv_path, index=False)
        except Exception as e:
            raise RuntimeError(f"Failed to write CSV: {e}")

    try:
        # If it's a pandas DF
        records = results_df.to_dict(orient="records")
    except Exception:
        # Spark DF fallback
        try:
            records = results_df.toPandas().to_dict(orient="records")
        except Exception as e:
            raise RuntimeError(f"Failed to write JSON: {e}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"rows": records}, f, indent=2, ensure_ascii=False)

    # Also write/update latest pointers for convenience
    latest_csv = outputs_dir / "report_latest.csv"
    latest_json = outputs_dir / "report_latest.json"
    try:
        latest_csv.write_bytes(csv_path.read_bytes())
        latest_json.write_text((json_path.read_text(encoding="utf-8")), encoding="utf-8")
    except Exception:
        pass

    # Return paths as web-facing hints (you can later add a download endpoint)
    return (str(csv_path), str(json_path))


# -----------------------------
# Routes
# -----------------------------
@process_bp.post("/process")
def process_dataset():
    """
    Run the processing pipeline synchronously:
      - resolve uploaded file
      - load dataframe
      - normalize schema
      - compute derived metrics (optional)
      - classify rows
      - write CSV/JSON outputs
    Body JSON:
    {
      "file_id": "abc123 or filename",
      "assumptions": { "albedo": 0.3 },
      "options": { "auto_derive": true, "engine": "auto|pandas|spark" }
    }
    Response:
    {
      "csv_path": ".../data/outputs/report_YYYYMMDD_HHMMSS.csv",
      "json_path": ".../data/outputs/report_YYYYMMDD_HHMMSS.json",
      "rows": <int>,
      "engine": "<pandas|spark>"
    }
    """
    try:
        payload = request.get_json(force=True) or {}
        file_id = payload.get("file_id")
        if not file_id:
            return _json_error("Missing 'file_id' in request body.", 400)

        assumptions = payload.get("assumptions") or {}
        options = payload.get("options") or {}
        auto_derive = bool(options.get("auto_derive", True))
        requested_engine = options.get("engine", "auto")

        # Import services
        services, missing = _import_services()

        # Resolve input
        input_path = _resolve_uploaded_file(str(file_id))

        # Select engine
        engine = _select_engine(services, input_path, requested_engine)

        # Load DF
        df, meta = _load_df(services, input_path, engine)
        rows_loaded = int(meta.get("rows") or 0)

        # Normalize schema
        df, mapping = _normalize_schema(services, df)

        # Derived metrics
        albedo = float(assumptions.get("albedo", 0.3))
        df = _enrich_metrics(services, df, assumptions={"albedo": albedo}, options={"auto_derive": auto_derive})

        # Classify
        results_df = _classify(services, df)

        # Export outputs
        csv_path, json_path = _write_outputs(services, results_df, _outputs_dir())

        return jsonify(
            {
                "csv_path": csv_path,
                "json_path": json_path,
                "rows": int(getattr(results_df, "shape", [0, 0])[0]),
                "engine": meta.get("engine", engine),
                "warnings": (missing if missing else []),
            }
        )
    except FileNotFoundError as e:
        return _json_error(str(e), 404)
    except Exception as e:
        return _json_error(f"Processing failed: {e}", 500)
