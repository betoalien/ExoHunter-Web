# routes/features.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, current_app, jsonify, request

features_bp = Blueprint("features", __name__, url_prefix="/api/features")


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, code: int = 400, **extra):
    payload = {"error": message, **extra}
    return jsonify(payload), code


def _project_root() -> Path:
    # app.config["BASE_DIR"] is set in config.py/_DefaultConfig (string path)
    base = current_app.config.get("BASE_DIR")
    return Path(base) if base else Path(__file__).resolve().parents[1]


def _feature_version() -> str:
    return str(current_app.config.get("FEATURE_VERSION", "v1"))


def _latest_manifest_path() -> Path:
    root = _project_root()
    return root / "feature_store" / _feature_version() / "features_latest.manifest.json"


def _load_latest_manifest() -> Dict[str, Any]:
    mpath = _latest_manifest_path()
    if not mpath.exists():
        raise FileNotFoundError(
            f"Latest manifest not found: {mpath}. "
            "Run ml/pipelines/make_dataset.py to generate features."
        )
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_features_df(parquet_path: Path):
    # We read as pandas for API responses
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas not available: {e}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Features parquet not found: {parquet_path}")
    try:
        return pd.read_parquet(parquet_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read parquet: {parquet_path} ({e})")


def _paginate(df, page: int, page_size: int):
    total = int(getattr(df, "shape", [0, 0])[0])
    if page_size <= 0:
        page_size = 50
    start = max((page - 1), 0) * page_size
    end = min(start + page_size, total)
    return df.iloc[start:end], total


def _filter_df(df, query_id: Optional[str] = None, category: Optional[str] = None):
    if query_id:
        mask = df["object_id"].astype(str).str.contains(query_id, case=False, na=False)
        df = df[mask]
    if category and "category" in df.columns:
        df = df[df["category"].astype(str) == category]
    return df


# -----------------------------
# Routes
# -----------------------------
@features_bp.get("/meta")
def features_meta():
    """
    Return metadata from the latest manifest:
    {
      "feature_version": "...",
      "features_file": "...",
      "rows_loaded": ...,
      "feature_names": [...]
    }
    """
    try:
        manifest = _load_latest_manifest()
        meta = {
            "feature_version": manifest.get("feature_version"),
            "features_file": manifest.get("features_file"),
            "rows_loaded": manifest.get("rows_loaded"),
            "feature_names": manifest.get("feature_names"),
            "timestamp": manifest.get("timestamp"),
            "source_file": manifest.get("source_file"),
        }
        return jsonify(meta)
    except FileNotFoundError as e:
        return _json_error(str(e), 404)
    except Exception as e:
        return _json_error(f"Failed to load meta: {e}", 500)


@features_bp.get("/")
def list_features():
    """
    List feature rows (paginated) with optional filters.
    Query params:
      - q: substring match on object_id
      - category: exact category match (if column exists)
      - page: 1-based page number (default 1)
      - page_size: items per page (default 50)
      - columns: comma-separated subset of columns to return
    """
    q = request.args.get("q") or request.args.get("search")
    category = request.args.get("category")
    page = int(request.args.get("page", "1"))
    page_size = int(request.args.get("page_size", "50"))
    columns_param = request.args.get("columns", "").strip()

    try:
        manifest = _load_latest_manifest()
        parquet_path = Path(manifest["features_file"])
        df = _load_features_df(parquet_path)

        # Optional filters
        df = _filter_df(df, q, category)

        # Pagination
        page_df, total = _paginate(df, page, page_size)

        # Optional column projection
        if columns_param:
            cols = [c.strip() for c in columns_param.split(",") if c.strip()]
            # keep only existing to avoid KeyError
            cols = [c for c in cols if c in page_df.columns]
            if cols:
                page_df = page_df[cols]

        # JSON response
        rows = page_df.to_dict(orient="records")
        return jsonify(
            {
                "total": total,
                "page": page,
                "page_size": page_size,
                "rows": rows,
            }
        )
    except FileNotFoundError as e:
        return _json_error(str(e), 404)
    except Exception as e:
        return _json_error(f"Failed to list features: {e}", 500)


@features_bp.get("/<object_id>")
def get_features(object_id: str):
    """
    Return the feature vector for a single object_id.
    Optional query param:
      - columns: comma-separated subset of columns (e.g., 'object_id,koi_period,koi_prad')
    """
    columns_param = request.args.get("columns", "").strip()
    try:
        manifest = _load_latest_manifest()
        parquet_path = Path(manifest["features_file"])
        df = _load_features_df(parquet_path)

        if "object_id" not in df.columns:
            return _json_error("Column 'object_id' not found in features.", 500)

        row = df[df["object_id"].astype(str) == str(object_id)]
        if row.empty:
            return _json_error(f"object_id not found: {object_id}", 404)

        if columns_param:
            cols = [c.strip() for c in columns_param.split(",") if c.strip()]
            cols = [c for c in cols if c in row.columns]
            if cols:
                row = row[cols]

        payload = row.iloc[0].to_dict()
        return jsonify(payload)
    except FileNotFoundError as e:
        return _json_error(str(e), 404)
    except Exception as e:
        return _json_error(f"Failed to fetch features for {object_id}: {e}", 500)
