# routes/uploads.py
from __future__ import annotations

import io
import os
import json
import time
import shutil
import string
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request, current_app

upload_bp = Blueprint("upload", __name__, url_prefix="/api")


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, code: int = 400, **extra):
    return jsonify({"error": message, **extra}), code


def _root() -> Path:
    base = current_app.config.get("BASE_DIR")
    return Path(base) if base else Path(__file__).resolve().parents[1]


def _uploads_dir() -> Path:
    d = _root() / "data" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _uploads_index_path() -> Path:
    return _uploads_dir() / "uploads_index.json"


def _allowed_extensions() -> set[str]:
    exts = current_app.config.get("ALLOWED_EXTENSIONS")
    if isinstance(exts, (set, list, tuple)):
        return set(exts)
    return {"csv", "parquet"}


def _safe_filename(name: str) -> str:
    valid = f"-_.() {string.ascii_letters}{string.digits}"
    cleaned = "".join(c if c in valid else "_" for c in name)
    return cleaned.strip("._ ") or f"upload_{int(time.time())}"


def _new_file_id() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"f_{ts}_{rand}"


def _ext_from_name(name: str) -> str:
    return (Path(name).suffix or "").lower().lstrip(".")


def _save_uploads_index(mapping: Dict[str, str]) -> None:
    idx = _uploads_index_path()
    idx.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_uploads_index() -> Dict[str, str]:
    idx = _uploads_index_path()
    if not idx.exists():
        return {}
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _import_service(modname: str):
    try:
        import importlib
        return importlib.import_module(modname)
    except Exception:
        return None


def _select_engine(local_path: Path) -> str:
    mod = _import_service("services.engine_select")
    if mod and hasattr(mod, "select_engine"):
        try:
            return mod.select_engine(str(local_path))
        except Exception:
            pass
    # fallback heuristic
    try:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        return "spark" if size_mb > 500 else "pandas"
    except Exception:
        return "pandas"


def _preview_columns(local_path: Path) -> Tuple[list[str], Optional[int]]:
    """
    Try to read column names without loading the whole file.
    Returns (columns, approx_rows or None).
    """
    cols: list[str] = []
    approx_rows: Optional[int] = None

    # Try schema_detect preview if available
    sdet = _import_service("services.schema_detect")
    if sdet and hasattr(sdet, "preview_columns"):
        try:
            preview = sdet.preview_columns(str(local_path))
            if isinstance(preview, dict):
                cols = list(preview.get("columns") or [])
                approx_rows = preview.get("rows")  # may be None
                return cols, approx_rows
        except Exception:
            pass

    # Fallback: use pandas with nrows=0 to get header
    try:
        import pandas as pd  # type: ignore
        suffix = local_path.suffix.lower()
        if suffix == ".parquet":
            # Parquet needs full open; still cheap for schema
            df = pd.read_parquet(local_path, engine="pyarrow")
            cols = list(df.columns)
            approx_rows = int(df.shape[0])
        else:
            df = pd.read_csv(local_path, nrows=0)
            cols = list(df.columns)
            approx_rows = None  # unknown without full read
    except Exception:
        cols = []
        approx_rows = None

    return cols, approx_rows


def _detect_mapping(local_path: Path) -> list[Dict[str, Any]]:
    """
    If services.schema_detect.normalize_columns can map to KOI names,
    return a compact mapping description without modifying the file.
    """
    mapping_list: list[Dict[str, Any]] = []
    sdet = _import_service("services.schema_detect")
    if not sdet:
        return mapping_list

    # Some implementations accept a list of names
    try:
        if hasattr(sdet, "suggest_mapping_from_path"):
            mapping_list = sdet.suggest_mapping_from_path(str(local_path)) or []
        elif hasattr(sdet, "suggest_mapping_from_columns"):
            cols, _ = _preview_columns(local_path)
            mapping_list = sdet.suggest_mapping_from_columns(cols) or []
    except Exception:
        pass

    # Fallback: trivial identity mapping using previewed columns
    if not mapping_list:
        cols, _ = _preview_columns(local_path)
        mapping_list = [{"detected": c, "mapped_to": c, "status": "ok"} for c in cols]

    return mapping_list


def _store_stream_to_file(stream: io.BufferedReader, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(stream, f)


def _resolve_from_url(url: str, dest_dir: Path) -> Path:
    """
    Use services.storage.resolve_input to fetch/copy the URL/bucket to local uploads dir.
    Fallback: if it's a local path, just copy.
    """
    storage = _import_service("services.storage")
    if storage and hasattr(storage, "resolve_input"):
        local_path = storage.resolve_input(url, dest_dir=str(dest_dir))
        return Path(local_path)

    # Fallback: treat as local path
    src = Path(url)
    if not src.exists():
        raise FileNotFoundError("Storage service not available and URL/path not found.")
    dest = dest_dir / src.name
    if str(src.resolve()) != str(dest.resolve()):
        shutil.copy2(src, dest)
    return dest


# -----------------------------
# Routes
# -----------------------------
@upload_bp.post("/upload")
def upload():
    """
    Accepts either:
      - multipart/form-data with a 'file' field (CSV/Parquet)
      - application/json with {"url": "<bucket or local path>"}
    Returns:
      {
        "file_id": "...",
        "path": ".../data/uploads/<saved-file>",
        "detected_columns": [... or mapping objects],
        "engine": "pandas|spark",
        "rows": <int|null>
      }
    """
    try:
        uploads_dir = _uploads_dir()
        allowed = _allowed_extensions()
        index = _load_uploads_index()

        saved_path: Optional[Path] = None
        original_name: Optional[str] = None

        # JSON with URL/bucket
        if request.is_json:
            data = request.get_json(silent=True) or {}
            url = (data.get("url") or "").strip()
            if not url:
                return _json_error("Body JSON must include 'url' for URL uploads.", 400)

            # Try to preserve extension from URL if present
            ext = _ext_from_name(url)
            if ext and ext not in allowed:
                return _json_error(f"Extension '.{ext}' not allowed. Allowed: {sorted(allowed)}", 400)

            # Resolve to local file
            saved_path = _resolve_from_url(url, uploads_dir)
            original_name = Path(url).name

        else:
            # Multipart file
            file = request.files.get("file")
            if not file:
                return _json_error("Missing file upload (field name 'file').", 400)

            original_name = _safe_filename(file.filename or "upload")
            ext = _ext_from_name(original_name)
            if ext not in allowed:
                return _json_error(f"Extension '.{ext}' not allowed. Allowed: {sorted(allowed)}", 400)

            file_id = _new_file_id()
            saved_name = f"{file_id}.{ext}"
            saved_path = uploads_dir / saved_name
            _store_stream_to_file(file.stream, saved_path)

        if not saved_path or not saved_path.exists():
            return _json_error("Failed to store uploaded file.", 500)

        # Assign/ensure file_id and index mapping
        file_id = _new_file_id()
        ext = _ext_from_name(saved_path.name) or "csv"
        final_name = f"{file_id}.{ext}"
        final_path = uploads_dir / final_name

        # If saved_path is already that path, keep; otherwise move/rename
        if str(saved_path.resolve()) != str(final_path.resolve()):
            shutil.move(str(saved_path), str(final_path))
        saved_path = final_path

        # Update index
        index[file_id] = str(saved_path)
        _save_uploads_index(index)

        # Preview columns and mapping
        detected_cols, approx_rows = _preview_columns(saved_path)
        mapping = _detect_mapping(saved_path)
        engine = _select_engine(saved_path)

        return jsonify(
            {
                "file_id": file_id,
                "path": str(saved_path),
                "original_name": original_name,
                "detected_columns": mapping if mapping else detected_cols,
                "engine": engine,
                "rows": approx_rows,
            }
        )
    except Exception as e:
        return _json_error(f"Upload failed: {e}", 500)


@upload_bp.get("/upload/index")
def upload_index():
    """
    List known uploads (from uploads_index.json).
    """
    try:
        idx = _load_uploads_index()
        items = [{"file_id": k, "path": v} for k, v in sorted(idx.items(), key=lambda x: x[0], reverse=True)]
        return jsonify({"count": len(items), "items": items})
    except Exception as e:
        return _json_error(f"Failed to load uploads index: {e}", 500)


@upload_bp.get("/upload/<file_id>/meta")
def upload_meta(file_id: str):
    """
    Metadata and quick preview for an uploaded file_id.
    """
    try:
        idx = _load_uploads_index()
        path_str = idx.get(file_id)
        if not path_str:
            return _json_error(f"Unknown file_id: {file_id}", 404)

        p = Path(path_str)
        if not p.exists():
            return _json_error(f"File not found for file_id: {file_id}", 404)

        cols, approx_rows = _preview_columns(p)
        engine = _select_engine(p)

        return jsonify(
            {
                "file_id": file_id,
                "path": str(p),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "engine": engine,
                "detected_columns": cols,
                "rows": approx_rows,
            }
        )
    except Exception as e:
        return _json_error(f"Failed to read meta: {e}", 500)
