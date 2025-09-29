# services/storage.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _root_dir() -> Path:
    # .../web_app
    return Path(__file__).resolve().parents[1]


def _data_dir() -> Path:
    d = _root_dir() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Paths
# -----------------------------
def uploads_dir() -> Path:
    return _ensure_dir(_data_dir() / "uploads")


def outputs_dir() -> Path:
    return _ensure_dir(_data_dir() / "outputs")


def raw_dir() -> Path:
    return _ensure_dir(_data_dir() / "raw")


def interim_dir() -> Path:
    return _ensure_dir(_data_dir() / "interim")


def labeled_dir() -> Path:
    return _ensure_dir(_data_dir() / "labeled")


# -----------------------------
# JSON helpers
# -----------------------------
def read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8")
    return p


# -----------------------------
# File I/O
# -----------------------------
def save_upload(src_path: str | Path, dest_name: Optional[str] = None) -> Path:
    """
    Copy an uploaded file into data/uploads/, returning the destination path.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Upload not found: {src}")
    dest = uploads_dir() / (dest_name or src.name)
    shutil.copy2(src, dest)
    return dest


def list_uploads(limit: Optional[int] = None) -> list[Dict[str, Any]]:
    """
    List files in data/uploads/ with metadata.
    """
    items: list[Dict[str, Any]] = []
    for f in sorted(uploads_dir().glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        items.append(
            {
                "name": f.name,
                "path": str(f),
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
        )
        if limit and len(items) >= limit:
            break
    return items


def save_dataframe(df, path: str | Path, fmt: str = "csv") -> Path:
    """
    Save a Pandas DataFrame to CSV or Parquet.
    """
    import pandas as pd  # type: ignore
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "parquet":
        df.to_parquet(p, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    return p


def load_dataframe(path: str | Path, fmt: Optional[str] = None):
    """
    Load a DataFrame from CSV or Parquet.
    """
    import pandas as pd  # type: ignore
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    suffix = p.suffix.lower()
    f = fmt or ("parquet" if suffix == ".parquet" else "csv")
    if f == "csv":
        return pd.read_csv(p)
    if f == "parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported format: {f}")


# -----------------------------
# Outputs helpers
# -----------------------------
def latest_output() -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (csv_path, json_path) of the latest report in data/outputs/.
    """
    out = outputs_dir()
    csvs = sorted(out.glob("report_*.csv"), reverse=True)
    jsons = sorted(out.glob("report_*.json"), reverse=True)
    return (csvs[0] if csvs else None, jsons[0] if jsons else None)


def save_output(df, fmt: str = "csv", prefix: str = "report") -> Path:
    """
    Save DataFrame as a timestamped report in data/outputs/.
    """
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.{fmt}"
    return save_dataframe(df, outputs_dir() / fname, fmt=fmt)
