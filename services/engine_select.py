# services/engine_select.py
from __future__ import annotations

import os
import gzip
from pathlib import Path
from typing import Dict, Tuple

# -----------------------------
# Defaults / thresholds (override via env)
# -----------------------------
# Tamaños en MB
CSV_PANDAS_MAX_MB = float(os.getenv("CSV_PANDAS_MAX_MB", "200"))    # <= 200MB → pandas
CSV_SPARK_MIN_MB  = float(os.getenv("CSV_SPARK_MIN_MB", "1000"))    # >= 1GB   → spark
PARQUET_SPARK_MIN_MB = float(os.getenv("PARQUET_SPARK_MIN_MB", "2000"))  # >= 2GB → spark

# Extensiones soportadas
CSV_EXTS = {".csv", ".tsv", ".txt"}
PARQUET_EXTS = {".parquet"}
GZIP_EXTS = {".gz", ".gzip"}

# Overrides
ENGINE_FORCE = os.getenv("ENGINE_FORCE", "").strip().lower()       # "pandas" | "spark" | ""
ENGINE_DEFAULT = os.getenv("ENGINE_DEFAULT", "pandas").strip().lower()


def _file_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except Exception:
        return -1.0


def _is_gzip(path: Path) -> bool:
    # Si termina en .gz/.gzip o el primer par de bytes es GZIP magic
    if path.suffix.lower() in GZIP_EXTS:
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        return head == b"\x1f\x8b"
    except Exception:
        return False


def _ext(path: Path) -> str:
    return path.suffix.lower()


def _csv_like(path: Path) -> bool:
    return _ext(path) in CSV_EXTS or (_is_gzip(path) and any(str(path).lower().endswith(e + ".gz") for e in CSV_EXTS))


def _parquet_like(path: Path) -> bool:
    return _ext(path) in PARQUET_EXTS


def _reason_map(engine: str, **extra) -> Dict:
    return {"engine": engine, **extra}


def select_engine_with_meta(path: str) -> Tuple[str, Dict]:
    """
    Heurística para elegir engine con explicación.
    Orden de decisión:
      1) ENGINE_FORCE env var (override duro).
      2) Extensión + tamaño:
         - CSV/TSV:
             <= CSV_PANDAS_MAX_MB → pandas
             >= CSV_SPARK_MIN_MB  → spark
             en medio → si gzip → spark; si no → pandas (por simplicidad)
         - Parquet:
             >= PARQUET_SPARK_MIN_MB → spark
             si no → pandas
      3) Por defecto → ENGINE_DEFAULT (pandas).
    """
    p = Path(path)

    # 0) Validaciones básicas
    if not p.exists():
        # Si no existe, devolvemos default con nota
        return ENGINE_DEFAULT, _reason_map(
            ENGINE_DEFAULT, reason="file_not_found", path=str(p)
        )

    # 1) Override duro
    if ENGINE_FORCE in {"pandas", "spark"}:
        return ENGINE_FORCE, _reason_map(
            ENGINE_FORCE,
            reason="forced_by_env",
            env="ENGINE_FORCE",
            value=ENGINE_FORCE,
            path=str(p),
        )

    size_mb = _file_size_mb(p)
    ext = _ext(p)
    is_gz = _is_gzip(p) if size_mb >= 0 else False

    # 2) Extensión + tamaño
    if _csv_like(p):
        # CSV / TSV (posible gzip)
        if size_mb >= 0:
            if size_mb <= CSV_PANDAS_MAX_MB:
                return "pandas", _reason_map(
                    "pandas",
                    reason="csv_small",
                    size_mb=round(size_mb, 2),
                    limit_mb=CSV_PANDAS_MAX_MB,
                    gzip=is_gz,
                    path=str(p),
                )
            if size_mb >= CSV_SPARK_MIN_MB:
                return "spark", _reason_map(
                    "spark",
                    reason="csv_large",
                    size_mb=round(size_mb, 2),
                    limit_mb=CSV_SPARK_MIN_MB,
                    gzip=is_gz,
                    path=str(p),
                )
            # Zona intermedia
            if is_gz:
                return "spark", _reason_map(
                    "spark",
                    reason="csv_mid_but_gzip",
                    size_mb=round(size_mb, 2),
                    pandas_max_mb=CSV_PANDAS_MAX_MB,
                    spark_min_mb=CSV_SPARK_MIN_MB,
                    gzip=True,
                    path=str(p),
                )
            else:
                return "pandas", _reason_map(
                    "pandas",
                    reason="csv_mid_plain",
                    size_mb=round(size_mb, 2),
                    pandas_max_mb=CSV_PANDAS_MAX_MB,
                    spark_min_mb=CSV_SPARK_MIN_MB,
                    gzip=False,
                    path=str(p),
                )
        else:
            # tamaño desconocido → usar default para CSV
            return ENGINE_DEFAULT, _reason_map(
                ENGINE_DEFAULT, reason="csv_unknown_size", path=str(p)
            )

    if _parquet_like(p):
        if size_mb >= 0 and size_mb >= PARQUET_SPARK_MIN_MB:
            return "spark", _reason_map(
                "spark",
                reason="parquet_large",
                size_mb=round(size_mb, 2),
                limit_mb=PARQUET_SPARK_MIN_MB,
                path=str(p),
            )
        # Parquet suele ir bien con pandas/pyarrow en tamaños medianos
        return "pandas", _reason_map(
            "pandas",
            reason="parquet_default",
            size_mb=round(size_mb, 2) if size_mb >= 0 else None,
            path=str(p),
        )

    # 3) Desconocido: fallback por tamaño
    if size_mb >= 0:
        # Umbral genérico
        if size_mb >= max(CSV_SPARK_MIN_MB, PARQUET_SPARK_MIN_MB):
            return "spark", _reason_map(
                "spark",
                reason="unknown_large",
                size_mb=round(size_mb, 2),
                path=str(p),
            )
        return "pandas", _reason_map(
            "pandas",
            reason="unknown_small",
            size_mb=round(size_mb, 2),
            path=str(p),
        )

    # Último recurso
    return ENGINE_DEFAULT, _reason_map(
        ENGINE_DEFAULT, reason="unknown_ext_unknown_size", path=str(p)
    )


def select_engine(path: str) -> str:
    """
    Interfaz simple (usada por routes y pipelines):
      returns: "pandas" | "spark"
    """
    engine, _meta = select_engine_with_meta(path)
    return engine
