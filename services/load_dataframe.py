# services/load_dataframe.py
from __future__ import annotations

import io
import os
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Public API
# -----------------------------
def load_df(path: str, engine: str = "pandas", nrows: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a CSV/Parquet file into a DataFrame.

    Args:
        path: file path (local)
        engine: "pandas" | "spark"
        nrows: optional row cap for quick tests (pandas only; spark ignored)

    Returns:
        (df, meta) where:
          - df: pandas.DataFrame or pyspark.sql.DataFrame
          - meta: {
              "rows": <int or None>,
              "cols": <int or None>,
              "engine": "pandas"|"spark",
              "columns": [ ... ],
              "path": "<path>",
              "format": "csv"|"parquet"|"unknown",
              "encoding": "<encoding if detected for csv>",
              "delimiter": "<delimiter if csv>",
              "koi_disposition_present": <bool>
            }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()
    if engine not in ("pandas", "spark"):
        engine = "pandas"

    if engine == "spark":
        df = _load_with_spark(p)
        rows = _spark_count_safe(df)
        cols = len(df.columns)
        # --- preserva/garantiza koi_disposition tal cual o vacío ---
        if "koi_disposition" not in df.columns:
            # Crear columna nula (Spark)
            from pyspark.sql import functions as F  # type: ignore
            df = df.withColumn("koi_disposition", F.lit(None))
            koi_disp_present = False
        else:
            koi_disp_present = True

        meta = {
            "rows": rows,
            "cols": cols if "koi_disposition" in df.columns else cols + 1,
            "engine": "spark",
            "columns": df.columns,
            "path": str(p),
            "format": "parquet" if suffix == ".parquet" else "csv" if suffix in (".csv", ".tsv", ".txt") else "unknown",
            "encoding": None,
            "delimiter": None,
            "koi_disposition_present": koi_disp_present,
        }
        return df, meta

    # pandas
    df, meta = _load_with_pandas(p, nrows=nrows)
    # --- preserva/garantiza koi_disposition tal cual o vacío ---
    koi_disp_present = "koi_disposition" in df.columns
    if not koi_disp_present:
        # Crear columna con None (no transformar ni inferir)
        df["koi_disposition"] = None

    meta["engine"] = "pandas"
    meta["koi_disposition_present"] = koi_disp_present
    # Actualizar columnas en meta (por si agregamos la nueva)
    meta["columns"] = list(df.columns)
    meta["cols"] = int(meta.get("cols") or len(df.columns))
    return df, meta


# -----------------------------
# Pandas loaders
# -----------------------------
def _load_with_pandas(p: Path, nrows: Optional[int] = None):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas is required to load data with engine='pandas': {e}")

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        # Prefer pyarrow; fallback to fastparquet si no está
        try:
            df = pd.read_parquet(p, engine="pyarrow")  # type: ignore
        except Exception:
            df = pd.read_parquet(p)  # let pandas choose available engine
        rows, cols = _shape_safe(df)
        return df, {
            "rows": rows,
            "cols": cols,
            "columns": list(df.columns),
            "path": str(p),
            "format": "parquet",
            "encoding": None,
            "delimiter": None,
        }

    # CSV / TSV / TXT
    # Intentar detectar encoding y delimitador con una muestra
    encoding = _detect_encoding(p)
    delim = _detect_delimiter(p, encoding=encoding)

    # dtype_backend: usar pyarrow si está disponible (mejor memoria y nulls)
    dtype_backend = None
    try:
        import importlib
        if importlib.util.find_spec("pyarrow"):  # type: ignore
            dtype_backend = "pyarrow"  # pandas >= 2.0
    except Exception:
        dtype_backend = None

    read_csv_kwargs = dict(
        sep=delim or ",",
        encoding=encoding or "utf-8",
        low_memory=False,
        na_values=["", "NA", "NaN", "null", "NULL", "None"],
        dtype_backend=dtype_backend,  # type: ignore
    )

    # Si es TSV, forzar \t
    if p.suffix.lower() == ".tsv":
        read_csv_kwargs["sep"] = "\t"

    # Compresión: pandas la infiere por extensión; no fuerces aquí
    try:
        df = pd.read_csv(p, nrows=nrows, **read_csv_kwargs)  # type: ignore
    except UnicodeDecodeError:
        # fallback latino-1 si el encoding fue incorrecto
        read_csv_kwargs["encoding"] = "latin-1"
        df = pd.read_csv(p, nrows=nrows, **read_csv_kwargs)  # type: ignore

    rows, cols = _shape_safe(df)
    meta = {
        "rows": rows,
        "cols": cols,
        "columns": list(df.columns),
        "path": str(p),
        "format": "csv",
        "encoding": read_csv_kwargs["encoding"],
        "delimiter": read_csv_kwargs["sep"],
    }
    return df, meta


def _detect_encoding(p: Path, sample_bytes: int = 262144) -> Optional[str]:
    """
    Try to detect text encoding using chardet (optional dependency).
    Returns 'utf-8' if unsure, but leaves None to let pandas decide by default.
    """
    try:
        import chardet  # type: ignore
        with open(p, "rb") as f:
            raw = f.read(sample_bytes)
        res = chardet.detect(raw)
        enc = (res or {}).get("encoding")
        if enc:
            # Normalizar algunos nombres
            enc_low = enc.lower()
            if "utf" in enc_low:
                return "utf-8"
            return enc
        return "utf-8"
    except Exception:
        # Si no hay chardet, intenta utf-8
        return "utf-8"


def _detect_delimiter(p: Path, encoding: Optional[str] = "utf-8", sample_bytes: int = 131072) -> Optional[str]:
    """
    Detect CSV delimiter using csv.Sniffer on a small sample.
    Returns ',', '\\t', ';', '|' or None if unknown.
    """
    try:
        with open(p, "rb") as fb:
            raw = fb.read(sample_bytes)
        sample = raw.decode(encoding or "utf-8", errors="ignore")
        # Limpiar cabeceras muy largas
        sample = "\n".join(sample.splitlines()[:50])  # primeras 50 líneas son suficientes
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        # Heurística por extensión
        if p.suffix.lower() == ".tsv":
            return "\t"
        return ","


def _shape_safe(df) -> tuple[int, int]:
    try:
        return int(df.shape[0]), int(df.shape[1])
    except Exception:
        return (None, None)


# -----------------------------
# Spark loaders
# -----------------------------
def _load_with_spark(p: Path):
    """
    Load CSV/Parquet with Spark.
    Returns a pyspark.sql.DataFrame.
    """
    try:
        # Prefer helper if user defined it (workers/spark_session.py)
        from workers import spark_session as _spark_helper  # type: ignore
        if hasattr(_spark_helper, "get_spark"):
            spark = _spark_helper.get_spark(app_name=os.getenv("SPARK_APP_NAME", "ExoHunter"))
        else:
            spark = _spark_default()
    except Exception:
        spark = _spark_default()

    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return spark.read.parquet(str(p))

    # CSV/TSV/TXT
    if suffix in (".csv", ".tsv", ".txt"):
        delim = "\t" if suffix == ".tsv" else ","
        return (
            spark.read.format("csv")
            .option("header", "true")
            .option("inferSchema", "true")
            .option("sep", delim)
            .load(str(p))
        )

    # Desconocido: intentar CSV por defecto
    return (
        spark.read.format("csv")
        .option("header", "true")
        .option("inferSchema", "true")
        .load(str(p))
    )


def _spark_default():
    """
    Create or get a default SparkSession with sane defaults.
    """
    from pyspark.sql import SparkSession  # type: ignore

    builder = (
        SparkSession.builder.appName(os.getenv("SPARK_APP_NAME", "ExoHunter"))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.session.timeZone", "UTC")
    )
    spark = builder.getOrCreate()
    return spark


def _spark_count_safe(df) -> Optional[int]:
    try:
        return int(df.count())
    except Exception:
        return None
