# ml/pipelines/make_dataset.py
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------------------
# Optional dotenv
# -----------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# -----------------------------------------
# Project paths
# -----------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../web_app
FEATURE_STORE_DIR = ROOT_DIR / "feature_store"
CONFIGS_DIR = ROOT_DIR / "configs"
SCHEMAS_DIR = ROOT_DIR / "schemas"
ML_DIR = ROOT_DIR / "ml"

DEFAULT_FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v1")
DEFAULT_ALBEDO = float(os.getenv("DEFAULT_ALBEDO", "0.3"))

DEFAULT_FEATURE_SPECS = ML_DIR / "feature_specs.yaml"
DEFAULT_THRESHOLDS = CONFIGS_DIR / "thresholds.yaml"
DEFAULT_SCHEMA_MAP = SCHEMAS_DIR / "expected_columns.json"

# -----------------------------------------
# Logging
# -----------------------------------------
logger = logging.getLogger("make_dataset")

def setup_logging(level: str = "INFO") -> None:
    level_obj = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=level_obj,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.setLevel(level_obj)


# -----------------------------------------
# Lazy imports from services (with friendly errors)
# -----------------------------------------
def _import_services():
    try:
        from services import storage, engine_select, load_dataframe, schema_detect, compute_metrics, feature_builder  # type: ignore
        return storage, engine_select, load_dataframe, schema_detect, compute_metrics, feature_builder
    except Exception as e:
        logger.error(
            "Unable to import required services. Ensure the following files exist:\n"
            " - services/storage.py\n"
            " - services/engine_select.py\n"
            " - services/load_dataframe.py\n"
            " - services/schema_detect.py\n"
            " - services/compute_metrics.py\n"
            " - services/feature_builder.py\n"
            "Original import error: %s", e
        )
        raise


# -----------------------------------------
# Core pipeline steps
# -----------------------------------------
def resolve_input_path(path_or_url: str, storage_mod) -> Path:
    """
    Use services.storage to resolve local path from local file or bucket URL.
    Expected to return a local filesystem path to the data file.
    """
    # Prefer service method if available
    if hasattr(storage_mod, "resolve_input"):
        local_path = storage_mod.resolve_input(path_or_url, dest_dir=str(ROOT_DIR / "data" / "raw"))
        return Path(local_path)

    # Fallback: assume local file
    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path_or_url}")
    return p


def choose_engine(input_path: Path, requested: str, engine_select_mod) -> str:
    """
    Decide engine = pandas|spark based on user request or engine_select service.
    """
    if requested in ("pandas", "spark"):
        return requested

    # auto
    if hasattr(engine_select_mod, "select_engine"):
        try:
            return engine_select_mod.select_engine(str(input_path))
        except Exception as e:
            logger.warning("engine_select.select_engine failed (%s). Falling back to pandas.", e)

    return "pandas"


def read_dataframe(input_path: Path, engine: str, load_dataframe_mod) -> Tuple[Any, Dict[str, Any]]:
    """
    Read CSV/Parquet into a DataFrame using services.load_dataframe.
    Returns (df, metadata)
    metadata may include: {"rows": int, "cols": int, "engine": "pandas"|"spark"}
    """
    if not hasattr(load_dataframe_mod, "load_df"):
        raise RuntimeError("services.load_dataframe.load_df(...) not found.")

    df, meta = load_dataframe_mod.load_df(str(input_path), engine=engine)
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("engine", engine)
    return df, meta


def normalize_schema(df: Any, schema_detect_mod, schema_map_path: Path) -> Tuple[Any, Any]:
    """
    Normalize column names to KOI standard using services.schema_detect.
    Returns (df_normalized, mapping)
    mapping example: [{"detected": "period", "mapped_to": "koi_period", "status": "ok"}, ...]
    """
    if not hasattr(schema_detect_mod, "normalize_columns"):
        raise RuntimeError("services.schema_detect.normalize_columns(...) not found.")

    mapping = None
    if schema_map_path.exists() and hasattr(schema_detect_mod, "load_expected_map"):
        try:
            schema_detect_mod.load_expected_map(str(schema_map_path))
        except Exception as e:
            logger.warning("Could not load expected_columns.json map: %s", e)

    df_norm, mapping = schema_detect_mod.normalize_columns(df)
    return df_norm, mapping


def enrich_metrics(df: Any, compute_metrics_mod, albedo: float, auto_derive: bool = True) -> Any:
    """
    Compute derived metrics (a, Teq, Insolation, depth if possible).
    """
    if not hasattr(compute_metrics_mod, "enrich"):
        logger.info("services.compute_metrics.enrich(...) not found. Skipping derived metrics.")
        return df

    assumptions = {"albedo": albedo}
    options = {"auto_derive": auto_derive}
    try:
        df2 = compute_metrics_mod.enrich(df, assumptions=assumptions, options=options)
        return df2
    except Exception as e:
        logger.warning("compute_metrics.enrich failed: %s. Continuing with original dataframe.", e)
        return df


def build_features(df: Any, feature_builder_mod, feature_specs_path: Path) -> Tuple[Any, Dict[str, Any]]:
    """
    Build feature vectors for ML. Returns (features_df, meta)
    meta example: {"features": ["f1","f2",...]}
    """
    if not hasattr(feature_builder_mod, "build"):
        raise RuntimeError("services.feature_builder.build(...) not found.")

    return feature_builder_mod.build(df, specs_path=str(feature_specs_path))


def write_parquet(df: Any, out_path: Path, engine: str = "pandas") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # For pandas
    if engine == "pandas":
        try:
            import pandas as pd  # type: ignore
            if not isinstance(df, pd.DataFrame):
                # Try to convert Spark → Pandas if necessary
                try:
                    df = df.toPandas()
                except Exception:
                    raise TypeError("Expected pandas.DataFrame; got different type and cannot convert.")
            df.to_parquet(out_path, index=False)
            return
        except Exception as e:
            raise RuntimeError(f"Failed writing parquet with pandas: {e}")

    # For spark
    if engine == "spark":
        try:
            # Spark DataFrame has write.parquet
            df.write.mode("overwrite").parquet(str(out_path))
            return
        except Exception as e:
            raise RuntimeError(f"Failed writing parquet with spark: {e}")

    raise ValueError(f"Unsupported engine: {engine}")


def write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# -----------------------------------------
# CLI
# -----------------------------------------
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ExoHunter — make_dataset: build feature parquet from raw KOI file (CSV/Parquet)."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path or bucket URL to input data file (CSV/Parquet)."
    )
    p.add_argument(
        "--feature-version",
        default=DEFAULT_FEATURE_VERSION,
        help=f"Feature store version (default: {DEFAULT_FEATURE_VERSION})."
    )
    p.add_argument(
        "--output-dir",
        default=str(FEATURE_STORE_DIR / DEFAULT_FEATURE_VERSION),
        help="Destination directory for features parquet (default: feature_store/<version>)."
    )
    p.add_argument(
        "--feature-specs",
        default=str(DEFAULT_FEATURE_SPECS),
        help=f"Path to feature_specs.yaml (default: {DEFAULT_FEATURE_SPECS})."
    )
    p.add_argument(
        "--schema-map",
        default=str(DEFAULT_SCHEMA_MAP),
        help=f"Path to schemas/expected_columns.json (default: {DEFAULT_SCHEMA_MAP})."
    )
    p.add_argument(
        "--engine",
        choices=("auto", "pandas", "spark"),
        default="auto",
        help="Force engine or auto-select (default: auto)."
    )
    p.add_argument(
        "--albedo",
        type=float,
        default=DEFAULT_ALBEDO,
        help=f"Assumed albedo for Teq (default: {DEFAULT_ALBEDO})."
    )
    p.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional: limit rows for quick tests."
    )
    p.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)."
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    logger.info("Starting make_dataset with args: %s", vars(args))

    # Import services
    storage_mod, engine_select_mod, load_dataframe_mod, schema_detect_mod, compute_metrics_mod, feature_builder_mod = _import_services()

    # Resolve input
    input_path = resolve_input_path(args.input, storage_mod)
    logger.info("Resolved input to local path: %s", input_path)

    # Choose engine
    engine = choose_engine(input_path, args.engine, engine_select_mod)
    logger.info("Selected engine: %s", engine)

    # Read dataframe
    df, meta = read_dataframe(input_path, engine, load_dataframe_mod)
    logger.info("Loaded dataframe (%s): %s", engine, meta)

    # Optional limit rows (dev/testing)
    if args.limit_rows is not None and hasattr(df, "head"):
        try:
            import pandas as pd  # type: ignore
            if engine == "spark":
                # convert to pandas and head
                df = df.limit(args.limit_rows).toPandas()
                engine = "pandas"
            else:
                # pandas
                df = df.head(args.limit_rows)
        except Exception as e:
            logger.warning("Failed to apply --limit-rows: %s", e)

    # Normalize schema
    df, mapping = normalize_schema(df, schema_detect_mod, Path(args.schema_map))
    logger.info("Schema normalized. Example mapping (first 10): %s", (mapping[:10] if isinstance(mapping, list) else "n/a"))

    # Derived metrics
    df = enrich_metrics(df, compute_metrics_mod, albedo=args.albedo, auto_derive=True)
    logger.info("Derived metrics computed (if service available).")

    # Build features
    features_df, feats_meta = build_features(df, feature_builder_mod, Path(args.feature_specs))
    feature_names = feats_meta.get("features") if isinstance(feats_meta, dict) else None
    logger.info("Built features: %d columns.", (len(feature_names) if feature_names else getattr(features_df, "shape", ['?','?'])[1]))

    # Output paths
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"features_{ts}.parquet"
    out_manifest = out_dir / f"features_{ts}.manifest.json"

    # Write features
    write_parquet(features_df, out_parquet, engine="pandas")  # save as pandas parquet for portability
    logger.info("Wrote features parquet: %s", out_parquet)

    # Manifest
    manifest = {
        "feature_version": args.feature_version,
        "source_file": str(input_path),
        "engine_loaded": meta.get("engine"),
        "rows_loaded": meta.get("rows"),
        "features_file": str(out_parquet),
        "feature_specs": str(args.feature_specs),
        "schema_map": str(args.schema_map),
        "albedo": args.albedo,
        "timestamp": ts,
        "feature_names": feature_names,
    }
    write_manifest(out_manifest, manifest)
    logger.info("Wrote manifest: %s", out_manifest)

    # Also write/refresh a convenience symlink/copy "latest" pointer
    try:
        latest_manifest = out_dir / "features_latest.manifest.json"
        with open(latest_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        logger.info("Updated latest manifest: %s", latest_manifest)
    except Exception as e:
        logger.warning("Could not update latest manifest: %s", e)

    logger.info("make_dataset finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
