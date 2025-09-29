# workers/tasks.py
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# -----------------------------
# Project paths & logging
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))  # allow "services.*" imports

def _setup_logger():
    try:
        from services.logging_utils import setup_logging
        return setup_logging(
            name="exo.tasks",
            level=os.getenv("LOG_LEVEL", "INFO"),
            base_dir=str(BASE_DIR),
            console=True,
            file=True,
            json_output=os.getenv("LOG_JSON", "0") in ("1", "true", "True"),
        )
    except Exception:
        import logging
        logging.basicConfig(
            level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        return logging.getLogger("exo.tasks")

log = _setup_logger()

# -----------------------------
# Imports (services & pipelines)
# -----------------------------
def _imp(name: str):
    import importlib
    try:
        return importlib.import_module(name)
    except Exception as e:
        log.warning("Module '%s' not available: %s", name, e)
        return None

engine_select = _imp("services.engine_select")
load_dataframe = _imp("services.load_dataframe")
schema_detect = _imp("services.schema_detect")
compute_metrics = _imp("services.compute_metrics")
classify = _imp("services.classify")
export_report = _imp("services.export_report")
feature_builder = _imp("services.feature_builder")
labeling_queue = _imp("services.labeling_queue")
storage = _imp("services.storage")
make_dataset_py = _imp("ml.pipelines.make_dataset")


# -----------------------------
# Helpers
# -----------------------------
def _resolve_upload(file_id_or_path: str) -> Path:
    """
    If a file_id exists in data/uploads/uploads_index.json, use mapping.
    Else treat as direct path (absolute or relative).
    """
    idx_path = BASE_DIR / "data" / "uploads" / "uploads_index.json"
    p = None
    if idx_path.exists():
        try:
            mapping = json.loads(idx_path.read_text(encoding="utf-8"))
            mp = mapping.get(file_id_or_path)
            if mp and Path(mp).exists():
                p = Path(mp)
        except Exception:
            pass
    if p is None:
        cand = Path(file_id_or_path)
        if cand.exists():
            p = cand
    if p is None:
        # Try name inside uploads/
        up = BASE_DIR / "data" / "uploads" / file_id_or_path
        if up.exists():
            p = up
    if p is None:
        raise FileNotFoundError(f"Could not resolve '{file_id_or_path}' to an uploaded file.")
    return p


def _select_engine(path: Path, requested: Optional[str] = None) -> str:
    if requested in ("pandas", "spark"):
        return requested
    if engine_select and hasattr(engine_select, "select_engine"):
        try:
            return engine_select.select_engine(str(path))
        except Exception as e:
            log.warning("engine_select failed (%s); using pandas", e)
    return "pandas"


def _to_pandas(df):
    if hasattr(df, "to_dict"):
        return df
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError("Expected a Pandas or Spark DataFrame")


# -----------------------------
# Tasks
# -----------------------------
def task_process(
    file_id_or_path: str,
    engine: str = "auto",
    auto_derive: bool = True,
    albedo: float = 0.3,
) -> Dict[str, Any]:
    """
    Run the same pipeline as /api/process but from CLI.
    Returns dict with csv_path, json_path, rows, engine.
    """
    if not all([load_dataframe, schema_detect, compute_metrics, classify, export_report]):
        raise RuntimeError("Some required services are missing (load_dataframe/schema_detect/compute_metrics/classify/export_report).")

    input_path = _resolve_upload(file_id_or_path)
    eng = _select_engine(input_path, None if engine == "auto" else engine)
    log.info("Processing %s with engine=%s", input_path, eng)

    # Load
    df, meta = load_dataframe.load_df(str(input_path), engine=eng)
    log.info("Loaded rows=%s cols=%s", meta.get("rows"), meta.get("cols"))

    # Normalize schema
    # Load expected map if available
    exp_map = BASE_DIR / "schemas" / "expected_columns.json"
    if exp_map.exists() and hasattr(schema_detect, "load_expected_map"):
        schema_detect.load_expected_map(str(exp_map))
    df, mapping = schema_detect.normalize_columns(df)
    log.debug("Schema mapping applied: %s", mapping[:5])

    # Compute metrics
    df = compute_metrics.enrich(df, assumptions={"albedo": albedo}, options={"auto_derive": auto_derive})
    log.info("Computed derived metrics (teq/insol/transit_depth/durations)")

    # Classify
    thr = BASE_DIR / "configs" / "thresholds.yaml"
    thr_path = str(thr) if thr.exists() else None
    results_df = classify.classify_df(df, thresholds_path=thr_path)
    rows = int(getattr(results_df, "shape", [0, 0])[0])
    log.info("Classified %d rows", rows)

    # Export
    out_dir = BASE_DIR / "data" / "outputs"
    paths = export_report.write_outputs(results_df, out_dir=str(out_dir), extra_meta={"generated_by": "workers.tasks"})
    log.info("Wrote outputs: %s", paths)

    return {"csv_path": paths["csv_path"], "json_path": paths["json_path"], "rows": rows, "engine": meta.get("engine", eng)}


def task_make_dataset(
    input_path: str,
    feature_version: str = "v1",
    engine: str = "auto",
    albedo: float = 0.3,
    auto_derive: bool = True,
) -> Dict[str, Any]:
    """
    Build features & manifest via ml/pipelines/make_dataset.py.
    """
    if not make_dataset_py or not hasattr(make_dataset_py, "run"):
        raise RuntimeError("ml/pipelines/make_dataset.py not available or missing 'run' function.")
    res = make_dataset_py.run(
        input_path=input_path,
        feature_version=feature_version,
        engine=engine,
        albedo=albedo,
        auto_derive=auto_derive,
    )
    log.info("make_dataset result: %s", res)
    return res


def task_update_label_queue(limit: Optional[int] = None, cfg_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Build/update the labeling queue from latest outputs.
    """
    if not labeling_queue:
        raise RuntimeError("services/labeling_queue.py not available.")
    summary = labeling_queue.update_queue_from_latest(limit=limit, cfg_path=cfg_path)
    log.info("Labeling queue updated: %s", summary)
    return summary


# -----------------------------
# CLI
# -----------------------------
def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exohunter-tasks",
        description="Task runner for ExoHunter pipelines (process, make-dataset, labeling queue).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # process
    p_proc = sub.add_parser("process", help="Process an uploaded file (ingest → metrics → classify → outputs)")
    p_proc.add_argument("file", help="file_id from uploads_index.json OR path to CSV/Parquet")
    p_proc.add_argument("--engine", choices=["auto", "pandas", "spark"], default="auto")
    p_proc.add_argument("--albedo", type=float, default=0.3)
    p_proc.add_argument("--no-auto-derive", action="store_true", help="Disable automatic derivation of metrics (a, Teq, Insol)")

    # make-dataset
    p_mk = sub.add_parser("make-dataset", help="Build features & manifest (feature store)")
    p_mk.add_argument("input", help="Path to CSV/Parquet to turn into a featureset")
    p_mk.add_argument("--version", default="v1", help="Feature store version (e.g., v1)")
    p_mk.add_argument("--engine", choices=["auto", "pandas", "spark"], default="auto")
    p_mk.add_argument("--albedo", type=float, default=0.3)
    p_mk.add_argument("--no-auto-derive", action="store_true")

    # label-queue
    p_lq = sub.add_parser("label-queue", help="Update labeling queue from the latest outputs")
    p_lq.add_argument("--limit", type=int, default=None)
    p_lq.add_argument("--config", default=None, help="Path to configs/labeling.yaml (optional)")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    try:
        if args.cmd == "process":
            res = task_process(
                file_id_or_path=args.file,
                engine=args.engine,
                auto_derive=not args.no_auto_derive,
                albedo=args.albedo,
            )
            print(json.dumps(res, indent=2))
            return 0

        if args.cmd == "make-dataset":
            res = task_make_dataset(
                input_path=args.input,
                feature_version=args.version,
                engine=args.engine,
                albedo=args.albedo,
                auto_derive=not args.no_auto_derive,
            )
            print(json.dumps(res, indent=2))
            return 0

        if args.cmd == "label-queue":
            res = task_update_label_queue(limit=args.limit, cfg_path=args.config)
            print(json.dumps(res, indent=2))
            return 0

        parser.print_help()
        return 1
    except Exception as e:
        log.exception("Task failed: %s", e)
        print(json.dumps({"error": str(e)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
