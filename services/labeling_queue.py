# services/labeling_queue.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Paths & helpers
# -----------------------------
def _root_dir() -> Path:
    # .../web_app
    return Path(__file__).resolve().parents[1]

def _configs_dir() -> Path:
    return _root_dir() / "configs"

def _labeled_dir() -> Path:
    d = _root_dir() / "data" / "labeled"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _outputs_dir() -> Path:
    d = _root_dir() / "data" / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _queue_path() -> Path:
    return _labeled_dir() / "labeling_queue.json"

def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _file_exists(p: Optional[Path]) -> bool:
    return bool(p and p.exists())

# -----------------------------
# Config loader
# -----------------------------
def load_labeling_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configs/labeling.yaml. Returns defaults if missing.
    """
    defaults: Dict[str, Any] = {
        "batch": {"size": 50, "strategy": "priority"},
        "priority": ["anomalous_signal", "candidate", "habitable_zone_candidate", "rocky_candidate"],
        "interesting_flags": [
            "FLAG_ANOMALOUS_PERIOD",
            "FLAG_ANOMALOUS_DEPTH",
            "FLAG_DURATION_INCONSISTENT",
            "FLAG_MISSING_PRAD",
            "FLAG_ASSUMED_ALBEDO",
        ],
        "weights": {
            "category": {
                "anomalous_signal": 5.0,
                "candidate": 3.0,
                "habitable_zone_candidate": 2.5,
                "rocky_candidate": 2.0,
                "hot_jupiter_candidate": 1.5,
                "likely_false_positive": -1.0,
                "no_result": -2.0,
            },
            "flags": {"default": 0.5},
            "uncertainty": {"enabled": True, "factor": 2.0},
        },
        "exclude": {"dispositions": ["confirmed_planet", "likely_false_positive"]},
    }
    try:
        import yaml  # type: ignore
        ypath = Path(path) if path else (_configs_dir() / "labeling.yaml")
        if ypath.exists():
            with open(ypath, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            # shallow merge
            for k, v in user_cfg.items():
                if isinstance(v, dict) and isinstance(defaults.get(k), dict):
                    defaults[k].update(v)
                else:
                    defaults[k] = v
    except Exception:
        pass
    return defaults

# -----------------------------
# Input Readers
# -----------------------------
def _latest_outputs() -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    """
    Return (csv_path, json_path, ts) of the latest report in data/outputs/.
    Prefer report_latest.*; otherwise pick highest timestamp.
    """
    out = _outputs_dir()
    latest_csv = out / "report_latest.csv"
    latest_json = out / "report_latest.json"
    if latest_csv.exists() or latest_json.exists():
        return (latest_csv if latest_csv.exists() else None, latest_json if latest_json.exists() else None, "latest")

    def ts_from(p: Path) -> Optional[str]:
        m = re.search(r"report_(\d{8}_\d{6})\.", p.name)
        return m.group(1) if m else None

    csvs = sorted(out.glob("report_*.csv"), reverse=True)
    jsons = sorted(out.glob("report_*.json"), reverse=True)

    best_csv = csvs[0] if csvs else None
    best_json = jsons[0] if jsons else None
    ts = ts_from(best_json or best_csv) if (best_csv or best_json) else None
    return best_csv, best_json, ts

def _read_results_any(csv_path: Optional[Path], json_path: Optional[Path]):
    """
    Read results into a pandas DataFrame from JSON (preferred) or CSV fallback.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas is required to read outputs: {e}")

    if _file_exists(json_path):
        data = _read_json(json_path)  # expects {"rows": [...]}
        rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
        return pd.DataFrame(rows)
    if _file_exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError("No outputs found in data/outputs/. Run the process pipeline first.")

# -----------------------------
# Scoring
# -----------------------------
def _split_flags(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v).strip()
    return [f for f in s.split(";") if f]

def _uncertainty_boost(row: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    If model uncertainty is available (e.g., 'uncertainty' column or 'score' near 0.5),
    return a bonus weight. Configurable via weights.uncertainty.
    """
    wcfg = cfg.get("weights", {}).get("uncertainty", {})
    if not bool(wcfg.get("enabled", True)):
        return 0.0
    factor = float(wcfg.get("factor", 2.0))
    # Heurística: si hay columna 'uncertainty', úsala; si no, usa |0.5 - score|
    ucol = row.get("uncertainty")
    if ucol is not None:
        try:
            u = float(ucol)
            return max(0.0, min(1.0, u)) * factor
        except Exception:
            pass
    # sin uncertainty explícita, boost si score ~0.5 (zona gris)
    try:
        s = float(row.get("score", 0.0))
        return (1.0 - abs(0.5 - s) * 2.0) * factor  # máx en 0.5, min en 0 o 1
    except Exception:
        return 0.0

def _labeling_score(row: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Compute a labeling score using:
      - category weight
      - interesting flags count
      - uncertainty boost
    """
    w = cfg.get("weights", {})
    w_cat = w.get("category", {})
    w_flags = float(w.get("flags", {}).get("default", 0.5))

    cat = str(row.get("category") or "").strip()
    base = float(w_cat.get(cat, 0.0))

    flags = _split_flags(row.get("flags"))
    interesting = set(cfg.get("interesting_flags", []))
    n_interesting = sum(1 for f in flags if f in interesting)

    score = base + (n_interesting * w_flags) + _uncertainty_boost(row, cfg)
    return float(score)

# -----------------------------
# Public API
# -----------------------------
def build_queue_from_df(df, cfg: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Create a labeling queue from a Pandas (or Spark) DataFrame of results.

    Returns a list of items:
      { "object_id": "...", "suggested_label": "<category>", "score": <float>, "flags": [...], "source_ts": "...?" }
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Pandas is required: {e}")

    # Spark → Pandas
    if not hasattr(df, "to_dict") and hasattr(df, "toPandas"):
        df = df.toPandas()
    if not hasattr(df, "to_dict"):
        raise TypeError("build_queue_from_df expects a Pandas or Spark DataFrame.")

    cfg = cfg or load_labeling_config()

    # Exclusions
    ex = cfg.get("exclude", {})
    ex_labels = set(ex.get("labels", []) or [])
    ex_disp = set((ex.get("dispositions", []) or []))

    # Fields present?
    has_obj = "object_id" in df.columns
    has_cat = "category" in df.columns
    has_score = "score" in df.columns

    # Prepare rows
    items: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        cat = str(rec.get("category") or "")
        if cat in ex_labels:
            continue
        disp = str(rec.get("koi_disposition") or "").lower()
        if disp in ex_disp:
            continue

        oid = rec.get("object_id") if has_obj else None
        score = float(rec.get("score") or 0.0) if has_score else 0.0
        flags = _split_flags(rec.get("flags"))
        item = {
            "object_id": oid or f"OBJ-{len(items)}",
            "suggested_label": cat or "candidate",
            "score_model": score,
            "score_labeling": _labeling_score(rec, cfg),
            "flags": flags,
        }
        items.append(item)

    # Ranking strategy
    strat = (cfg.get("batch", {}).get("strategy") or "priority").lower()
    if strat == "random":
        import random
        random.shuffle(items)
    elif strat in ("uncertainty", "low_confidence"):
        # Ascending by model confidence (score near 0.5 first)
        items.sort(key=lambda x: abs(0.5 - float(x.get("score_model", 0.0))))
    else:
        # Default: priority by score_labeling (desc)
        items.sort(key=lambda x: float(x.get("score_labeling", 0.0)), reverse=True)

    # Apply priority list to push categories to the front (stable sort trick)
    priority_list = cfg.get("priority", [])
    if priority_list:
        pr_index = {name: i for i, name in enumerate(priority_list)}
        items.sort(key=lambda x: pr_index.get(str(x.get("suggested_label")), len(pr_index)))

    if limit:
        items = items[: int(limit)]
    return items

def build_queue_from_latest_outputs(limit: Optional[int] = None, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Load the latest report from data/outputs/ and build a queue.
    """
    csv_path, json_path, _ts = _latest_outputs()
    df = _read_results_any(csv_path, json_path)
    return build_queue_from_df(df, cfg=cfg, limit=limit or load_labeling_config().get("batch", {}).get("size", 50))

def save_queue(queue_items: List[Dict[str, Any]], path: Optional[str] = None) -> Path:
    """
    Persist queue to data/labeled/labeling_queue.json (or custom path).
    """
    qpath = Path(path) if path else _queue_path()
    _write_json(qpath, queue_items)
    return qpath

def update_queue_from_latest(limit: Optional[int] = None, cfg_path: Optional[str] = None) -> Dict[str, Any]:
    """
    One-shot helper:
      - load labeling config
      - build queue from latest outputs
      - save to labeling_queue.json
      - return summary
    """
    cfg = load_labeling_config(cfg_path)
    queue = build_queue_from_latest_outputs(limit=limit, cfg=cfg)
    qpath = save_queue(queue)
    return {"queue_size": len(queue), "queue_path": str(qpath)}

def merge_with_existing_queue(new_items: List[Dict[str, Any]], dedup_on: str = "object_id") -> List[Dict[str, Any]]:
    """
    Merge new items with existing labeling_queue.json, deduplicating by object_id.
    """
    existing = []
    qpath = _queue_path()
    if qpath.exists():
        try:
            existing = _read_json(qpath) or []
        except Exception:
            existing = []

    seen = set()
    merged: List[Dict[str, Any]] = []

    # Keep existing order for already queued items
    for it in existing:
        key = str(it.get(dedup_on))
        if key not in seen:
            merged.append(it)
            seen.add(key)

    # Append new ones if not present
    for it in new_items:
        key = str(it.get(dedup_on))
        if key not in seen:
            merged.append(it)
            seen.add(key)

    save_queue(merged)
    return merged
