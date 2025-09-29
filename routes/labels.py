# routes/labels.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request, current_app

labels_bp = Blueprint("labels", __name__, url_prefix="/api/labels")


# -----------------------------
# Helpers
# -----------------------------
def _json_error(message: str, code: int = 400, **extra):
    return jsonify({"error": message, **extra}), code


def _project_root() -> Path:
    base = current_app.config.get("BASE_DIR")
    return Path(base) if base else Path(__file__).resolve().parents[1]


def _load_labeling_config() -> Dict[str, Any]:
    cfg_path = _project_root() / "configs" / "labeling.yaml"
    try:
        import yaml  # type: ignore
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load labeling.yaml: {e}")


def _queue_file() -> Path:
    return _project_root() / "data" / "labeled" / "labeling_queue.json"


def _load_queue() -> list[Dict[str, Any]]:
    qpath = _queue_file()
    if not qpath.exists():
        return []
    try:
        with open(qpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_queue(queue: list[Dict[str, Any]]) -> None:
    qpath = _queue_file()
    qpath.parent.mkdir(parents=True, exist_ok=True)
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2, ensure_ascii=False)


# -----------------------------
# Routes
# -----------------------------
@labels_bp.get("/config")
def labels_config():
    """Return current labeling.yaml configuration."""
    try:
        cfg = _load_labeling_config()
        return jsonify(cfg)
    except Exception as e:
        return _json_error(str(e), 500)


@labels_bp.get("/queue")
def labels_queue():
    """
    Return current labeling queue.
    Query params:
      - limit: max items to return
    """
    try:
        limit = int(request.args.get("limit", "50"))
        queue = _load_queue()
        if limit > 0:
            queue = queue[:limit]
        return jsonify({"count": len(queue), "items": queue})
    except Exception as e:
        return _json_error(f"Failed to load queue: {e}", 500)


@labels_bp.post("/queue")
def labels_enqueue():
    """
    Add items to labeling queue manually.
    Expected body: JSON array of objects with at least {"object_id": "..."}
    """
    try:
        items = request.get_json(force=True)
        if not isinstance(items, list):
            return _json_error("Body must be a JSON array of objects.", 400)

        queue = _load_queue()
        added = 0
        for item in items:
            if not isinstance(item, dict) or "object_id" not in item:
                continue
            queue.append(item)
            added += 1

        _save_queue(queue)
        return jsonify({"status": "ok", "added": added, "queue_size": len(queue)})
    except Exception as e:
        return _json_error(f"Failed to enqueue: {e}", 500)


@labels_bp.post("/apply")
def labels_apply():
    """
    Apply a label to an object.
    Expected body: {"object_id": "...", "label": "...", "notes": "..."}
    Stores result in data/labeled/labeled_results.json
    """
    try:
        data = request.get_json(force=True)
        if not data or "object_id" not in data or "label" not in data:
            return _json_error("Body must include object_id and label.", 400)

        results_file = _project_root() / "data" / "labeled" / "labeled_results.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = []

        entry = {
            "object_id": data["object_id"],
            "label": data["label"],
            "notes": data.get("notes"),
        }
        results.append(entry)

        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Remove from queue if present
        queue = [item for item in _load_queue() if item.get("object_id") != data["object_id"]]
        _save_queue(queue)

        return jsonify({"status": "ok", "labeled": entry, "queue_size": len(queue)})
    except Exception as e:
        return _json_error(f"Failed to apply label: {e}", 500)


@labels_bp.get("/results")
def labels_results():
    """Return all labeled results so far."""
    try:
        results_file = _project_root() / "data" / "labeled" / "labeled_results.json"
        if not results_file.exists():
            return jsonify({"count": 0, "items": []})
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        return jsonify({"count": len(results), "items": results})
    except Exception as e:
        return _json_error(f"Failed to load results: {e}", 500)
