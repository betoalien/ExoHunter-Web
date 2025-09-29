# routes/health.py
from __future__ import annotations

import os
import sys
import platform
from pathlib import Path
from datetime import datetime

from flask import Blueprint, jsonify, current_app

health_bp = Blueprint("health", __name__, url_prefix="/api/health")


@health_bp.get("/")
def health_root():
    """
    Basic health endpoint.
    Returns metadata about the app, environment, and feature store.
    """
    try:
        base_dir = current_app.config.get("BASE_DIR", str(Path(__file__).resolve().parents[1]))
        feature_version = current_app.config.get("FEATURE_VERSION", "v1")
        feature_store = Path(base_dir) / "feature_store" / feature_version

        payload = {
            "status": "ok",
            "app": "ExoHunter",
            "mode": "api",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "python": sys.version.split()[0],
            "platform": platform.system(),
            "base_dir": str(base_dir),
            "feature_version": feature_version,
            "feature_store_exists": feature_store.exists(),
            "pid": os.getpid(),
        }
        return jsonify(payload), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@health_bp.get("/ping")
def health_ping():
    """Simple ping → pong for quick connectivity tests."""
    return jsonify({"ping": "pong"}), 200


@health_bp.get("/ready")
def health_ready():
    """
    Readiness probe.
    Checks that feature store directory is accessible.
    """
    try:
        base_dir = current_app.config.get("BASE_DIR", str(Path(__file__).resolve().parents[1]))
        feature_version = current_app.config.get("FEATURE_VERSION", "v1")
        feature_store = Path(base_dir) / "feature_store" / feature_version

        if feature_store.exists() and feature_store.is_dir():
            return jsonify({"ready": True, "feature_store": str(feature_store)}), 200
        else:
            return jsonify({"ready": False, "error": "feature_store not found"}), 503
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)}), 500
