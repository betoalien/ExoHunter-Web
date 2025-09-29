# app.py
from __future__ import annotations

import os
import logging
import importlib
from pathlib import Path
from typing import Iterable, Optional

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass

# -----------------------------
# Paths (no las pises después)
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parent  # <- antes BASE_DIR
REQUIRED_DIRS = [
    ROOT_DIR / "data" / "raw",
    ROOT_DIR / "data" / "interim",
    ROOT_DIR / "data" / "labeled",
    ROOT_DIR / "data" / "uploads",
    ROOT_DIR / "data" / "outputs",
    ROOT_DIR / "feature_store" / "v1",
    ROOT_DIR / "feature_store" / "v2",
    ROOT_DIR / "ml" / "artifacts" / "models",
    ROOT_DIR / "ml" / "artifacts" / "scalers_encoders",
    ROOT_DIR / "ml" / "eval" / "confusion_plots",
]

# -----------------------------
# Config loader
# -----------------------------
class _DefaultConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "500")) * 1024 * 1024  # 500 MB
    JSON_SORT_KEYS = False
    JSON_AS_ASCII = False
    # CORS
    CORS_RESOURCES = {r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}}
    # Logging level
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    # Folders (usa ROOT_DIR; no sobreescribas a Path con un str)
    BASE_DIR = str(ROOT_DIR)  # si quieres tener la ruta como string en config
    UPLOADS_DIR = str(ROOT_DIR / "data" / "uploads")
    OUTPUTS_DIR = str(ROOT_DIR / "data" / "outputs")

def _load_config_object() -> object:
    """Try to import Config from config.py. If missing, use _DefaultConfig."""
    try:
        spec = importlib.import_module("config")
        if hasattr(spec, "Config"):
            return getattr(spec, "Config")
    except Exception:
        pass
    return _DefaultConfig

# -----------------------------
# Blueprint registration helper
# -----------------------------
def _import_blueprint(module_name: str, attr_candidates: Iterable[str]) -> Optional[object]:
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        logging.getLogger("exo").warning("Module '%s' not available: %s", module_name, e)
        return None

    for attr in attr_candidates:
        bp = getattr(mod, attr, None)
        if bp is not None:
            return bp

    logging.getLogger("exo").warning(
        "No blueprint attribute %s found in '%s'", list(attr_candidates), module_name
    )
    return None

def _register_blueprints(app: Flask) -> None:
    candidates = [
        ("routes.upload",   ("upload_bp", "bp", "blueprint")),
        ("routes.process",  ("process_bp", "bp", "blueprint")),
        ("routes.results",  ("results_bp", "bp", "blueprint")),
        ("routes.labels",   ("labels_bp", "bp", "blueprint")),
        ("routes.features", ("features_bp", "bp", "blueprint")),
        ("routes.health",   ("health_bp", "bp", "blueprint")),
    ]
    for module_name, attrs in candidates:
        bp = _import_blueprint(module_name, attrs)
        if bp is not None:
            app.register_blueprint(bp)
            app.logger.info("Registered blueprint from %s", module_name)

# -----------------------------
# App factory
# -----------------------------
def create_app(config_object: object | None = None) -> Flask:
    # Ensure directories exist
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(ROOT_DIR / "templates"),
        static_folder=str(ROOT_DIR / "static"),
        static_url_path="/static",
    )

    # Load config
    cfg = config_object or _load_config_object()
    app.config.from_object(cfg)

    # Logging
    log_level = getattr(logging, str(getattr(cfg, "LOG_LEVEL", "INFO")).upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app.logger = logging.getLogger("exo")
    app.logger.setLevel(log_level)

    # CORS for API routes
    CORS(app, resources=getattr(cfg, "CORS_RESOURCES", {r"/api/*": {"origins": "*"}}))

    # Register blueprints
    _register_blueprints(app)

    # -------------------------
    # UI Routes
    # -------------------------
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/results", methods=["GET"])
    def results_ui():
        return render_template("results.html")

    # -------------------------
    # Health & Error Handlers
    # -------------------------
    @app.route("/healthz", methods=["GET"])
    def healthz():
        return jsonify(status="ok", app="ExoHunter", mode="ui+api"), 200

    @app.errorhandler(404)
    def not_found(err):
        if request.path.startswith("/api/"):
            return jsonify(error="Not Found", path=request.path), 404
        return render_template("index.html"), 404

    @app.errorhandler(500)
    def server_error(err):
        app.logger.exception("Unhandled server error: %s", err)
        if request.path.startswith("/api/"):
            return jsonify(error="Internal Server Error"), 500
        return render_template("index.html"), 500

    return app

# -----------------------------
# Expose app for ASGI/WSGI servers
# -----------------------------
# Esto permite: `uvicorn app:app --reload` y también `flask run` (usando FLASK_APP=app.py)
app = create_app()

# -----------------------------
# Dev entrypoint
# -----------------------------
if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True", "YES", "yes")
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug)
