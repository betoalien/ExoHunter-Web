# services/logging_utils.py
from __future__ import annotations

import os
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_LOG_NAME = "exo"
DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_DIR = os.getenv("LOG_DIR", "logs")
DEFAULT_FILE = os.getenv("LOG_FILE", "exohunter.log")
DEFAULT_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10MB
DEFAULT_BACKUPS = int(os.getenv("LOG_BACKUPS", "5"))
DEFAULT_JSON = os.getenv("LOG_JSON", "0") in ("1", "true", "True", "YES", "yes")


# -----------------------------
# Formatters
# -----------------------------
class _PlainFormatter(logging.Formatter):
    DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt or self.DEFAULT_FMT, datefmt or self.DEFAULT_DATEFMT)

    def formatTime(self, record, datefmt=None):
        # Always UTC ISO-like
        return datetime.utcfromtimestamp(record.created).strftime(datefmt or self.DEFAULT_DATEFMT)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.utcfromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)  # merge extras if provided
        return json.dumps(payload, ensure_ascii=False)


def _level_from_string(level_str: str | int | None) -> int:
    if isinstance(level_str, int):
        return level_str
    try:
        return getattr(logging, str(level_str or DEFAULT_LEVEL).upper())
    except Exception:
        return logging.INFO


# -----------------------------
# Handlers
# -----------------------------
def _make_console_handler(level: int, json_output: bool) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter() if json_output else _PlainFormatter())
    return handler


def _make_file_handler(base_dir: Path, level: int, json_output: bool) -> logging.Handler:
    log_dir = base_dir / DEFAULT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / DEFAULT_FILE

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUPS,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter() if json_output else _PlainFormatter())
    return handler


# -----------------------------
# Public API
# -----------------------------
def setup_logging(
    name: str = DEFAULT_LOG_NAME,
    level: str | int | None = None,
    base_dir: str | Path | None = None,
    console: bool = True,
    file: bool = True,
    json_output: bool | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger with console and rotating file handlers.

    Args:
        name: logger name (e.g., "exo", "workers", "api")
        level: log level ("DEBUG","INFO","WARNING","ERROR")
        base_dir: base directory to place logs/ (defaults to process CWD or app BASE_DIR)
        console: add console handler
        file: add rotating file handler logs/exohunter.log
        json_output: force JSON logs; defaults to env LOG_JSON
        propagate: control propagation to root logger

    Returns:
        logging.Logger
    """
    lvl = _level_from_string(level)
    json_out = DEFAULT_JSON if json_output is None else bool(json_output)

    # Determine base dir
    if base_dir is None:
        # Try Flask app config if available
        try:
            from flask import current_app  # type: ignore
            base_dir = current_app.config.get("BASE_DIR", str(Path.cwd()))
        except Exception:
            base_dir = str(Path.cwd())

    base_dir = Path(base_dir)

    logger = logging.getLogger(name)
    logger.setLevel(lvl)
    logger.propagate = propagate

    # Avoid duplicate handlers if re-setup
    _remove_handlers(logger)

    if console:
        logger.addHandler(_make_console_handler(lvl, json_out))
    if file:
        try:
            logger.addHandler(_make_file_handler(base_dir, lvl, json_out))
        except Exception:
            # If file handler fails (read-only FS), keep console only
            pass

    return logger


def _remove_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
            h.close()
        except Exception:
            pass


# -----------------------------
# Flask integration
# -----------------------------
def patch_flask_logging(app, logger: Optional[logging.Logger] = None) -> None:
    """
    Wire Flask logs to our logger and add a simple request logger.

    Usage:
        from services.logging_utils import setup_logging, patch_flask_logging
        log = setup_logging("exo", level="INFO", base_dir=app.config["BASE_DIR"])
        patch_flask_logging(app, log)
    """
    log = logger or setup_logging(base_dir=app.config.get("BASE_DIR", Path.cwd()))

    # Silence overly chatty loggers (optional)
    for noisy in ("werkzeug", "urllib3", "botocore", "s3transfer"):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass

    # Redirect Flask's app.logger to ours
    app.logger.handlers = log.handlers
    app.logger.setLevel(log.level)
    app.logger.propagate = False

    # Add request logging hooks
    @app.before_request
    def _log_request():
        try:
            from flask import request  # type: ignore
            app.logger.info(
                "REQ %s %s", request.method, request.path,
                extra={"extra": {"remote_addr": request.remote_addr, "agent": request.user_agent.string}}
            )
        except Exception:
            pass

    @app.after_request
    def _log_response(response):
        try:
            app.logger.info("RES %s %s → %s", response.request.method, response.request.path, response.status_code)  # type: ignore
        except Exception:
            pass
        return response

    @app.errorhandler(Exception)
    def _log_exception(err):
        try:
            app.logger.exception("Unhandled exception: %s", err)
        except Exception:
            pass
        # Let Flask handle the response (app's errorhandler or default)
        return err
