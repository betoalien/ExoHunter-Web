# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """
    Global configuration for ExoHunter.
    Values are loaded from environment variables when available.
    """

    # -------------------------------------------------
    # Core Flask
    # -------------------------------------------------
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
    DEBUG = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True", "yes", "YES")
    TESTING = os.getenv("FLASK_TESTING", "0") in ("1", "true", "True", "yes", "YES")

    # -------------------------------------------------
    # File upload / limits
    # -------------------------------------------------
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "500")) * 1024 * 1024  # default 500 MB
    ALLOWED_EXTENSIONS = {"csv", "parquet"}

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------
    BASE_DIR = str(BASE_DIR)
    UPLOADS_DIR = str(BASE_DIR / "data" / "uploads")
    OUTPUTS_DIR = str(BASE_DIR / "data" / "outputs")
    RAW_DIR = str(BASE_DIR / "data" / "raw")
    INTERIM_DIR = str(BASE_DIR / "data" / "interim")
    LABELED_DIR = str(BASE_DIR / "data" / "labeled")
    FEATURE_STORE_DIR = str(BASE_DIR / "feature_store")

    # -------------------------------------------------
    # CORS
    # -------------------------------------------------
    CORS_RESOURCES = {
        r"/api/*": {
            "origins": os.getenv("CORS_ORIGINS", "*").split(","),
        }
    }

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # -------------------------------------------------
    # ML / Feature store
    # -------------------------------------------------
    FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v1")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "baseline")

    # -------------------------------------------------
    # External storage (optional)
    # -------------------------------------------------
    STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # options: local, s3, gcs, azure
    BUCKET_URL = os.getenv("BUCKET_URL", "")  # e.g. s3://mybucket/exohunter

    # -------------------------------------------------
    # Misc
    # -------------------------------------------------
    DEFAULT_ALBEDO = float(os.getenv("DEFAULT_ALBEDO", "0.3"))  # used for Teq calculations
    DEFAULT_THRESHOLD_FILE = str(BASE_DIR / "configs" / "thresholds.yaml")
    DEFAULT_LABELING_FILE = str(BASE_DIR / "configs" / "labeling.yaml")
    DEFAULT_MLFLOW_FILE = str(BASE_DIR / "configs" / "mlflow.yaml")
