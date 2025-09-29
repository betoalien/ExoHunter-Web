# workers/spark_session.py
from __future__ import annotations

import os
from typing import Optional

# Valores por defecto (puedes sobreescribir con variables de entorno)
_DEF_APP_NAME = os.getenv("SPARK_APP_NAME", "ExoHunter")
_DEF_MASTER = os.getenv("SPARK_MASTER", "local[*]")  # usa todos los cores locales
_DEF_DRIVER_MEM = os.getenv("SPARK_DRIVER_MEMORY", "4g")
_DEF_EXECUTOR_MEM = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
_DEF_SQL_SHUFFLE_PARTS = os.getenv("SPARK_SQL_SHUFFLE_PARTITIONS", "200")
_DEF_WAREHOUSE_DIR = os.getenv("SPARK_WAREHOUSE_DIR", "./spark-warehouse")
_DEF_TIMEZONE = os.getenv("SPARK_TIMEZONE", "UTC")

# Arrow acelera las conversiones Spark <-> Pandas
_DEF_ARROW = os.getenv("SPARK_ARROW_ENABLED", "true").lower() in ("1", "true", "yes")


def _apply_common_configs(builder, app_name: str):
    """
    Configuraciones comunes y razonables para workloads de datos tabulares.
    """
    builder = (
        builder.appName(app_name)
        .config("spark.sql.session.timeZone", _DEF_TIMEZONE)
        .config("spark.sql.execution.arrow.pyspark.enabled", str(_DEF_ARROW).lower())
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", _DEF_SQL_SHUFFLE_PARTS)
        .config("spark.driver.memory", _DEF_DRIVER_MEM)
        .config("spark.executor.memory", _DEF_EXECUTOR_MEM)
        .config("spark.sql.warehouse.dir", _DEF_WAREHOUSE_DIR)
    )

    # Soporte opcional para S3 si las credenciales están en el entorno
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        builder = (
            builder.config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID"))
            .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY"))
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config("spark.hadoop.fs.s3a.path.style.access", os.getenv("S3_PATH_STYLE", "true"))
        )
        # Endpoint alternativo (minio/ceph)
        if os.getenv("S3_ENDPOINT"):
            builder = builder.config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT"))

    return builder


def _winutils_hint():
    """
    En Windows, algunos builds de Hadoop requieren winutils.exe.
    No es estrictamente necesario para la mayoría de lecturas locales/Parquet,
    pero dejamos el hint por si el usuario lo necesita.
    """
    if os.name == "nt" and not os.getenv("HADOOP_HOME"):
        # No lanzamos error; solo informativo en logs de Spark.
        pass


def get_spark(app_name: Optional[str] = None):
    """
    Crea (o retorna) una SparkSession configurada.
    Respeta variables de entorno:
      - SPARK_MASTER (default: local[*])
      - SPARK_APP_NAME
      - SPARK_DRIVER_MEMORY, SPARK_EXECUTOR_MEMORY
      - SPARK_SQL_SHUFFLE_PARTITIONS
      - SPARK_ARROW_ENABLED
      - SPARK_TIMEZONE
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_ENDPOINT (opcional)
    """
    _winutils_hint()

    from pyspark.sql import SparkSession  # type: ignore

    master = os.getenv("SPARK_MASTER", _DEF_MASTER)
    app = app_name or _DEF_APP_NAME

    builder = SparkSession.builder.master(master)
    builder = _apply_common_configs(builder, app)

    # Permite que el usuario pase configs arbitrarias via SPARK_EXTRA_CONFIG (clave1=val1,clave2=val2)
    extra = os.getenv("SPARK_EXTRA_CONFIG", "")
    if extra:
        for kv in extra.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                builder = builder.config(k.strip(), v.strip())

    spark = builder.getOrCreate()

    # En algunos entornos conviene reducir verbosidad
    try:
        spark.sparkContext.setLogLevel(os.getenv("SPARK_LOG_LEVEL", "WARN"))
    except Exception:
        pass

    return spark


def stop_spark(spark=None):
    """
    Cierra la SparkSession si existe.
    """
    try:
        if spark is None:
            from pyspark.sql import SparkSession  # type: ignore
            spark = SparkSession.getActiveSession()
        if spark is not None:
            spark.stop()
    except Exception:
        pass
