# services/schema_detect.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Cache global del expected map (standard_name -> [aliases...])
_EXPECTED_MAP: Optional[Dict[str, List[str]]] = None
# Índice invertido (alias_normalizado -> standard_name)
_ALIAS_INDEX: Optional[Dict[str, str]] = None


# -----------------------------
# Helpers de normalización
# -----------------------------
def _norm(s: str) -> str:
    """
    Normaliza un nombre de columna:
      - lower
      - strip espacios
      - reemplaza separadores comunes por '_'
      - quita múltiples '_'
      - remove non-alnum/_ (mantener letras y números)
    """
    x = (s or "").strip().lower()
    x = re.sub(r"[ \t\-\./]+", "_", x)
    x = re.sub(r"[^a-z0-9_]", "", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def _project_root() -> Path:
    # .../web_app
    return Path(__file__).resolve().parents[1]


def _expected_columns_path() -> Path:
    return _project_root() / "schemas" / "expected_columns.json"


def _build_alias_index(expected: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Construye índice invertido alias_normalizado -> standard_name.
    Incluye el propio standard_name como alias de sí mismo.
    """
    idx: Dict[str, str] = {}
    for std, aliases in expected.items():
        idx[_norm(std)] = std
        for a in aliases or []:
            idx[_norm(str(a))] = std
    return idx


def _load_default_expected_map() -> Dict[str, List[str]]:
    """
    Carga schemas/expected_columns.json si existe, con fallback mínimo.
    Incluye alias extendidos para koi_disposition.
    """
    path = _expected_columns_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Asegura listas
            for k, v in list(data.items()):
                if isinstance(v, str):
                    data[k] = [v]
                elif not isinstance(v, list):
                    data[k] = []
            # Asegura que koi_disposition tenga alias extendidos (merge)
            _ensure_disposition_aliases(data)
            return data
        except Exception:
            pass

    # Fallback mínimo si no hay archivo
    data = {
        "object_id": ["koi_name", "kepoi_name", "kepid", "id"],
        "koi_period": ["period", "orbital_period", "p", "pl_orbper"],
        "koi_prad": ["planet_radius", "rp", "pl_rade"],
        "koi_sma": ["semi_major_axis", "a", "pl_orbsmax"],
        "koi_teq": ["equilibrium_temp", "teq", "pl_eqt", "equilibrium_temperature"],
        "koi_insol": ["insolation", "flux", "pl_insol"],
        "koi_smass": ["stellar_mass", "st_mass"],
        "koi_srad": ["stellar_radius", "st_rad", "rs", "r_star"],
        "koi_slogg": ["logg", "st_logg"],
        "koi_steff": ["teff", "st_teff", "t_eff", "stellar_teff"],
        "koi_disposition": [
            "disposition", "pl_disposition",
            "koi_dispo", "dispo", "status",
            "koi_pdisposition", "koi_disp", "planet_disposition"
        ],
        "koi_duration": ["duration_hours", "tdur", "transit_duration"],
        "koi_impact": ["impact_parameter", "b"],
    }
    return data


def _ensure_disposition_aliases(data: Dict[str, List[str]]) -> None:
    """
    Garantiza que 'koi_disposition' tenga un set amplio de alias,
    incluso si el JSON no los traía todos.
    """
    base_aliases = set([
        "disposition", "pl_disposition",
        "koi_dispo", "dispo", "status",
        "koi_pdisposition", "koi_disp", "planet_disposition",
    ])
    if "koi_disposition" not in data:
        data["koi_disposition"] = sorted(base_aliases)
        return
    # Merge sin duplicados
    merged = set(data.get("koi_disposition") or [])
    merged |= base_aliases
    data["koi_disposition"] = sorted(merged)


# -----------------------------
# API de carga del expected map
# -----------------------------
def load_expected_map(path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Carga y cachea el expected map (standard_name -> [aliases]).
    Si 'path' es None, usa schemas/expected_columns.json.
    """
    global _EXPECTED_MAP, _ALIAS_INDEX
    if path is None and _EXPECTED_MAP is not None and _ALIAS_INDEX is not None:
        return _EXPECTED_MAP

    if path:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        for k, v in list(data.items()):
            if isinstance(v, str):
                data[k] = [v]
            elif not isinstance(v, list):
                data[k] = []
        _ensure_disposition_aliases(data)
        _EXPECTED_MAP = data
    else:
        _EXPECTED_MAP = _load_default_expected_map()

    _ALIAS_INDEX = _build_alias_index(_EXPECTED_MAP)
    return _EXPECTED_MAP


# -----------------------------
# Preview de columnas (archivo)
# -----------------------------
def preview_columns(path: str) -> Dict[str, Any]:
    """
    Devuelve una previsualización de columnas sin cargar todo el archivo.
    Soporta CSV/TSV/TXT/Parquet.
    Return:
      {
        "columns": [...],
        "rows": <int|None>,
        "format": "csv"|"parquet"|"unknown",
        "path": "<path>"
      }
    """
    p = Path(path)
    out = {"columns": [], "rows": None, "format": "unknown", "path": str(p)}

    if not p.exists():
        return out

    suf = p.suffix.lower()
    try:
        import pandas as pd  # type: ignore
    except Exception:
        # Sin pandas, devolvemos vacío
        return out

    if suf == ".parquet":
        try:
            df = pd.read_parquet(p)
            out["columns"] = list(df.columns)
            out["rows"] = int(df.shape[0])
            out["format"] = "parquet"
            return out
        except Exception:
            return out

    # CSV/TSV
    try:
        if suf == ".tsv":
            df = pd.read_csv(p, nrows=0, sep="\t")
        else:
            df = pd.read_csv(p, nrows=0)  # pandas infiere delimitador básico
        out["columns"] = list(df.columns)
        out["format"] = "csv"
        return out
    except Exception:
        return out


# -----------------------------
# Sugerencias de mapeo
# -----------------------------
def suggest_mapping_from_columns(columns: Iterable[str]) -> List[Dict[str, Any]]:
    """
    A partir de una lista de columnas, propone el mapeo a nombres estándar.
    Devuelve una lista de objetos:
      { "detected": "<col>", "mapped_to": "<standard|detected>", "status": "ok|alias|unknown" }
    """
    load_expected_map()  # asegura cache
    mapping: List[Dict[str, Any]] = []
    assert _ALIAS_INDEX is not None

    for col in columns:
        n = _norm(str(col))
        if n in _ALIAS_INDEX:
            std = _ALIAS_INDEX[n]
            status = "ok" if n == _norm(std) else "alias"
            mapping.append({"detected": col, "mapped_to": std, "status": status})
            continue

        # Heurística simple: si empieza por koi_ y existe en expected, aceptar
        if n.startswith("koi_") and _EXPECTED_MAP and n in _EXPECTED_MAP:
            mapping.append({"detected": col, "mapped_to": n, "status": "ok"})
            continue

        # Fallback: desconocido → identidad
        mapping.append({"detected": col, "mapped_to": col, "status": "unknown"})

    return mapping


def suggest_mapping_from_path(path: str) -> List[Dict[str, Any]]:
    """
    Previsualiza columnas de un archivo y ejecuta suggest_mapping_from_columns.
    """
    prev = preview_columns(path)
    cols = prev.get("columns") or []
    return suggest_mapping_from_columns(cols)


# -----------------------------
# Normalización de DataFrames
# -----------------------------
def _ensure_pandas(df):
    """Acepta Spark DF convirtiéndolo a Pandas si es necesario."""
    if hasattr(df, "toPandas") and not hasattr(df, "to_dict"):
        return df.toPandas()
    return df


def normalize_columns(df_in):
    """
    Renombra columnas de un DataFrame de entrada a los nombres estándar usando
    'schemas/expected_columns.json'. Devuelve:
      (df_normalizado, mapping_list)

    mapping_list es una lista de:
      { "detected": "...", "mapped_to": "...", "status": "ok|alias|unknown" }

    Reglas clave para koi_disposition:
      - Si ya existe 'koi_disposition', NO modificar su contenido (solo renombre si viene con alias).
      - Si no existe ninguna variante, crear 'koi_disposition' con None (vacía).
    """
    load_expected_map()  # asegurar cache
    assert _ALIAS_INDEX is not None

    df = _ensure_pandas(df_in).copy()
    original_cols = list(df.columns)

    mapping_list = suggest_mapping_from_columns(original_cols)

    # Construir dict de renombre solo cuando 'mapped_to' difiera del original
    rename_map: Dict[str, str] = {}
    for m in mapping_list:
        src = m["detected"]
        dst = m["mapped_to"]
        if src != dst:
            rename_map[src] = dst

    if rename_map:
        df = df.rename(columns=rename_map)

    # Asegurar 'object_id' (sintético si no hay candidato)
    if "object_id" not in df.columns:
        for cand in ("koi_name", "kepoi_name", "kepid", "id", "rowid"):
            if cand in df.columns:
                df["object_id"] = df[cand].astype(str)
                break
        else:
            df["object_id"] = [f"OBJ-{i}" for i in range(len(df))]
        mapping_list.append({"detected": "(synthetic)", "mapped_to": "object_id", "status": "generated"})

    # --- Reglas específicas para koi_disposition ---
    # 1) Si ya está, no tocar valores.
    if "koi_disposition" in df.columns:
        # Nos aseguramos de registrar el mapping si no estaba
        if not any(m["mapped_to"] == "koi_disposition" for m in mapping_list):
            mapping_list.append({"detected": "koi_disposition", "mapped_to": "koi_disposition", "status": "ok"})
    else:
        # 2) No existe: intentar detectar alias manualmente contra el índice
        #    (por si no fueron capturados arriba, aunque debería).
        disp_alias = None
        for col in original_cols:
            if _ALIAS_INDEX.get(_norm(col)) == "koi_disposition":
                disp_alias = col
                break

        if disp_alias:
            # Renombrado tardío (conserva valores tal cual)
            if disp_alias != "koi_disposition":
                df = df.rename(columns={disp_alias: "koi_disposition"})
            mapping_list.append({"detected": disp_alias, "mapped_to": "koi_disposition", "status": "alias"})
        else:
            # 3) Crear columna vacía
            df["koi_disposition"] = None
            mapping_list.append({"detected": "(created)", "mapped_to": "koi_disposition", "status": "generated_empty"})

    return df, mapping_list
