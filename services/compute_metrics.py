# services/compute_metrics.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional

R_EARTH_PER_R_SUN = 109.0        # 1 R_sun ≈ 109 R_earth
T_SUN = 5772.0                   # Sun effective temperature, K

def _as_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None

def _add_col(df, name: str, values):
    """Set or overwrite column safely on a pandas DataFrame."""
    df[name] = values
    return df

def _ensure_pandas(df):
    """Accept Spark DF by converting to Pandas if needed."""
    if hasattr(df, "toPandas") and not hasattr(df, "to_dict"):
        try:
            return df.toPandas()
        except Exception as e:
            raise RuntimeError(f"Expected a pandas.DataFrame or convertible Spark DF. Error: {e}")
    return df

def _safe_series(df, name: str):
    return df[name] if name in df.columns else [None] * len(df)

def _kepler_a_au(period_days: Optional[float], m_star_sun: Optional[float]) -> Optional[float]:
    """
    a [AU] ≈ ( P[yr]^2 * (M_star [M_sun]) )^(1/3)
    Neglect planet mass, constants collapse in AU/yr/M_sun units.
    """
    P = _as_float(period_days)
    M = _as_float(m_star_sun)
    if P is None or P <= 0 or M is None or M <= 0:
        return None
    P_year = P / 365.25
    return (P_year * P_year * M) ** (1.0 / 3.0)

def _teq_K(t_star_K: Optional[float], r_star_sun: Optional[float], a_au: Optional[float], albedo: float) -> Optional[float]:
    """
    T_eq = T_star * sqrt(R_star / (2 a)) * (1 - A)^(1/4)
    R_star en R_sun, a en AU.
    """
    T = _as_float(t_star_K)
    Rs = _as_float(r_star_sun)
    a = _as_float(a_au)
    if T is None or Rs is None or a is None or Rs <= 0 or a <= 0:
        return None
    try:
        return T * math.sqrt(Rs / (2.0 * a)) * ((1.0 - float(albedo)) ** 0.25)
    except Exception:
        return None

def _insol_relative(t_star_K: Optional[float], r_star_sun: Optional[float], a_au: Optional[float]) -> Optional[float]:
    """
    F/F_earth ≈ ( (R_star/R_sun)^2 * (T_star/T_sun)^4 ) / (a/AU)^2
    """
    T = _as_float(t_star_K)
    Rs = _as_float(r_star_sun)
    a = _as_float(a_au)
    if T is None or Rs is None or a is None or Rs <= 0 or a <= 0:
        return None
    try:
        return (Rs * Rs) * ((T / T_SUN) ** 4.0) / (a * a)
    except Exception:
        return None

def _transit_depth(prad_re: Optional[float], rstar_rsun: Optional[float]) -> Optional[float]:
    """
    δ = (Rp/Rs)^2 with Rp in R_earth and R_s in R_sun.
    Rp/Rs = (Rp [R_earth]) / (R_s [R_sun] * 109)
    """
    Rp = _as_float(prad_re)
    Rs = _as_float(rstar_rsun)
    if Rp is None or Rs is None or Rp <= 0 or Rs <= 0:
        return None
    ratio = Rp / (Rs * R_EARTH_PER_R_SUN)
    return ratio * ratio

def _expected_duration_hours(period_days: Optional[float], rstar_rsun: Optional[float], a_au: Optional[float], impact_b: Optional[float]) -> Optional[float]:
    """
    Approx transit duration (first to last contact), circular orbit:
      T_dur ≈ (P/π) * sqrt(1 - b^2) * (R_s / a)
    Return in hours.
    """
    P = _as_float(period_days)
    Rs = _as_float(rstar_rsun)
    a = _as_float(a_au)
    b = _as_float(impact_b) if impact_b is not None else None
    if P is None or P <= 0 or Rs is None or Rs <= 0 or a is None or a <= 0:
        return None
    if b is None:
        b = 0.5  # suposición moderada si no se conoce
    try:
        dur_days = (P / math.pi) * math.sqrt(max(0.0, 1.0 - b * b)) * (Rs / a)
        return dur_days * 24.0
    except Exception:
        return None

def enrich(df_in, assumptions: Optional[Dict[str, Any]] = None, options: Optional[Dict[str, Any]] = None):
    """
    Enrich a DataFrame with derived metrics used across ExoHunter:
      - koi_sma (if missing): from koi_period & koi_smass (AU)
      - koi_teq (if missing): from koi_steff, koi_srad, koi_sma, albedo
      - koi_insol (if missing): relative insolation
      - transit_depth: (koi_prad, koi_srad)
      - expected_duration_hours: (koi_period, koi_srad, koi_sma, koi_impact?)
      - duration_ratio: koi_duration / expected_duration_hours (if koi_duration present)
      - soft flags: e.g., FLAG_NUMERIC_OUT_OF_RANGE
    Notes:
      * Accepts pandas DF, or Spark DF (will convert to pandas).
      * Returns pandas DF (para portabilidad en el pipeline).
    """
    assumptions = assumptions or {}
    options = options or {}
    auto_derive = bool(options.get("auto_derive", True))
    albedo = float(assumptions.get("albedo", 0.3))

    df = _ensure_pandas(df_in).copy()

    # Column handles (may be missing)
    col_period     = "koi_period"     # days
    col_prad       = "koi_prad"       # Earth radii
    col_sma        = "koi_sma"        # AU
    col_smass      = "koi_smass"      # Solar masses
    col_srad       = "koi_srad"       # Solar radii
    col_teq        = "koi_teq"        # K
    col_steff      = "koi_steff"      # K
    col_insol      = "koi_insol"      # F/F_earth
    col_duration   = "koi_duration"   # hours (Kepler KOI suele estar en horas)
    col_impact     = "koi_impact"     # b

    n = len(df)

    # Prepare arrays for new columns
    sma_vals = list(_safe_series(df, col_sma))
    teq_vals = list(_safe_series(df, col_teq))
    insol_vals = list(_safe_series(df, col_insol))
    depth_vals = [None] * n
    exp_dur_vals = [None] * n
    dur_ratio_vals = [None] * n
    flags_soft = [[] for _ in range(n)]

    # Accessors
    getv = lambda i, c: _as_float(df.iloc[i][c]) if c in df.columns else None

    for i in range(n):
        P = getv(i, col_period)
        Mstar = getv(i, col_smass)
        Rs = getv(i, col_srad)
        Rp = getv(i, col_prad)
        a = getv(i, col_sma)
        Teff = getv(i, col_steff)
        Teq = getv(i, col_teq)
        Insol = getv(i, col_insol)
        b = getv(i, col_impact)
        Tobs = getv(i, col_duration)

        # 1) a (AU) si falta y auto_derive activo
        if a is None and auto_derive:
            a = _kepler_a_au(P, Mstar)
            sma_vals[i] = a

        # 2) Teq (K) si falta y auto_derive activo
        if Teq is None and auto_derive:
            Teq = _teq_K(Teff, Rs, a, albedo)
            teq_vals[i] = Teq

        # 3) Insolación relativa si falta y auto_derive activo
        if Insol is None and auto_derive:
            Insol = _insol_relative(Teff, Rs, a)
            insol_vals[i] = Insol

        # 4) Profundidad de tránsito δ
        depth_vals[i] = _transit_depth(Rp, Rs)

        # 5) Duración esperada y ratio
        exp_h = _expected_duration_hours(P, Rs, a, b)
        exp_dur_vals[i] = exp_h
        if Tobs is not None and exp_h is not None and exp_h > 0:
            dur_ratio_vals[i] = Tobs / exp_h

        # 6) Soft flags por rangos físicos obvios
        if Rp is not None and Rp <= 0:
            flags_soft[i].append("SOFT_NEGATIVE_RADIUS")
        if Rs is not None and Rs <= 0:
            flags_soft[i].append("SOFT_NEGATIVE_STELLAR_RADIUS")
        if P is not None and P <= 0:
            flags_soft[i].append("SOFT_NEGATIVE_PERIOD")
        if Teff is not None and (Teff < 2000 or Teff > 10000):
            # No es error, pero marcamos fuera de rango solar típico (F,G,K)
            flags_soft[i].append("SOFT_STELLAR_TEFF_OUT_OF_RANGE")

    # Escribir de vuelta
    if col_sma not in df.columns:
        _add_col(df, col_sma, sma_vals)
    else:
        # sólo rellena donde esté NaN
        df[col_sma] = [sma_vals[i] if _as_float(df.iloc[i][col_sma]) is None else df.iloc[i][col_sma] for i in range(n)]

    if col_teq not in df.columns:
        _add_col(df, col_teq, teq_vals)
    else:
        df[col_teq] = [teq_vals[i] if _as_float(df.iloc[i][col_teq]) is None else df.iloc[i][col_teq] for i in range(n)]

    if col_insol not in df.columns:
        _add_col(df, col_insol, insol_vals)
    else:
        df[col_insol] = [insol_vals[i] if _as_float(df.iloc[i][col_insol]) is None else df.iloc[i][col_insol] for i in range(n)]

    _add_col(df, "transit_depth", depth_vals)
    _add_col(df, "expected_duration_hours", exp_dur_vals)
    _add_col(df, "duration_ratio", dur_ratio_vals)
    _add_col(df, "soft_flags", [";".join(f) if f else "" for f in flags_soft])

    return df
