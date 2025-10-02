# local_SLSNe_metric
# Survey-ready SLSN templates from catalog photometry via 2D GP (time, wavelength).
from __future__ import annotations
import re
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.slicers import UserPointsSlicer
from dustmaps.sfd import SFDQuery
#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues
from itertools import islice
from tqdm.auto import tqdm
import sys
import pyarrow
import unicodedata
from collections import Counter
from shared_utils import sample_rate_from_volume  # Add this to your existing import line
from rubin_sim.phot_utils import Sed, Bandpass


import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, apply_spectral_index, evaluate, compare_flux_diff_to_error

import matplotlib.pyplot as plt 
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from astropy.coordinates import SkyCoord
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
from astropy.cosmology import z_at_value
from scipy.stats import truncnorm
import numpy as np
import glob
import os

import pickle 
from pathlib import Path

from collections import OrderedDict

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import pickle
import warnings

# GP: same library used in Tyler's code
import george
import scipy.optimize as op
from functools import partial

# Cosmology & units for DM(z) and rest-frame scaling
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from shared_utils import (equatorialFromGalactic, uniform_sphere_degrees, 
                         inject_uniform_healpix, apply_spectral_index, 
                         evaluate, compare_flux_diff_to_error,
                         sample_rate_from_volume) 

DEBUG = False
dust_model = DustValues()


# Cache Rubin bandpasses once per process

_LSST_BANDS = None
def _get_lsst_bands():
    global _LSST_BANDS
    if _LSST_BANDS is None:
        thru_dir = os.path.join(get_data_dir(), "throughputs", "baseline")
        bands = {}
        for b in "ugrizy":
            bp = Bandpass()
            bp.read_throughput(os.path.join(thru_dir, f"total_{b}.dat"))
            bands[b] = bp
        _LSST_BANDS = bands
    return _LSST_BANDS
    

_Z_INTERP_TABLE = None

def _get_z_from_comoving_fast(d_cm_array, z_min=0.001, z_max=10, n_points=10000):
    """Fast z(d_comoving) lookup via pre-computed interpolation."""
    global _Z_INTERP_TABLE
    
    if _Z_INTERP_TABLE is None:
        z_grid = np.logspace(np.log10(z_min), np.log10(z_max), n_points)
        d_grid = cosmo.comoving_distance(z_grid).value
        _Z_INTERP_TABLE = (d_grid, z_grid)
    
    d_grid, z_grid = _Z_INTERP_TABLE
    return np.interp(d_cm_array, d_grid, z_grid)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Constants & filter mappings
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------




# Speed of light (m/s)
# -------- SED helpers consistent with saved sed_grid --------
_C_MS = 2.99792458e8
_C_CM_S = 2.99792458e10        # cm/s
_A_TO_CM = 1e-8                 # 1 Å = 1e-8 cm
_JY_TO_CGS = 1e-23              # 1 Jy = 1e-23 erg/s/cm^2/Hz

def _interp_Fnu_abs_at_phase(sed_grid: dict, phase_rest: float):
    """
    Interpolate the stored absolute rest-frame SED at a given phase.

    sed_grid keys:
      - 'phase'      : 1D array (days, rest frame)
      - 'lam_rest_A' : 1D array (Å, rest frame)
      - 'Fnu_abs'    : 2D array [N_phase, N_lambda] in Jy at 10 pc (rest-frame Fν)

    Returns
    -------
    lam_rest_A : (N_lambda,) Å
    Fnu_abs    : (N_lambda,) Jy at 10 pc
    """
    ph = np.asarray(sed_grid["phase"], float)
    lam_rest_A = np.asarray(sed_grid["lam_rest_A"], float)
    Fnu_abs_grid = np.asarray(sed_grid["Fnu_abs"], float)  # shape (Nt, Nλ)

    if not (np.isfinite(phase_rest) and ph.min() <= phase_rest <= ph.max()):
        return None, None

    # Interpolate along the phase axis for every wavelength bin
    Fnu_abs = np.empty_like(lam_rest_A, dtype=float)
    for j in range(lam_rest_A.size):
        Fnu_abs[j] = np.interp(phase_rest, ph, Fnu_abs_grid[:, j], left=np.nan, right=np.nan)

    return lam_rest_A, Fnu_abs


def synthesize_mag_at_z(sed_grid: dict, phase_rest: float, z: float, filt: str) -> float:
    """
    Throughput-integrated apparent AB magnitude in Rubin filter 'filt' at redshift z.

    Uses your saved absolute rest-frame SED (Fν at 10 pc), transforms to *observed*
    Fλ at Earth for the target z, and integrates with the LSST bandpass.
    """
    bands = _get_lsst_bands()
    if filt not in bands:
        return np.nan

    lam_rest_A, Fnu_abs_10pc = _interp_Fnu_abs_at_phase(sed_grid, phase_rest)
    if lam_rest_A is None:
        return np.nan

    # observed frame λ
    lam_obs_A  = lam_rest_A * (1.0 + z)
    lam_obs_cm = lam_obs_A * _A_TO_CM

    DL_pc = cosmo.luminosity_distance(float(z)).to_value(u.pc)
    scale = (DL_pc / 10.0)**2 * (1.0 + z)
    Fnu_obs_Jy  = Fnu_abs_10pc / scale
    Fnu_obs_cgs = Fnu_obs_Jy * _JY_TO_CGS

    # Fλ = Fν c / λ^2  (λ in cm)
    Flambda_obs = Fnu_obs_cgs * (_C_CM_S / (lam_obs_cm**2))  # erg/s/cm^2/Å

    # overlap check with LSST bandpass
    bp = bands[filt]
    wmin, wmax = float(np.nanmin(bp.wavelen)), float(np.nanmax(bp.wavelen))
    if (np.nanmax(lam_obs_A) < wmin) or (np.nanmin(lam_obs_A) > wmax):
        return np.nan

    # integrate
    sed = Sed(wavelen=lam_obs_A, flambda=Flambda_obs)
    try:
        mag = float(sed.calc_mag(bp))  # AB mag
    except Exception:
        mag = np.nan
    return mag


# AB system
F0_JY = 3631.0
LN10_OVER_2P5 = np.log(10.0) / 2.5

# ZTF effective wavelengths (Å) — used if your catalogs reference ztfg/ztfr/ztfi
ZTF_EFF_LAMBDA = {
    "ztfg": 4800.0,
    "ztfr": 6400.0,
    "ztfi": 7900.0,
}

# LSST effective *frequencies* (Hz) where we will PREDICT final templates
LSST_EFF_FREQ = {
    'u': 8.088e14,
    'g': 6.293e14,
    'r': 4.844e14,
    'i': 3.979e14,
    'z': 3.461e14,
    'y': 3.080e14,
}

def angstrom_to_hz(lambda_A):
    """Å to Hz (accepts scalar or array)."""
    lam_m = np.asarray(lambda_A, dtype=float) * 1e-10
    return _C_MS / lam_m

def hz_to_angstrom(nu_hz):
    """Hz to Å (accepts scalar or array)."""
    nu = np.asarray(nu_hz, dtype=float)
    return (_C_MS / nu) * 1e10


# LSST central λ (Å), derived from LSST_EFF_FREQ
LSST_EFF_LAMBDA = {b: hz_to_angstrom(nu) for b, nu in LSST_EFF_FREQ.items()}

# SDSS aliases (optional convenience)
SDSS_ALIAS = {'sdssu':'u','sdssg':'g','sdssr':'r','sdssi':'i','sdssz':'z'}

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Unit helpers
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

def mag_to_flux_jy(mag: np.ndarray) -> np.ndarray:
    m = np.asarray(mag, float)
    return F0_JY * 10.0 ** (-0.4 * m)

def magerr_to_fluxerr_jy(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    f = mag_to_flux_jy(mag)
    return LN10_OVER_2P5 * f * np.asarray(mag_err, float)

def flux_jy_to_mag(flux_jy: np.ndarray) -> np.ndarray:
    f = np.asarray(flux_jy, float)
    out = np.full_like(f, np.nan, dtype=float)
    good = f > 0
    out[good] = -2.5 * np.log10(f[good] / F0_JY)
    return out

def dm_from_z(z: float) -> float:
    DL = cosmo.luminosity_distance(float(z)).to_value(u.Mpc)
    return 5.0 * np.log10(DL) + 25.0

def _filter_key_series(df: pd.DataFrame) -> pd.Series:
    """Key: '<System>|<CanonicalFilter>' if System present; else just '<CanonicalFilter>'."""
    filt = df["Filter"].astype(str).str.strip().map(canonical_filter)
    if "System" in df.columns:
        sys = df["System"].astype(str).str.strip()
        # if System is blank/NA, fall back to just filter
        only_filter = sys.isna() | (sys == "")
        key = sys.str.cat(filt, sep="|")
        key[only_filter] = filt[only_filter]
        return key
    return filt

def _pick_event_file_with_cenwave(phot_dir: Path, name: str, filename_pattern: str) -> Path | None:
    """
    Return a Path to a per-event file that actually contains a 'Cenwave' column.
    Priority:
      1) If filename_pattern already points to *_cenwave.csv / .parquet and exists, use it.
      2) Otherwise prefer {name}_cenwave.csv, then {name}_cenwave.parquet.
      3) Otherwise try filename_pattern as-is ({name}.csv) and verify it has Cenwave.
    """
    phot_dir = Path(phot_dir)

    def has_cenwave(p: Path) -> bool:
        if not p.exists():
            return False
        try:
            import pandas as pd
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, nrows=5)
            return any(c.lower() == "cenwave" for c in map(str, df.columns))
        except Exception:
            return False

    # 0) If user already passed a pattern that resolves to a file with Cenwave, use it
    direct = phot_dir / filename_pattern.format(name=name)
    if has_cenwave(direct):
        return direct

    # 1) Prefer explicit *_cenwave files
    c1 = phot_dir / f"{name}_cenwave.csv"
    c2 = phot_dir / f"{name}_cenwave.parquet"
    if has_cenwave(c1): return c1
    if has_cenwave(c2): return c2

    # 2) Fall back to the non-cenwave pattern only if it *does* have Cenwave
    return direct if has_cenwave(direct) else None

def pick_t0_hybrid(
    mjd, band, lamA, *, z, gp_predict, t0_catalog,
    window_days=15.0, min_pts_in_window=5, agree_days=4.0,
    mag=None
) -> tuple[float, dict]:
    """
    Choose t0 using a GP-based, data-driven peak time, with the catalog peak as a prior.
    Returns (t0_used, info_dict).
    """
    import numpy as np

    # 1) per-band prediction frequency from per-band median Cenwave
    med_lam_by_band = (
        pd.DataFrame({"band": band, "lamA": lamA})
        .groupby("band", sort=False)["lamA"].median()
    )
    target_freq_hz = {b: float(angstrom_to_hz(L)) for b, L in med_lam_by_band.items()}

    # 2) Dense time grid across observed span
    t_dense = np.linspace(np.nanmin(mjd), np.nanmax(mjd), 800)

    # 3) GP mean → peak time per band
    pred = gp_predict_surface(gp_predict, t_dense, target_freq_hz)
    t0_by_band = {}
    for b, (f_mu, _f_sig) in pred.items():
        if np.isfinite(f_mu).any():
            i = int(np.nanargmax(f_mu))   # max flux
            t0_by_band[b] = float(t_dense[i])

    # --- robust fallback if we couldn't get per-band peaks
    if not t0_by_band:
        if t0_catalog is not None and np.isfinite(t0_catalog):
            return float(t0_catalog), {
                "t0_cat": float(t0_catalog), "t0_data": np.nan, "delta_days": np.nan,
                "n_pts_win": 0, "snr_med": np.nan, "bands_used": [], "source": "catalog_fallback"
            }
        if mag is not None and np.isfinite(mag).any():
            t0_minmag = float(mjd[np.nanargmin(mag)])
            return t0_minmag, {
                "t0_cat": np.nan, "t0_data": t0_minmag, "delta_days": np.nan,
                "n_pts_win": 0, "snr_med": np.nan, "bands_used": [], "source": "minmag_fallback"
            }
        t0_mid = float(np.nanmedian(mjd))
        return t0_mid, {
            "t0_cat": np.nan, "t0_data": t0_mid, "delta_days": np.nan,
            "n_pts_win": 0, "snr_med": np.nan, "bands_used": [], "source": "median_mjd_fallback"
        }

    # 4) Prefer rest-frame g/r-ish bands
    bands_pref = []
    for b, lam_obs in med_lam_by_band.items():
        lam_rest = float(lam_obs) / (1.0 + float(z))
        if 4500.0 <= lam_rest <= 7000.0:
            bands_pref.append(b)
    bands_used = bands_pref if bands_pref else list(t0_by_band.keys())

    # 5) Robust combine
    t_candidates = np.array([t0_by_band[b] for b in bands_used if b in t0_by_band], float)
    t0_data = float(np.nanmedian(t_candidates))

    # 6) Coverage metric in +/- window
    center = float(t0_data if (t0_catalog is None or not np.isfinite(t0_catalog)) else t0_catalog)
    in_win = np.abs(mjd - center) <= float(window_days)
    n_pts_win = int(np.isfinite(mjd[in_win]).sum())
    snr_med = np.nan  # (optional: compute from flux/err if available)
    strong_coverage = (n_pts_win >= int(min_pts_in_window))

    # 7) Decide with prior
    if t0_catalog is None or not np.isfinite(t0_catalog):
        t0_used, source = t0_data, "data_no_prior"
        delta = np.nan
    else:
        delta = float(abs(t0_data - float(t0_catalog)))
        if delta <= float(agree_days):
            t0_used, source = t0_data, "data_agrees_with_prior"
        else:
            if strong_coverage:
                t0_used, source = t0_data, "data_overrode_prior"
            else:
                t0_used, source = float(t0_catalog), "catalog_due_to_weak_coverage"

    info = {
        "t0_cat": float(t0_catalog) if (t0_catalog is not None and np.isfinite(t0_catalog)) else np.nan,
        "t0_data": float(t0_data),
        "delta_days": float(delta),
        "n_pts_win": n_pts_win,
        "snr_med": snr_med,
        "bands_used": list(bands_used),
        "source": source,
    }
    return float(t0_used), info



#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Processing from Sebastian's git to csv files saved locally 
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

def _clean_header(cols):
    cleaned = []
    for c in cols:
        # normalize unicode; strip BOM & weird whitespace
        if not isinstance(c, str):
            c = str(c)
        c = unicodedata.normalize("NFKC", c)
        c = c.replace("\ufeff", "")  # BOM
        c = c.strip()
        # collapse internal whitespace runs to single space, then remove spaces
        c = re.sub(r"\s+", " ", c)
        # canonicalize a few expected labels
        c = c.replace("Mag Err", "MagErr")
        cleaned.append(c)
    return cleaned

def read_supernova_table_txt(path: Path) -> pd.DataFrame:
    """
    Tolerant reader for Sebastian's per-object .txt files.
    Uses a whitespace regex separator and cleans headers.
    """
    # Use explicit regex separator, python engine, ignore comments
    df = pd.read_csv(
        path, 
        sep=r"\s+", 
        engine="python", 
        comment="#", 
        dtype=str,   # read everything as string first; coerce later
        skip_blank_lines=True
    )
    df.columns = _clean_header(df.columns)
    return df

def coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip() #.str.lower()
    return s.isin(["1","true","t","yes","y"])

def norm_filter(x):
    key = str(x).strip()
    low = key.lower()
    return FILTER_MAP.get(low, key)



# 1) You can keep FILTER_MAP / norm_filter defined (for future use),
#    but DO NOT call it in to_export().

def to_export(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a per-object table to Felipe's 5 columns:
      mjd, mag, mag_err, filter, detected
    Interprets UL=True as non-detection -> detected=0, mag_err=inf.
    Treats MagErr = -1.0 as 'unknown' (kept as NaN) for detections; for UL rows we set inf anyway.
    """
    need = ["MJD","Mag","MagErr","Filter","UL"]
    cols = {c: c for c in df_raw.columns}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df_raw.columns)}")

    df = df_raw.copy()

    # numerics
    mjd = pd.to_numeric(df["MJD"], errors="coerce")
    mag = pd.to_numeric(df["Mag"], errors="coerce")

    mag_err = pd.to_numeric(df["MagErr"], errors="coerce").astype(float)
    mag_err[mag_err < 0] = np.nan

    ul = coerce_bool(df["UL"])           # True = upper limit
    detected = (~ul).astype(np.int8)
    System = df["System"].astype("string").map(lambda s: s.strip() if isinstance(s, str) else s)

    # *** PRESERVE FILTER STRINGS relatively AS IN SOURCE ***
    filt = df["Filter"].astype("string").map(lambda s: s.strip() if isinstance(s, str) else s)

    out_perevent = pd.DataFrame({
        "mjd": mjd.astype(float),
        "mag": mag.astype(float),
        "mag_err": mag_err,
        "UL": ul,
        "filter": filt,
        "System": System,
        "detected": detected
    }).sort_values("mjd").reset_index(drop=True)

    keep = out_perevent[["mjd","mag","mag_err","UL","filter"]].notna().any(axis=1)
    out_perevent = out_perevent[keep].reset_index(drop=True)

    out_perevent = out_perevent.astype({
    "mjd": "float64",
    "mag": "float64",
    "mag_err": "float64",
    "UL": "boolean",           # pandas Nullable Boolean
    "filter": "string",
    "System": "string",
    "detected": "int8",
})

    # return columns in agreed order (keeping 'UL' if you’ve been writing it)
    return out_perevent


# Cell 3 — process a single event for quick sanity check

def process_one(event_dir: Path, event_name: str,
                out_perevent: Path, *, write_parquet: bool, write_csv: bool):
    infile = event_dir / f"{event_name}.txt"   # <-- use the directory you passed
    if not infile.exists():
        raise FileNotFoundError(f"Could not find {infile}")

    df_out = to_export(read_supernova_table_txt(infile))

    out_perevent.mkdir(parents=True, exist_ok=True)
    if write_parquet:
        df_out.to_parquet(out_perevent / f"{event_name}.parquet",
                          index=False, engine="pyarrow",
                          compression="zstd", compression_level=7)
    if write_csv:
        df_out.to_csv(out_perevent / f"{event_name}.csv", index=False)
    return df_out

# --------------------------------------
# --------------------------------------
# All Event Functions 
# --------------------------------------
# --------------------------------------

def process_all_events(supernovae_dir: Path,
                       out_perevent: Path,
                       out_allevent: Path,
                       *, include_ul: bool = True,
                       make_combined: bool = True,
                       write_parquet: bool = True,
                       write_csv: bool = True):
    rows = []
    combined_frames = [] if make_combined else None

    event_dirs = sorted(p for p in supernovae_dir.iterdir() if p.is_dir())
    for d in tqdm(event_dirs, desc="Exporting SLSNe per-object"):
        event = d.name
        infile = d / f"{event}.txt"
        if not infile.exists():
            rows.append({
                "event": event, "status": "missing_file", "path": str(infile),
                "n_rows": 0, "n_detected": 0, "n_limits": 0
            })
            continue

        try:
            df_raw = read_supernova_table_txt(infile)
            df_out = to_export(df_raw)
            if not include_ul:
                df_out = df_out[df_out["detected"] == 1].reset_index(drop=True)

            # write per-event
            if write_csv:
                df_out.to_csv(out_perevent / f"{event}.csv", index=False)
            if write_parquet:
                df_out.to_parquet(
                    out_perevent / f"{event}.parquet",
                    index=False, engine="pyarrow", compression="zstd", compression_level=7
                )

            # accumulate for combined dataset
            if make_combined and len(df_out):
                tmp = df_out.copy()
                tmp.insert(0, "object_id", event)
                combined_frames.append(tmp)

            rows.append({
                "event": event, "status": "ok", "path": str(infile),
                "n_rows": int(len(df_out)),
                "n_detected": int(df_out["detected"].sum()),
                "n_limits": int((df_out["detected"] == 0).sum())
            })

        except Exception as e:
            rows.append({
                "event": event, "status": f"error: {e}", "path": str(infile),
                "n_rows": 0, "n_detected": 0, "n_limits": 0
            })

    # write index summary
    index_df = pd.DataFrame(rows).sort_values(["status","event"]).reset_index(drop=True)
    index_df.to_csv(out_perevent / "_index.csv", index=False)

    # write combined (optional)
    combined_csv = combined_parq = None
    if make_combined and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        if write_csv:
            combined_csv = out_allevent / "all_objects.csv"
            combined.to_csv(combined_csv, index=False)
        if write_parquet:
            combined_parq = out_allevent / "all_objects.parquet"
            combined.to_parquet(
                combined_parq, index=False, engine="pyarrow",
                compression="zstd", compression_level=7
            )

    return index_df, combined_csv, combined_parq

# --------------------------------------
# --------------------------------------
# All Parameter Table Process  Functions 
# --------------------------------------
# --------------------------------------

# ---- robust loader for all_parameters.txt (takes a PATH ARGUMENT) ----
def load_allparams_robust(path: Path) -> pd.DataFrame:
    import re
    # split on runs of spaces OR tabs; preserve single spaces inside values
    df = pd.read_csv(
        path, sep=r"\s{2,}|\t+", engine="python", header=0,
        comment="#", dtype=str, skip_blank_lines=True, on_bad_lines="warn"
    )
    # if header width mismatch, re-read with explicit names inferred from first non-comment line
    if df.shape[1] < 2:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip() and not line.lstrip().startswith("#"):
                    cols = re.split(r"\s{2,}|\t+", line.strip())
                    break
        df = pd.read_csv(
            path, sep=r"\s{2,}|\t+", engine="python", header=None, names=cols,
            comment="#", dtype=str, skip_blank_lines=True, on_bad_lines="warn"
        )
    return df

# ---- optional: CLI entrypoint (won't run on import) ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--allparams", type=Path, help="Path to all_parameters.txt")
    args = ap.parse_args()
    if args.allparams:
        df = load_allparams_robust(args.allparams)
        print(df.head())

def resolve_cols(df, requested=None):
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    anchor = lower.get("name", cols[0])  # prefer 'name'; else first
    name_out = [anchor]
    if requested is None:
        return name_out
    if not isinstance(requested, (list, tuple)):
        requested = [requested]
    for r in requested:
        if isinstance(r, int):
            if 0 <= r < len(cols) and cols[r] not in name_out:
                name_out.append(cols[r]); continue
        key = str(r).strip().lower()
        if key in lower:
            c = lower[key]
            if c not in name_out: name_out.append(c)
        else:
            # soft partial match
            for c in cols:
                if key in c.lower() and c not in name_out:
                    name_out.append(c)
    return name_out

def print_cols(df, requested=None, head=10):
    show = resolve_cols(df, requested)
    display(df[show].head(head))
    return show

def _key_cols(df):
    f = df["Filter"].astype(str).str.strip()
    if "System" in df.columns:
        s = df["System"].astype(str).str.strip()
        return (s + "|" + f)   # case-sensitive, preserves info
    return f


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Future-proof: use sep=r"\s+"
    return pd.read_csv(path, comment="#", sep=r"\s+", engine="python", header=0)

# --- helper once near your small utilities ---
PREFIXES = ("swift_", "ps1_", "panstarrs_", "lsst_", "ztf_")
SUFFIXES = ("-AB", "-Vega")

def canonical_filter(s: str) -> str:
    s = str(s).strip()
    for p in PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s  # keep case: R != r

# --- per-filter cenwave map ---
def per_filter_cenwave(event_root: Path, name: str, rel_bin=0.02, verbose: bool = True) -> dict[str, float]:
    """
    Build {Filter_can: Cenwave[Å]} using mode clustering.
    Precedence: entries from *_model.txt override/fill before *_rest.txt.

    Prints a per-event diagnostic summary when verbose=True.
    """
    d = Path(event_root) / name
    mod = _read_table(d / f"{name}_model.txt")
    rst = _read_table(d / f"{name}_rest.txt")

    cen_map: dict[str, float] = {}

    # diagnostics
    rows_seen = {"model": 0, "rest": 0}
    groups_seen = {"model": 0, "rest": 0}
    groups_kept  = {"model": 0, "rest": 0}
    groups_skipped = {"model": 0, "rest": 0}   # skipped because already filled by higher priority
    overlaps = []  # filters present in both sources (model wins)

    def _add_from(df: pd.DataFrame, src: str):
        nonlocal overlaps
        if df.empty or not {"Filter", "Cenwave"}.issubset(df.columns):
            return
        sub = df[np.isfinite(df["Cenwave"])].copy()
        if sub.empty:
            return
        rows_seen[src] += int(len(sub))
        sub["Filter_can"] = sub["Filter"].astype(str).str.strip().map(canonical_filter)

        for lab, g in sub.groupby("Filter_can"):
            groups_seen[src] += 1
            lam = g["Cenwave"].to_numpy(float)
            center = cenwave_mode(lam, rel_bin=rel_bin)
            if not np.isfinite(center):
                continue
            if lab in cen_map:
                groups_skipped[src] += 1
                if src == "rest":
                    overlaps.append(lab)
                continue
            cen_map[lab] = float(center)
            groups_kept[src] += 1

    # precedence: model then rest
    _add_from(mod, "model")
    _add_from(rst, "rest")

    if verbose:
        n_model = groups_kept["model"]
        n_rest  = groups_kept["rest"]
        total   = n_model + n_rest
        msg = (f"[cenwave] {name}: entries kept -> model={n_model}, rest={n_rest}, total={total} | "
               f"rows seen -> model={rows_seen['model']}, rest={rows_seen['rest']} | "
               f"groups seen -> model={groups_seen['model']}, rest={groups_seen['rest']} | "
               f"rest skipped due to precedence={groups_skipped['rest']}")
        print(msg)
        if overlaps:
            # show a few overlapping filters to help QA
            sample = ", ".join(sorted(set(overlaps))[:8])
            print(f"[cenwave] {name}: model/rest overlap filters (model wins): {sample}{' ...' if len(set(overlaps))>8 else ''}")

    return cen_map
    
# --- audit (mode): same canonical key ---
def audit_cenwave_mode(event_root: Path, name: str, rel_bin=0.02,
                       outlier_frac_thresh=0.25,
                       widen_factor=2.0, mad_floor_factor=1.0):
    d = Path(event_root) / name
    mod = _read_table(d / f"{name}_model.txt")
    rst = _read_table(d / f"{name}_rest.txt")

    rows = []
    for src_name, df in [("model", mod), ("rest", rst)]:
        if df.empty or not {"MJD","Cenwave","Filter"}.issubset(df.columns):
            continue
        sub = df[np.isfinite(df["Cenwave"])].copy()
        sub["Filter_can"] = sub["Filter"].astype(str).str.strip().map(canonical_filter)

        for lab, g in sub.groupby("Filter_can"):
            lam = g["Cenwave"].to_numpy(float)
            n = lam.size
            if n == 0:
                rows.append([name, src_name, lab, 0, np.nan, np.nan, False]); continue
            center = cenwave_mode(lam, rel_bin=rel_bin)
            if not np.isfinite(center) or center <= 0:
                rows.append([name, src_name, lab, n, np.nan, np.nan, True]); continue
            bin_width = rel_bin * center
            mad = np.median(np.abs(lam - np.median(lam))) if n >= 3 else 0.0
            mad_sigma = 1.4826 * mad
            tol = max(widen_factor * bin_width, mad_floor_factor * mad_sigma)
            frac_outside = float(np.mean(np.abs(lam - center) > tol))
            flag = frac_outside > outlier_frac_thresh
            rows.append([name, src_name, lab, n, center, frac_outside, flag])

    return pd.DataFrame(rows, columns=[
        "event","source","filter","n_rows","mode_center_A","frac_outside","flag_outliers"
    ])

# --- attach using the SAME canonical key built from the CSV's Filter column ---
def attach_cenwave_to_perevent_csv(csv_path: Path, cen_map: dict[str, float]) -> Path:
    df = pd.read_csv(csv_path)
    if "Filter" not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        if "filter" in cols:
            df.rename(columns={cols["filter"]: "Filter"}, inplace=True)
        else:
            raise ValueError(f"No 'Filter' column in {csv_path.name}")

    df["Filter_can"] = df["Filter"].astype(str).str.strip().map(canonical_filter)
    df["Cenwave"] = df["Filter_can"].map(cen_map).astype(float)

    # optional: quick debug if lots of NaNs
    n_ok = int(np.isfinite(df["Cenwave"]).sum())
    if n_ok == 0:
        missing_counts = df["Filter_can"].value_counts().head(8)
        print(f"[attach][warn] 0 matches in {csv_path.name}. Top missing keys:\n{missing_counts}")

    out = csv_path.with_name(csv_path.stem + "_cenwave" + csv_path.suffix)
    df.to_csv(out, index=False)
    print(f"[attach] wrote {out.name} with Cenwave for {n_ok}/{len(df)} rows")
    return out




def audit_cenwave_variability(event_dir: Path, name: str,
                              rel_scatter_thresh=0.02,   # 2%
                              slope_thresh_A_per_day=5.0, # tweak as needed
                              r2_thresh=0.2):
    """
    Returns a dict summary + per-filter diagnostics DataFrame.
    Flags filters where Cenwave appears time-variable.
    """
    d = Path(event_dir) / name
    mod = _read_table(d / f"{name}_model.txt")   # MJD, Cenwave, Filter
    rst = _read_table(d / f"{name}_rest.txt")    # MJD, Cenwave, Filter

    def _norm(df):
        if df.empty: return df
        need = {"MJD","Cenwave","Filter"}
        if not need.issubset(df.columns): return pd.DataFrame()
        df = df.copy()
        df = df[np.isfinite(df["MJD"]) & np.isfinite(df["Cenwave"])].copy()
        df["Filter_can"] = df["Filter"].astype(str).str.strip().map(canonical_filter)

        return df


    mod, rst = _norm(mod), _norm(rst)

    # Combine sources, keeping provenance
    src_frames = []
    if not mod.empty:
        m = mod[["MJD","Cenwave","Filter_can"]].copy(); m["src"] = "model"; src_frames.append(m)
    if not rst.empty:
        r = rst[["MJD","Cenwave","Filter_can"]].copy(); r["src"] = "rest";  src_frames.append(r)

    if not src_frames:
        return {"event": name, "status": "no_data"}, pd.DataFrame()

    allf = pd.concat(src_frames, ignore_index=True)

    rows = []
    for lab, g in allf.groupby("Filter_can"):
        if len(g) < 3:
            rows.append((lab, len(g), np.nan, np.nan, np.nan, np.nan, np.nan, False, False, ""))
            continue

        lam = g["Cenwave"].to_numpy(float)
        mjd = g["MJD"].to_numpy(float)
        med = np.nanmedian(lam)
        mad = np.nanmedian(np.abs(lam - med))
        rel_scatter = mad / med if med > 0 else np.nan
        flag_scatter = np.isfinite(rel_scatter) and (rel_scatter > rel_scatter_thresh)

        # Trend test: center time to reduce collinearity
        t0 = np.nanmedian(mjd)
        t = mjd - t0
        # simple OLS
        A = np.vstack([np.ones_like(t), t]).T
        try:
            coef, *_ = np.linalg.lstsq(A, lam, rcond=None)
            a, b = coef
            lam_hat = a + b*t
            ss_res = np.nansum((lam - lam_hat)**2)
            ss_tot = np.nansum((lam - np.nanmean(lam))**2)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        except Exception:
            b, r2 = np.nan, np.nan
        flag_trend = (np.isfinite(b) and np.abs(b) > slope_thresh_A_per_day) and (np.isfinite(r2) and r2 > r2_thresh)

        note = ""
        if flag_scatter: note += "scatter>thr; "
        if flag_trend:   note += "trend>thr; "

        rows.append((lab, len(g), med, mad, rel_scatter, b, r2, flag_scatter, flag_trend, note.strip()))

    diag = pd.DataFrame(rows, columns=[
        "filter","n_rows","cenwave_med_A","cenwave_mad_A","rel_scatter",
        "slope_A_per_day","r2","flag_scatter","flag_trend","note"
    ])

    # Cross-source consistency (median per source)
    cross = []
    for lab in diag["filter"]:
        vals = []
        for src in ("model","rest"):
            df = (mod if src=="model" else rst)
            if df.empty: continue
            sub = df[df["Filter_can"] == lab]["Cenwave"].to_numpy(float)
            if sub.size: vals.append((src, float(np.nanmedian(sub))))
        if len(vals)==2:
            delta = abs(vals[0][1]-vals[1][1])
            rel = delta / max(vals[0][1], vals[1][1])
            cross.append((lab, vals[0][1], vals[1][1], delta, rel))
    cross_df = pd.DataFrame(cross, columns=["filter","median_model_A","median_rest_A","delta_A","rel_delta"])

    summary = {
        "event": name,
        "any_time_variation": bool(diag["flag_trend"].fillna(False).any() or diag["flag_scatter"].fillna(False).any()),
        "filters_flagged": diag.loc[diag["flag_trend"] | diag["flag_scatter"], "filter"].tolist(),
    }
    return summary, diag.merge(cross_df, on="filter", how="left")


def cenwave_mode(vals, rel_bin=0.02):
    """
    Return the median of the dominant relative-width bin (mode cluster).
    rel_bin=0.02 → 2% bins around the central wavelength scale.
    """
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    med = np.median(vals)
    if med <= 0:
        return np.nan
    width = rel_bin * med
    if width <= 0:
        return np.nan
    keys = np.round(vals / width).astype(int)  # bin index
    cnt = Counter(keys)
    k_mode, _ = cnt.most_common(1)[0]
    cluster = vals[keys == k_mode]
    return float(np.median(cluster))



#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#Building LC templates for SLSNe - Rubin Specific begins for gaussian regression interpolation
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# -------------
# Filter resolving
# --------------


# ----------------------------
# GP helpers (Tyler-style)
# ----------------------------

def _default_kernel(scale_guess: float) -> george.kernels.Kernel:
    """
    Matérn-3/2 in 2D (time [days], frequency [Hz]).
    Frequency metric element is frozen to stabilize sparse color fits.
    """
    time_scale_days = 20.0
    freq_scale_hz   = 1.0e14
    k = (0.5 * max(scale_guess, 1e-6))**2 * george.kernels.Matern32Kernel(
        [time_scale_days**2, freq_scale_hz**2], ndim=2
    )
    k.freeze_parameter("k2:metric:log_M_1_1")
    return k

def fit_2d_gp(time_obs_days: np.ndarray,
              freq_obs_hz: np.ndarray,
              flux_jy: np.ndarray,
              fluxerr_jy: np.ndarray,
              kernel: george.kernels.Kernel | None = None) -> partial:
    """
    Fit a 2D GP in (time, frequency) to flux data. Returns gp_predict partial(y|X).
    """
    t   = np.asarray(time_obs_days, float)
    nu  = np.asarray(freq_obs_hz, float)
    y   = np.asarray(flux_jy, float)
    yerr= np.asarray(fluxerr_jy, float)

    good = np.isfinite(t) & np.isfinite(nu) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if good.sum() < 5:
        raise ValueError("Insufficient finite points for GP fit.")

    t, nu, y, yerr = t[good], nu[good], y[good], yerr[good]
    X = np.vstack([t, nu]).T

    scale_guess = np.nanmedian(np.abs(y))
    if not np.isfinite(scale_guess) or scale_guess <= 0:
        scale_guess = 1e-3

    if kernel is None:
        kernel = _default_kernel(scale_guess)

    gp = george.GP(kernel)
    gp.compute(X, yerr)

    p0 = gp.get_parameter_vector()

    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    bounds = [(p0[0] - 10.0, p0[0] + 10.0)] + [(None, None) for _ in p0[1:]]
    try:
        res = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", bounds=bounds, tol=1e-6)
        if res.success:
            gp.set_parameter_vector(res.x)
        else:
            warnings.warn(f"GP optimizer failed; using initial parameters. msg={res.message}")
    except Exception as e:
        warnings.warn(f"GP optimization raised {e!r}; using initial parameters.")

    return partial(gp.predict, y)

def gp_predict_surface(gp_predict: partial,
                       t_eval_days: np.ndarray,
                       target_freq_hz: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Predict (flux, sigma) at each target frequency across t_eval_days.
    Returns: {band: (flux_mu[Jy], flux_sigma[Jy])}
    """
    out = {}
    t_eval = np.asarray(t_eval_days, float)
    for band, nu in target_freq_hz.items():
        Xp = np.vstack([t_eval, np.full_like(t_eval, nu)]).T
        mu, var = gp_predict(Xp, return_var=True)
        sig = np.sqrt(np.clip(var, 0.0, np.inf))
        out[band] = (mu, sig)
    return out


# ----------------------------
# Core builder (tighter IO)
# ----------------------------

@dataclass
class CatalogInputs:
    photometry_dir: Path          # per-event CSV/Parquet from export_slsne_photometry.py
    params_table: pd.DataFrame    # parameters.txt as DataFrame
    name_col: str = "name"
    z_col: str = "redshift_med"
    peak_mjd_col: str = "Peak_MJD_med"   # optional

class LC:
    """
    Rubin-ready SLSN light-curve model (templates) with an LC-like interface
    compatible with shared_utils.evaluate(...).

    Stored values:
      - time axis: rest-frame phase (days since t0)
      - mags: absolute magnitudes (AB)

    data: list of templates; each template: {band: {"ph": array, "mag": array}}
    """

    def __init__(self, num_lightcurves=None, load_from=None,
                 lightcurves=None, t_grid=None, names=None):
        if lightcurves is not None:
            self.data   = lightcurves
            self.t_grid = t_grid
            self.names  = names if names is not None else [f"tpl_{i}" for i in range(len(self.data))]
            # new fields when constructed in-memory
            self.template_file = None
            self.sed_grid = None
        elif load_from:
            if not os.path.exists(load_from):
                raise FileNotFoundError(f"SLSN templates not found: {load_from}")
            with open(load_from, "rb") as f:
                obj = pickle.load(f)
            if "lightcurves" not in obj:
                raise ValueError("templates.pkl missing key 'lightcurves'")
            self.data   = obj["lightcurves"]
            self.t_grid = obj.get("t_grid", None)
            self.names  = obj.get("names", [f"tpl_{i}" for i in range(len(self.data))])
            # NEW: remember where this came from + keep sed_grid in memory
            self.template_file = load_from
            self.sed_grid = obj.get("sed_grid", None)
        else:
            self.data, self.t_grid = [], None
            self.names = []
            self.template_file = None
            self.sed_grid = None

    def interp(self, t, filtername, lc_indx=0):
        """Interpolate absolute magnitude at rest-frame phase t (days) in `filtername`."""
        if lc_indx >= len(self.data) or filtername not in self.data[lc_indx]:
            return np.full_like(np.asarray(t, float), np.nan, dtype=float)
        t   = np.asarray(t, float)
        ph  = np.asarray(self.data[lc_indx][filtername]["ph"],  float)
        mag = np.asarray(self.data[lc_indx][filtername]["mag"], float)

        if ph.size == 0:
            return np.full_like(t, np.nan, dtype=float)

        # Log-time interpolation if support is strictly post-peak (>0)
        if np.all(ph > 0):
            x  = np.log10(np.clip(t, ph.min(), ph.max()))
            xp = np.log10(ph)
            out = np.interp(x, xp, mag, left=np.nan, right=np.nan)
        else:
            out = np.interp(t, ph, mag, left=np.nan, right=np.nan)

        out[(t < ph.min()) | (t > ph.max())] = np.nan
        return out




    # --------- Builder: from catalog (fits GP, saves pickle, returns LC) ---------
    @classmethod
    def from_catalog(cls,
                     inputs: CatalogInputs,
                     *,
                     filename_pattern: str = "{name}.csv",
                     use_ul: bool = False,
                     tpad_pre_days: float = 5.0,
                     tpad_post_days: float = 160.0,
                     n_time: int = 220,
                     kernel: george.kernels.Kernel | None = None,
                     min_points_for_fit: int = 6,
                     save_to: Path | None = None,
                     # legacy args kept for compatibility; ignored now:
                     filters_dir: Path | None = None,
                     user_filter_map: dict[str, float] | None = None,
                     per_row_lambda_cols: tuple[str, ...] = ("Cenwave",)
                     ) -> "LC":
        """
        Build SLSN templates from per-event photometry using a 2D GP in (time, frequency).
        **This version requires a per-row Cenwave (Å) column**, ideally from `{name}_cenwave.csv`.

        - We do NOT infer wavelengths from names anymore.
        - Band labels come from the per-event `filter` column (case preserved: 'R' != 'r').
        - For prediction, each band uses the *median* Cenwave measured for that band.
        """

        phot_dir = Path(inputs.photometry_dir)
        params = inputs.params_table.copy()
        params.columns = [str(c).strip() for c in params.columns]

        def resolve_col(df, key):
            if key in df.columns:
                return key
            kl = key.lower()
            for c in df.columns:
                if str(c).lower() == kl:
                    return c
            return key

        name_col = resolve_col(params, inputs.name_col)
        z_col    = resolve_col(params, inputs.z_col)
        peak_col = inputs.peak_mjd_col if inputs.peak_mjd_col in params.columns else None

        # inside LC.from_catalog, before the loop
        templates, names, meta, sed_grid = [], [], [], []
        t0_cat_list, t0_data_list, z_used, dm_list = [], [], [], []
        saved_t_grid = None

        for _, row in params.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue

            # redshift
            try:
                z = float(row[z_col])
            except Exception:
                warnings.warn(f"[{name}] missing/invalid redshift; skipping.")
                continue
            if not np.isfinite(z) or z <= 0:
                warnings.warn(f"[{name}] non-positive redshift; skipping.")
                continue

            # find a per-event file that *has* Cenwave
            path = _pick_event_file_with_cenwave(phot_dir, name, filename_pattern)
            if path is None:
                warnings.warn(f"[{name}] no per-event file with 'Cenwave' found; skipping.")
                continue

            # load photometry
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            cols = {c.lower().strip(): c for c in df.columns}
            need = ["mjd", "mag", "filter", "cenwave"]
            miss = [k for k in need if k not in cols]
            if miss:
                warnings.warn(f"[{name}] missing columns {miss} in {path.name}; skipping.")
                continue

            def colget(k): return df[cols[k]]

            mjd  = pd.to_numeric(colget("mjd"), errors="coerce").to_numpy(float)
            mag  = pd.to_numeric(colget("mag"), errors="coerce").to_numpy(float)
            merr = (pd.to_numeric(colget("mag_err"), errors="coerce").to_numpy(float)
                    if "mag_err" in cols else np.full_like(mag, np.nan))
            band = colget("filter").astype(str).to_numpy()
            lamA = pd.to_numeric(colget("cenwave"), errors="coerce").to_numpy(float)

            # optional UL handling
            if "ul" in cols and not use_ul:
                ul = colget("ul").astype(bool).to_numpy()
                keep = ~ul
            else:
                keep = np.ones_like(mjd, dtype=bool)

            mjd, mag, merr, band, lamA = mjd[keep], mag[keep], merr[keep], band[keep], lamA[keep]

            # drop non-finite & require positive Cenwave
            good = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(merr) & np.isfinite(lamA) & (lamA > 0)
            if good.sum() < min_points_for_fit:
                warnings.warn(f"[{name}] too few finite points after cuts; skipping.")
                continue

            mjd, mag, merr, band, lamA = mjd[good], mag[good], merr[good], band[good], lamA[good]

            # --- fit the 2D GP so we can (a) pick t0 via GP peaks and (b) later predict
            nu = angstrom_to_hz(lamA)
            flux    = mag_to_flux_jy(mag)
            fluxerr = magerr_to_fluxerr_jy(mag, merr)
            
            try:
                gp_predict = fit_2d_gp(mjd, nu, flux, fluxerr, kernel=kernel)
            except Exception as e:
                warnings.warn(f"[{name}] GP fit failed: {e!r}")
                continue


            # choose reference epoch t0 (observed frame)
            # --- catalog prior for t0 (if present) ---
            t0_cat = None
            if peak_col is not None:
                try:
                    t0_val = float(row[peak_col])
                    if np.isfinite(t0_val):
                        t0_cat = t0_val
                except Exception:
                    t0_cat = None
            
            # --- pick hybrid t0 using the GP + prior ---
            t0_used, t0_info = pick_t0_hybrid(
                mjd, band, lamA,
                z=z,
                gp_predict=gp_predict,
                t0_catalog=t0_cat,
                window_days=15.0,
                min_pts_in_window=5,
                agree_days=4.0,
                mag=mag,
            )

            t0_cat_list.append(float(t0_info.get("t0_cat", np.nan)))
            t0_data_list.append(float(t0_used))
            z_used.append(float(z))
            dm_list.append(float(dm_from_z(z)))
            
            # evaluation grid (observed frame) centered on t0_used
            t_eval_obs = np.linspace(t0_used - tpad_pre_days, t0_used + tpad_post_days, n_time)
            
            # per-band median Cenwave (same as before)
            med_lam_by_band = (
                pd.DataFrame({"band": band, "lamA": lamA})
                .groupby("band", sort=False)["lamA"].median()
            )
            target_freq_hz = {b: float(angstrom_to_hz(L)) for b, L in med_lam_by_band.items()}
            
            # predict GP surface at those bands
            pred = gp_predict_surface(gp_predict, t_eval_obs, target_freq_hz)
            
            # rest-frame phase & absolute magnitudes
            DM = dm_from_z(z)
            phase = (t_eval_obs - t0_used) / (1.0 + z)

            tpl = {}
            for b, (f_mu, _f_sig) in pred.items():
                m_app = flux_jy_to_mag(f_mu)
                M_abs = m_app - DM
                goodp = np.isfinite(phase) & np.isfinite(M_abs)
                ph_b  = phase[goodp]
                mag_b = M_abs[goodp]

                if ph_b.size < 5:
                    tpl[b] = {"ph": np.array([], float), "mag": np.array([], float)}
                    continue

                # enforce post-peak positive support for log-time interpolation
                floor = 1e-4
                keep_pos = ph_b >= floor
                if keep_pos.sum() >= 3:
                    ph_b  = ph_b[keep_pos]
                    mag_b = mag_b[keep_pos]

                order = np.argsort(ph_b)
                tpl[b] = {"ph": ph_b[order], "mag": mag_b[order]}

            meta.append({
                "name": name,
                "z": float(z),
                "t0_used": float(t0_used),
                **t0_info
            })

            # --- Rubin-ready SED grid (rest frame) ---
            # Choose a modest, safe rest-frame lambda grid based on THIS event's coverage
            lam_obs_min = np.nanpercentile(lamA, 5)   # observed Å in the input rows
            lam_obs_max = np.nanpercentile(lamA, 95)
            lam_rest_min = lam_obs_min / (1.0 + z)
            lam_rest_max = lam_obs_max / (1.0 + z)
            
            # Pad a little, but don't extrapolate wildly
            pad = 0.05 * (lam_rest_max - lam_rest_min)
            lam_rest = np.linspace(max(1000.0, lam_rest_min - pad),
                                   min(25000.0, lam_rest_max + pad),
                                   180).astype(float)  # ~100–200 bins is fine
            
            # We'll evaluate the GP at the observed-frame frequency that corresponds to each rest lambda:
            #   λ_obs = (1+z)*λ_rest  ->  ν_obs = c / λ_obs
            lam_obs_for_eval = (1.0 + z) * lam_rest
            nu_obs_for_eval  = angstrom_to_hz(lam_obs_for_eval)
            
            # Evaluate GP on a common observed-time grid (you already have t_eval_obs)
            # Build the full (t, ν) prediction matrix by looping through ν columns
            F_app = np.empty((t_eval_obs.size, lam_rest.size), dtype=float)
            Var_app = np.empty_like(F_app)
            
            for j, nu_j in enumerate(nu_obs_for_eval):
                Xp = np.vstack([t_eval_obs, np.full_like(t_eval_obs, nu_j)]).T
                mu_j, var_j = gp_predict(Xp, return_var=True)
                F_app[:, j]  = mu_j
                Var_app[:, j] = np.clip(var_j, 0.0, np.inf)
            
            # Convert apparent flux → absolute rest-frame flux at 10 pc (per-frequency)
            # Fnu_abs = ((DL/10pc)^2) * (1+z) * Fnu_obs
            DL_pc = cosmo.luminosity_distance(z).to_value(u.pc)
            scale_abs = (DL_pc / 10.0)**2 * (1.0 + z)
            Fnu_abs = scale_abs * F_app  # shape (n_time, n_lambda)
            
            # Build rest-frame phase axis (already computed as 'phase')
            # Optionally mark a coverage mask where GP is trustworthy:
            # Simple heuristic: require var < (k * median(var_in_band)) and inside original ν range
            nu_obs_min = np.nanmin(angstrom_to_hz(lamA))
            nu_obs_max = np.nanmax(angstrom_to_hz(lamA))
            coverage = (
                (nu_obs_for_eval[None, :] >= 0.95*nu_obs_min) &
                (nu_obs_for_eval[None, :] <= 1.05*nu_obs_max) &
                np.isfinite(Fnu_abs)
            )
            
            # Store for this event
            sed_entry = {
                "phase": phase.astype(float),          # 1D
                "lam_rest_A": lam_rest.astype(float),  # 1D
                "Fnu_abs": Fnu_abs.astype(float),      # 2D [phase, lam]
                "coverage": coverage.astype(bool),     # 2D
            }

            # inside the per-event loop, after templates.append(tpl):
            templates.append(tpl)
            sed_grid.append(sed_entry)
            names.append(name)
            #t0_cat_list.append(float(row[peak_col]) if (peak_col is not None and np.isfinite(row[peak_col])) else np.nan)
            #t0_data_list.append(float(t0))
            #z_used.append(float(z))
            #dm_list.append(float(dm_from_z(z)))


            if saved_t_grid is None:
                saved_t_grid = phase.tolist()

        # Build once, save (with names), and return the same instance
        model = cls(lightcurves=templates, t_grid=saved_t_grid, names=names)
        model.sed_grid = sed_grid

        if save_to is not None:
            atomic_save_pickle({
                "lightcurves": templates,
                "t_grid": saved_t_grid,
                "names": names,
                "sed_grid": sed_grid,           # <--- new
                "meta": {
                    "z": z_used,
                    "dm": dm_list,
                    "t0_cat": t0_cat_list,
                    "t0_data": t0_data_list,
                }
            }, save_to)
            model.template_file = str(save_to)


        return model

    def build_magnitude_grid(self, 
                             z_grid=None, 
                             phase_grid=None,
                             filters='ugrizy',
                             save_to=None):
        """
        Pre-compute LSST apparent magnitudes on a (template, z, phase, filter) grid.
        
        This replaces thousands of on-the-fly SED syntheses with fast 2D interpolation.
        Takes ~5-10 minutes to build once, then all simulations are 100x+ faster.
        
        Parameters
        ----------
        z_grid : array-like, optional
            Redshift sampling points. Default: 50 points from 0.02 to 2.0
        phase_grid : array-like, optional
            Rest-frame phase sampling (days). Default: 0.1 to 160 days (log-spaced)
        filters : str or list
            LSST filters to pre-compute
        save_to : Path, optional
            If provided, save grid to disk for reuse
        """
        print("[build_magnitude_grid] Starting pre-computation...")
        
        if z_grid is None:
            z_grid = np.linspace(0.02, 2.0, 50)
        if phase_grid is None:
            # Log-spaced to match template support
            phase_grid = np.geomspace(0.1, 160, 200)
        
        z_grid = np.asarray(z_grid, float)
        phase_grid = np.asarray(phase_grid, float)
        
        if isinstance(filters, str):
            filters = list(filters)
        
        n_templates = len(self.sed_grid)
        n_z = len(z_grid)
        n_phase = len(phase_grid)
        
        # Storage: [n_templates, n_filters, n_z, n_phase]
        self.mag_grid = {}
        for filt in filters:
            self.mag_grid[filt] = np.full((n_templates, n_z, n_phase), np.nan, dtype=np.float32)
        
        # Pre-compute for all templates
        from tqdm.auto import tqdm
        for i_tpl in tqdm(range(n_templates), desc="Templates"):
            sed = self.sed_grid[i_tpl]
            
            for i_z, z in enumerate(z_grid):
                for filt in filters:
                    # Vectorized: compute all phases at once for this (template, z, filter)
                    mags = np.array([
                        synthesize_mag_at_z(sed, phase, z, filt) 
                        for phase in phase_grid
                    ])
                    self.mag_grid[filt][i_tpl, i_z, :] = mags
        
        # Store grid axes for interpolation
        self.mag_grid_axes = {
            'z': z_grid,
            'phase': phase_grid
        }
        
        print(f"[build_magnitude_grid] Complete. Grid shape per filter: "
              f"{self.mag_grid[filters[0]].shape}")
        
        if save_to:
            grid_data = {
                'mag_grid': self.mag_grid,
                'mag_grid_axes': self.mag_grid_axes,
                'filters': filters
            }
            atomic_save_pickle(grid_data, save_to)
            print(f"[build_magnitude_grid] Saved to {save_to}")
        
        return self

    def load_magnitude_grid(self, grid_file):
        """Load pre-computed magnitude grid from disk."""
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
        
        with open(grid_file, 'rb') as f:
            grid_data = pickle.load(f)
        
        self.mag_grid = grid_data['mag_grid']
        self.mag_grid_axes = grid_data['mag_grid_axes']
        
        print(f"[load_magnitude_grid] Loaded grid with filters: {list(self.mag_grid.keys())}")
        return self


# ----------------------------
# helper to find the template index for a given event:
# ----------------------------

def _load_names(templates_or_path):
    import pickle
    if hasattr(templates_or_path, "names"):  # LC instance
        return list(getattr(templates_or_path, "names", []))
    with open(templates_or_path, "rb") as f:
        obj = pickle.load(f)
    return list(obj.get("names", []))

def template_index_for_event(templates_or_path, event_name: str) -> int:
    names = _load_names(templates_or_path)
    if not names:
        raise RuntimeError("No template names available. Rebuild and save with 'names'.")
    try:
        return names.index(event_name)
    except ValueError as e:
        raise ValueError(f"'{event_name}' not found. Available: {len(names)} templates.") from e

def template_name_for_index(templates_or_path, idx: int) -> str:
    names = _load_names(templates_or_path)
    if not names:
        return f"tpl_{idx}"
    if not (0 <= idx < len(names)):
        raise IndexError(f"Index {idx} out of range (0..{len(names)-1}).")
    return names[idx]


# ----------------------------
# Atomic pickle (safe writes)
# ----------------------------

def atomic_save_pickle(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))
        print(f"[INFO] Saved atomically to {path}")
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

#--------------------------------
# Read filters from Sebastian's git
#--------------------------------

# --- central wavelength definitions (choose one) ---
def effective_wavelength(lam_A, trans):
    """
    λ_eff = ∫ λ T(λ) dλ / ∫ T(λ) dλ  (in Å)
    """
    lam = np.asarray(lam_A, float)
    T   = np.asarray(trans, float)
    good = np.isfinite(lam) & np.isfinite(T) & (T > 0)
    if good.sum() < 2: 
        return np.nan
    num = np.trapz(lam[good] * T[good], lam[good])
    den = np.trapz(T[good], lam[good])
    return num / den if den > 0 else np.nan

def pivot_wavelength(lam_A, trans):
    """
    λ_p = sqrt( ∫ T(λ) λ dλ  /  ∫ T(λ) dλ/λ )  (in Å)
    More robust across SED differences; commonly used in synthetic photometry.
    """
    lam = np.asarray(lam_A, float)
    T   = np.asarray(trans, float)
    good = np.isfinite(lam) & np.isfinite(T) & (T > 0)
    if good.sum() < 2:
        return np.nan
    num = np.trapz(T[good] * lam[good], lam[good])
    den = np.trapz(T[good] / lam[good], lam[good])
    return np.sqrt(num / den) if den > 0 else np.nan


def _read_filter_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a 2-column filter file (wavelength, throughput).
    Tries space/tab CSV, falls back to pandas if needed.
    Returns arrays in Å and unitless throughput in [0,1] (not enforced).
    """
    txt = path.read_text(errors="ignore")
    # drop comment lines
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    # try fast numpy load
    try:
        arr = np.loadtxt(lines)
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(-1, 2)
        lam, T = arr[:, 0], arr[:, 1]
        return lam, T
    except Exception:
        # fallback: pandas with flexible separators
        df = pd.read_csv(path, comment="#", header=None, delim_whitespace=True)
        if df.shape[1] < 2:
            # try comma
            df = pd.read_csv(path, comment="#", header=None)
        lam = df.iloc[:, 0].to_numpy(dtype=float, copy=False)
        T   = df.iloc[:, 1].to_numpy(dtype=float, copy=False)
        return lam, T


def _name_from_filename(path: Path) -> str:
    """
    Turn filenames into short filter keys; you can tune this.
    e.g., 'decam_g.txt' -> 'decam_g', 'swift_UVW1.dat' -> 'swift_uvw1'
    """
    stem = path.stem.lower()
    stem = re.sub(r"[^a-z0-9_+.-]+", "", stem)
    stem = stem.replace("+", "")  # clean things like 'UVM2+atm'
    return stem

# -----------------------------
# 1) Observations-only plot
# -----------------------------
def plot_event_obs(
    event_name: str,
    per_event_dir: Path,
    *,
    use_phase: bool = False,           # if True, you must pass (z, t0)
    z: float | None = None,
    t0: float | None = None,           # peak/explosion MJD used for phase
    bands: list[str] | None = None,    # subset of filters to show (case preserved)
    case_sensitive: bool = True,
    scale_to_peak: bool = False,       # subtract min apparent mag per-event
    s_obs: float = 14.0,
    alpha_obs: float = 0.8,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot {event}_cenwave.csv grouped by filter (no GP curves)."""
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    need = {"mjd", "mag", "filter"}
    if not need.issubset(cols):
        raise ValueError(f"{csv.name} is missing {need - set(cols)}")

    x = df[cols["mjd"]].to_numpy(float)
    y = df[cols["mag"]].to_numpy(float)
    f = df[cols["filter"]].astype(str)

    if use_phase:
        if z is None or t0 is None:
            raise ValueError("use_phase=True requires z and t0.")
        x = (x - float(t0)) / (1.0 + float(z))

    if scale_to_peak and np.isfinite(y).any():
        y = y - np.nanmin(y)

    # choose bands to draw
    band_vals = np.unique(f) if bands is None else np.array(bands, dtype=str)

    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(band_vals)}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, b in enumerate(band_vals):
        mask = (f == b) if case_sensitive else (f.str.lower() == str(b).lower())
        if not mask.any():
            continue
        xi = x[mask]
        yi = y[mask]
        ok = np.isfinite(xi) & np.isfinite(yi)
        if ok.sum() == 0:
            continue
        ax.scatter(
            xi[ok], yi[ok],
            s=s_obs, alpha=alpha_obs, facecolors="none",
            edgecolors=color_map[b], linewidths=0.9, label=b
        )

    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Rest-frame Phase [days]" if use_phase else "MJD")
    ax.set_ylabel("Scaled Apparent Mag" if scale_to_peak else "Apparent Mag")
    ax.legend(frameon=False, ncol=min(8, len(band_vals)))
    ax.set_title(title or event_name)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 2) Model-only plot
# -----------------------------
def plot_event_model(
    templates_file: Path,
    template_idx: int,
    *,
    bands: list[str] | None = None,
    case_sensitive: bool = True,
    use_log_time: bool = False,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    lw: float = 1.6,
    alpha: float = 0.9,
):
    """Plot one template’s model curves (absolute mag vs rest-frame phase)."""

    with open(templates_file, "rb") as f:
        obj = pickle.load(f)

    # make sure lcs exists before using it
    lcs: list[dict] = list(obj.get("lightcurves", []))
    names = obj.get("names", [f"tpl_{i}" for i in range(len(lcs))])

    if not (0 <= template_idx < len(lcs)):
        raise IndexError(f"template_idx out of range (0..{len(lcs)-1}).")

    tpl = lcs[template_idx]

    # discover bands in this template
    def keys_for_plot():
        ks = []
        for k, comp in tpl.items():
            if isinstance(comp, dict) and {"ph","mag"} <= set(comp):
                ks.append(k)
        return ks

    draw_keys = bands if bands is not None else keys_for_plot()

    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(draw_keys)}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for b in draw_keys:
        key = b if case_sensitive else next((k for k in tpl if str(k).lower()==str(b).lower()), None)
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph","mag"} <= set(comp)):
            continue
        t = np.asarray(comp["ph"], float)
        m = np.asarray(comp["mag"], float)
        ok = np.isfinite(t) & np.isfinite(m)
        if use_log_time:
            ok &= (t > 0)
        if ok.sum() < 2:
            continue
        x = np.log10(t[ok]) if use_log_time else t[ok]
        y = m[ok]
        ax.plot(x, y, color=color_map[b], lw=lw, alpha=alpha, label=b)

    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("log10(Phase [days])" if use_log_time else "Rest-frame Phase [days]")
    ax.set_ylabel("Absolute Magnitude (AB)")
    ax.legend(frameon=False, ncol=min(8, len(draw_keys)))
    # show event name if present
    ax.set_title(title or f"{names[template_idx]} (template #{template_idx})")
    plt.tight_layout()
    plt.show()


# -----------------------------
# 3) Overlay: obs + model
# -----------------------------
def plot_event_obs_vs_model(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    *,
    # x-axis: obs in phase requires z,t0
    use_phase: bool = True,
    z: float | None = None,
    t0: float | None = None,
    bands: list[str] | None = None,      # bands to draw (both obs + model)
    case_sensitive: bool = True,
    scale_to_peak: bool = False,
    use_log_time_for_model: bool = False,
    ylim: tuple[float, float] | None = None,
    s_obs: float = 12.0,
    lw_model: float = 1.6,
):
    """Overlay {event}_cenwave.csv (points) with template curves (lines)."""
    # load model
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs: list[dict] = obj.get("lightcurves", [])
    tpl = lcs[template_idx]

    # observations
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    x_obs = df[cols["mjd"]].to_numpy(float)
    y_obs = df[cols["mag"]].to_numpy(float)
    f_obs = df[cols["filter"]].astype(str)

    if use_phase:
        if z is None or t0 is None:
            raise ValueError("use_phase=True requires z and t0.")
        x_obs = (x_obs - float(t0)) / (1.0 + float(z))

    if scale_to_peak and np.isfinite(y_obs).any():
        y_obs = y_obs - np.nanmin(y_obs)

    # choose band keys present in either obs or model
    model_keys = [k for k, comp in tpl.items() if isinstance(comp, dict) and {"ph", "mag"} <= set(comp)]
    want = bands or sorted(set(model_keys) | set(np.unique(f_obs)))
    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(want)}

    fig, ax = plt.subplots(figsize=(8, 5))
    # plot obs
    for b in want:
        mask = (f_obs == b) if case_sensitive else (f_obs.str.lower() == str(b).lower())
        if mask.any():
            xi = x_obs[mask]
            yi = y_obs[mask]
            ok = np.isfinite(xi) & np.isfinite(yi)
            if ok.sum():
                ax.scatter(
                    xi[ok], yi[ok], s=s_obs, alpha=0.75,
                    facecolors="none", edgecolors=color_map[b], linewidths=0.9, label=f"{b} (obs)"
                )
    # plot model
    for b in want:
        key = b if case_sensitive else next((k for k in tpl if str(k).lower() == str(b).lower()), None)
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph", "mag"} <= set(comp)):
            continue
        t = np.asarray(comp["ph"], float)
        m = np.asarray(comp["mag"], float)
        ok = np.isfinite(t) & np.isfinite(m)
        if use_log_time_for_model:
            ok &= (t > 0)
        if ok.sum() == 0:
            continue
        x = np.log10(t[ok]) if use_log_time_for_model else t[ok]
        y = m[ok]
        ax.plot(x, y, color=color_map[b], lw=lw_model, alpha=0.9, label=f"{b} (model)")

    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Rest-frame Phase [days]" if use_phase else "MJD")
    ax.set_ylabel("Scaled Apparent Mag" if scale_to_peak else "Magnitude")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.set_title(f"{event_name}  —  template #{template_idx}")
    plt.tight_layout()
    plt.show()

# -----------------
# Diagnostics 
#-----------------

__all__ = [
    "diagnose_abs_from_templates",
    "diagnose_internal_consistency",
    "plot_residuals",
    "list_event_bands",
    "list_template_bands",
]

# ---------- helpers ----------

def _dm_from_z(z: float) -> float:
    DL_Mpc = cosmo.luminosity_distance(float(z)).to_value(u.Mpc)
    return 5.0 * np.log10(DL_Mpc) + 25.0

def list_event_bands(per_event_dir: Path, event_name: str) -> list[str]:
    """List distinct `filter` strings present in {event}_cenwave.csv."""
    p = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(p)
    # accept either 'Filter' or 'filter'
    cols = {c.lower(): c for c in df.columns}
    key = cols.get("filter")
    if key is None:
        raise KeyError(f"{p.name} has no 'filter' column (found {list(df.columns)})")
    return sorted(df[key].astype(str).unique().tolist())

def list_template_bands(templates_file: Path, template_idx: int) -> list[str]:
    """List band keys present in a saved template."""
    obj = pickle.load(open(templates_file, "rb"))
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]
    return sorted([k for k, v in tpl.items() if isinstance(v, dict) and {"ph","mag"} <= set(v)])

# ---------- Diagnostic 1: observed vs. (template+DM) ----------

def diagnose_abs_from_templates(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    *,
    z: float,
    t0: float,
    bands: list[str] | None = None,     # e.g. ["g","r","i","z"] or None for all shared
    case_sensitive: bool = True,
    use_log_time_for_model: bool = False,
    return_with_phase: bool = False,    # if True, include phase for plotting residual vs phase
    verbose: bool = True
) -> pd.DataFrame:
    """
    Residual = m_obs  -  ( M_template(phase) + DM(z) )
    Expect ~0 mag (per band) if absolute-magnitude construction is correct.
    """
    # load model
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]

    # load observations
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    for need in ("mjd","mag","filter"):
        if need not in cols:
            raise ValueError(f"{csv.name} missing column '{need}'")
    mjd  = df[cols["mjd"]].to_numpy(float)
    mobs = df[cols["mag"]].to_numpy(float)
    fobs = df[cols["filter"]].astype(str)
    phase_obs = (mjd - float(t0)) / (1.0 + float(z))

    # bands to compare (intersection of obs+model unless explicit)
    model_bands = [k for k, v in tpl.items() if isinstance(v, dict) and {"ph","mag"} <= set(v)]
    want = bands or sorted(set(model_bands) | set(np.unique(fobs)))

    DM = _dm_from_z(z)
    rows = []
    for b in want:
        # model component
        key = b if case_sensitive else next((k for k in model_bands if k.lower() == str(b).lower()), None)
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph","mag"} <= set(comp)):
            continue
        ph = np.asarray(comp["ph"], float)
        Ma = np.asarray(comp["mag"], float)
        okm = np.isfinite(ph) & np.isfinite(Ma)
        if okm.sum() < 2:
            continue

        # obs in this band
        mask = (fobs == b) if case_sensitive else (fobs.str.lower() == str(b).lower())
        if not mask.any():
            continue
        ph_i = phase_obs[mask]
        m_i  = mobs[mask]

        # choose linear vs log-time interpolation
        if use_log_time_for_model:
            pos = okm & (ph > 0)
            if pos.sum() < 2:
                continue
            xp = np.log10(ph[pos]); fp = Ma[pos]
            sel = ph_i > 0
            Mi = np.full_like(ph_i, np.nan, float)
            Mi[sel] = np.interp(np.log10(np.clip(ph_i[sel], 10**xp.min(), 10**xp.max())), xp, fp)
        else:
            xp, fp = ph[okm], Ma[okm]
            Mi = np.interp(np.clip(ph_i, xp.min(), xp.max()), xp, fp, left=np.nan, right=np.nan)

        resid = m_i - (Mi + DM)
        for rr, phv in zip(resid, ph_i):
            rows.append((b, float(rr), float(phv)))

    if not rows:
        if verbose:
            print("[diagnose_abs_from_templates] No overlapping points to compare.")
        return pd.DataFrame(columns=["band","resid_mag","phase"])

    out = pd.DataFrame(rows, columns=["band","resid_mag","phase"])
    if not return_with_phase:
        out = out[["band","resid_mag"]]

    if verbose:
        g = out.groupby("band")["resid_mag"]
        summ = pd.DataFrame({"N": g.size(), "median": g.median(), "mean": g.mean(), "std": g.std()})
        print("[abs-from-templates] m_obs - (M_abs + DM) [mag]:")
        print(summ.sort_index())
        print("Global median:", np.nanmedian(out["resid_mag"]))

    return out

# ---------- Diagnostic 2: re-fit GP and compare to template ----------

def diagnose_internal_consistency(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    *,
    z: float,
    t0: float,
    n_time: int = 220,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Re-fit 2D GP on the event; compare:
        m_app_from_GP(t, band_medCenwave)  vs  (M_template(phase) + DM)
    Expect residual ~ 0 if flux→mag and DM usage are consistent.
    """
    # load template
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]

    # event data
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    need = ("mjd","mag","mag_err","filter","cenwave")
    for k in need:
        if k not in cols:
            raise ValueError(f"{csv.name} missing column '{k}'")
    mjd  = df[cols["mjd"]].to_numpy(float)
    mag  = df[cols["mag"]].to_numpy(float)
    merr = df[cols["mag_err"]].to_numpy(float)
    band = df[cols["filter"]].astype(str)
    lamA = df[cols["cenwave"]].to_numpy(float)

    # flux domain (AB)
    F0 = 3631.0
    ln10_over_2p5 = np.log(10.0)/2.5
    def mag_to_flux(m): return F0 * 10.0**(-0.4*np.asarray(m,float))
    def magerr_to_fluxerr(m, me):
        f = mag_to_flux(m)
        return ln10_over_2p5 * f * np.asarray(me,float)
    flux    = mag_to_flux(mag)
    fluxerr = magerr_to_fluxerr(mag, merr)

    # frequencies (Hz)
    C = 2.99792458e8
    nu = C / (lamA*1e-10)

    # fit GP (Matérn-3/2 in 2D)
    import george, scipy.optimize as op
    from functools import partial

    def _default_kernel(scale_guess):
        time_scale_days = 20.0; freq_scale_hz = 1e14
        k = (0.5*max(scale_guess,1e-6))**2 * george.kernels.Matern32Kernel([time_scale_days**2, freq_scale_hz**2], ndim=2)
        k.freeze_parameter("k2:metric:log_M_1_1")
        return k

    good = np.isfinite(mjd)&np.isfinite(nu)&np.isfinite(flux)&np.isfinite(fluxerr)&(fluxerr>0)
    mjd_, nu_, y_, yerr_ = mjd[good], nu[good], flux[good], fluxerr[good]
    X = np.vstack([mjd_, nu_]).T
    scale_guess = np.nanmedian(np.abs(y_))
    if not np.isfinite(scale_guess) or scale_guess <= 0:
        scale_guess = 1e-3
    kernel = _default_kernel(scale_guess)
    gp = george.GP(kernel)
    gp.compute(X, yerr_)

    p0 = gp.get_parameter_vector()
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y_, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    def grad(p):
        gp.set_parameter_vector(p); return -gp.grad_log_likelihood(y_, quiet=True)
    try:
        res = op.minimize(nll, p0, jac=grad, method="L-BFGS-B")
        if res.success: gp.set_parameter_vector(res.x)
    except Exception:
        pass
    gp_predict = partial(gp.predict, y_)

    # evaluation grid in observed frame
    t_eval = np.linspace(mjd_.min(), mjd_.max(), int(n_time))

    # per-band median Cenwave -> predict apparent mag curves
    med_lam = pd.DataFrame({"b": band, "lam": lamA}).groupby("b", sort=False)["lam"].median()
    DM = _dm_from_z(z)
    rows = []
    for b, L in med_lam.items():
        nu0 = C/(L*1e-10)
        Xp = np.vstack([t_eval, np.full_like(t_eval, nu0)]).T
        mu, var = gp_predict(Xp, return_var=True)
        # flux->AB mag
        m_gp = np.full_like(mu, np.nan, float); ok = mu > 0
        m_gp[ok] = -2.5*np.log10(mu[ok]/F0)

        # template abs at matching rest-frame phases
        comp = tpl.get(b) or next((tpl[k] for k in tpl if k.lower()==str(b).lower()), None)
        if not (isinstance(comp, dict) and {"ph","mag"} <= set(comp)):
            continue
        ph = np.asarray(comp["ph"], float)
        Ma = np.asarray(comp["mag"], float)
        okm = np.isfinite(ph) & np.isfinite(Ma)
        if okm.sum() < 2:
            continue

        phase_eval = (t_eval - float(t0)) / (1.0 + float(z))
        M_on_grid  = np.interp(np.clip(phase_eval, ph[okm].min(), ph[okm].max()), ph[okm], Ma[okm], left=np.nan, right=np.nan)
        resid = m_gp - (M_on_grid + DM)
        for r, phv in zip(resid, phase_eval):
            if np.isfinite(r):
                rows.append((b, float(r), float(phv)))

    out = pd.DataFrame(rows, columns=["band","resid_mag","phase"])
    if verbose and not out.empty:
        g = out.groupby("band")["resid_mag"]
        summ = pd.DataFrame({"N": g.size(), "median": g.median(), "std": g.std()})
        print("[internal] m_GP - (M_abs + DM) [mag]:")
        print(summ.sort_index())
        print("Global median:", np.nanmedian(out["resid_mag"]))

    return out

# ---------- quick plotting ----------

def plot_residuals(df: pd.DataFrame, *, by_band=True, vs_phase=False, title=None):
    """
    df can be the output of either diagnostic (with columns 'band','resid_mag', and optional 'phase').
    """
    if df is None or df.empty:
        print("[plot_residuals] nothing to plot.")
        return
    if vs_phase and "phase" not in df.columns:
        print("[plot_residuals] 'phase' column not present; plotting by band only.")
        vs_phase = False

    if vs_phase:
        # residual vs phase with median per band
        bands = sorted(df["band"].unique())
        cmap = plt.get_cmap("tab20")
        color_map = {b: cmap(i % 20) for i, b in enumerate(bands)}
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for b in bands:
            sub = df[df["band"] == b]
            ax.scatter(sub["phase"], sub["resid_mag"], s=12, alpha=0.6, label=b, facecolors="none",
                       edgecolors=color_map[b], linewidths=0.8)
            # running median (coarse)
            if len(sub) >= 8:
                q = sub.sort_values("phase")
                k = max(5, len(q)//12)
                med = q["resid_mag"].rolling(window=k, center=True, min_periods=max(3, k//2)).median()
                ax.plot(q["phase"], med, alpha=0.9, lw=1.5, color=color_map[b])
        ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
        ax.set_xlabel("Rest-frame Phase [days]")
        ax.set_ylabel("Residual  m_obs − (M + DM)  [mag]")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=min(8, len(bands)))
        if title: ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return

    if by_band:
        g = df.groupby("band")["resid_mag"]
        summ = pd.DataFrame({"N": g.size(), "median": g.median(), "std": g.std()}).sort_index()
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(summ.index, summ["median"], yerr=summ["std"], fmt="o", capsize=3)
        ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
        ax.set_ylabel("Residual  m_obs − (M + DM)  [mag]")
        ax.set_xlabel("Band")
        ax.grid(True, axis="y", alpha=0.25)
        if title: ax.set_title(title)
        plt.tight_layout()
        plt.show()

# --- helper: get (z, t0) the builder would use for an event ---
def get_event_z_t0(event_name: str,
                   params_table: pd.DataFrame,
                   per_event_dir: Path,
                   peak_mjd_col: str = "Peak_MJD_med") -> tuple[float, float]:
    # z from params table
    dfp = params_table.copy()
    cols = {c.lower(): c for c in dfp.columns}
    name_col  = cols.get("name", list(dfp.columns)[0])
    z_col     = cols.get("redshift_med", "redshift_med")
    peak_col  = cols.get(peak_mjd_col.lower(), None)

    row = dfp.loc[dfp[name_col].astype(str) == str(event_name)]
    if row.empty:
        raise ValueError(f"{event_name} not in params table.")
    z = float(pd.to_numeric(row[z_col], errors="coerce").iloc[0])

    # t0 = Peak_MJD_med if finite, else min(mag) time from per-event CSV
    t0 = None
    if peak_col is not None:
        t0 = pd.to_numeric(row[peak_col], errors="coerce").iloc[0]
        t0 = float(t0) if np.isfinite(t0) else None
    if t0 is None:
        csv = Path(per_event_dir) / f"{event_name}.csv"
        df  = pd.read_csv(csv)
        c   = {c.lower(): c for c in df.columns}
        mjd = pd.to_numeric(df[c["mjd"]], errors="coerce").to_numpy(float)
        mag = pd.to_numeric(df[c["mag"]], errors="coerce").to_numpy(float)
        t0  = mjd[np.nanargmin(mag)]
    return z, t0

# -- iny utility to inspect t₀ choices per event

def list_t0_meta(templates_file):
    import pickle, pandas as pd
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    meta = obj.get("meta", [])
    return pd.DataFrame(meta).sort_values("name").reset_index(drop=True)

def _trapz_norm(lam, T):
    good = np.isfinite(lam) & np.isfinite(T) & (T > 0)
    if good.sum() < 2: return 1.0
    return np.trapz(T[good], lam[good])



def _m52snr(mag: np.ndarray, m5: np.ndarray) -> np.ndarray:
    """
    LSST single-visit SNR from 5σ depth. SNR ≈ 5 * 10^{0.4 (m5 - m)}.
    """
    snr = np.full_like(mag, np.nan, dtype=float)
    finite = np.isfinite(mag) & np.isfinite(m5)
    snr[finite] = 5.0 * (10.0 ** (0.4 * (m5[finite] - mag[finite])))
    return snr

def _phase_key(phase_rest, step=0.2):
    # 0.05 d grid; adjust to 0.02 if you need a bit finer
    return float(np.round(phase_rest/step)*step)

def synthesize_mag_at_z_cached(cache, sed_grid, phase_rest, z, filt):
    key = (id(sed_grid), filt, _phase_key(phase_rest), round(float(z), 5))
    if key in cache:
        return cache[key]
    val = synthesize_mag_at_z(sed_grid, phase_rest, z, filt)  # your integrated method (now uses calc_mag/correct API)
    cache[key] = val
    return val

def _project_template_to_lsst_abs(metric, sed_grid, z):
    """
    Returns dict: {'phase': phase, 'u': M_u(phase), ...} cached on metric._proj_cache
    """
    cache = getattr(metric, "_proj_cache", None)
    if cache is None:
        cache = {}
        setattr(metric, "_proj_cache", cache)
    key = (id(sed_grid), round(float(z),5))
    if key in cache:
        return cache[key]

    bands = _get_lsst_bands()
    phase = np.asarray(sed_grid["phase"], float)
    lam_rest = np.asarray(sed_grid["lam_rest_A"], float)
    Fnu_abs = np.asarray(sed_grid["Fnu_abs"], float)  # shape (nphase, nlam)

    out = {"phase": phase}
    for f,bp in bands.items():
        # build a rest-frame bandpass sampled on lam_rest grid
        # sample T_obs at lam_obs = (1+z)*lam_rest
        lam_obs = (1.0 + z) * lam_rest
        # Interpolate bp throughput onto lam_obs, then reuse on lam_rest domain
        # (Bandpass exposes arrays; we use numpy interp.)
        T_obs = np.interp(lam_obs, bp.wavelen, bp.sb, left=0.0, right=0.0)

        # Convert Fnu_abs(ν) -> Flambda_abs(λ) for integration with T(λ)
        # Fλ = Fν * c / λ^2 ; λ in meters for unit-consistency, but relative constant cancels in AB mag calc_mag.
        lam_m = lam_rest * 1e-10
        Flambda = Fnu_abs * (2.99792458e8) / (lam_m**2)

        # Build a Sed per phase is costly; integrate directly to AB is non-trivial.
        # Pragmatic: build a Sed once per phase with (lam_rest, Flambda) and use a synthetic Bandpass with (lam_rest, T_obs).
        from rubin_sim.phot_utils import Bandpass, Sed
        bp_rest = Bandpass(wavelen=lam_rest, sb=T_obs)
        M = np.full_like(phase, np.nan, float)
        for i in range(len(phase)):
            sed = Sed(wavelen=lam_rest, flambda=Flambda[i])
            try:
                M[i] = float(sed.calc_mag(bp_rest))
            except Exception:
                M[i] = np.nan
        out[f] = M

    cache[key] = out
    return out

def evaluate(metric, dataSlice, slice_point, return_full_obs=False):
    """
    SLSN-specific evaluate using pre-computed magnitude grid for speed.
    Falls back to on-the-fly synthesis if grid unavailable (but warns).
    """
    use_grid = hasattr(metric.lc_model, 'mag_grid') and metric.lc_model.mag_grid
    
    if not use_grid:
        warnings.warn(
            "No magnitude grid found on lc_model. "
            "Falling back to slow SED synthesis. "
            "Run lc_model.build_magnitude_grid() once to speed up by 100x+",
            RuntimeWarning
        )
    
    # Get SED grid
    sed_grid_all = getattr(metric.lc_model, 'sed_grid', None)
    if sed_grid_all is None:
        template_path = getattr(metric.lc_model, 'template_file', None)
        if template_path is None:
            raise AttributeError("LC object missing both sed_grid and template_file")
        with open(template_path, 'rb') as f:
            templates = pickle.load(f)
        sed_grid_all = templates['sed_grid']
        metric.lc_model.sed_grid = sed_grid_all
    
    # Bounds-checked template index
    tpl_idx = int(slice_point['file_indx'])
    tpl_idx = max(0, min(tpl_idx, len(sed_grid_all) - 1))
    sed_grid = sed_grid_all[tpl_idx]
    
    z   = float(slice_point['z'])
    t0  = float(slice_point['peak_time'])
    dm  = float(slice_point.get('distance_modulus', cosmo.distmod(z).value))
    ebv = float(slice_point.get('ebv', 0.0))
    
    dust_model_cached = getattr(metric, 'dust_model', None) or DustValues()
    setattr(metric, 'dust_model', dust_model_cached)
    
    # Extract arrays
    names = dataSlice.dtype.names or ()
    filters = dataSlice[metric.filterCol].astype(str)
    mjds    = dataSlice[metric.mjdCol].astype(float)
    m5s     = dataSlice[metric.m5Col].astype(float) if metric.m5Col in names else np.full(len(dataSlice), np.nan)
    
    phase_rest = (mjds - t0) / (1.0 + z)
    
    # ========== FAST PATH: Grid Interpolation ==========
    if use_grid:
        from scipy.interpolate import RegularGridInterpolator
        
        z_grid = metric.lc_model.mag_grid_axes['z']
        phase_grid = metric.lc_model.mag_grid_axes['phase']
        
        mags = np.full(len(dataSlice), np.nan, dtype=float)
        
        for filt in np.unique(filters):
            mask = (filters == filt)
            if not np.any(mask):
                continue
            
            if filt not in metric.lc_model.mag_grid:
                continue
            
            grid_3d = metric.lc_model.mag_grid[filt][tpl_idx, :, :]  # Now tpl_idx is safe
            
            interp = RegularGridInterpolator(
                (z_grid, phase_grid),
                grid_3d,
                bounds_error=False,
                fill_value=np.nan,
                method='linear'
            )
            
            phases_f = phase_rest[mask]
            z_repeated = np.full_like(phases_f, z)
            points = np.column_stack([z_repeated, phases_f])
            
            mags[mask] = interp(points)
        
        mags = mags + dm
        
    # ========== SLOW PATH: Synthesis ==========
    else:
        phase_keys = np.array([_phase_key(p) for p in phase_rest])
        uniq_pairs, inv = np.unique(
            np.stack([filters, phase_keys], axis=1),
            axis=0, return_inverse=True
        )
        
        mag_abs_uniq = np.full(len(uniq_pairs), np.nan, float)
        for k, (filt_k, pkey) in enumerate(uniq_pairs):
            mag_abs_uniq[k] = synthesize_mag_at_z_cached(
                metric._mag_cache, sed_grid, float(pkey), z, filt_k
            )
        
        mag_abs = mag_abs_uniq[inv]
        mags = mag_abs + dm
    
    # ========== Common Post-Processing ==========
    # Extinction
    if ebv != 0.0:
        ax1 = dust_model_cached.ax1
        mags = mags + np.array([ax1.get(f, 0.0) for f in filters]) * ebv
    
    # SNR
    snr = _m52snr(mags, m5s)
    
    if not return_full_obs:
        return snr, filters, mjds, None
    
    # Build obs_record dict
    obs_record = {
        "filter": filters.tolist() if isinstance(filters, np.ndarray) else list(filters),
        "mjd_obs": mjds.tolist() if isinstance(mjds, np.ndarray) else list(mjds),
        "mag_obs": mags.tolist() if isinstance(mags, np.ndarray) else list(mags),
        "snr_obs": snr.tolist() if isinstance(snr, np.ndarray) else list(snr),
        "phase_rest": phase_rest.tolist() if isinstance(phase_rest, np.ndarray) else list(phase_rest),
    }
    
    return snr, filters, mjds, obs_record
# ------- 

def compute_slsn_properties(df_cov, templates_file):
    """
    Compute SLSN-specific observable properties from templates.
    """
    import pickle
    import numpy as np
    
    with open(templates_file, 'rb') as f:
        obj = pickle.load(f)
    
    lcs = obj['lightcurves']
    
    rows = []
    for i, tpl in enumerate(lcs):
        row = {'tpl_idx': i}
        
        # Get r-band curve
        if 'r' in tpl and isinstance(tpl['r'], dict):
            ph = np.asarray(tpl['r']['ph'], float)
            mag = np.asarray(tpl['r']['mag'], float)
            
            if len(ph) > 5:
                # Peak absolute magnitude
                peak_idx = np.argmin(mag)
                row['M_peak_r'] = float(mag[peak_idx])
                
                # Decline rate: 15-50 days post-peak
                post = (ph > 0) & (ph >= 15) & (ph <= 50)
                if post.sum() >= 3:
                    from scipy.stats import linregress
                    slope, _, _, _, _ = linregress(ph[post], mag[post])
                    row['decline_rate_r'] = float(slope)
                
                # Width at M_peak + 1 mag
                thresh = row['M_peak_r'] + 1.0
                above = mag < thresh
                if above.sum() >= 2:
                    row['width_r'] = float(ph[above].max() - ph[above].min())
        
        # g-r color at peak
        if 'g' in tpl and 'r' in tpl:
            g_mag = np.asarray(tpl['g']['mag'], float)
            g_ph = np.asarray(tpl['g']['ph'], float)
            r_mag = np.asarray(tpl['r']['mag'], float)
            r_ph = np.asarray(tpl['r']['ph'], float)
            
            # Find mags closest to phase=0
            if len(g_ph) > 0 and len(r_ph) > 0:
                g_near_peak = g_mag[np.argmin(np.abs(g_ph))]
                r_near_peak = r_mag[np.argmin(np.abs(r_ph))]
                row['g_minus_r'] = float(g_near_peak - r_near_peak)
        
        rows.append(row)
    
    df_props = pd.DataFrame(rows)
    
    # Merge with coverage info
    return df_cov.merge(df_props, on='tpl_idx', how='left')

# --------

def characterize_template_coverage(templates_file, save_summary=True):
    """Fast summary of template characteristics."""
    import pickle
    import pandas as pd
    import numpy as np
    
    with open(templates_file, 'rb') as f:
        obj = pickle.load(f)
    
    lcs = obj['lightcurves']
    names = obj.get('names', [f'tpl_{i}' for i in range(len(lcs))])
    meta = obj.get('meta', {})  # Dict with keys: z, dm, t0_cat, t0_data
    
    # Extract meta arrays (NOT indexing meta directly)
    z_arr = meta.get('z', [])
    dm_arr = meta.get('dm', [])
    t0_cat_arr = meta.get('t0_cat', [])
    t0_data_arr = meta.get('t0_data', [])
    
    rows = []
    for i, (name, tpl) in enumerate(zip(names, lcs)):
        row = {
            'tpl_idx': i, 
            'name': name,
            'z': z_arr[i] if i < len(z_arr) else np.nan,
            'dm': dm_arr[i] if i < len(dm_arr) else np.nan,
            't0_cat': t0_cat_arr[i] if i < len(t0_cat_arr) else np.nan,
            't0_data': t0_data_arr[i] if i < len(t0_data_arr) else np.nan,
        }
        
        # Count bands
        bands = [b for b, d in tpl.items() 
                 if isinstance(d, dict) and 'ph' in d and 'mag' in d and len(d['ph']) > 0]
        row['n_bands'] = len(bands)
        row['bands'] = ','.join(sorted(bands))
        
        # Get reference band characteristics
        for check_band in ['r', 'g', 'i']:
            if check_band in tpl and isinstance(tpl[check_band], dict):
                ph = np.asarray(tpl[check_band].get('ph', []), float)
                mag = np.asarray(tpl[check_band].get('mag', []), float)
                if len(ph) > 0:
                    row['ref_band'] = check_band
                    row['phase_min'] = float(ph.min())
                    row['phase_max'] = float(ph.max())
                    row['phase_span'] = float(ph.max() - ph.min())
                    row['n_phases'] = len(ph)
                    row['mag_peak'] = float(mag[np.argmin(np.abs(ph))])
                    break
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_summary:
        summary_path = templates_file.parent / f"{templates_file.stem}_coverage_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"Saved to {summary_path}")
    
    return df

# -------

def assess_literature_coverage(df_cov):
    """
    Compare your template distributions to published SLSN samples.
    """
    import matplotlib.pyplot as plt
    
    # Known SLSN ranges from literature
    lit_ranges = {
        'M_peak_r': (-23.0, -19.5),      # Quimby+13
        'decline_rate_r': (0.005, 0.08), # mag/day, Nicholl+17
        'g_minus_r': (-0.3, 0.5),        # typical colors
        'width_r': (20, 80),             # days, Inserra+13
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    gaps = {}
    
    for ax, (param, (lit_min, lit_max)) in zip(axes, lit_ranges.items()):
        if param not in df_cov.columns:
            ax.text(0.5, 0.5, f'{param}\nNo Data', ha='center', va='center')
            gaps[param] = "no_data"
            continue
        
        vals = df_cov[param].dropna()
        if len(vals) == 0:
            gaps[param] = "no_data"
            continue
        
        # Plot histogram
        ax.hist(vals, bins=20, alpha=0.7, edgecolor='black', label='Your templates')
        
        # Literature range
        ax.axvline(lit_min, color='red', ls='--', lw=2, label='Lit range')
        ax.axvline(lit_max, color='red', ls='--', lw=2)
        ax.axvspan(lit_min, lit_max, alpha=0.2, color='red')
        
        # Check coverage
        obs_min, obs_max = vals.min(), vals.max()
        left_gap = max(0, lit_min - obs_min)
        right_gap = max(0, obs_max - lit_max)
        
        coverage_frac = np.sum((vals >= lit_min) & (vals <= lit_max)) / len(vals)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(f'{param}\nCoverage: {coverage_frac*100:.0f}% in lit range')
        ax.legend()
        
        # Assess gaps
        if coverage_frac < 0.5:
            gaps[param] = "poor_coverage"
        elif left_gap > 0.2 * (lit_max - lit_min):
            gaps[param] = "missing_faint_end"
        elif right_gap > 0.2 * (lit_max - lit_min):
            gaps[param] = "missing_bright_end"
        else:
            gaps[param] = "good"
    
    plt.tight_layout()
    plt.savefig("SLSN_template_coverage_vs_literature.png", dpi=150)
    plt.show()
    
    # Print summary
    print("\n=== Coverage Assessment ===")
    for param, status in gaps.items():
        print(f"{param:20s}: {status}")
    
    return gaps

# --------

def find_missing_archetypes(df_cov):
    """Identify specific SLSN types not well-represented."""
    missing = []
    
    # Check for fast vs slow evolvers
    if 'decline_rate_r' in df_cov.columns:
        vals = df_cov['decline_rate_r'].dropna()
        # CORRECTED: positive slope = declining (getting fainter)
        fast = vals > 0.05  # >0.05 mag/day decline
        slow = vals < 0.02  # <0.02 mag/day decline
        print(f"Fast evolvers (>0.05 mag/day): {fast.sum()}")
        print(f"Slow evolvers (<0.02 mag/day): {slow.sum()}")
        if fast.sum() < 10:
            missing.append("fast_declining")
        if slow.sum() < 10:
            missing.append("slow_declining")
    
    # Check for bright vs faint
    if 'M_peak_r' in df_cov.columns:
        vals = df_cov['M_peak_r'].dropna()
        bright = vals < -22  # More negative = brighter
        faint = vals > -20.5
        print(f"Very bright (M_r < -22): {bright.sum()}")
        print(f"Faint (M_r > -20.5): {faint.sum()}")
        if bright.sum() < 10:
            missing.append("super_luminous")
        if faint.sum() < 10:
            missing.append("faint_end")
    
    # Check color diversity
    if 'g_minus_r' in df_cov.columns:
        vals = df_cov['g_minus_r'].dropna()
        blue = vals < -0.1
        red = vals > 0.3
        print(f"Blue (g-r < -0.1): {blue.sum()}")
        print(f"Red (g-r > 0.3): {red.sum()}")
        if blue.sum() < 5:
            missing.append("very_blue")
        if red.sum() < 5:
            missing.append("red_events")
    
    if missing:
        print(f"\nMissing archetypes: {', '.join(missing)}")
        print("Consider: (1) adding more events to catalog, (2) restricting z_max to improve completeness")
    else:
        print("\n✓ Good coverage across major SLSN types")
    
    return missing

#---------------
# Rubin ready 
# -----------------

class SLSN_Base_Metric(BaseMetric):
    def __init__(self, metricName='BaseSLSNMetric', 
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', mjd0=60980.5,
                 lc_model=None, use_extinction=True, use_kcorrect=False,
                 k_correct_type=None, k_correct_arg=None,
                 **kwargs):
        
        if lc_model is None:
            raise ValueError("lc_model required")
        
        self._mag_cache = {} 
        self.lc_model = lc_model
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.use_extinction = use_extinction
        self.use_kcorrect = use_kcorrect
        self.k_correct_type = k_correct_type
        self.k_correct_arg = k_correct_arg
        
        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName, 
                         units='Detection Efficiency', **kwargs)
    
    def detect(self, filters, snr, times, obs_record):
        """SLSN-specific detection logic matching Base_Metric.detect signature."""
        mags = np.asarray(obs_record.get('mag_obs', []))
        return detect_slsn(filters, snr, times, mags, obs_record)

class SLSN_Detect_Metric(SLSN_Base_Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SLSN_Detect')
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        # Call evaluate with return_full_obs=True to get the 4-tuple
        snr, filters, times, obs_record = evaluate(
            self, dataSlice, slice_point, return_full_obs=True
        )
        
        if obs_record is None or len(snr) == 0:
            return self.badval
        
        # Use the detect method from base class
        detected = self.detect(filters, snr, times, obs_record)
        
        # Add metadata
        obs_record.update({
            'detected': bool(detected),
            'sid_duplicate': slice_point['sid'],
            'file_indx': slice_point['file_indx'],
            'z': slice_point['z'],
            'ra': slice_point['ra'],
            'dec': slice_point['dec'],
            'distance_Mpc': slice_point['distance'],
            'ebv': slice_point['ebv'],
            'peak_time': slice_point['peak_time'],
        })
        
        self.obs_records[slice_point['sid']] = obs_record
        return 1.0 if detected else 0.0


def generate_SLSN_PopSlicer(lc_model, t_start=1, t_end=3652,
                            z_min=0.1, z_max=2.0,
                            rate_density=1e-7,
                            gal_lat_cut=None,
                            seed=42,
                            save_to=None,
                            load_from=None):
    """
    Generate population of SLSNe with redshift-dependent distances.
    """
    
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        print(f"Loaded SLSN population from {load_from}")
        return slicer  # ← WAS MISSING THIS RETURN
    
    rng = np.random.default_rng(seed)
    
    # Sample from volumetric rate
    n_events = sample_rate_from_volume(
        rate_density=rate_density,
        t_start=t_start, t_end=t_end,
        z_min=z_min, z_max=z_max
    )
    
    print(f"Simulating {n_events} SLSNe (rate={rate_density:.1e} Mpc^-3 yr^-1)")
    
    # Uniform sky positions
    nside = 64
    ra, dec = inject_uniform_healpix(nside, n_events, seed=seed)
    
    # Sample redshifts uniformly in comoving volume
    z_min_cm = cosmo.comoving_distance(z_min).value
    z_max_cm = cosmo.comoving_distance(z_max).value
    d_cm = rng.uniform(z_min_cm, z_max_cm, n_events)
    
    # Convert back to redshift (vectorized)
    z_vals = _get_z_from_comoving_fast(d_cm)
    distances = d_cm  # comoving distance in Mpc
    
    # Random template assignment
    num_templates = len(lc_model.data)
    file_indx = rng.integers(0, num_templates, n_events)
    
    # Peak times uniformly across survey
    peak_times = rng.uniform(t_start, t_end, n_events)
    
    # Extinction
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    sfd = SFDQuery()
    ebv_vals = sfd(coords)
    
    # Build slicer
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    slicer.slice_points['sid'] = np.arange(n_events)  # ← CRITICAL: ADD THIS
    slicer.slice_points['z'] = z_vals
    slicer.slice_points['distance'] = distances
    slicer.slice_points['distance_modulus'] = np.array([dm_from_z(z) for z in z_vals])
    slicer.slice_points['peak_time'] = peak_times
    slicer.slice_points['file_indx'] = file_indx
    slicer.slice_points['ebv'] = ebv_vals
    
    # Store per-filter extinction
    ax1 = dust_model.ax1
    for f in ['u','g','r','i','z','y']:
        slicer.slice_points[f'A_{f}'] = ax1[f] * ebv_vals
    
    # ← ADD: Store injected peak magnitudes for run_detect
    for f in ['u','g','r','i','z','y']:
        peak_abs = []
        peak_noebv = []
        peak_ebv = []
        
        for idx, z_val, dm, ebv in zip(file_indx, z_vals, 
                                        slicer.slice_points['distance_modulus'], 
                                        ebv_vals):
            # Get absolute peak from template
            if f in lc_model.data[idx] and len(lc_model.data[idx][f]['mag']) > 0:
                M_abs = float(np.min(lc_model.data[idx][f]['mag']))
            else:
                M_abs = np.nan
            
            m_noebv = M_abs + dm
            m_ebv = m_noebv + ax1[f] * ebv
            
            peak_abs.append(M_abs)
            peak_noebv.append(m_noebv)
            peak_ebv.append(m_ebv)
        
        slicer.slice_points[f'peak_mag_abs_{f}'] = np.array(peak_abs)
        slicer.slice_points[f'peak_app_mag_noebv_{f}'] = np.array(peak_noebv)
        slicer.slice_points[f'peak_app_mag_ebv_{f}'] = np.array(peak_ebv)
    
    if save_to:
        atomic_save_pickle(dict(slicer.slice_points), save_to)
        print(f"Saved SLSN population to {save_to}")
    
    return slicer

def detect_slsn(filters, snr, times, mags, obs_record):
    """
    SLSN detection criterion following Firth+2015.
    
    Returns True if:
    1. ≥2 filters with SNR≥5 detections
    2. Rising light curve observed (at least one pair showing brightening)
    3. Observations span ≥15 days
    """
    
    # Criterion 1: Multi-band detection
    detected_filters = []
    for f in np.unique(filters):
        mask = (filters == f) & (snr >= 5)
        if np.sum(mask) >= 1:
            detected_filters.append(f)
    
    if len(detected_filters) < 2:
        return False
    
    # Criterion 2: Observe rising light curve
    # SLSNe rise for ~20-40 days, so we need to catch the rise
    rising_observed = False
    for f in detected_filters:
        mask = (filters == f) & (snr >= 5)
        t_filt = times[mask]
        m_filt = mags[mask]
        
        if len(t_filt) >= 2:
            # Sort by time
            order = np.argsort(t_filt)
            t_sorted = t_filt[order]
            m_sorted = m_filt[order]
            
            # Check for consecutive points showing rise (mag decreasing)
            for i in range(len(m_sorted) - 1):
                dt = t_sorted[i+1] - t_sorted[i]
                dm = m_sorted[i+1] - m_sorted[i]
                
                # Rising: dm < -0.1 mag over dt > 0.5 days
                if dm < -0.1 and dt > 0.5 and dt < 30:
                    rising_observed = True
                    break
        
        if rising_observed:
            break
    
    if not rising_observed:
        return False
    
    # Criterion 3: Temporal baseline
    detected_mask = snr >= 5
    if np.sum(detected_mask) < 2:
        return False
    
    duration = np.ptp(times[detected_mask])
    if duration < 15:  # days
        return False
    
    return True

class SLSN_CharacterizeMetric(SLSN_Base_Metric):
    """
    Characterization metric following Inserra+2024 catalog criteria.
    
    An SLSN is 'characterized' if:
    1. It's detected (bronze criterion)
    2. ≥5 epochs with SNR≥5 across all filters
    3. ≥3 different filters with detections
    4. ≥2 epochs within ±10 days of peak
    5. Observations extend to +30 days post-peak (for decline rate)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = 'SLSN_Characterize'
        self.obs_records = {}
    
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate(
            self, dataSlice, slice_point, return_full_obs=False
        )
        
        if obs_record is None:
            return 0.0
        
        # Must pass detection first
        detected = detect_slsn(filters, snr, times, 
                              obs_record['mag_obs'], obs_record)
        if not detected:
            return 0.0
        
        # Characterization criteria
        good = snr >= 5
        
        # 1. Sufficient epochs
        n_epochs = np.sum(good)
        if n_epochs < 5:
            return 0.0
        
        # 2. Multi-band coverage
        n_filters = len(np.unique(filters[good]))
        if n_filters < 3:
            return 0.0
        
        # 3. Coverage near peak (peak_time from slice_point)
        peak_mjd = self.mjd0 + slice_point['peak_time']
        near_peak = good & (np.abs(obs_record['mjd_obs'] - peak_mjd) <= 10)
        if np.sum(near_peak) < 2:
            return 0.0
        
        # 4. Post-peak coverage for decline rate
        post_peak = good & (obs_record['mjd_obs'] > peak_mjd + 30)
        if np.sum(post_peak) < 1:
            return 0.0
        
        # Store metadata
        obs_record['characterized'] = True
        obs_record['n_epochs'] = n_epochs
        obs_record['n_filters_char'] = n_filters
        self.obs_records[slice_point['sid']] = obs_record
        
        return 1.0




class SLSN_SpecTriggerMetric(SLSN_Base_Metric):
    def run(self, dataSlice, slice_point=None):
        out = evaluate(self, dataSlice, slice_point, return_full_obs=False)
        if out is None or len(out.get("rows", [])) == 0:
            return 0.0

        filters = out["filter"]
        snr     = out["snr"]
        times   = out["mjd_obs"]
        mags    = out["mag_obs"]

        detected = detect_slsn(filters, snr, times, mags, out)
        if not detected:
            return 0.0

        # Near-peak epochs
        peak_mjd = self.mjd0 + slice_point['peak_time']
        near_peak = (snr >= 5) & (np.abs(times - peak_mjd) <= 5)
        if not np.any(near_peak):
            return 0.0

        # Brightness requirement
        if np.min(mags[near_peak]) > 21.0:
            return 0.0

        # Color check if both bands exist near-peak
        has_g = np.any(near_peak & (filters == 'g'))
        has_r = np.any(near_peak & (filters == 'r'))
        if has_g and has_r:
            g_mag = np.min(mags[near_peak & (filters == 'g')])
            r_mag = np.min(mags[near_peak & (filters == 'r')])
            if (g_mag - r_mag) > 0.3:
                return 0.0

        return 1.0



Detect_Metric = SLSN_Detect_Metric