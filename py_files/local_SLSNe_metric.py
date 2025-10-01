# local_SLSNe_metric
# Survey-ready SLSN templates from catalog photometry via 2D GP (time, wavelength).
from __future__ import annotations
import re
from rubin_sim.maf.metrics import BaseMetric

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


DEBUG = False

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Constants & filter mappings
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


user_filter_map = {
    "b": 4332.70,
    "c": 5627.80,
    "f475w": 4708.87,
    "f625w": 6266.20,
    "f775w": 7652.44,
    "f850lp": 9004.99,
    "g": 4671.78,
    "h": 16230.17,
    "i": 7682.36,
    "j": 12317.97,
    "k": 21682.12,
    "ks": 21454.68,
    "r": 6141.12,
    "rs": 6734.34,
    "u": 3608.04,
    "uvm2": 2245.03,
    "uvw1": 2681.67,
    "uvw2": 2083.95,
    "v": 3878.68,
    "w1": 33526.00,
    "w2": 46028.00,
    "y": 9613.60,
    "cyan": 5182.42,
    "orange": 6629.82,
    "w": 5980.70,
    "z": 8906.54,}

# Speed of light (m/s)
_C_MS = 2.99792458e8

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
                # NEW: names passed directly (may be None)
                self.names  = names if names is not None else [f"tpl_{i}" for i in range(len(self.data))]
            elif load_from:
                if not os.path.exists(load_from):
                    raise FileNotFoundError(f"SLSN templates not found: {load_from}")
                with open(load_from, "rb") as f:
                    obj = pickle.load(f)
                if "lightcurves" not in obj:
                    raise ValueError("templates.pkl missing key 'lightcurves'")
                self.data   = obj["lightcurves"]
                self.t_grid = obj.get("t_grid", None)
                # NEW: load names from pickle, fallback if missing
                self.names  = obj.get("names", [f"tpl_{i}" for i in range(len(self.data))])
            else:
                self.data, self.t_grid = [], None
                self.names = []
    
            # discover bands (unchanged)
            self.filts = []
            for tpl in (self.data or []):
                if isinstance(tpl, dict) and tpl:
                    self.filts = sorted(tpl.keys())
                    break

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
        sed_grid = []   # <--- new
        templates: list[dict] = []
        names: list[str] = []
        meta: list[dict] = []    # <--- NEW
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
            t0_cat_list.append(float(row[peak_col]) if (peak_col is not None and np.isfinite(row[peak_col])) else np.nan)
            t0_data_list.append(float(t0))
            z_used.append(float(z))
            dm_list.append(float(dm_from_z(z)))


            if saved_t_grid is None:
                saved_t_grid = phase.tolist()

        # Build once, save (with names), and return the same instance
        model = cls(lightcurves=templates, t_grid=saved_t_grid, names=names)
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


        return model


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

def synthesize_mag_at_z(templates_file: Path, template_idx: int,
                        filt_lam_obs_A: np.ndarray, filt_T: np.ndarray,
                        z_target: float, phase_rest: np.ndarray,
                        extinction_Av: float | None = None,
                        Rv: float = 3.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (phase_rest, m_AB) for a chosen filter at a target redshift using the stored sed_grid.
    filt_lam_obs_A, filt_T: observed-frame filter curve (Å, unitless T).
    """
    import pickle
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    sed = obj["sed_grid"][template_idx]
    phase0 = np.asarray(sed["phase"], float)                     # rest-frame
    lam_rest = np.asarray(sed["lam_rest_A"], float)              # rest-frame Å
    Fnu_abs = np.asarray(sed["Fnu_abs"], float)                  # shape (n_phase, n_lambda)

    # Shift rest SED to observed frame at z_target: lam_obs = (1+z_tar)*lam_rest
    lam_obs = (1.0 + z_target) * lam_rest[None, :]               # broadcast
    # Interpolate SED (per phase) onto the filter λ grid
    lam_f = np.asarray(filt_lam_obs_A, float)
    T_f   = np.asarray(filt_T, float)
    T_norm = _trapz_norm(lam_f, T_f)

    # Convert absolute (10pc) Fν_rest to observed apparent Fν at z_target:
    #   Fν_obs = Fν_abs / ((DL(z_tar)/10pc)^2 * (1+z_tar))
    DL_tar_pc = cosmo.luminosity_distance(z_target).to_value(u.pc)
    inv_scale = 1.0 / ((DL_tar_pc/10.0)**2 * (1.0 + z_target))
    # Sample per phase onto filter grid and integrate
    from numpy import interp, trapz, log10
    m_AB = np.full(phase_rest.shape, np.nan, float)

    # Interp along phase axis (so caller can pick arbitrary phase_rest)
    # 1) Interp absolute SED onto requested phases (bilinear in phase, nearest in lambda via 1D interp per phase)
    # First, for each requested phase, get Fnu_abs(phase, lam_rest)
    # linear interp along phase:
    def interp_phase(F2D, x_old, x_new):
        return np.vstack([np.interp(x_new, x_old, F2D[:, j], left=np.nan, right=np.nan)
                          for j in range(F2D.shape[1])]).T  # -> (len(x_new), n_lambda)

    F_abs_at_phase = interp_phase(Fnu_abs, phase0, np.asarray(phase_rest, float))  # (nP, nL)
    # 2) For each phase, convert to observed apparent Fν at z_target and integrate over filter
    for i in range(F_abs_at_phase.shape[0]):
        F_abs_row = F_abs_at_phase[i, :]                        # vs lam_rest
        lam_obs_row = (1.0 + z_target) * lam_rest               # 1D, Å
        # Interp onto filter λ grid in observed frame
        F_abs_on_filt = np.interp(lam_f, lam_obs_row, F_abs_row, left=np.nan, right=np.nan)
        Fnu_obs_on_f  = inv_scale * F_abs_on_filt               # apparent at Earth
        good = np.isfinite(Fnu_obs_on_f) & np.isfinite(T_f) & (T_f > 0)
        if good.sum() < 2:
            continue
        # AB magnitude via Fν-weighted throughput. For simplicity, treat T as effective in Fν-space:
        num = trapz(Fnu_obs_on_f[good] * T_f[good], lam_f[good])
        den = T_norm if T_norm > 0 else 1.0
        Fnu_eff = num / den
        if Fnu_eff > 0:
            m_AB[i] = -2.5 * log10(Fnu_eff / F0_JY)

    # Optional: extinction in observed frame (very simple Aλ≈ const per band):
    # If you want, you can apply a scalar A_band here.
    return phase_rest, m_AB

