"""
export_slsne_photometry.py — Read, clean, and export per-event photometry.

Functions for processing raw SLSN catalogs, attaching cenwave, and exporting
to clean CSV/Parquet format.
"""
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# =============================================================================
# Filter canonicalization
# =============================================================================

PREFIXES = ("swift_", "ps1_", "panstarrs_", "lsst_", "ztf_")
SUFFIXES = ("-AB", "-Vega")

def canonical_filter(s: str) -> str:
    """
    Strip common prefixes/suffixes from filter names, preserving case.
    Example: 'swift_UVW1' -> 'UVW1', 'decam_g-AB' -> 'g'
    """
    s = str(s).strip()
    for p in PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
    for suf in SUFFIXES:
        if s.endswith(suf):
            s = s[:-len(suf)]
    return s

# =============================================================================
# Header cleaning and file reading
# =============================================================================

def _clean_header(cols):
    """Normalize column names: strip BOM, collapse whitespace, standardize labels."""
    cleaned = []
    for c in cols:
        if not isinstance(c, str):
            c = str(c)
        c = unicodedata.normalize("NFKC", c)
        c = c.replace("\ufeff", "")  # BOM
        c = c.strip()
        c = re.sub(r"\s+", " ", c)
        c = c.replace("Mag Err", "MagErr")
        cleaned.append(c)
    return cleaned

def read_supernova_table_txt(path: Path) -> pd.DataFrame:
    """
    Tolerant reader for per-object .txt files with whitespace separation.
    Uses regex separator and cleans headers.
    """
    df = pd.read_csv(
        path, sep=r"\s+", engine="python", comment="#", 
        dtype=str, skip_blank_lines=True
    )
    df.columns = _clean_header(df.columns)
    return df

def coerce_bool(series: pd.Series) -> pd.Series:
    """Convert string series to boolean (1/true/yes → True)."""
    s = series.astype(str).str.strip()
    return s.isin(["1", "true", "t", "yes", "y"])

# =============================================================================
# Export to standard 5-column format
# =============================================================================

def to_export(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert per-object table to standard format:
      mjd, mag, mag_err, filter, detected, UL, System
    
    UL=True → detected=0, mag_err=inf
    MagErr=-1 → kept as NaN for detections
    """
    need = ["MJD", "Mag", "MagErr", "Filter", "UL"]
    cols = {c: c for c in df_raw.columns}
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {list(df_raw.columns)}")

    df = df_raw.copy()
    mjd = pd.to_numeric(df["MJD"], errors="coerce")
    mag = pd.to_numeric(df["Mag"], errors="coerce")
    mag_err = pd.to_numeric(df["MagErr"], errors="coerce").astype(float)
    mag_err[mag_err < 0] = np.nan

    ul = coerce_bool(df["UL"])
    detected = (~ul).astype(np.int8)
    System = df["System"].astype("string").map(lambda s: s.strip() if isinstance(s, str) else s)
    filt = df["Filter"].astype("string").map(lambda s: s.strip() if isinstance(s, str) else s)

    out = pd.DataFrame({
        "mjd": mjd.astype(float),
        "mag": mag.astype(float),
        "mag_err": mag_err,
        "UL": ul,
        "filter": filt,
        "System": System,
        "detected": detected
    }).sort_values("mjd").reset_index(drop=True)

    keep = out[["mjd", "mag", "mag_err", "UL", "filter"]].notna().any(axis=1)
    out = out[keep].reset_index(drop=True)

    out = out.astype({
        "mjd": "float64", "mag": "float64", "mag_err": "float64",
        "UL": "boolean", "filter": "string", "System": "string", "detected": "int8"
    })
    return out

# =============================================================================
# Batch processing
# =============================================================================

def process_one(event_dir: Path, event_name: str, out_perevent: Path, 
                *, write_parquet: bool = True, write_csv: bool = False):
    """Process a single event and export to CSV/Parquet."""
    infile = event_dir / f"{event_name}.txt"
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

def process_all_events(supernovae_dir: Path, out_perevent: Path, out_allevent: Path,
                       *, include_ul: bool = True, make_combined: bool = True,
                       write_parquet: bool = True, write_csv: bool = True):
    """Process all events in a directory and optionally create combined dataset."""
    rows = []
    combined_frames = [] if make_combined else None
    
    event_dirs = sorted(p for p in supernovae_dir.iterdir() if p.is_dir())
    for d in tqdm(event_dirs, desc="Exporting SLSNe"):
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
            
            if write_csv:
                df_out.to_csv(out_perevent / f"{event}.csv", index=False)
            if write_parquet:
                df_out.to_parquet(out_perevent / f"{event}.parquet", 
                                  index=False, engine="pyarrow", 
                                  compression="zstd", compression_level=7)
            
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
    
    index_df = pd.DataFrame(rows).sort_values(["status", "event"]).reset_index(drop=True)
    index_df.to_csv(out_perevent / "_index.csv", index=False)
    
    combined_csv = combined_parq = None
    if make_combined and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        if write_csv:
            combined_csv = out_allevent / "all_objects.csv"
            combined.to_csv(combined_csv, index=False)
        if write_parquet:
            combined_parq = out_allevent / "all_objects.parquet"
            combined.to_parquet(combined_parq, index=False, engine="pyarrow",
                                compression="zstd", compression_level=7)
    
    return index_df, combined_csv, combined_parq

# =============================================================================
# Cenwave calculation and attachment
# =============================================================================

def cenwave_mode(vals, rel_bin=0.02):
    """
    Return median of dominant bin cluster (mode).
    rel_bin=0.02 → 2% bins around central wavelength scale.
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
    keys = np.round(vals / width).astype(int)
    cnt = Counter(keys)
    k_mode, _ = cnt.most_common(1)[0]
    cluster = vals[keys == k_mode]
    return float(np.median(cluster))

def _read_table(path: Path) -> pd.DataFrame:
    """Helper to read model/rest tables."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, comment="#", sep=r"\s+", engine="python", header=0)

def per_filter_cenwave(event_root: Path, name: str, rel_bin=0.02, 
                       verbose: bool = True) -> dict[str, float]:
    """
    Build {Filter_canonical: Cenwave[Å]} using mode clustering.
    Precedence: *_model.txt overrides/fills before *_rest.txt.
    """
    d = Path(event_root) / name
    mod = _read_table(d / f"{name}_model.txt")
    rst = _read_table(d / f"{name}_rest.txt")
    
    cen_map: dict[str, float] = {}
    rows_seen = {"model": 0, "rest": 0}
    groups_seen = {"model": 0, "rest": 0}
    groups_kept = {"model": 0, "rest": 0}
    groups_skipped = {"model": 0, "rest": 0}
    overlaps = []
    
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
    
    _add_from(mod, "model")
    _add_from(rst, "rest")
    
    if verbose:
        n_model = groups_kept["model"]
        n_rest = groups_kept["rest"]
        total = n_model + n_rest
        msg = (f"[cenwave] {name}: entries kept -> model={n_model}, rest={n_rest}, "
               f"total={total} | rows seen -> model={rows_seen['model']}, "
               f"rest={rows_seen['rest']}")
        print(msg)
        if overlaps:
            sample = ", ".join(sorted(set(overlaps))[:8])
            print(f"[cenwave] {name}: model/rest overlap (model wins): {sample}"
                  f"{' ...' if len(set(overlaps)) > 8 else ''}")
    
    return cen_map

def attach_cenwave_to_perevent_csv(csv_path: Path, cen_map: dict[str, float]) -> Path:
    """Attach Cenwave column to per-event CSV using canonical filter keys."""
    df = pd.read_csv(csv_path)
    if "Filter" not in df.columns:
        cols = {c.lower(): c for c in df.columns}
        if "filter" in cols:
            df.rename(columns={cols["filter"]: "Filter"}, inplace=True)
        else:
            raise ValueError(f"No 'Filter' column in {csv_path.name}")
    
    df["Filter_can"] = df["Filter"].astype(str).str.strip().map(canonical_filter)
    df["Cenwave"] = df["Filter_can"].map(cen_map).astype(float)
    
    n_ok = int(np.isfinite(df["Cenwave"]).sum())
    if n_ok == 0:
        missing_counts = df["Filter_can"].value_counts().head(8)
        print(f"[attach][warn] 0 matches in {csv_path.name}. Top missing keys:\n{missing_counts}")
    
    out = csv_path.with_name(csv_path.stem + "_cenwave" + csv_path.suffix)
    df.to_csv(out, index=False)
    print(f"[attach] wrote {out.name} with Cenwave for {n_ok}/{len(df)} rows")
    return out

# =============================================================================
# Audit functions
# =============================================================================

def audit_cenwave_mode(event_root: Path, name: str, rel_bin=0.02,
                       outlier_frac_thresh=0.25, widen_factor=2.0, 
                       mad_floor_factor=1.0):
    """Audit cenwave consistency using mode-based clustering."""
    d = Path(event_root) / name
    mod = _read_table(d / f"{name}_model.txt")
    rst = _read_table(d / f"{name}_rest.txt")
    
    rows = []
    for src_name, df in [("model", mod), ("rest", rst)]:
        if df.empty or not {"MJD", "Cenwave", "Filter"}.issubset(df.columns):
            continue
        sub = df[np.isfinite(df["Cenwave"])].copy()
        sub["Filter_can"] = sub["Filter"].astype(str).str.strip().map(canonical_filter)
        
        for lab, g in sub.groupby("Filter_can"):
            lam = g["Cenwave"].to_numpy(float)
            n = lam.size
            if n == 0:
                rows.append([name, src_name, lab, 0, np.nan, np.nan, False])
                continue
            center = cenwave_mode(lam, rel_bin=rel_bin)
            if not np.isfinite(center) or center <= 0:
                rows.append([name, src_name, lab, n, np.nan, np.nan, True])
                continue
            bin_width = rel_bin * center
            mad = np.median(np.abs(lam - np.median(lam))) if n >= 3 else 0.0
            mad_sigma = 1.4826 * mad
            tol = max(widen_factor * bin_width, mad_floor_factor * mad_sigma)
            frac_outside = float(np.mean(np.abs(lam - center) > tol))
            flag = frac_outside > outlier_frac_thresh
            rows.append([name, src_name, lab, n, center, frac_outside, flag])
    
    return pd.DataFrame(rows, columns=[
        "event", "source", "filter", "n_rows", "mode_center_A", 
        "frac_outside", "flag_outliers"
    ])

def audit_cenwave_variability(event_dir: Path, name: str,
                               rel_scatter_thresh=0.02, 
                               slope_thresh_A_per_day=5.0,
                               r2_thresh=0.2):
    """Check for time-variable Cenwave (flags problematic filters)."""
    d = Path(event_dir) / name
    mod = _read_table(d / f"{name}_model.txt")
    rst = _read_table(d / f"{name}_rest.txt")
    
    def _norm(df):
        if df.empty:
            return df
        need = {"MJD", "Cenwave", "Filter"}
        if not need.issubset(df.columns):
            return pd.DataFrame()
        df = df.copy()
        df = df[np.isfinite(df["MJD"]) & np.isfinite(df["Cenwave"])].copy()
        df["Filter_can"] = df["Filter"].astype(str).str.strip().map(canonical_filter)
        return df
    
    mod, rst = _norm(mod), _norm(rst)
    src_frames = []
    if not mod.empty:
        m = mod[["MJD", "Cenwave", "Filter_can"]].copy()
        m["src"] = "model"
        src_frames.append(m)
    if not rst.empty:
        r = rst[["MJD", "Cenwave", "Filter_can"]].copy()
        r["src"] = "rest"
        src_frames.append(r)
    
    if not src_frames:
        return {"event": name, "status": "no_data"}, pd.DataFrame()
    
    allf = pd.concat(src_frames, ignore_index=True)
    rows = []
    
    for lab, g in allf.groupby("Filter_can"):
        if len(g) < 3:
            rows.append((lab, len(g), np.nan, np.nan, np.nan, np.nan, np.nan, 
                         False, False, ""))
            continue
        
        lam = g["Cenwave"].to_numpy(float)
        mjd = g["MJD"].to_numpy(float)
        med = np.nanmedian(lam)
        mad = np.nanmedian(np.abs(lam - med))
        rel_scatter = mad / med if med > 0 else np.nan
        flag_scatter = np.isfinite(rel_scatter) and (rel_scatter > rel_scatter_thresh)
        
        t0 = np.nanmedian(mjd)
        t = mjd - t0
        A = np.vstack([np.ones_like(t), t]).T
        try:
            coef, *_ = np.linalg.lstsq(A, lam, rcond=None)
            a, b = coef
            lam_hat = a + b * t
            ss_res = np.nansum((lam - lam_hat) ** 2)
            ss_tot = np.nansum((lam - np.nanmean(lam)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except Exception:
            b, r2 = np.nan, np.nan
        
        flag_trend = ((np.isfinite(b) and np.abs(b) > slope_thresh_A_per_day) and 
                      (np.isfinite(r2) and r2 > r2_thresh))
        
        note = ""
        if flag_scatter:
            note += "scatter>thr; "
        if flag_trend:
            note += "trend>thr; "
        
        rows.append((lab, len(g), med, mad, rel_scatter, b, r2, 
                     flag_scatter, flag_trend, note.strip()))
    
    diag = pd.DataFrame(rows, columns=[
        "filter", "n_rows", "cenwave_med_A", "cenwave_mad_A", "rel_scatter",
        "slope_A_per_day", "r2", "flag_scatter", "flag_trend", "note"
    ])
    
    summary = {
        "event": name,
        "any_time_variation": bool(diag["flag_trend"].fillna(False).any() or 
                                    diag["flag_scatter"].fillna(False).any()),
        "filters_flagged": diag.loc[diag["flag_trend"] | diag["flag_scatter"], 
                                     "filter"].tolist(),
    }
    return summary, diag

# =============================================================================
# Parameter table loader
# =============================================================================

def load_allparams_robust(path: Path) -> pd.DataFrame:
    """Robust loader for all_parameters.txt with flexible delimiters."""
    df = pd.read_csv(
        path, sep=r"\s{2,}|\t+", engine="python", header=0,
        comment="#", dtype=str, skip_blank_lines=True, on_bad_lines="warn"
    )
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
