#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# ---------- Helpers ----------

def parse_redshift_from_name(name: str) -> Optional[float]:
    """
    Try to parse a redshift from a filename. Accepts patterns like:
      *_z0.51*, *_z0p51*, *z=0.51*, *(z0.51)*
    Returns float or None.
    """
    # common token replacements
    s = name.replace('p', '.')
    # regex variants
    patterns = [
        r'[_\-]z([0-9]+\.?[0-9]*)',
        r'z=([0-9]+\.?[0-9]*)',
        r'\(z([0-9]+\.?[0-9]*)\)',
        r'[_\-]z([0-9]+)p([0-9]+)',
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            if len(m.groups()) == 2:
                # handle z12p34 -> 12.34
                return float(f"{m.group(1)}.{m.group(2)}")
            try:
                return float(m.group(1))
            except Exception:
                continue
    return None

def distance_modulus_from_z(z: float) -> float:
    """
    Compute distance modulus mu = 5 log10(d_L/10pc) using Planck18.
    """
    dl = cosmo.luminosity_distance(z)  # in Mpc
    mu = 5 * np.log10(dl.to(u.pc).value / 10.0)
    return float(mu)

def k_correction_powerlaw(z: float, beta: float) -> float:
    """
    Very simple approximate K-correction for a power-law f_nu ~ nu^(-beta).
    For AB magnitudes, K ~ -2.5*(1 - beta) * log10(1+z).
    If beta is None, return 0.0.
    Reference: Hogg et al. (2002), simplified same-filter approximation.
    """
    if beta is None:
        return 0.0
    return -2.5 * (1.0 - beta) * np.log10(1.0 + z)

def infer_object_id(df: pd.DataFrame, file_stem: str, id_col: Optional[str]) -> str:
    if id_col and id_col in df.columns and pd.notnull(df[id_col]).any():
        val = df[id_col].dropna().astype(str).iloc[0]
        return str(val)
    return file_stem

def maybe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# ---------- Core ----------

def process_file(
    path: Path,
    meta_z: Optional[float],
    args,
) -> Optional[pd.DataFrame]:
    # Read CSV/TSV or try whitespace-delimited
    sep = ','
    if path.suffix.lower() in ['.tsv', '.tab']:
        sep = '\t'
    try:
        df = pd.read_csv(path, sep=sep, comment='#')
    except Exception:
        try:
            df = pd.read_csv(path, delim_whitespace=True, comment='#')
        except Exception as e:
            print(f"[WARN] Could not read {path}: {e}", file=sys.stderr)
            return None

    # Required columns
    for col in [args.time_col, args.mag_col]:
        if col not in df.columns:
            print(f"[WARN] Missing required column '{col}' in {path}", file=sys.stderr)
            return None

    # Infer id
    obj_id = infer_object_id(df, path.stem, args.id_col)

    # Determine z: priority meta -> filename -> column
    z = None
    if meta_z is not None and np.isfinite(meta_z):
        z = float(meta_z)
    if z is None:
        z_guess = parse_redshift_from_name(path.name)
        if z_guess is not None:
            z = z_guess
    if z is None and args.z_col and args.z_col in df.columns:
        z_vals = pd.to_numeric(df[args.z_col], errors='coerce').dropna()
        if not z_vals.empty:
            z = float(z_vals.iloc[0])

    if z is None or not np.isfinite(z) or z <= 0:
        print(f"[WARN] No valid redshift for {path}. Skipping.", file=sys.stderr)
        return None

    # Distance modulus
    try:
        mu = distance_modulus_from_z(z)
    except Exception as e:
        print(f"[WARN] Could not compute distance modulus for z={z} in {path}: {e}", file=sys.stderr)
        return None

    # Optional: extinction columns
    # Supports either a single per-row E(B-V) column or a scalar from metadata (not implemented here).
    ebv_col = args.ebv_col if args.ebv_col in df.columns else None
    Rv = args.rv if args.rv is not None else 3.1
    # Very simple A_lambda ~ Rv * E(B-V); if you have band-specific A_lambda, provide it in the input instead.
    # (Users can refine to A_lambda = R_lambda * E(B-V) with proper CCM/Fitzpatrick curves by band.)
    if ebv_col:
        A_lambda = Rv * pd.to_numeric(df[ebv_col], errors='coerce').fillna(0.0)
    else:
        A_lambda = 0.0

    # K-correction (optional simple power-law)
    beta = None if (args.kbeta is None or str(args.kbeta).lower() == 'none') else float(args.kbeta)
    K = k_correction_powerlaw(z, beta)

    # Compute absolute magnitude: M = m - mu - K - A_lambda
    mags = pd.to_numeric(df[args.mag_col], errors='coerce')
    M = mags - mu - K - A_lambda

    out = pd.DataFrame({
        'object_id': obj_id,
        'filename': path.name,
        'z': z,
        'mu': mu,
        'K_pl_beta': np.nan if beta is None else beta,
        'A_lambda': A_lambda if np.isscalar(A_lambda) else A_lambda.values,
        'time': df[args.time_col],
        'mag_app': mags,
        'mag_abs': M
    })
    # carry optional columns if present
    for c in [args.magerr_col, args.filt_col]:
        if c and c in df.columns:
            out[c] = df[c]

    return out

def load_metadata(meta_csv: Optional[str], id_col: Optional[str], z_col: Optional[str]) -> Dict[str, float]:
    if not meta_csv:
        return {}
    meta = pd.read_csv(meta_csv)
    if id_col not in meta.columns or z_col not in meta.columns:
        raise ValueError(f"Metadata CSV must contain columns '{id_col}' and '{z_col}'")
    m = {}
    for _, row in meta.iterrows():
        oid = str(row[id_col])
        z = maybe_float(row[z_col])
        if np.isfinite(z) and z > 0:
            m[oid] = float(z)
    return m

def main():
    p = argparse.ArgumentParser(description="Convert apparent to absolute magnitudes for SLSN light curves using redshifts.")
    p.add_argument('--lc_dir', required=True, help='Directory containing light-curve files (CSV/TSV/whitespace).')
    p.add_argument('--meta_csv', default=None, help='Optional metadata CSV mapping object_id -> redshift.')
    p.add_argument('--id_col', default='object_id', help='Column name for object id in metadata and/or LC files.')
    p.add_argument('--z_col', default='redshift', help='Column name for redshift if present in LC or metadata.')
    p.add_argument('--time_col', default='mjd', help='Time column name in LC files.')
    p.add_argument('--mag_col', default='mag', help='Apparent magnitude column name in LC files.')
    p.add_argument('--magerr_col', default=None, help='Optional magnitude error column name in LC files.')
    p.add_argument('--filt_col', default=None, help='Optional filter/band column name in LC files.')
    p.add_argument('--ebv_col', default='ebv', help='Optional E(B-V) column for MW extinction correction (per row).')
    p.add_argument('--rv', default=3.1, type=float, help='R_V to convert E(B-V) to A_lambda if ebv_col present.')
    p.add_argument('--kbeta', default=None, help='Power-law spectral index beta for simple K-correction; use "None" to disable.')
    p.add_argument('--out_dir', required=True, help='Output directory.')
    args = p.parse_args()

    lc_dir = Path(args.lc_dir)
    out_dir = Path(args.out_dir)
    obj_dir = out_dir / 'objects'
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)

    meta_map = load_metadata(args.meta_csv, args.id_col, args.z_col) if args.meta_csv else {}

    rows = []
    for pth in sorted(lc_dir.glob('*')):
        if not pth.is_file():
            continue

        # map redshift via metadata by object_id if possible
        meta_z = None
        # attempt to look up by file stem
        stem = pth.stem
        if stem in meta_map:
            meta_z = meta_map[stem]

        df_abs = process_file(pth, meta_z, args)
        if df_abs is None:
            continue
        rows.append(df_abs)

        # write per-object file
        oid = df_abs['object_id'].iloc[0]
        df_abs.to_csv(obj_dir / f"{oid}.csv", index=False)

    if not rows:
        print("[INFO] No valid light curves processed.", file=sys.stderr)
        sys.exit(1)

    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_dir / 'slsn_absolute_magnitudes.csv', index=False)
    print(f"[OK] Wrote {len(all_df)} rows to {out_dir / 'slsn_absolute_magnitudes.csv'}")
    print(f"[OK] Per-object CSVs in {obj_dir}")

if __name__ == '__main__':
    main()
