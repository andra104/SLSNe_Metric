
#####

# Cell 1 — setup
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow
import re
import unicodedata
from itertools import islice
from slsne.lcurve import get_all_lcs
from tqdm.auto import tqdm

# at top of export_slsne_photometry.py, after imports
try:
    from IPython.display import display  # for notebooks
except Exception:
    def display(x):
        print(x)




# simple filter normalization
FILTER_MAP = {
    "ztfg":"g","ztfr":"r","ztfi":"i",
    "sdssu":"u","sdssg":"g","sdssr":"r","sdssi":"i","sdssz":"z",
    "ps1_g":"g","ps1_r":"r","ps1_i":"i","ps1_z":"z","ps1_y":"y",
    # Johnson-Cousins -> coarse mapping for the ML handoff
    "b":"u","v":"g","r":"r",
    "u":"u","g":"g","i":"i","z":"z","y":"y",
    "R":"r","V":"g","B":"u",
}

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
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1","true","t","yes","y"])

def norm_filter(x):
    key = str(x).strip()
    low = key.lower()
    return FILTER_MAP.get(low, key)

def to_export(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a per-object table to Felipe's 5 columns:
      mjd, mag, mag_err, filter, detected
    Interprets UL=True as non-detection -> detected=0, mag_err=inf.
    Treats MagErr = -1.0 as 'unknown' (kept as NaN) for detections; for UL rows we set inf anyway.
    """
    # columns we expect in these tables
    need = ["MJD","Mag","MagErr","Filter","UL"]
    cols = {c: c for c in df_raw.columns}
    missing = [c for c in need if c not in cols]
    if missing:
        # show for debugging
        raise ValueError(f"Missing columns {missing}. Found: {list(df_raw.columns)}")

    df = df_raw.copy()

    # coerce numerics
    mjd = pd.to_numeric(df["MJD"], errors="coerce")
    mag = pd.to_numeric(df["Mag"], errors="coerce")

    # MagErr handling:
    # - Many ROTSE UL rows use -1.0; we will coerce to NaN first
    mag_err = pd.to_numeric(df["MagErr"], errors="coerce").astype(float)
    mag_err[mag_err < 0] = np.nan

    # UL flag: True means upper limit
    ul = coerce_bool(df["UL"])

    # detected flag
    detected = (~ul).astype(np.int8)

    # for UL rows, enforce Felipe convention
    mag_err[ul.values] = np.inf

    # filters normalized
    filt = df["Filter"].map(norm_filter).astype("string")

    out_perevent = pd.DataFrame({
        "mjd": mjd.astype(float),
        "mag": mag.astype(float),
        "mag_err": mag_err,
        "filter": filt,
        "UL": ul,
        "detected": detected
    }).sort_values("mjd").reset_index(drop=True)

    # drop rows that are totally empty (all NaN except 'detected' which would be 1 by default if UL missing)
    keep = out_perevent[["mjd","mag","mag_err","UL","filter"]].notna().any(axis=1)
    out_perevent = out_perevent[keep].reset_index(drop=True)

    return out_perevent[["mjd","mag","mag_err","UL", "filter","detected"]]

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
            df_out = to_export(df_raw) if include_ul is None else to_export(df_raw)  # your to_export already adds UL

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



