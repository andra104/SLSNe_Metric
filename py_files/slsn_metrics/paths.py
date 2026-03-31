"""
paths.py — Self-contained path configuration for slsn_metrics.

All paths are derived relative to this file's location so the repository
works on any machine (local Mac, MSI, any server) without toggling flags
or hardcoding paths.

Actual repository layout on disk
---------------------------------
SLSNe_Metric/                       <- repo root  (get_repo_root())
├── py_files/
│   └── slsn_metrics/               <- this package (get_package_dir())
│       ├── __init__.py
│       ├── paths.py                <- this file
│       └── ...
├── cadences/                       <- OpSim .db files
├── output/
│   ├── SLSNe/
│   │   ├── shared/                 <- templates, populations
│   │   └── baseline_v4.3.1_10yrs/ <- per-cadence results
│   └── per_event_files/            <- raw per-event photometry CSVs
└── notebooks/
"""

import os
from pathlib import Path


# =============================================================================
# Core path resolution
# =============================================================================

def get_package_dir() -> Path:
    """
    Absolute path to the slsn_metrics package directory (where paths.py lives).
    e.g. .../SLSNe_Metric/py_files/slsn_metrics/
    """
    return Path(os.path.abspath(os.path.dirname(__file__)))


def get_repo_root() -> Path:
    """
    Absolute path to the repository root.

    Layout:
        repo_root/                   <- this is what we return
        └── py_files/
            └── slsn_metrics/        <- paths.py lives here

    Goes up TWO levels from package directory:
        package_dir = .../SLSNe_Metric/py_files/slsn_metrics
        py_files    = .../SLSNe_Metric/py_files
        repo_root   = .../SLSNe_Metric
    """
    package_dir = get_package_dir()   # .../py_files/slsn_metrics
    py_files    = package_dir.parent  # .../py_files
    repo_root   = py_files.parent     # .../SLSNe_Metric
    return repo_root


# =============================================================================
# Directory accessors
# =============================================================================

def get_cadences_dir() -> Path:
    """Path to directory containing OpSim .db cadence files."""
    return get_repo_root() / "cadences"


def get_output_dir(science_case: str = "SLSNe", subdir=None) -> Path:
    """
    Path to output directory for a given science case, created if missing.

    Examples
    --------
    get_output_dir()                     -> output/SLSNe/
    get_output_dir(subdir='evolving')    -> output/SLSNe/evolving/
    get_output_dir(subdir='sensitivity') -> output/SLSNe/sensitivity/
    """
    base = get_repo_root() / "output" / science_case
    if subdir:
        base = base / subdir
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_data_dir() -> Path:
    """
    Path to per-event photometry files.
    Location: output/per_event_files/
    """
    return get_repo_root() / "output" / "per_event_files"


def get_per_event_dir() -> Path:
    """
    Path to per-event cenwave photometry CSVs.
    Same as get_data_dir() — points to output/per_event_files/
    """
    return get_data_dir()


def get_all_events_dir() -> Path:
    """
    Path to compiled all-events catalog files.
    Location: output/all_events/
    Contains: allparameter.csv, all_objects.csv, allevent_redshift_med.csv, etc.
    """
    return get_repo_root() / "output" / "all_events"


def get_shared_output_dir(science_case: str = "SLSNe") -> Path:
    """
    Path to shared output files (templates, populations).
    Location: output/SLSNe/shared/
    """
    d = get_output_dir(science_case) / "shared"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rate_csv_path(filename: str = "fiducial_models.csv") -> Path:
    """
    Default path to Ben's tabulated R(z) CSV file.
    Location: output/SLSNe/shared/fiducial_models.csv

    Override at CLI with --rate-csv. When Ben sends an updated CSV,
    drop it here and it is picked up automatically.

    CSV column structure (must not change between versions):
        col 0 : redshift z       (0 to 6, step 0.06)
        col 1 : f_OH             (O-dependent metallicity fraction)
        col 2 : f_Fe_mixed       (Fe-dependent metallicity fraction)
    """
    return get_shared_output_dir() / filename


# =============================================================================
# Cadence path helpers
# =============================================================================

def get_cadence_path(cadence_name):
    """
    Full path to an OpSim cadence .db file (or list of paths).

    Parameters
    ----------
    cadence_name : str or list of str
        Cadence name(s) without the .db extension.
    """
    cadences_dir = get_cadences_dir()
    if isinstance(cadence_name, (list, tuple)):
        return [cadences_dir / f"{c}.db" for c in cadence_name]
    return cadences_dir / f"{cadence_name}.db"


def get_cadence_output_dir(cadence_name: str, science_case: str = "SLSNe") -> Path:
    """
    Per-cadence results directory.
    e.g. output/SLSNe/baseline_v4.3.1_10yrs/
    """
    d = get_output_dir(science_case) / cadence_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# Rubin sim data
# =============================================================================

def get_rubin_sim_data_dir():
    """Return RUBIN_SIM_DATA_DIR from environment if set, else None."""
    val = os.environ.get("RUBIN_SIM_DATA_DIR")
    return Path(val) if val else None


def set_rubin_sim_data_dir(path) -> None:
    """
    Set RUBIN_SIM_DATA_DIR environment variable.
    Call this BEFORE importing rubin_sim modules.
    """
    os.environ["RUBIN_SIM_DATA_DIR"] = str(path)
    print(f"[paths] RUBIN_SIM_DATA_DIR = {path}")


# =============================================================================
# Diagnostic / verification
# =============================================================================

def print_paths(cadence_name=None, science_case: str = "SLSNe", verbose: bool = True) -> None:
    """
    Print the full resolved path structure for debugging and verification.
    Call at the start of any notebook to confirm everything resolves correctly.
    """
    if cadence_name is not None and not isinstance(cadence_name, (list, tuple)):
        cadence_names = [cadence_name]
    else:
        cadence_names = list(cadence_name) if cadence_name else []

    def _check(p: Path) -> str:
        return "✓ exists" if p.exists() else "✗ missing"

    print("\n" + "=" * 70)
    print("📁  SLSN METRICS — PATH STRUCTURE")
    print("=" * 70)

    repo_root    = get_repo_root()
    cadences_dir = get_cadences_dir()
    output_base  = get_repo_root() / "output" / science_case
    data_dir     = get_data_dir()
    shared_dir   = get_shared_output_dir(science_case)
    rubin_data   = get_rubin_sim_data_dir()

    print(f"\n📂 Repo root      : {repo_root}")
    print(f"                    {_check(repo_root)}")
    print(f"\n📂 Cadences dir   : {cadences_dir}")
    print(f"                    {_check(cadences_dir)}")
    print(f"\n📂 Output dir     : {output_base}")
    print(f"                    {_check(output_base)}")
    print(f"\n📂 Shared outputs : {shared_dir}")
    print(f"                    {_check(shared_dir)}")
    print(f"\n📂 Data dir       : {data_dir}")
    print(f"                    {_check(data_dir)}")
    print(f"\n🌌 RUBIN_SIM_DATA : {rubin_data or 'not set (env var missing)'}")
    if rubin_data:
        print(f"                    {_check(rubin_data)}")

    if cadence_names:
        print(f"\n📂 Cadence databases:")
        for c in cadence_names:
            db_path = get_cadence_path(c)
            print(f"   {_check(db_path)}  {c}.db")
            print(f"              {db_path}")
        print(f"\n📂 Cadence output dirs:")
        for c in cadence_names:
            out = get_repo_root() / "output" / science_case / c
            print(f"   {_check(out)}  {c}/")
            print(f"              {out}")

    if verbose:
        print(f"\n📋 Actual layout:")
        print(f"   <repo_root>/                    ({repo_root})")
        print(f"   ├── py_files/")
        print(f"   │   └── slsn_metrics/           <- Python package")
        print(f"   ├── cadences/                   <- OpSim .db files")
        print(f"   ├── output/")
        print(f"   │   ├── per_event_files/        <- raw photometry CSVs (shared, no science case)")
        print(f"   │   └── {science_case}/")
        if cadence_names:
            for c in cadence_names:
                print(f"   │       ├── {c}/")
        print(f"   │       └── shared/             <- templates, populations")
        print(f"   └── notebooks/")

    print("\n" + "=" * 70 + "\n")


def quick_path_check(cadence_name: str, science_case: str = "SLSNe") -> None:
    """One-line path sanity check. Call before running metrics."""
    db_path  = get_cadence_path(cadence_name)
    out_path = get_repo_root() / "output" / science_case / cadence_name
    db_ok    = "✓" if db_path.exists()  else "✗ MISSING"
    out_ok   = "✓" if out_path.exists() else "(will be created)"
    print(f"Cadence DB: [{db_ok}]  Results dir: [{out_ok}]")
    if not db_path.exists():
        print(f"  -> Expected: {db_path}")