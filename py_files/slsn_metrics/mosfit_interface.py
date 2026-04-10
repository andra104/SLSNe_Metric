"""
mosfit_interface.py — Physical SED templates from Sebastian Gomez's magnetar model.

Replaces GP-based templates for high-z events where optical data cannot
constrain rest-frame UV (z > ~1.2). Uses the slsnni() magnetar+ejecta model
from Gomez+2024 with fitted parameters from all_parameters.txt to generate
physically motivated SEDs from UV through NIR at rest frame.

The output sed_grid format matches exactly what synthesize_mag_at_z() in
model.py expects, so the rest of the pipeline is unchanged.

Key function
------------
build_physical_templates(params_file, repo_root)
    Reads all_parameters.txt, calls slsnni() per event, returns LC object.

Usage
-----
from slsn_metrics.mosfit_interface import build_physical_templates
from slsn_metrics.paths import get_repo_root

templates = build_physical_templates(
    params_file=get_repo_root() / "SLSNe/slsne/ref_data/all_parameters.txt",
    repo_root=get_repo_root() / "SLSNe/slsne",
)
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .paths import get_repo_root

log = logging.getLogger(__name__)

# =============================================================================
# Constants for F_lambda -> F_nu conversion
# =============================================================================
C_CGS        = 2.99792458e10   # cm/s
ANGSTROM_CM  = 1.0e-8          # cm per Angstrom
JY           = 1.0e-23         # erg/s/cm^2/Hz per Jansky
# Distance of 10 pc in cm
PC_CM        = 3.085677581e18  # cm per parsec
D_10PC_CM    = 10.0 * PC_CM    # 10 pc in cm

# =============================================================================
# Phase grid for SED evaluation
# =============================================================================
# Rest-frame phases in days. Dense near peak, sparser at late times.
# Matches the range used in process_rest_frame() in mosutils.py.
PHASE_GRID = np.concatenate([
    np.linspace(-30,  100, 131),   # -30 to +100 days, 1-day steps
    np.linspace(100,  400,  61),   # +100 to +400 days, 5-day steps
])

# =============================================================================
# Wavelength grid for SED sampling
# =============================================================================
# Dense enough to resolve UV suppression feature and LSST filter profiles.
# Range 500-12000 Å covers Swift UVW2 (1600Å) through LSST y (11000Å)
# with margin. At z=2 the LSST y band samples ~3600Å rest — well within range.
WAVE_GRID_A = np.linspace(500.0, 12000.0, 3000)  # Angstroms, rest frame


def _flam_to_fnu_jy_at_10pc(flam_ergs_per_s_per_A: np.ndarray,
                              lam_A: np.ndarray) -> np.ndarray:
    """
    Convert F_lambda (erg/s/Å at 10 pc luminosity distance) to
    F_nu in Jansky at 10 pc.

    F_nu = F_lam * lam^2 / c
    then scale from luminosity (erg/s/Å) to flux at 10 pc.

    Parameters
    ----------
    flam_ergs_per_s_per_A : array [N_lambda]
        Specific luminosity in erg/s/Å  (slsnni output at z=0 is
        already normalised to 10 pc via observations() when redshift=0,
        BUT when we call with cenwaves= we get raw erg/s/Å luminosity,
        so we must divide by 4*pi*D_10pc^2 ourselves)
    lam_A : array [N_lambda]
        Wavelengths in Angstroms

    Returns
    -------
    fnu_jy : array [N_lambda]
        F_nu in Jansky at 10 pc
    """
    lam_cm  = lam_A * ANGSTROM_CM                        # Å → cm
    # F_lam in erg/s/cm^2/Å at 10 pc
    flam_at_10pc = flam_ergs_per_s_per_A / (4.0 * np.pi * D_10PC_CM**2)
    # F_nu in erg/s/cm^2/Hz
    fnu_cgs = flam_at_10pc * lam_cm**2 / C_CGS
    # Convert to Jansky
    fnu_jy  = fnu_cgs / JY
    return fnu_jy


def _load_gomez_models(gomez_slsne_dir: Path) -> object:
    """
    Import slsnni from Sebastian's models.py without installing his package.
    Adds gomez_slsne_dir to sys.path temporarily.

    Parameters
    ----------
    gomez_slsne_dir : Path
        Path to the SLSNe/slsne/ directory (contains models.py)

    Returns
    -------
    slsnni : callable
    """
    slsne_dir = str(gomez_slsne_dir.resolve())
    if slsne_dir not in sys.path:
        sys.path.insert(0, slsne_dir)
    # models.py in slsne/ imports from within the slsne package
    # so we also need the parent on path
    parent_dir = str(gomez_slsne_dir.parent.resolve())
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from slsne.models import slsnni  # noqa: PLC0415
        log.info("[mosfit_interface] slsnni imported from %s", slsne_dir)
        return slsnni
    except ImportError as e:
        raise ImportError(
            f"Could not import slsnni from {gomez_slsne_dir}. "
            f"Make sure SLSNe/ repo is cloned at that path.\n"
            f"Original error: {e}"
        )


def build_physical_sed_grid(
    params_file: Path | str,
    gomez_slsne_dir: Path | str | None = None,
    phase_grid: np.ndarray | None = None,
    wave_grid_A: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[list[dict], list[str]]:
    """
    Build rest-frame SED grids for all events in all_parameters.txt
    using Sebastian's physical magnetar+ejecta model (slsnni).

    Each SED grid has format:
        {
            'phase':      1D array [N_phase]  rest-frame days since peak
            'lam_rest_A': 1D array [N_lambda] wavelengths in Å
            'Fnu_abs':    2D array [N_phase, N_lambda] F_nu in Jy at 10 pc
            'coverage':   2D bool  [N_phase, N_lambda] True where flux > 0
        }

    This format is exactly what _interp_Fnu_abs_at_phase() in model.py
    expects (lines 83-103 of model.py).

    Parameters
    ----------
    params_file : Path or str
        Path to all_parameters.txt from Gomez+2024
    gomez_slsne_dir : Path or str or None
        Path to SLSNe/slsne/ directory. If None, defaults to
        repo_root/SLSNe/slsne/
    phase_grid : array or None
        Rest-frame phases in days. Defaults to PHASE_GRID.
    wave_grid_A : array or None
        Wavelengths in Angstroms. Defaults to WAVE_GRID_A.
    verbose : bool
        Show progress bar.

    Returns
    -------
    sed_grid : list of dict
        One entry per event in all_parameters.txt
    names : list of str
        Event names in same order as sed_grid
    """
    params_file = Path(params_file)
    if not params_file.exists():
        raise FileNotFoundError(f"all_parameters.txt not found at {params_file}")

    # Resolve gomez directory
    if gomez_slsne_dir is None:
        gomez_slsne_dir = get_repo_root() / "SLSNe" / "slsne"
    gomez_slsne_dir = Path(gomez_slsne_dir)

    # Import slsnni from Sebastian's code
    slsnni = _load_gomez_models(gomez_slsne_dir)

    # Load parameter table
    params = pd.read_csv(params_file, delim_whitespace=True)
    log.info("[mosfit_interface] Loaded %d events from %s",
             len(params), params_file.name)

    # Use default grids if not provided
    if phase_grid is None:
        phase_grid = PHASE_GRID
    if wave_grid_A is None:
        wave_grid_A = WAVE_GRID_A

    sed_grid = []
    names    = []
    n_failed = 0

    iterator = tqdm(params.iterrows(), total=len(params),
                    desc="Building physical SED grid",
                    disable=not verbose)

    for _, row in iterator:
        name = str(row['name']).strip()

        try:
            # ── Extract median parameters ──────────────────────────────────
            # Note: log(Bfield)_med is log10(B / 1e14 G) in Sebastian's convention
            # slsnni expects Bfield in units of 1e14 G, so:
            #   Bfield = 10^(log(Bfield)_med) / 1e14  ... but wait —
            #   looking at process_rest_frame: Bfield = 10**param['log(Bfield)'] / 1e14
            #   so the column is log10(B_in_Gauss), and we divide by 1e14
            Pspin        = float(row['Pspin_med'])
            Bfield       = 10.0 ** float(row['log(Bfield)_med']) / 1e14
            Mns          = float(row['Mns_med'])
            thetaPB      = float(row['thetaPB_med'])
            kappa        = float(row['kappa_med'])
            kappagamma   = float(row['kappagamma_med'])
            mejecta      = float(row['mejecta_med'])
            fnickel      = float(row['fnickel_med'])
            vejecta      = float(row['vejecta_med'])
            temperature  = float(row['temperature_med'])
            cut_wave     = float(row['cutoff_wavelength_med'])
            alpha        = float(row['alpha_med'])

            # log_kappa_gamma: slsnni expects log10(kappagamma)
            log_kappa_gamma = np.log10(kappagamma) if kappagamma > 0 else -2.0

            # texplosion: use 0.0 for rest-frame relative phases
            # (same convention as process_rest_frame in mosutils.py)
            texplosion = 0.0

            # No host extinction or MW reddening — we want intrinsic SED
            log_nh_host = 16.0   # minimal host absorption
            ebv         = 0.0
            redshift    = 0.0    # rest frame

            # ── Call slsnni with cenwaves to get raw SEDs ──────────────────
            # Returns seds: array of length N_phase, each element is
            # F_lambda array [N_lambda] in erg/s/Å (luminosity, not flux)
            raw_seds = slsnni(
                times           = phase_grid,
                Pspin           = Pspin,
                Bfield          = Bfield,
                Mns             = Mns,
                thetaPB         = thetaPB,
                texplosion      = texplosion,
                kappa           = kappa,
                log_kappa_gamma = log_kappa_gamma,
                mejecta         = mejecta,
                fnickel         = fnickel,
                v_ejecta        = vejecta,
                temperature     = temperature,
                cut_wave        = cut_wave,
                alpha           = alpha,
                redshift        = redshift,
                log_nh_host     = log_nh_host,
                ebv             = ebv,
                cenwaves        = wave_grid_A,   # ← raw SED mode
            )
            # raw_seds is shape (N_phase,) array of arrays — stack to 2D
            # shape: [N_phase, N_lambda]
            Flam_2d = np.vstack([np.asarray(s, dtype=float) for s in raw_seds])

            # ── Convert F_lambda → F_nu in Jy at 10 pc ────────────────────
            # _flam_to_fnu_jy_at_10pc works per-phase row
            Fnu_2d = np.zeros_like(Flam_2d)
            for i_ph in range(Flam_2d.shape[0]):
                Fnu_2d[i_ph, :] = _flam_to_fnu_jy_at_10pc(
                    Flam_2d[i_ph, :], wave_grid_A
                )

            # ── Replace NaN/negative with 0 ───────────────────────────────
            Fnu_2d = np.where(np.isfinite(Fnu_2d) & (Fnu_2d > 0), Fnu_2d, 0.0)
            coverage = Fnu_2d > 0

            # ── Build sed_entry matching model.py format ───────────────────
            # model.py lines 83-86:
            #   sed_grid['phase']      → 1D [N_phase]
            #   sed_grid['lam_rest_A'] → 1D [N_lambda]
            #   sed_grid['Fnu_abs']    → 2D [N_phase, N_lambda] Jy at 10 pc
            sed_entry = {
                'phase':      phase_grid.astype(float),
                'lam_rest_A': wave_grid_A.astype(float),
                'Fnu_abs':    Fnu_2d.astype(np.float32),
                'coverage':   coverage.astype(bool),
            }

            sed_grid.append(sed_entry)
            names.append(name)

        except Exception as exc:
            log.warning("[mosfit_interface] FAILED %s: %s", name, exc)
            n_failed += 1
            continue

    log.info("[mosfit_interface] Built %d SED grids (%d failed)",
             len(sed_grid), n_failed)

    return sed_grid, names


def build_physical_templates(
    params_file: Path | str | None = None,
    gomez_slsne_dir: Path | str | None = None,
    save_to: Path | str | None = None,
    phase_grid: np.ndarray | None = None,
    wave_grid_A: np.ndarray | None = None,
    verbose: bool = True,
):
    """
    Build an LC object with physical magnetar-model SED templates.

    Drop-in replacement for LC.from_catalog() — returns the same LC class
    so runners.py, metrics.py, and population.py need zero changes.

    Parameters
    ----------
    params_file : Path or None
        Path to all_parameters.txt. Defaults to
        repo_root/SLSNe/slsne/ref_data/all_parameters.txt
    gomez_slsne_dir : Path or None
        Path to SLSNe/slsne/. Defaults to repo_root/SLSNe/slsne/
    save_to : Path or None
        If given, save the resulting LC object to this path as a .pkl
        (same format as existing templates — uses joblib).
    phase_grid : array or None
        Rest-frame phase grid in days.
    wave_grid_A : array or None
        Wavelength grid in Angstroms.
    verbose : bool
        Show progress bar.

    Returns
    -------
    model : LC
        Template model ready to pass to generate_SLSN_PopSlicer()
        and all metric classes.

    Reload flags
    ------------
    GENERATE_NEW_TEMPLATES = True   ← must set this when switching to
                                       physical templates for the first time
    GENERATE_NEW_POPULATION = True  ← must rebuild population with new templates
    FORCE_REBUILD_MAG_GRID  = True  ← must rebuild mag grid (new SED coverage)
    """
    from .model import LC  # local import to avoid circular import

    repo_root = get_repo_root()

    # Default paths
    if params_file is None:
        params_file = repo_root / "SLSNe" / "slsne" / "ref_data" / "all_parameters.txt"
    if gomez_slsne_dir is None:
        gomez_slsne_dir = repo_root / "SLSNe" / "slsne"

    # Build SED grids
    sed_grid, names = build_physical_sed_grid(
        params_file     = params_file,
        gomez_slsne_dir = gomez_slsne_dir,
        phase_grid      = phase_grid,
        wave_grid_A     = wave_grid_A,
        verbose         = verbose,
    )

    if len(sed_grid) == 0:
        raise RuntimeError("No SED grids were built — check all_parameters.txt and slsnni import.")

    # Use phase_grid as t_grid for LC (rest-frame days)
    if phase_grid is None:
        phase_grid = PHASE_GRID

    # Build minimal LC-compatible lightcurves list
    # LC.from_catalog() stores lightcurves as list of per-band dicts.
    # For the physical templates we store a placeholder so LC initialises
    # correctly — magnitude synthesis goes through sed_grid directly via
    # synthesize_mag_at_z(), bypassing the per-band interp path.
    lightcurves = [{} for _ in names]

    model = LC(
        lightcurves = lightcurves,
        t_grid      = phase_grid.tolist(),
        names       = names,
    )
    model.sed_grid = sed_grid

    if save_to is not None:
        import joblib
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'lightcurves': lightcurves,
            't_grid':      phase_grid.tolist(),
            'names':       names,
            'sed_grid':    sed_grid,
            'meta': {
                'source':     'physical_magnetar_model',
                'params_file': str(params_file),
                'n_events':   len(names),
            },
        }
        joblib.dump(payload, save_to)
        log.info("[mosfit_interface] Saved physical templates to %s", save_to)

    return model
