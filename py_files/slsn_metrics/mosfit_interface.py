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
)

Reload flags (when switching from GP to physical templates)
------------------------------------------------------------
GENERATE_NEW_TEMPLATES  = True   <- rebuild physical templates
GENERATE_NEW_POPULATION = True   <- rebuild population with new templates
FORCE_REBUILD_MAG_GRID  = True   <- rebuild mag grid (new SED coverage)
"""

from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .paths import get_repo_root
from .gomez_models import (
    total_luminosity,
    diffusion,
    photosphere,
    blackbody_supressed,
)

log = logging.getLogger(__name__)

# =============================================================================
# Constants for F_lambda -> F_nu conversion
# =============================================================================
C_CGS       = 2.99792458e10   # cm/s
ANGSTROM_CM = 1.0e-8          # cm per Angstrom
JY          = 1.0e-23         # erg/s/cm^2/Hz per Jansky
PC_CM       = 3.085677581e18  # cm per parsec
D_10PC_CM   = 10.0 * PC_CM   # 10 pc in cm

# =============================================================================
# Phase grid for SED evaluation
# =============================================================================
# Rest-frame phases in days relative to peak.
# Dense near peak, sparser at late times.
# Starts at +1 (not 0) to avoid T=0 at explosion boundary.
# Matches range used in process_rest_frame() in mosutils.py.
PHASE_GRID = np.concatenate([
    np.linspace(1,   100, 100),   # +1 to +100 days, ~1-day steps
    np.linspace(100, 400,  61),   # +100 to +400 days, 5-day steps
])

# =============================================================================
# Wavelength grid for SED sampling
# =============================================================================
# Range 500-12000 Å covers Swift UVW2 (1600Å) through LSST y (11000Å).
# At z=2 the LSST y band samples ~3600Å rest — well within range.
WAVE_GRID_A = np.linspace(500.0, 12000.0, 3000)  # Angstroms, rest frame


def _flam_to_fnu_jy_at_10pc(flam_ergs_per_s_per_A: np.ndarray,
                              lam_A: np.ndarray) -> np.ndarray:
    """
    Convert F_lambda (erg/s/Å, luminosity) to F_nu in Jansky at 10 pc.

    slsnni() with cenwaves= returns raw luminosity in erg/s/Å, not flux.
    We must divide by 4*pi*D_10pc^2 to get flux, then convert units.

    F_nu [Jy] = F_lam [erg/s/cm^2/Å] * lam^2 [cm^2] / c [cm/s] / JY
    """
    lam_cm       = lam_A * ANGSTROM_CM
    flam_at_10pc = flam_ergs_per_s_per_A / (4.0 * np.pi * D_10PC_CM**2)
    fnu_cgs      = flam_at_10pc * lam_cm**2 / C_CGS
    return fnu_cgs / JY


def _call_slsnni_safe(
    phases, Pspin, Bfield, Mns, thetaPB, texplosion,
    kappa, log_kappa_gamma, mejecta, fnickel, vejecta,
    temperature, cut_wave, alpha, wave_grid_A,
):
    """
    Call Sebastian's physical model step by step, masking out T=0 phases.

    The problem: blackbody_supressed() computes a normalisation array
    over ALL phases simultaneously. If any phase has T=0 (pre-explosion),
    its norm = 0/0 = nan, which propagates into every phase via
    `seds *= norms`. This corrupts the entire output array.

    The fix: identify valid phases (T > 0) before calling
    blackbody_supressed(), call it only on those phases, then
    reconstruct the full array with zeros for invalid phases.

    Parameters
    ----------
    phases : array [N_phase]
        Rest-frame days (relative to explosion, texplosion convention)
    ... (physical parameters matching slsnni() signature)
    wave_grid_A : array [N_lambda]
        Wavelengths in Angstroms

    Returns
    -------
    Flam_2d : array [N_phase, N_lambda]
        F_lambda in erg/s/Å. Zero where T=0 (pre-explosion phases).
    """
    rest_t_explosion = texplosion   # already in rest frame (redshift=0)
    kappa_gamma      = 10.0 ** log_kappa_gamma

    # ── Step 1: Input luminosity (magnetar + nickel) ───────────────────────
    lum_in = total_luminosity(
        phases, fnickel, mejecta, Pspin, Bfield, Mns, thetaPB,
        rest_t_explosion=rest_t_explosion,
    )

    # ── Step 2: Radiation diffusion through ejecta ─────────────────────────
    lum_out = diffusion(
        phases, lum_in, kappa, kappa_gamma, mejecta, vejecta,
        rest_t_explosion=rest_t_explosion,
    )

    # ── Step 3: Photospheric radius and temperature ────────────────────────
    rphot, Tphot = photosphere(
        phases, lum_out, vejecta, temperature,
        rest_t_explosion=rest_t_explosion,
    )

    # ── Step 4: Mask T=0 phases before blackbody call ─────────────────────
    # T=0 occurs at phases before diffusion wave reaches photosphere.
    # Passing T=0 to blackbody_supressed corrupts the normalisation array
    # via nan propagation — see docstring above.
    valid    = np.asarray(Tphot) > 0
    n_valid  = valid.sum()

    # Initialise output with zeros (pre-explosion phases stay zero)
    Flam_2d = np.zeros((len(phases), len(wave_grid_A)), dtype=np.float64)

    if n_valid == 0:
        log.warning("[mosfit_interface] No valid phases (all T=0) — "
                    "check texplosion parameter")
        return Flam_2d

    # ── Step 5: Blackbody SED on valid phases only ─────────────────────────
    seds_valid = blackbody_supressed(
        phases[valid],
        np.asarray(lum_out)[valid],
        rphot[valid],
        Tphot[valid],
        cutoff_wavelength = cut_wave,
        alpha             = alpha,
        sample_wavelengths= wave_grid_A,
        redshift          = 0.0,   # rest frame — no redshifting here
    )

    # Stack object array → 2D float [n_valid, N_lambda]
    seds_stack = np.vstack([np.asarray(s, dtype=float) for s in seds_valid])

    # Replace nan/negative with 0
    seds_stack = np.where(np.isfinite(seds_stack) & (seds_stack > 0),
                          seds_stack, 0.0)

    Flam_2d[valid] = seds_stack
    return Flam_2d


def build_physical_sed_grid(
    params_file: "Path | str",
    phase_grid: "np.ndarray | None" = None,
    wave_grid_A: "np.ndarray | None" = None,
    verbose: bool = True,
) -> "tuple[list[dict], list[str]]":
    """
    Build rest-frame SED grids for all events in all_parameters.txt
    using Sebastian's physical magnetar+ejecta model.

    Each SED grid matches the format expected by _interp_Fnu_abs_at_phase()
    in model.py (lines 83-103):

        {
            'phase':      1D array [N_phase]         rest-frame days
            'lam_rest_A': 1D array [N_lambda]        wavelengths in Å
            'Fnu_abs':    2D array [N_phase, N_lam]  F_nu Jy at 10 pc
            'coverage':   2D bool  [N_phase, N_lam]  True where flux > 0
        }

    Parameters
    ----------
    params_file : Path or str
        Path to all_parameters.txt from Gomez+2024
    phase_grid : array or None
        Rest-frame phases in days. Defaults to PHASE_GRID.
    wave_grid_A : array or None
        Wavelengths in Angstroms. Defaults to WAVE_GRID_A.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    sed_grid : list of dict
    names : list of str
    """
    params_file = Path(params_file)
    if not params_file.exists():
        raise FileNotFoundError(f"all_parameters.txt not found: {params_file}")
    # Load parameter table
    params = pd.read_csv(params_file, sep=r'\s+')
    log.info("[mosfit_interface] Loaded %d events from %s",
             len(params), params_file.name)

    if phase_grid is None:
        phase_grid = PHASE_GRID
    if wave_grid_A is None:
        wave_grid_A = WAVE_GRID_A

    sed_grid = []
    names    = []
    n_failed = 0

    iterator = tqdm(params.iterrows(), total=len(params),
                    desc="Building physical SED grids",
                    disable=not verbose)

    for _, row in iterator:
        name = str(row['name']).strip()
        try:
            # ── Extract median posterior parameters ────────────────────────
            # Bfield convention: column is log10(B in Gauss),
            # slsnni expects B in units of 1e14 G
            # → Bfield = 10^col / 1e14
            # (confirmed from process_rest_frame in mosutils.py)
            Pspin       = float(row['Pspin_med'])
            Bfield      = 10.0 ** float(row['log(Bfield)_med']) / 1e14
            Mns         = float(row['Mns_med'])
            thetaPB     = float(row['thetaPB_med'])
            kappa       = float(row['kappa_med'])
            kappagamma  = float(row['kappagamma_med'])
            mejecta     = float(row['mejecta_med'])
            fnickel     = float(row['fnickel_med'])
            vejecta     = float(row['vejecta_med'])
            temperature = float(row['temperature_med'])
            cut_wave    = float(row['cutoff_wavelength_med'])
            alpha       = float(row['alpha_med'])

            # log_kappa_gamma: slsnni expects log10(kappagamma)
            log_kappa_gamma = (np.log10(kappagamma)
                               if kappagamma > 0 else -2.0)

            # texplosion: use the actual fitted value from the table.
            # This is in observer-frame days relative to peak MJD.
            # At redshift=0 (rest frame), rest_t_explosion = texplosion.
            # Negative = explosion happened before peak (typical: -20 to -80d)
            texplosion = float(row['texplosion_med'])

            # ── Build SED grid with T=0 masking ───────────────────────────
            Flam_2d = _call_slsnni_safe(
                phases          = phase_grid,
                Pspin           = Pspin,
                Bfield          = Bfield,
                Mns             = Mns,
                thetaPB         = thetaPB,
                texplosion      = texplosion,
                kappa           = kappa,
                log_kappa_gamma = log_kappa_gamma,
                mejecta         = mejecta,
                fnickel         = fnickel,
                vejecta         = vejecta,
                temperature     = temperature,
                cut_wave        = cut_wave,
                alpha           = alpha,
                wave_grid_A     = wave_grid_A,
            )

            # ── Convert F_lambda → F_nu in Jy at 10 pc ────────────────────
            Fnu_2d = np.zeros_like(Flam_2d)
            for i_ph in range(Flam_2d.shape[0]):
                Fnu_2d[i_ph] = _flam_to_fnu_jy_at_10pc(
                    Flam_2d[i_ph], wave_grid_A
                )

            Fnu_2d   = np.where(np.isfinite(Fnu_2d) & (Fnu_2d > 0),
                                Fnu_2d, 0.0)
            coverage = Fnu_2d > 0

            # ── Build sed_entry matching model.py format ───────────────────
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
    params_file: "Path | str | None" = None,
    save_to: "Path | str | None" = None,
    phase_grid: "np.ndarray | None" = None,
    wave_grid_A: "np.ndarray | None" = None,
    verbose: bool = True,
):
    """
    Build an LC object with physical magnetar-model SED templates.

    Drop-in replacement for LC.from_catalog(). Returns the same LC class
    so runners.py, metrics.py, and population.py need zero changes.

    Parameters
    ----------
    params_file : Path or None
        Path to all_parameters.txt.
        Default: repo_root/SLSNe/slsne/ref_data/all_parameters.txt
    save_to : Path or None
        Save resulting LC object as .pkl (joblib format, same as GP templates).
    phase_grid : array or None
        Rest-frame phase grid in days.
    wave_grid_A : array or None
        Wavelength grid in Angstroms.
    verbose : bool
        Show progress bar.

    Returns
    -------
    model : LC
        Template model ready for generate_SLSN_PopSlicer() and metrics.
    """
    from .model import LC   # local import — avoids circular import

    repo_root = get_repo_root()

    if params_file is None:
        params_file = (repo_root / "SLSNe" / "slsne" /
                       "ref_data" / "all_parameters.txt")

    sed_grid, names = build_physical_sed_grid(
        params_file     = params_file,
        phase_grid      = phase_grid,
        wave_grid_A     = wave_grid_A,
        verbose         = verbose,
    )

    if len(sed_grid) == 0:
        raise RuntimeError(
            "No SED grids built — check all_parameters.txt and slsnni import."
        )

    if phase_grid is None:
        phase_grid = PHASE_GRID

    # Placeholder lightcurves list — LC requires it but magnitude synthesis
    # goes through sed_grid → synthesize_mag_at_z(), not per-band interp
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
                'source':      'physical_magnetar_model',
                'params_file': str(params_file),
                'n_events':    len(names),
            },
        }
        joblib.dump(payload, save_to)
        log.info("[mosfit_interface] Saved physical templates → %s", save_to)

    return model
