"""
constants.py — Physical constants, cosmology lookups, bandpass cache.

Provides fast vectorized z↔distance and DM(z) interpolation using a shared table.
"""
import os
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from rubin_sim.phot_utils import Bandpass
from rubin_scheduler.data import get_data_dir

# =============================================================================
# Cosmology table (built once per process)
# =============================================================================

_COSMO_TABLE = None

def _ensure_cosmo_table(z_min=1e-4, z_max=10.0, n_points=10000):
    """Build shared log-spaced z grid → comoving distance (Mpc) and DM (mag)."""
    global _COSMO_TABLE
    if _COSMO_TABLE is not None:
        return
    z_grid = np.logspace(np.log10(z_min), np.log10(z_max), int(n_points))
    d_Mpc = cosmo.comoving_distance(z_grid).value
    dm_grid = cosmo.distmod(z_grid).value
    _COSMO_TABLE = {
        "z": z_grid, "d_Mpc": d_Mpc, "dm": dm_grid,
        "z_min": float(z_grid.min()), "z_max": float(z_grid.max())
    }

def dm_from_z(z, *, clip=True):
    """Vectorized distance modulus from redshift using cached table."""
    _ensure_cosmo_table()
    zarr = np.asarray(z, float)
    if clip:
        zarr = np.clip(zarr, _COSMO_TABLE["z_min"], _COSMO_TABLE["z_max"])
    dm = np.interp(zarr, _COSMO_TABLE["z"], _COSMO_TABLE["dm"])
    return float(dm) if np.isscalar(z) else dm

def dm_from_z_fast(z):
    """Alias for dm_from_z with clipping."""
    return dm_from_z(z, clip=True)

def z_from_comoving_fast(d_Mpc, *, clip=True):
    """Vectorized z(d_comoving in Mpc) using the same table."""
    _ensure_cosmo_table()
    darr = np.asarray(d_Mpc, float)
    if clip:
        darr = np.clip(darr, _COSMO_TABLE["d_Mpc"].min(), _COSMO_TABLE["d_Mpc"].max())
    z = np.interp(darr, _COSMO_TABLE["d_Mpc"], _COSMO_TABLE["z"])
    return float(z) if np.isscalar(d_Mpc) else z

def reset_cosmo_cache():
    """Reset cosmology cache (call if you change cosmology parameters)."""
    global _COSMO_TABLE
    _COSMO_TABLE = None

# =============================================================================
# Physical constants
# =============================================================================

C_MS = 2.99792458e8        # Speed of light (m/s)
C_CM_S = 2.99792458e10     # Speed of light (cm/s)
A_TO_CM = 1e-8             # 1 Å = 1e-8 cm
JY_TO_CGS = 1e-23          # 1 Jy = 1e-23 erg/s/cm^2/Hz

# AB magnitude system
F0_JY = 3631.0             # Zero-point flux (Jy)
LN10_OVER_2P5 = np.log(10.0) / 2.5

# =============================================================================
# Filter effective wavelengths and frequencies
# =============================================================================

# ZTF effective wavelengths (Å)
ZTF_EFF_LAMBDA = {
    "ztfg": 4800.0,
    "ztfr": 6400.0,
    "ztfi": 7900.0,
}

# LSST effective frequencies (Hz) for prediction
LSST_EFF_FREQ = {
    'u': 8.088e14,
    'g': 6.293e14,
    'r': 4.844e14,
    'i': 3.979e14,
    'z': 3.461e14,
    'y': 3.080e14,
}

def angstrom_to_hz(lambda_A):
    """Convert wavelength (Å) to frequency (Hz)."""
    lam_m = np.asarray(lambda_A, dtype=float) * 1e-10
    return C_MS / lam_m

def hz_to_angstrom(nu_hz):
    """Convert frequency (Hz) to wavelength (Å)."""
    nu = np.asarray(nu_hz, dtype=float)
    return (C_MS / nu) * 1e10

# LSST central wavelengths (Å) derived from frequencies
LSST_EFF_LAMBDA = {b: hz_to_angstrom(nu) for b, nu in LSST_EFF_FREQ.items()}

# SDSS filter aliases
SDSS_ALIAS = {'sdssu':'u', 'sdssg':'g', 'sdssr':'r', 'sdssi':'i', 'sdssz':'z'}

# =============================================================================
# Unit conversion helpers
# =============================================================================

def mag_to_flux_jy(mag: np.ndarray) -> np.ndarray:
    """Convert AB magnitude to flux (Jy)."""
    m = np.asarray(mag, float)
    return F0_JY * 10.0 ** (-0.4 * m)

def magerr_to_fluxerr_jy(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    """Convert magnitude error to flux error (Jy)."""
    f = mag_to_flux_jy(mag)
    return LN10_OVER_2P5 * f * np.asarray(mag_err, float)

def flux_jy_to_mag(flux_jy: np.ndarray) -> np.ndarray:
    """Convert flux (Jy) to AB magnitude."""
    f = np.asarray(flux_jy, float)
    out = np.full_like(f, np.nan, dtype=float)
    good = f > 0
    out[good] = -2.5 * np.log10(f[good] / F0_JY)
    return out

# =============================================================================
# Bandpass cache (lazy load once per process)
# =============================================================================

_LSST_BANDS = None

def get_lsst_bands():
    """Load and cache LSST bandpasses."""
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

# =============================================================================
# Phase binning for evaluator cache
# =============================================================================

PHASE_BIN_STEP = 0.2  # days (rest-frame)

def phase_bucket_vec(phase_rest, step=PHASE_BIN_STEP, *, return_index=True):
    """
    Vectorized phase quantization to stable bin indices or centers.
    
    Parameters
    ----------
    phase_rest : array-like
        Rest-frame phase values (days)
    step : float
        Bin width (days)
    return_index : bool
        If True, return integer indices; else return bin centers (float)
    """
    pr = np.asarray(phase_rest, dtype=float)
    idx = np.rint(pr / float(step)).astype(np.int32)
    if return_index:
        return idx
    return idx.astype(float) * float(step)
