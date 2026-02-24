"""
population.py — SLSN population generation with cached sky distributions.

Generates volumetric populations with redshift sampling, Galactic extinction,
and cached uniform HEALPix sky coordinates.

Rate Models
-----------
'constant'  : Original flat volumetric rate (Mpc^-3 yr^-1), uniform comoving
              volume sampling.  Equivalent to old generate_SLSN_PopSlicer.
'evolving'  : Metallicity-dependent rate R(z) = R(z_ref) × [Ψ(z)·f(z)] /
              [Ψ(z_ref)·f(z_ref)], following Madau & Dickinson 2014 CSFRD and
              Tremonti+04 MZR with Andrews & Martini 2013 redshift evolution.
              Redshifts are importance-sampled from P(z) ∝ R(z)·dV/dz.

References
----------
- Quimby+2013    : R(z=0.17) = 1e-7 Mpc^-3 yr^-1
- Prajs+2017     : R(z=0.3)  = 3.5e-7 Mpc^-3 yr^-1
- Cooke+2012     : R(z=2.0)  = 1.5e-6 Mpc^-3 yr^-1
- Madau & Dickinson 2014  : Cosmic SFR density (Eq. 15)
- Tremonti+2004  : Mass-metallicity relation (Eq. 3)
- Andrews & Martini 2013  : MZR redshift evolution
- Leja+2020, 2022: Stellar mass function, star-forming main sequence
- Schulze+2021   : OH_max = 8.3 metallicity threshold
"""
import os
import pickle
from pathlib import Path
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo
from dustmaps.sfd import SFDQuery
from rubin_sim.maf.slicers import UserPointsSlicer
from astropy.cosmology import z_at_value
from rubin_sim.phot_utils import DustValues
from .constants import z_from_comoving_fast, dm_from_z
from .runners import get_distance_bounds
from .model import atomic_save_pickle
from .diagnostics import plot_population_diagnostics


# Dust model (singleton)
dust_model = DustValues()

# =============================================================================
# Observed rate measurements from literature
# Used for diagnostic plots and rate validation.
# =============================================================================

OBSERVED_RATES = [
    {'z': 0.17, 'rate': 1e-7,   'err_low': 0.3e-7, 'err_high': 0.3e-7, 'ref': 'Quimby+13'},
    {'z': 0.3,  'rate': 3.5e-7, 'err_low': 1.0e-7, 'err_high': 1.5e-7, 'ref': 'Prajs+17'},
    {'z': 2.0,  'rate': 1.5e-6, 'err_low': 0.5e-6, 'err_high': 0.8e-6, 'ref': 'Cooke+12'},
]

# --------------------------------------------
# Uniform Sphere Healpix
# --------------------------------------------
def inject_uniform_healpix(nside, n_events, seed=42):
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(seed)
    pix = rng.choice(npix, size=n_events)
    theta, phi = hp.pix2ang(nside, pix)
    ra = np.degrees(phi)
    dec = np.degrees(0.5 * np.pi - theta)
    return ra, dec


# --------------------------------------------
# Volumetric rate model (constant)
# For GRBs, on-axis ≈ 10⁻⁹ Mpc⁻³ yr⁻¹
# --------------------------------------------
def sample_rate_from_volume(rate_density, t_start, t_end,
                            d_min=None, d_max=None,
                            z_min=None, z_max=None):
    """
    Estimate the number of events from comoving volume and a CONSTANT volumetric rate.

    Parameters
    ----------
    rate_density : float
        Volumetric event rate in events Mpc^-3 yr^-1.
    t_start : float
        Start of the time window (days).
    t_end : float
        End of the time window (days).
    d_min : float, optional
        Minimum comoving distance in Mpc (supply OR z_min/z_max).
    d_max : float, optional
        Maximum comoving distance in Mpc.
    z_min : float, optional
        Minimum redshift (alternative to d_min).
    z_max : float, optional
        Maximum redshift (alternative to d_max).

    Returns
    -------
    int
        Poisson-sampled number of events in the survey volume and time window.
    """
    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)
    years = (t_end - t_start) / 365.25
    if d_max > 1:
        z_min_val = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
        z_max_val = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)
        V = (cosmo.comoving_volume(z_max_val).to(u.Mpc**3).value
             - cosmo.comoving_volume(z_min_val).to(u.Mpc**3).value)
    else:
        V = (4/3) * np.pi * (d_max**3 - d_min**3)
    return np.random.poisson(rate_density * V * years)


# =============================================================================
# Cached uniform HEALPix draws (in-memory + disk)
# =============================================================================

_HEALPIX_MEMO: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _cached_uniform_healpix(nside: int, n_events: int, seed: int,
                             cache_file: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Return RA, DEC for uniform HEALPix draw with caching.

    Uses (nside, n_events, seed) as key. Checks memory cache first,
    then optional disk cache, finally generates fresh draw.

    Parameters
    ----------
    nside : int
        HEALPix resolution
    n_events : int
        Number of sky positions
    seed : int
        Random seed
    cache_file : Path, optional
        Disk cache location

    Returns
    -------
    ra, dec : arrays
        Sky coordinates (degrees)
    """
    key = (int(nside), int(n_events), int(seed))

    # Memory cache
    if key in _HEALPIX_MEMO:
        return _HEALPIX_MEMO[key]

    # Disk cache
    if cache_file is not None:
        cache_file = Path(cache_file)
        try:
            if cache_file.exists():
                store = pickle.load(open(cache_file, "rb"))
            else:
                store = {}
            if key in store:
                ra, dec = store[key]
                _HEALPIX_MEMO[key] = (np.asarray(ra), np.asarray(dec))
                return _HEALPIX_MEMO[key]
        except Exception:
            pass  # Fall through to fresh draw

    # Fresh draw
    ra, dec = inject_uniform_healpix(nside, n_events, seed=seed)

    # Update caches
    _HEALPIX_MEMO[key] = (ra, dec)
    if cache_file is not None:
        try:
            if cache_file.exists():
                store = pickle.load(open(cache_file, "rb"))
            else:
                store = {}
            store[key] = (ra, dec)
            pickle.dump(store, open(cache_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # Non-fatal

    return ra, dec


# =============================================================================
# NEW: Metallicity-dependent rate evolution physics
# Implements the rate model from the notebook (Cells 2-3).
# These functions are used by generate_SLSN_PopSlicer when rate_model='evolving'.
# =============================================================================

def cosmic_sfr_density_MD14(z):
    """
    Madau & Dickinson 2014 cosmic SFR density [M_sun yr^-1 Mpc^-3].

    Equation 15 from their paper.

    Parameters
    ----------
    z : float or array
        Redshift.

    Returns
    -------
    psi : float or array
        Star formation rate density.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    psi = 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)
    return psi.item() if psi.size == 1 else psi


def tremonti04_mzr(log_M_star, z=0):
    """
    Tremonti+2004 Mass-Metallicity Relation (their Equation 3).

    12 + log(O/H) as a function of stellar mass, with redshift evolution
    from Andrews & Martini 2013.

    Parameters
    ----------
    log_M_star : float or array
        log10(M_* / M_sun)
    z : float
        Redshift. Tremonti+04 is z~0; evolution applied via Andrews & Martini 2013.

    Returns
    -------
    OH : float or array
        12 + log10(O/H)  [oxygen abundance]

    Notes
    -----
    Tremonti+04 Eq. 3:
        12 + log(O/H) = -1.492 + 1.847x - 0.08026x²   where x = log(M_*/M_sun) - 10
    Redshift evolution (Andrews & Martini 2013):
        OH(z) = OH(z=0) - 0.14 * z
    """
    log_M_star = np.atleast_1d(np.asarray(log_M_star, dtype=float))
    x = log_M_star - 10.0
    OH_z0 = -1.492 + 1.847 * log_M_star - 0.08026 * log_M_star**2
    OH = OH_z0 - 0.45 * z
    return OH.item() if OH.size == 1 else OH


def leja20_stellar_mass_function(log_M_star, z):
    """
    Leja+2020 stellar mass function Φ(M_* | z)  [Mpc^-3 dex^-1].

    Simplified single-Schechter function with redshift-evolving parameters
    approximating the Leja+20 double Schechter fits.

    Parameters
    ----------
    log_M_star : float or array
        log10(M_* / M_sun)
    z : float
        Redshift.

    Returns
    -------
    phi_per_dex : float or array
        Number density [Mpc^-3 dex^-1].

    Notes
    -----
    Parameters at z~0.5 (Leja+20 Table 2 approximate):
        φ*  ~ 10^-2.5 Mpc^-3 dex^-1
        M*  ~ 10^10.7 M_sun
        α   ~ -0.4
    Redshift scalings are approximate linear interpolations.
    """
    log_M_star = np.atleast_1d(np.asarray(log_M_star, dtype=float))

    log_phi_star = -2.5 - 0.3 * z
    log_M_star_char = 10.7 - 0.1 * z
    alpha = -0.4 - 0.15 * z

    phi_star = 10**log_phi_star
    M_star_char = 10**log_M_star_char
    M_star = 10**log_M_star

    # Schechter: φ(M) = φ* (M/M*)^α exp(-M/M*)
    phi = phi_star * (M_star / M_star_char)**alpha * np.exp(-M_star / M_star_char)

    # Convert to per dex: Φ(log M) = ln(10) * M * φ(M)
    phi_per_dex = np.log(10) * M_star * phi
    return phi_per_dex.item() if phi_per_dex.size == 1 else phi_per_dex


def leja22_mean_sfr(log_M_star, z):
    """
    Leja+2022 mean SFR on the star-forming main sequence.

    log(SFR) = α + β·log(M_*) + γ·log(1+z)

    Parameters
    ----------
    log_M_star : float or array
        log10(M_* / M_sun)
    z : float
        Redshift.

    Returns
    -------
    sfr : float or array
        Mean star formation rate [M_sun yr^-1].

    Notes
    -----
    Parameters from Speagle+2014 updated by Leja+2022:
        α = -0.5,  β = 0.8,  γ = 2.5
    """
    log_M_star = np.atleast_1d(np.asarray(log_M_star, dtype=float))
    log_sfr = -0.5 + 0.8 * log_M_star + 2.5 * np.log10(1 + z)
    sfr = 10**log_sfr
    return sfr.item() if sfr.size == 1 else sfr


def metallicity_fraction(z, OH_max=8.3, n_mass_bins=200):
    """
    Fraction f(z) of star formation occurring in low-metallicity galaxies.

    Implements the MZR integral (Eq. 2 from abstract):

        f(z) = ∫ Φ(M_*|z) · ⟨SFR⟩(M_*|z) · Θ{OH[M_*,z] < OH_max} d log M_*
               ─────────────────────────────────────────────────────────────────
               ∫ Φ(M_*|z) · ⟨SFR⟩(M_*|z) d log M_*

    where Θ is the Heaviside step function.

    Parameters
    ----------
    z : float or array
        Redshift.
    OH_max : float
        Metallicity threshold: 12 + log10(O/H)_max.
        Default 8.3 (~0.4 Z_sun) from Schulze+2021.
    n_mass_bins : int
        Integration resolution over stellar mass. Default 200.

    Returns
    -------
    f : float or array
        Fraction of SF in galaxies below metallicity threshold [0, 1].
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    f_z = np.zeros_like(z, dtype=float)

    # Integration over log stellar mass: 10^8 to 10^12 M_sun
    log_M_grid = np.linspace(8.0, 12.0, n_mass_bins)
    d_log_M = log_M_grid[1] - log_M_grid[0]

    for i, z_val in enumerate(z):
        phi = leja20_stellar_mass_function(log_M_grid, z_val)   # [Mpc^-3 dex^-1]
        sfr = leja22_mean_sfr(log_M_grid, z_val)                # [M_sun yr^-1]
        OH  = tremonti04_mzr(log_M_grid, z_val)                 # [12 + log(O/H)]

        # Heaviside: 1 where OH < OH_max (low metallicity = SLSN-favorable)
        low_Z = (OH < OH_max).astype(float)

        numerator   = np.trapezoid(phi * sfr * low_Z, dx=d_log_M)
        denominator = np.trapezoid(phi * sfr,          dx=d_log_M)

        f_z[i] = numerator / denominator if denominator > 0 else 0.0

    return f_z.item() if f_z.size == 1 else f_z


def slsn_rate_evolution(z, R_ref=1e-7, z_ref=0.17, OH_max=8.3):
    """
    Redshift-dependent SLSN volumetric rate [Mpc^-3 yr^-1].

    Implements Eq. 3 from abstract:

        R(z) = R(z_ref) × [Ψ(z) · f(z)] / [Ψ(z_ref) · f(z_ref)]

    where Ψ(z) is the Madau & Dickinson 2014 CSFRD and f(z) is the
    fraction of SF in low-metallicity galaxies.

    Parameters
    ----------
    z : float or array
        Redshift.
    R_ref : float
        Reference volumetric rate at z_ref [Mpc^-3 yr^-1].
        Default 1e-7 from Quimby+2013 at z=0.17.
    z_ref : float
        Reference redshift for normalization. Default 0.17 (Quimby+2013).
    OH_max : float
        Metallicity threshold 12 + log10(O/H)_max. Default 8.3 (Schulze+2021).

    Returns
    -------
    rate : float or array
        Volumetric SLSN rate [Mpc^-3 yr^-1].
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))

    psi_z   = cosmic_sfr_density_MD14(z)
    psi_ref = cosmic_sfr_density_MD14(z_ref)

    f_z   = metallicity_fraction(z,     OH_max)
    f_ref = metallicity_fraction(z_ref, OH_max)

    rate = np.atleast_1d(R_ref * (psi_z * f_z) / (psi_ref * f_ref))
    return rate.item() if rate.size == 1 else rate


def sample_events_from_evolving_rate(R_ref, z_ref, z_min, z_max,
                                     t_start, t_end, OH_max, n_bins=100):
    """
    Poisson-sample the total number of SLSN events from ∫R(z)·dV/dz·dz·Δt.

    Parameters
    ----------
    R_ref : float
        Reference rate [Mpc^-3 yr^-1].
    z_ref : float
        Reference redshift.
    z_min, z_max : float
        Redshift integration limits.
    t_start, t_end : float
        Survey window in days.
    OH_max : float
        Metallicity threshold.
    n_bins : int
        Integration resolution.

    Returns
    -------
    n : int
        Poisson draw of expected event count.
    """
    years = (t_end - t_start) / 365.25

    z_grid = np.linspace(z_min, z_max, n_bins)
    dz = z_grid[1] - z_grid[0]

    total_rate = 0.0
    for z_val in z_grid:
        R_z  = slsn_rate_evolution(z_val, R_ref, z_ref, OH_max)
        # Full-sky differential comoving volume
        dV_dz = cosmo.differential_comoving_volume(z_val).to_value(u.Mpc**3 / u.sr) * 4 * np.pi
        total_rate += R_z * dV_dz * dz

    expected_n = total_rate * years

    print(f"Expected events from R(z) integration: {expected_n:.1f}")
    print(f"  [z_min={z_min}, z_max={z_max}, survey={years:.1f} yr]")

    return int(np.random.poisson(expected_n))


def sample_redshifts_weighted_by_rate(n_events, z_min, z_max,
                                      R_ref, z_ref, OH_max, n_bins=1000):
    """
    Importance-sample redshifts from P(z) ∝ R(z) · dV/dz.

    Uses inverse-transform (CDF) sampling.

    Parameters
    ----------
    n_events : int
        Number of redshifts to draw.
    z_min, z_max : float
        Redshift range.
    R_ref : float
        Reference rate [Mpc^-3 yr^-1].
    z_ref : float
        Reference redshift.
    OH_max : float
        Metallicity threshold.
    n_bins : int
        CDF resolution.

    Returns
    -------
    z_samples : ndarray, shape (n_events,)
        Drawn redshifts.
    """
    z_grid = np.linspace(z_min, z_max, n_bins)

    weights = np.array([
        slsn_rate_evolution(z_val, R_ref, z_ref, OH_max)
        * cosmo.differential_comoving_volume(z_val).to_value(u.Mpc**3 / u.sr)
        for z_val in z_grid
    ])

    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]  # normalize to [0, 1]

    # BUG FIX: renamed from 'u' to 'u_rand' to avoid shadowing astropy.units alias
    u_rand = np.random.uniform(0, 1, n_events)
    z_samples = np.interp(u_rand, cdf, z_grid)

    return z_samples


# =============================================================================
# Population generator (unified: supports both 'constant' and 'evolving')
# =============================================================================

def generate_SLSN_PopSlicer(lc_model,
                             t_start=1,
                             t_end=3652,
                             z_min=0.1,
                             z_max=2.0,
                             # ---- rate model (NEW) ----
                             rate_model='constant',        # 'constant' or 'evolving'
                             rate_density=1e-7,            # used when rate_model='constant'
                             R_ref=1e-7,                   # used when rate_model='evolving'
                             z_ref=0.17,                   # reference redshift for evolving
                             OH_max=8.3,                   # metallicity threshold
                             # ---- sky / time ----
                             peak_t_min=None,
                             peak_t_max=None,
                             gal_lat_cut=None,
                             seed=42,
                             healpix_cache_file: Path | None = None,
                             save_to=None,
                             load_from=None,
                             make_debug_plots=True):
    """
    Generate SLSN population with volumetric sampling and cached sky coordinates.

    Supports two rate models:

    rate_model='constant' (original behavior)
        Uniform comoving volume sampling. Uses ``rate_density`` to compute
        the expected number of events via ``sample_rate_from_volume``.

    rate_model='evolving' (new)
        Redshift-dependent rate R(z) ∝ Ψ(z)·f(z) where Ψ is the Madau &
        Dickinson 2014 CSFRD and f(z) is the fraction of SF in low-metallicity
        galaxies (Tremonti+04 MZR). Redshifts are importance-sampled from
        P(z) ∝ R(z)·dV/dz via inverse-transform sampling.

    Parameters
    ----------
    lc_model : LC
        Template model instance (must have `.sed_grid` and `.data`).
    t_start, t_end : float
        Survey time range (days from MJD0).
    z_min, z_max : float
        Redshift range.
    rate_model : str
        'constant' or 'evolving'. Default 'constant'.
    rate_density : float
        Constant volumetric rate [Mpc^-3 yr^-1]. Only used if rate_model='constant'.
    R_ref : float
        Reference rate at z_ref [Mpc^-3 yr^-1]. Only used if rate_model='evolving'.
        Default 1e-7 (Quimby+2013 at z=0.17).
    z_ref : float
        Reference redshift for rate normalization. Default 0.17 (Quimby+2013).
    OH_max : float
        Metallicity threshold 12 + log10(O/H)_max. Default 8.3 (Schulze+2021).
    peak_t_min, peak_t_max : float, optional
        Peak time window (defaults to t_start, t_end).
    gal_lat_cut : float, optional
        Minimum Galactic latitude (degrees). Events with |b| < gal_lat_cut are
        removed to avoid high-extinction regions. Recommended: 15.0.
    seed : int
        Random seed.
    healpix_cache_file : Path, optional
        Cache file for sky coordinate draws.
    save_to : Path, optional
        Save final population dict to pickle.
    load_from : Path, optional
        Load population from pickle (bypasses all generation logic).
    make_debug_plots : bool
        Whether to call plot_population_diagnostics. Default True.

    Returns
    -------
    slicer : UserPointsSlicer
        MAF slicer with population properties in slice_points.

    Notes on Reloading
    ------------------
    Templates  : NO reload needed — same GP fits.
    Mag grid   : NO reload needed — same templates.
    Population : MUST regenerate when changing rate_model, z_min/z_max, OH_max.
    Kernel     : NO reload needed — not touched here.
    """

    # ------------------------------------------------------------------
    # Fast reload path (unchanged from original)
    # ------------------------------------------------------------------
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        n_loaded = len(slice_data['ra'])
        print(f"[LOAD] Loaded {n_loaded} SLSNe from {load_from}")
        return slicer

    # ------------------------------------------------------------------
    # Template coverage check (unchanged)
    # ------------------------------------------------------------------
    valid_templates = []
    for i, sed in enumerate(lc_model.sed_grid):
        lam_min = sed['lam_rest_A'].min()
        lam_max = sed['lam_rest_A'].max()
        if (lam_max - lam_min) > 2000:
            valid_templates.append(i)

    print(f"Valid templates: {len(valid_templates)}/{len(lc_model.sed_grid)}")
    if len(valid_templates) == 0:
        raise RuntimeError("No valid templates with sufficient wavelength coverage!")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # STEP 1: Sample event count and raw redshifts
    # This is the KEY divergence between the two rate models.
    # ------------------------------------------------------------------
    if rate_model == 'evolving':
        print(f"\n{'='*70}")
        print(f"RATE MODEL: EVOLVING  [metallicity-dependent R(z)]")
        print(f"{'='*70}")
        print(f"Reference: R(z={z_ref}) = {R_ref:.2e} Mpc^-3 yr^-1")
        print(f"Metallicity threshold: 12+log(O/H) < {OH_max}")

        # Total events from ∫R(z)·dV/dz·dz·Δt
        n_events_raw = sample_events_from_evolving_rate(
            R_ref, z_ref, z_min, z_max, t_start, t_end, OH_max
        )

        # Redshifts importance-sampled from P(z) ∝ R(z)·dV/dz
        z_vals_raw = sample_redshifts_weighted_by_rate(
            n_events_raw, z_min, z_max, R_ref, z_ref, OH_max
        )

        print(f"Generated {n_events_raw} events  "
              f"(z̄ = {z_vals_raw.mean():.3f}, "
              f"z range {z_vals_raw.min():.3f}–{z_vals_raw.max():.3f})")
        print(f"{'='*70}\n")

    else:  # rate_model == 'constant'
        print(f"\n{'='*70}")
        print(f"RATE MODEL: CONSTANT  [flat rate = {rate_density:.2e} Mpc^-3 yr^-1]")
        print(f"{'='*70}")

        n_events_raw = sample_rate_from_volume(
            rate_density=rate_density, t_start=t_start, t_end=t_end,
            z_min=z_min, z_max=z_max
        )
        print(f"Simulating {n_events_raw} SLSNe")

        # Uniform comoving volume sampling (original logic)
        z_min_mpc = cosmo.comoving_distance(z_min).to_value(u.Mpc)
        z_max_mpc = cosmo.comoving_distance(z_max).to_value(u.Mpc)
        u_rand = rng.uniform(0.0, 1.0, n_events_raw)
        d_mpc = ((z_max_mpc**3 - z_min_mpc**3) * u_rand + z_min_mpc**3)**(1.0 / 3.0)
        z_vals_raw = z_from_comoving_fast(d_mpc)
        print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # STEP 2: Sky positions (uniform HEALPix)
    # Generated for n_events_raw; galactic cut applied below.
    # ------------------------------------------------------------------
    nside = 64
    ra, dec = _cached_uniform_healpix(nside, n_events_raw, seed,
                                       cache_file=healpix_cache_file)

    coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')

    # ------------------------------------------------------------------
    # STEP 3: Galactic latitude cut
    # IMPORTANT: z_vals_raw is filtered HERE too, keeping arrays aligned.
    # In the old generate_SLSN_PopSlicer, redshifts were generated AFTER
    # the cut which made n_events already post-cut. Here we generate
    # z_vals_raw before so we can apply the same mask to keep sky
    # positions and redshifts synchronized.
    # ------------------------------------------------------------------
    if gal_lat_cut is not None:
        mask = np.abs(coords.galactic.b.deg) > gal_lat_cut  # True = keep

        n_before = n_events_raw
        ra       = ra[mask]
        dec      = dec[mask]
        coords   = coords[mask]
        z_vals_raw = z_vals_raw[mask]  # ← critical: same mask on redshifts
        n_after  = len(ra)

        print(f"[GAL LAT CUT] Applied |b| > {gal_lat_cut}°:")
        print(f"  Before: {n_before}  →  After: {n_after} "
              f"({100 * n_after / n_before:.1f}% retained)")

    n_events = len(ra)  # final count after all cuts

    z_vals   = z_vals_raw
    distances = cosmo.comoving_distance(z_vals).to_value(u.Mpc)

    print(f"Distance range: {distances.min():.1f}–{distances.max():.1f} Mpc  "
          f"(mean {distances.mean():.1f} Mpc)")

    # ------------------------------------------------------------------
    # STEP 4: Per-event template, peak time, EBV
    # ------------------------------------------------------------------
    file_indx = rng.choice(valid_templates, size=n_events)

    peak_t_min = t_start if peak_t_min is None else peak_t_min
    peak_t_max = t_end   if peak_t_max is None else peak_t_max
    print(f"Peak times: {peak_t_min}–{peak_t_max} days")
    peak_times = rng.uniform(peak_t_min, peak_t_max, n_events)

    sfd = SFDQuery()
    ebv_vals = sfd(coords)

    # ------------------------------------------------------------------
    # STEP 5: Build MAF slicer and populate slice_points
    # ------------------------------------------------------------------
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    sp = slicer.slice_points

    sp['sid']              = np.arange(n_events)
    sp['z']               = z_vals
    sp['distance']        = distances
    sp['distance_modulus'] = dm_from_z(z_vals)
    sp['peak_time']       = peak_times
    sp['file_indx']       = file_indx
    sp['ebv']             = ebv_vals
    sp['gall']            = coords.galactic.l.deg
    sp['galb']            = coords.galactic.b.deg

    # Store rate model metadata so downstream code knows how this pop was made
    sp['rate_model'] = rate_model
    if rate_model == 'evolving':
        sp['R_ref']   = R_ref
        sp['z_ref']   = z_ref
        sp['OH_max']  = OH_max
    else:
        sp['rate_density'] = rate_density

    # Per-filter extinctions
    ax1 = dust_model.ax1
    for f in ['u', 'g', 'r', 'i', 'z', 'y']:
        sp[f'A_{f}'] = ax1[f] * ebv_vals

    # Peak magnitude summaries (absolute, apparent no-EBV, apparent with EBV)
    for f in ['u', 'g', 'r', 'i', 'z', 'y']:
        peak_abs, peak_noebv, peak_ebv = [], [], []
        for idx, z_val, dm, ebv in zip(file_indx, z_vals,
                                        sp['distance_modulus'], ebv_vals):
            if (f in lc_model.data[idx]
                    and len(lc_model.data[idx][f]['mag']) > 0):
                M_abs = float(np.min(lc_model.data[idx][f]['mag']))
            else:
                M_abs = np.nan
            m_noebv = M_abs + dm
            m_ebv   = m_noebv + ax1[f] * ebv
            peak_abs.append(M_abs)
            peak_noebv.append(m_noebv)
            peak_ebv.append(m_ebv)
        sp[f'peak_mag_abs_{f}']        = np.array(peak_abs)
        sp[f'peak_app_mag_noebv_{f}']  = np.array(peak_noebv)
        sp[f'peak_app_mag_ebv_{f}']    = np.array(peak_ebv)

    # ------------------------------------------------------------------
    # STEP 6: Diagnostic plots
    # ------------------------------------------------------------------
    if make_debug_plots:
        plot_population_diagnostics(
            ra_rad=sp['ra'],
            dec_rad=sp['dec'],
            peak_times=sp['peak_time'],
            distances_mpc=sp['distance'],
            z_vals=sp['z'],
            ebv=sp.get('ebv', None),
            gall=sp.get('gall', None),
            galb=sp.get('galb', None),
            outdir=Path(save_to).parent if save_to else None,
            prefix=f"population_{rate_model}"
        )

    # ------------------------------------------------------------------
    # STEP 7: Galactic cut verification + save
    # ------------------------------------------------------------------
    if gal_lat_cut is not None:
        n_bad = np.sum(np.abs(sp['galb']) < gal_lat_cut)
        print(f"[VERIFY] Events with |b| < {gal_lat_cut}°: {n_bad} (should be 0)")
        if n_bad > 0:
            print(f"  WARNING: Galactic cut verification failed!")

    if save_to:
        atomic_save_pickle(dict(sp), save_to)
        print(f"Saved SLSN population to {save_to}")

    return slicer


# =============================================================================
# Backward-compatibility alias
# =============================================================================
# Old call sites that used generate_SLSN_PopSlicer with only rate_density
# will continue to work unchanged because rate_model defaults to 'constant'.
# If you previously called:
#   generate_SLSN_PopSlicer(templates, rate_density=3e-8, ...)
# it still works with identical behavior.

generate_SLSN_PopSlicer_with_rate_evolution = generate_SLSN_PopSlicer