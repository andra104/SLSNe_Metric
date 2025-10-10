"""
population.py — SLSN population generation with cached sky distributions.

Generates volumetric populations with redshift sampling, Galactic extinction,
and cached uniform HEALPix sky coordinates.
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
#from shared_utils import sample_rate_from_volume, inject_uniform_healpix
from .constants import z_from_comoving_fast, dm_from_z
from .runners import get_distance_bounds 
from .model import atomic_save_pickle
from .diagnostics import plot_population_diagnostics


# Dust model (singleton)
dust_model = DustValues()

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
# Volumetric rate model (for GRBs, on-axis ≈ 10⁻⁹ Mpc⁻³ yr⁻¹)
# --------------------------------------------
def sample_rate_from_volume(rate_density, t_start, t_end, 
                                d_min=None, d_max=None,
                                z_min=None, z_max=None): #1e-8 for GRBs to account for dirty fireball and off axis, 1e-9 without
    """
    Estimate the number of event from comoving volume and volumetric rate.

    Parameters
    ----------
    t_start : float
        Start of the time window (days).
    t_end : float
        End of the time window (days).
    d_min : float
        Minimum luminosity distance in Mpc.
    d_max : float
        Maximum luminosity distance in Mpc.
    rate_density : float
        Volumetric event rate in events/Mpc^3/yr.

    Returns
    -------
    int
        Expected number of events in the survey.
    """

    d_min, d_max = get_distance_bounds(d_min=d_min, d_max=d_max, z_min=z_min, z_max=z_max)
    years = (t_end - t_start) / 365.25
    if d_max > 1:
        z_min = z_at_value(cosmo.comoving_distance, d_min * u.Mpc)
        z_max = z_at_value(cosmo.comoving_distance, d_max * u.Mpc)    
        V = cosmo.comoving_volume(z_max).to(u.Mpc**3).value - cosmo.comoving_volume(z_min).to(u.Mpc**3).value
    else: #for things that are close, regular volume
        V = ((4/3)*np.pi*(d_max**3 - d_min**3))
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
# Population generator
# =============================================================================

def generate_SLSN_PopSlicer(lc_model, t_start=1, t_end=3652,
                            z_min=0.1, z_max=2.0,
                            rate_density=1e-7,
                            peak_t_min=None, peak_t_max=None,
                            gal_lat_cut=None,
                            seed=42,
                            healpix_cache_file: Path | None = None,
                            save_to=None,
                            load_from=None, make_debug_plots=True):
    """
    Generate SLSN population with volumetric sampling and cached sky coordinates.
    
    Persists EBV and per-filter extinctions in slice_points for fast retrieval.
    
    Parameters
    ----------
    lc_model : LC
        Template model instance
    t_start, t_end : float
        Survey time range (days from mjd0)
    z_min, z_max : float
        Redshift range
    rate_density : float
        Volumetric rate (Mpc^-3 yr^-1)
    peak_t_min, peak_t_max : float, optional
        Peak time window (defaults to t_start, t_end)
    gal_lat_cut : float, optional
        Minimum Galactic latitude (degrees). If set, removes events with |b| < gal_lat_cut
        to avoid high-extinction regions near the Galactic plane. Default: None (no cut).
        Recommended: 15.0 degrees.)
    seed : int
        Random seed
    healpix_cache_file : Path, optional
        Cache file for sky coordinates
    save_to : Path, optional
        Save population to pickle
    load_from : Path, optional
        Load population from pickle (fast path)
    
    Returns
    -------
    slicer : UserPointsSlicer
        MAF slicer with population properties in slice_points
    """
    # Fast reload path
    if load_from and os.path.exists(load_from):
        with open(load_from, 'rb') as f:
            slice_data = pickle.load(f)
        slicer = UserPointsSlicer(ra=slice_data['ra'], dec=slice_data['dec'], badval=0)
        slicer.slice_points.update(slice_data)
        n_loaded = len(slice_data['ra'])
        print(f"[LOAD] Loaded {n_loaded} SLSNe from {load_from}")
        return slicer

    # After loading templates, check coverage
    valid_templates = []
    for i, sed in enumerate(lc_model.sed_grid):
        lam_min = sed['lam_rest_A'].min()
        lam_max = sed['lam_rest_A'].max()
        lam_range = lam_max - lam_min
        
        if lam_range > 2000:
            valid_templates.append(i)
    
    print(f"Valid templates: {len(valid_templates)}/{len(lc_model.sed_grid)}")
    
    if len(valid_templates) == 0:
        raise RuntimeError("No valid templates with sufficient wavelength coverage!")
    
    rng = np.random.default_rng(seed)
    
    # Sample event count from volumetric rate
    n_events = sample_rate_from_volume(
        rate_density=rate_density, t_start=t_start, t_end=t_end,
        z_min=z_min, z_max=z_max
    )
    print(f"Simulating {n_events} SLSNe (rate={rate_density:.1e} Mpc^-3 yr^-1)")
    
    # ===== STEP 1: Generate sky positions =====
    nside = 64
    ra, dec = _cached_uniform_healpix(nside, n_events, seed, 
                                      cache_file=healpix_cache_file)
    
    # ===== STEP 2: Convert to SkyCoord and apply galactic cut =====
    coords = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
    
    if gal_lat_cut is not None:
        galb_before = coords.galactic.b.deg
        mask = np.abs(galb_before) > gal_lat_cut  # Keep events AWAY from plane
        
        n_before = len(ra)
        ra = ra[mask]
        dec = dec[mask]
        coords = coords[mask]
        n_after = len(ra)
        
        print(f"[GAL LAT CUT] Applied |b| > {gal_lat_cut}° cut:")
        print(f"  Before: {n_before} events")
        print(f"  After:  {n_after} events ({100*n_after/n_before:.1f}% retained)")
        print(f"  Removed {n_before - n_after} events near Galactic plane")
        
        # CRITICAL: Update n_events for ALL subsequent arrays
        n_events = n_after
    
    # ===== STEP 3: Generate all other arrays using POST-CUT n_events =====
    
    # Redshift from comoving volume
    z_min_mpc = cosmo.comoving_distance(z_min).to_value(u.Mpc)
    z_max_mpc = cosmo.comoving_distance(z_max).to_value(u.Mpc)
    
    u_rand = rng.uniform(0.0, 1.0, n_events)  # Uses post-cut n_events
    d_mpc = ((z_max_mpc**3 - z_min_mpc**3) * u_rand + z_min_mpc**3)**(1.0/3.0)
    z_vals = z_from_comoving_fast(d_mpc)
    distances = d_mpc

    print(f"Distance range: {np.min(distances):.1f} - {np.max(distances):.1f} Mpc")
    print(f"Mean distance: {np.mean(distances):.1f} Mpc")
    
    # Random template per event
    file_indx = rng.choice(valid_templates, size=n_events)  # Uses post-cut n_events
    
    # Peak times
    peak_t_min = t_start if peak_t_min is None else peak_t_min
    peak_t_max = t_end if peak_t_max is None else peak_t_max
    print(f"Peak times: {peak_t_min} to {peak_t_max} days")
    peak_times = rng.uniform(peak_t_min, peak_t_max, n_events)  # Uses post-cut n_events
    
    # ===== STEP 4: Query EBV for final coordinates =====
    sfd = SFDQuery()
    ebv_vals = sfd(coords)
    
    # ===== STEP 5: Build slicer and store all properties =====
    slicer = UserPointsSlicer(ra=ra, dec=dec, badval=0)
    sp = slicer.slice_points
    
    # Core properties
    sp['sid'] = np.arange(n_events)
    sp['z'] = z_vals
    sp['distance'] = distances
    sp['distance_modulus'] = dm_from_z(z_vals)
    sp['peak_time'] = peak_times
    sp['file_indx'] = file_indx
    sp['ebv'] = ebv_vals
    
    # CRITICAL: Store galactic coordinates (these are from POST-CUT coords)
    sp['gall'] = coords.galactic.l.deg
    sp['galb'] = coords.galactic.b.deg
    
    # Per-filter extinctions
    ax1 = dust_model.ax1
    for f in ['u', 'g', 'r', 'i', 'z', 'y']:
        sp[f'A_{f}'] = ax1[f] * ebv_vals
    
    # Peak magnitude summaries
    for f in ['u', 'g', 'r', 'i', 'z', 'y']:
        peak_abs, peak_noebv, peak_ebv = [], [], []
        for idx, z_val, dm, ebv in zip(file_indx, z_vals, 
                                        sp['distance_modulus'], ebv_vals):
            if (f in lc_model.data[idx] and 
                len(lc_model.data[idx][f]['mag']) > 0):
                M_abs = float(np.min(lc_model.data[idx][f]['mag']))
            else:
                M_abs = np.nan
            m_noebv = M_abs + dm
            m_ebv = m_noebv + ax1[f] * ebv
            peak_abs.append(M_abs)
            peak_noebv.append(m_noebv)
            peak_ebv.append(m_ebv)
        sp[f'peak_mag_abs_{f}'] = np.array(peak_abs)
        sp[f'peak_app_mag_noebv_{f}'] = np.array(peak_noebv)
        sp[f'peak_app_mag_ebv_{f}'] = np.array(peak_ebv)

    # Diagnostic plots
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
            outdir=os.path.dirname(save_to) if save_to else None,
            prefix="population"
        )
    
    # Save
    if save_to:
        atomic_save_pickle(dict(sp), save_to)
        print(f"Saved SLSN population to {save_to}")
    
    # Verify cut was applied correctly
    if gal_lat_cut is not None:
        galb_final = sp['galb']
        n_bad = np.sum(np.abs(galb_final) < gal_lat_cut)
        print(f"\n[VERIFICATION] Events with |b| < {gal_lat_cut}°: {n_bad} (should be 0)")
        if n_bad > 0:
            print(f"  WARNING: Galactic cut verification failed!")
    
    return slicer
