"""
gomez_models.py — Vendored magnetar+ejecta physics from Gomez+2024.

This file contains a minimal subset of functions from:

    Gomez, S. et al. 2024, "The Type I Superluminous Supernova Catalog I:
    Light Curve Properties, Models, and Catalog Description"
    arXiv:2407.07946
    https://github.com/gmzsebastian/SLSNe (MIT License)

Original author: Sebastian Gomez, 2024.

Functions copied from slsne/models.py at commit main/HEAD (July 2024).
Only the SLSN-I magnetar+ejecta physics functions are included here.
TDE, viscous, and observation/filter functions are excluded.

Modifications from original:
    - np.trapz → np.trapezoid throughout (NumPy 2.0 compatibility fix)
    - Removed functions not used by this pipeline:
      slsnni, observations, mm83, tde_*, viscous, blackbody (TDE version)

Usage in this pipeline
----------------------
from slsn_metrics.gomez_models import (
    total_luminosity, diffusion, photosphere, blackbody_supressed
)
These four functions are the complete physical chain:
    total_luminosity  →  L_in(t)   magnetar + nickel input luminosity
    diffusion         →  L_out(t)  radiation diffusion through ejecta
    photosphere       →  T(t), R(t) photospheric temperature and radius
    blackbody_supressed → F_lambda(lambda, t) SED in erg/s/Angstrom
"""

# Standard imports (same as original)
import numpy as np
from astropy import constants as c
from astropy import units as u
from scipy.interpolate import interp1d
import numexpr as ne

# =============================================================================
# Constants (copied verbatim from gomez+2024 slsne/models.py)
# =============================================================================
NI56_LUM     = 6.45e43
CO56_LUM     = 1.45e43
NI56_LIFE    = 8.8
CO56_LIFE    = 111.3
DAY_CGS      = 86400.0
C_CGS        = 29979245800.0
KM_CGS       = 100000.0
M_SUN_CGS    = 1.9884754153381438e+33
FOUR_PI      = 12.566370614359172
N_INT_TIMES  = 100
MIN_LOG_SPACING = -3
DIFF_CONST   = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
TRAP_CONST   = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS ** 2)
STEF_CONST   = (4.0 * np.pi * c.sigma_sb).cgs.value
RAD_CONST    = KM_CGS * DAY_CGS
C_CONST      = c.c.cgs.value
FLUX_CONST   = (FOUR_PI * (2.0 * c.h * c.c ** 2 * np.pi).cgs.value
                * u.Angstrom.cgs.scale)
X_CONST      = (c.h * c.c / c.k_B).cgs.value
N_TERMS      = 1000
ANG_CGS      = u.Angstrom.cgs.scale


# =============================================================================
# Physical model functions
# =============================================================================

def nickelcobalt(times, fnickel, mejecta, rest_t_explosion):
    """
    Luminosity of a nickel-cobalt powered supernova light curve.
    From Nadyozhin 1994 (1994ApJS...92..527N).

    Parameters
    ----------
    times : array  Times in days.
    fnickel : float  Nickel mass fraction.
    mejecta : float  Total ejecta mass in solar masses.
    rest_t_explosion : float  Explosion time in rest-frame days.

    Returns
    -------
    luminosities : array  Luminosity in erg/s.
    """
    mnickel = fnickel * mejecta
    ts = np.empty_like(times)
    t_inds = times >= rest_t_explosion
    ts[t_inds] = times[t_inds] - rest_t_explosion

    luminosities = np.zeros_like(times)
    luminosities[t_inds] = mnickel * (
        NI56_LUM * np.exp(-ts[t_inds] / NI56_LIFE) +
        CO56_LUM * np.exp(-ts[t_inds] / CO56_LIFE))
    luminosities[np.isnan(luminosities)] = 0.0
    return luminosities


def magnetar(times, Pspin, Bfield, Mns, thetaPB, rest_t_explosion):
    """
    Luminosity of a magnetar-powered supernova light curve.
    From Ostriker & Gunn 1971, eq 4.

    Parameters
    ----------
    times : array  Times in days.
    Pspin : float  Spin period in milliseconds.
    Bfield : float  Magnetic field in units of 10^14 Gauss.
    Mns : float  Neutron star mass in solar masses.
    thetaPB : float  Magnetic/rotation axis angle in radians.
    rest_t_explosion : float  Explosion time in rest-frame days.

    Returns
    -------
    luminosities : list  Luminosity in erg/s.
    """
    Ep = 2.6e52 * (Mns / 1.4) ** (3. / 2.) * Pspin ** (-2)
    tp = (1.3e5 * Bfield ** (-2) * Pspin ** 2
          * (Mns / 1.4) ** (3. / 2.)
          * (np.sin(thetaPB)) ** (-2))

    ts = [np.inf if rest_t_explosion > x
          else (x - rest_t_explosion) for x in times]
    luminosities = [2 * Ep / tp / (1. + 2 * t * DAY_CGS / tp) ** 2
                    for t in ts]
    luminosities = [0.0 if np.isnan(x) else x for x in luminosities]
    return luminosities


def total_luminosity(times, fnickel, mejecta, Pspin, Bfield, Mns,
                     thetaPB, rest_t_explosion):
    """
    Total input luminosity = nickel-cobalt + magnetar spin-down.

    Parameters
    ----------
    times : array  Times in days.
    fnickel : float  Nickel mass fraction.
    mejecta : float  Ejecta mass in solar masses.
    Pspin : float  Spin period in milliseconds.
    Bfield : float  Magnetic field in units of 10^14 Gauss.
    Mns : float  Neutron star mass in solar masses.
    thetaPB : float  Magnetic/rotation axis angle in radians.
    rest_t_explosion : float  Explosion time in rest-frame days.

    Returns
    -------
    luminosities : array  Total luminosity in erg/s.
    """
    nickel_lum  = nickelcobalt(times, fnickel, mejecta, rest_t_explosion)
    magnetar_lum = magnetar(times, Pspin, Bfield, Mns, thetaPB,
                            rest_t_explosion)
    return nickel_lum + magnetar_lum


def diffusion(times, input_luminosities, kappa, kappa_gamma,
              mejecta, v_ejecta, rest_t_explosion):
    """
    Radiation diffusion of light through the supernova ejecta.

    Parameters
    ----------
    times : array  Times in days.
    input_luminosities : array  Input luminosity in erg/s.
    kappa : float  Ejecta opacity in cm^2/g.
    kappa_gamma : float  Gamma-ray opacity in cm^2/g.
    mejecta : float  Ejecta mass in solar masses.
    v_ejecta : float  Ejecta velocity in km/s.
    rest_t_explosion : float  Explosion time in rest-frame days.

    Returns
    -------
    luminosities : array  Diffused luminosity in erg/s.
    """
    tau_diff  = np.sqrt(DIFF_CONST * kappa * mejecta / v_ejecta) / DAY_CGS
    trap_coeff = (TRAP_CONST * kappa_gamma * mejecta
                  / (v_ejecta ** 2)) / DAY_CGS ** 2
    td2, A = tau_diff ** 2, trap_coeff

    times_since_explosion = times - rest_t_explosion
    luminosities = np.zeros_like(times_since_explosion)
    min_te = min(times_since_explosion)
    tb = max(0.0, min_te)
    linterp = interp1d(times_since_explosion, input_luminosities,
                       copy=False, assume_sorted=True)

    lu  = len(times_since_explosion)
    num = int(round(N_INT_TIMES / 2.0))
    lsp = np.logspace(
        np.log10(tau_diff / times_since_explosion[-1]) + MIN_LOG_SPACING,
        0, num)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))

    int_times = np.clip(
        tb + (times_since_explosion.reshape(lu, 1) - tb) * xm,
        tb, times_since_explosion[-1])
    int_te2s  = int_times[:, -1] ** 2
    int_lums  = linterp(int_times)           # noqa: F841
    int_args  = (int_lums * int_times
                 * np.exp((int_times ** 2
                            - int_te2s.reshape(lu, 1)) / td2))
    int_args[np.isnan(int_args)] = 0.0

    # MODIFICATION: np.trapz → np.trapezoid (NumPy 2.0 compatibility)
    uniq_lums = np.trapezoid(int_args, int_times)

    int_te2s[int_te2s <= 0] = np.nan
    luminosities = uniq_lums * (-2.0 * np.expm1(-A / int_te2s) / td2)
    luminosities[np.isnan(luminosities)] = 0.0
    return luminosities


def photosphere(times, luminosities, v_ejecta, temperature,
                rest_t_explosion):
    """
    Photospheric radius and temperature from bolometric luminosity.

    Parameters
    ----------
    times : array  Times in days.
    luminosities : array  Luminosity in erg/s.
    v_ejecta : float  Ejecta velocity in km/s.
    temperature : float  Temperature floor in Kelvin.
    rest_t_explosion : float  Explosion time in rest-frame days.

    Returns
    -------
    rphot : array  Photospheric radius in cm.
    Tphot : array  Photospheric temperature in Kelvin.
    """
    radius2_in = [(RAD_CONST * v_ejecta
                   * max(x - rest_t_explosion, 0.0)) ** 2
                  for x in times]
    rec_radius2_in = [x / (STEF_CONST * temperature ** 4)
                      for x in luminosities]
    rphot, Tphot = [], []
    for li, lum in enumerate(luminosities):
        radius2     = radius2_in[li]
        rec_radius2 = rec_radius2_in[li]
        if lum == 0.0:
            temperature_out = 0.0
        elif radius2 < rec_radius2:
            temperature_out = (lum / (STEF_CONST * radius2)) ** 0.25
        else:
            radius2         = rec_radius2
            temperature_out = temperature
        rphot.append(np.sqrt(radius2))
        Tphot.append(temperature_out)
    return np.array(rphot), np.array(Tphot)


def mod_blackbody(lam, T, R2, sup_lambda, power_lambda):
    """
    Modified blackbody with UV suppression blueward of sup_lambda.

    Parameters
    ----------
    lam : array  Wavelengths in Angstroms.
    T : float  Temperature in Kelvin.
    R2 : float  Photospheric radius squared in cm^2.
    sup_lambda : float  Suppression onset wavelength in Angstroms.
    power_lambda : float  Suppression power law index.

    Returns
    -------
    Radiance : array  Spectral radiance in erg/s/Angstrom.
    """
    h    = 6.62607e-27    # Planck constant (cm^2 g/s)
    c    = 2.99792458e10  # Speed of light (cm/s)
    k_B  = 1.38064852e-16 # Boltzmann constant (cm^2 g/s^2/K)
    lam_cm = lam * 1e-8

    if T > 0:
        exponential = (h * c) / (lam_cm * k_B * T)
        B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)
                 / (np.exp(exponential) - 1))
    else:
        B_lam = np.zeros_like(lam_cm) * np.nan

    A        = 4 * np.pi * R2
    Radiance = B_lam * A / 1e8
    blue     = lam < sup_lambda
    Radiance[blue] *= (lam[blue] / sup_lambda) ** power_lambda
    return Radiance


def blackbody_supressed(times, luminosities, rphot, Tphot,
                        cutoff_wavelength, alpha,
                        sample_wavelengths, redshift):
    """
    Modified blackbody SED suppressed blueward of cutoff_wavelength.

    IMPORTANT: All phases passed to this function must have Tphot > 0.
    Phases with T=0 (pre-explosion) must be masked out by the caller
    and handled separately (set to zero flux). Passing T=0 causes
    nan propagation in the normalisation step that corrupts all phases.
    See mosfit_interface._call_slsnni_safe() for the masking pattern.

    Parameters
    ----------
    times : array  Times in days (only valid/T>0 phases).
    luminosities : array  Luminosity in erg/s.
    rphot : array  Photospheric radius in cm.
    Tphot : array  Photospheric temperature in Kelvin (all must be > 0).
    cutoff_wavelength : float  UV suppression onset in Angstroms.
    alpha : float  UV suppression power.
    sample_wavelengths : array  Output wavelengths in Angstroms.
    redshift : float  Redshift (use 0.0 for rest-frame templates).

    Returns
    -------
    seds : array of arrays  F_lambda in erg/s/Angstrom, shape (N_phase,).
    """
    xc = X_CONST     # noqa: F841
    fc = FLUX_CONST  # noqa: F841
    cc = C_CONST     # noqa: F841
    ac = ANG_CGS
    cwave_ac  = cutoff_wavelength * ac
    cwave_ac2 = cwave_ac * cwave_ac         # noqa: F841
    cwave_ac3 = cwave_ac2 * cwave_ac        # noqa: F841
    zp1 = 1.0 + redshift

    lt   = len(times)
    seds = np.empty(lt, dtype=object)
    rp2  = np.array(rphot) ** 2
    tp   = Tphot

    rest_wavs  = sample_wavelengths * ac / zp1
    sup_power  = alpha
    wavs_power = (5 - sup_power)            # noqa: F841

    for li, lum in enumerate(luminosities):
        ab   = rest_wavs < cwave_ac         # noqa: F841
        tpi  = tp[li]                       # noqa: F841
        rp2i = rp2[li]                      # noqa: F841

        sed = ne.evaluate(
            "where(ab, fc * (rp2i / cwave_ac ** sup_power / "
            "rest_wavs ** wavs_power) / expm1(xc / rest_wavs / tpi), "
            "fc * (rp2i / rest_wavs ** 5) / "
            "expm1(xc / rest_wavs / tpi))"
        )
        sed[np.isnan(sed)] = 0.0
        seds[li] = sed

    bb_wavelengths = np.linspace(100, 100000, N_TERMS)

    # MODIFICATION: np.trapz → np.trapezoid (NumPy 2.0 compatibility)
    norms = np.array([
        (R2 * STEF_CONST * T ** 4) /
        np.trapezoid(
            mod_blackbody(bb_wavelengths, T, R2,
                          cutoff_wavelength, alpha),
            bb_wavelengths)
        for T, R2 in zip(tp, rp2)
    ])

    seds *= norms
    return seds
