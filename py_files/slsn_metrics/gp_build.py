"""
gp_build.py — 2D Gaussian Process fitting for SLSN light curves.

Provides kernel creation, GP fitting in (time, frequency) space, and hybrid t0 selection.
"""
import numpy as np
import pandas as pd
import warnings
import george
import scipy.optimize as op
from functools import partial
from .constants import angstrom_to_hz

# =============================================================================
# GP Kernel
# =============================================================================

def default_gp_kernel(scale_guess: float) -> george.kernels.Kernel:
    """
    Matérn-3/2 kernel in 2D (time [days], frequency [Hz]).
    Frequency metric element frozen for sparse color stability.
    
    Parameters
    ----------
    scale_guess : float
        Typical flux scale for amplitude initialization
    """
    time_scale_days = 20.0
    freq_scale_hz = 1.0e14
    k = (0.5 * max(scale_guess, 1e-6))**2 * george.kernels.Matern32Kernel(
        [time_scale_days**2, freq_scale_hz**2], ndim=2
    )
    k.freeze_parameter("k2:metric:log_M_1_1")
    return k

# =============================================================================
# GP Fitting
# =============================================================================

def fit_2d_gp(time_obs_days: np.ndarray,
              freq_obs_hz: np.ndarray,
              flux_jy: np.ndarray,
              fluxerr_jy: np.ndarray,
              kernel: george.kernels.Kernel | None = None) -> partial:
    """
    Fit 2D Gaussian Process in (time, frequency) to flux measurements.
    
    Returns a partial function: gp_predict(X_new, return_var=False)
    where X_new is shape (N, 2) with columns [time_days, freq_hz].
    
    Parameters
    ----------
    time_obs_days : array
        Observed times (MJD or days since reference)
    freq_obs_hz : array
        Observed frequencies (Hz)
    flux_jy : array
        Flux measurements (Jy)
    fluxerr_jy : array
        Flux uncertainties (Jy)
    kernel : george.kernels.Kernel, optional
        Custom kernel; if None, uses default_gp_kernel
    
    Returns
    -------
    gp_predict : partial
        Callable that returns (mean, [variance]) at new points
    """
    t = np.asarray(time_obs_days, float)
    nu = np.asarray(freq_obs_hz, float)
    y = np.asarray(flux_jy, float)
    yerr = np.asarray(fluxerr_jy, float)
    
    good = np.isfinite(t) & np.isfinite(nu) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if good.sum() < 5:
        raise ValueError("Insufficient finite points for GP fit (need ≥5).")
    
    t, nu, y, yerr = t[good], nu[good], y[good], yerr[good]
    X = np.vstack([t, nu]).T
    
    scale_guess = np.nanmedian(np.abs(y))
    if not np.isfinite(scale_guess) or scale_guess <= 0:
        scale_guess = 1e-3
    
    if kernel is None:
        kernel = default_gp_kernel(scale_guess)
    
    gp = george.GP(kernel)
    gp.compute(X, yerr)
    
    p0 = gp.get_parameter_vector()
    
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)
    
    bounds = [(p0[0] - 10.0, p0[0] + 10.0)] + [(None, None) for _ in p0[1:]]
    
    try:
        res = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", 
                          bounds=bounds, tol=1e-6)
        if res.success:
            gp.set_parameter_vector(res.x)
        else:
            warnings.warn(f"GP optimizer failed; using initial parameters. "
                          f"msg={res.message}")
    except Exception as e:
        warnings.warn(f"GP optimization raised {e!r}; using initial parameters.")
    
    return partial(gp.predict, y)

# =============================================================================
# GP Prediction Surface
# =============================================================================

def gp_predict_surface(gp_predict: partial,
                       t_eval_days: np.ndarray,
                       target_freq_hz: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Predict GP surface at specified frequencies across time grid.
    
    Parameters
    ----------
    gp_predict : partial
        Fitted GP predictor from fit_2d_gp
    t_eval_days : array
        Time points for prediction
    target_freq_hz : dict
        {band_name: frequency_hz} for each band to predict
    
    Returns
    -------
    predictions : dict
        {band: (flux_mean, flux_sigma)} for each band
    """
    out = {}
    t_eval = np.asarray(t_eval_days, float)
    for band, nu in target_freq_hz.items():
        Xp = np.vstack([t_eval, np.full_like(t_eval, nu)]).T
        mu, var = gp_predict(Xp, return_var=True)
        sig = np.sqrt(np.clip(var, 0.0, np.inf))
        out[band] = (mu, sig)
    return out

# =============================================================================
# Hybrid t0 Selection
# =============================================================================

def pick_t0_hybrid(
    mjd, band, lamA, *, z, gp_predict, t0_catalog,
    window_days=15.0, min_pts_in_window=5, agree_days=4.0,
    mag=None
) -> tuple[float, dict]:
    """
    Choose reference epoch (t0) using GP-derived peak time with catalog prior.
    
    Strategy:
    1. Use GP to find per-band peak times from flux curves
    2. Prefer rest-frame g/r-like bands (4500-7000 Å rest)
    3. Robust median across selected bands
    4. Compare with catalog peak: if close (≤agree_days), use GP value
    5. If far apart, check data coverage: use GP if strong, else catalog
    
    Parameters
    ----------
    mjd : array
        Observed MJDs
    band : array
        Filter names for each observation
    lamA : array
        Central wavelengths (Å) for each observation
    z : float
        Redshift
    gp_predict : partial
        Fitted GP predictor
    t0_catalog : float or None
        Catalog peak time (MJD), if available
    window_days : float
        Window around t0 for coverage check
    min_pts_in_window : int
        Minimum points required for strong coverage
    agree_days : float
        Agreement threshold (days)
    mag : array, optional
        Magnitudes for fallback if GP fails
    
    Returns
    -------
    t0_used : float
        Selected reference time (MJD)
    info : dict
        Diagnostic information about the selection
    """
    # 1) Per-band prediction frequency from median Cenwave
    med_lam_by_band = (
        pd.DataFrame({"band": band, "lamA": lamA})
        .groupby("band", sort=False)["lamA"].median()
    )
    target_freq_hz = {b: float(angstrom_to_hz(L)) for b, L in med_lam_by_band.items()}
    
    # 2) Dense time grid
    t_dense = np.linspace(np.nanmin(mjd), np.nanmax(mjd), 800)
    
    # 3) GP mean → peak time per band
    pred = gp_predict_surface(gp_predict, t_dense, target_freq_hz)
    t0_by_band = {}
    for b, (f_mu, _f_sig) in pred.items():
        if np.isfinite(f_mu).any():
            i = int(np.nanargmax(f_mu))
            t0_by_band[b] = float(t_dense[i])
    
    # --- Fallback if no peaks found ---
    if not t0_by_band:
        if t0_catalog is not None and np.isfinite(t0_catalog):
            return float(t0_catalog), {
                "t0_cat": float(t0_catalog), "t0_data": np.nan, "delta_days": np.nan,
                "n_pts_win": 0, "snr_med": np.nan, "bands_used": [],
                "source": "catalog_fallback"
            }
        if mag is not None and np.isfinite(mag).any():
            t0_minmag = float(mjd[np.nanargmin(mag)])
            return t0_minmag, {
                "t0_cat": np.nan, "t0_data": t0_minmag, "delta_days": np.nan,
                "n_pts_win": 0, "snr_med": np.nan, "bands_used": [],
                "source": "minmag_fallback"
            }
        t0_mid = float(np.nanmedian(mjd))
        return t0_mid, {
            "t0_cat": np.nan, "t0_data": t0_mid, "delta_days": np.nan,
            "n_pts_win": 0, "snr_med": np.nan, "bands_used": [],
            "source": "median_mjd_fallback"
        }
    
    # 4) Prefer rest-frame g/r-ish bands (4500-7000 Å)
    bands_pref = []
    for b, lam_obs in med_lam_by_band.items():
        lam_rest = float(lam_obs) / (1.0 + float(z))
        if 4500.0 <= lam_rest <= 7000.0:
            bands_pref.append(b)
    bands_used = bands_pref if bands_pref else list(t0_by_band.keys())
    
    # 5) Robust combine
    t_candidates = np.array([t0_by_band[b] for b in bands_used if b in t0_by_band], float)
    t0_data = float(np.nanmedian(t_candidates))
    
    # 6) Coverage metric in window
    center = float(t0_data if (t0_catalog is None or not np.isfinite(t0_catalog)) else t0_catalog)
    in_win = np.abs(mjd - center) <= float(window_days)
    n_pts_win = int(np.isfinite(mjd[in_win]).sum())
    snr_med = np.nan
    strong_coverage = (n_pts_win >= int(min_pts_in_window))
    
    # 7) Decide with prior
    if t0_catalog is None or not np.isfinite(t0_catalog):
        t0_used, source = t0_data, "data_no_prior"
        delta = np.nan
    else:
        delta = float(abs(t0_data - float(t0_catalog)))
        if delta <= float(agree_days):
            t0_used, source = t0_data, "data_agrees_with_prior"
        else:
            if strong_coverage:
                t0_used, source = t0_data, "data_overrode_prior"
            else:
                t0_used, source = float(t0_catalog), "catalog_due_to_weak_coverage"
    
    info = {
        "t0_cat": float(t0_catalog) if (t0_catalog is not None and np.isfinite(t0_catalog)) else np.nan,
        "t0_data": float(t0_data),
        "delta_days": float(delta),
        "n_pts_win": n_pts_win,
        "snr_med": snr_med,
        "bands_used": list(bands_used),
        "source": source,
    }
    return float(t0_used), info
