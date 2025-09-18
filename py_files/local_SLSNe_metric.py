# slsn_gp_lc.py
# Survey-ready SLSN templates from catalog photometry via 2D GP (time, wavelength).

from rubin_sim.maf.metrics import BaseMetric

#from rubin_sim.utils import uniformSphere
#from rubin_sim.data import get_data_dir
from rubin_scheduler.data import get_data_dir #local
from rubin_sim.phot_utils import DustValues

import sys
import os
sys.path.append(os.path.abspath(".."))
from shared_utils import equatorialFromGalactic, uniform_sphere_degrees, inject_uniform_healpix, apply_spectral_index, evaluate, compare_flux_diff_to_error

import matplotlib.pyplot as plt 
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import Galactic, ICRS as ICRSFrame
from astropy.coordinates import SkyCoord
#from rubin_sim.phot_utils import SFDMap
import astropy.units as u
import healpy as hp
from astropy.cosmology import z_at_value
from scipy.stats import truncnorm
import numpy as np
import glob
import os

import pickle 
from pathlib import Path

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import pickle
import warnings

# GP: same library used in Tyler's code
import george
import scipy.optimize as op
from functools import partial

# Cosmology & units for DM(z) and rest-frame scaling
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo


DEBUG = False

# ----------------------------
# Constants & filter mappings
# ----------------------------

# ZTF effective wavelengths (Å) used during training
ZTF_EFF_LAMBDA = {
    "ztfg": 4800.0,
    "ztfr": 6400.0,
    "ztfi": 7900.0,
}

# LSST central frequencies (Hz) for prediction / template standardization
LSST_EFF_FREQ = {
    'u': 8.088e14,
    'g': 6.293e14,
    'r': 4.844e14,
    'i': 3.979e14,
    'z': 3.461e14,
    'y': 3.080e14,
}

# Speed of light
_C_MS = 2.99792458e8

# AB zero-point (Jy) and derivative
F0_JY = 3631.0
LN10_OVER_2P5 = np.log(10.0) / 2.5


# ----------------------------
# Unit helpers
# ----------------------------

def mag_to_flux_jy(mag: np.ndarray) -> np.ndarray:
    """AB mag → flux (Jy)."""
    m = np.asarray(mag, float)
    return F0_JY * 10.0 ** (-0.4 * m)

def magerr_to_fluxerr_jy(mag: np.ndarray, mag_err: np.ndarray) -> np.ndarray:
    """σ_m → σ_f (Jy). First-order, adequate for GP noise model."""
    f = mag_to_flux_jy(mag)
    return LN10_OVER_2P5 * f * np.asarray(mag_err, float)

def flux_jy_to_mag(flux_jy: np.ndarray) -> np.ndarray:
    """Flux (Jy) → AB mag. NaN for non-positive flux."""
    f = np.asarray(flux_jy, float)
    out = np.full_like(f, np.nan, dtype=float)
    good = f > 0
    out[good] = -2.5 * np.log10(f[good] / F0_JY)
    return out

def dm_from_z(z: float) -> float:
    """Distance modulus from Planck18 cosmology."""
    DL = cosmo.luminosity_distance(float(z)).to_value(u.Mpc)
    return 5.0 * np.log10(DL) + 25.0

def angstrom_to_hz(lambda_A: float) -> float:
    """Å → Hz."""
    lam_m = float(lambda_A) * 1e-10
    return _C_MS / lam_m


# ----------------------------
# GP helpers (Tyler-style)
# ----------------------------

def _default_kernel(scale_guess: float) -> george.kernels.Kernel:
    """
    Matérn-3/2 in 2D (time [days], frequency [Hz]).
    Frequency metric element is frozen to stabilize sparse color fits.
    """
    time_scale_days = 20.0
    freq_scale_hz   = 1.0e14
    k = (0.5 * max(scale_guess, 1e-6))**2 * george.kernels.Matern32Kernel(
        [time_scale_days**2, freq_scale_hz**2], ndim=2
    )
    k.freeze_parameter("k2:metric:log_M_1_1")
    return k

def fit_2d_gp(time_obs_days: np.ndarray,
              freq_obs_hz: np.ndarray,
              flux_jy: np.ndarray,
              fluxerr_jy: np.ndarray,
              kernel: george.kernels.Kernel | None = None) -> partial:
    """
    Fit a 2D GP in (time, frequency) to flux data. Returns gp_predict partial(y|X).
    """
    t   = np.asarray(time_obs_days, float)
    nu  = np.asarray(freq_obs_hz, float)
    y   = np.asarray(flux_jy, float)
    yerr= np.asarray(fluxerr_jy, float)

    good = np.isfinite(t) & np.isfinite(nu) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if good.sum() < 5:
        raise ValueError("Insufficient finite points for GP fit.")

    t, nu, y, yerr = t[good], nu[good], y[good], yerr[good]
    X = np.vstack([t, nu]).T

    scale_guess = np.nanmedian(np.abs(y))
    if not np.isfinite(scale_guess) or scale_guess <= 0:
        scale_guess = 1e-3

    if kernel is None:
        kernel = _default_kernel(scale_guess)

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
        res = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", bounds=bounds, tol=1e-6)
        if res.success:
            gp.set_parameter_vector(res.x)
        else:
            warnings.warn(f"GP optimizer failed; using initial parameters. msg={res.message}")
    except Exception as e:
        warnings.warn(f"GP optimization raised {e!r}; using initial parameters.")

    return partial(gp.predict, y)

def gp_predict_surface(gp_predict: partial,
                       t_eval_days: np.ndarray,
                       target_freq_hz: dict[str, float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Predict (flux, sigma) at each target frequency across t_eval_days.
    Returns: {band: (flux_mu[Jy], flux_sigma[Jy])}
    """
    out = {}
    t_eval = np.asarray(t_eval_days, float)
    for band, nu in target_freq_hz.items():
        Xp = np.vstack([t_eval, np.full_like(t_eval, nu)]).T
        mu, var = gp_predict(Xp, return_var=True)
        sig = np.sqrt(np.clip(var, 0.0, np.inf))
        out[band] = (mu, sig)
    return out


# ----------------------------
# Core builder (tighter IO)
# ----------------------------

@dataclass
class CatalogInputs:
    photometry_dir: Path          # per-event CSV/Parquet from export_slsne_photometry.py
    params_table: pd.DataFrame    # parameters.txt as DataFrame
    name_col: str = "name"
    z_col: str = "redshift_med"
    peak_mjd_col: str = "Peak_MJD_med"   # optional

class LC:
    """
    Rubin-ready SLSN light-curve model (templates) with an LC-like interface
    compatible with shared_utils.evaluate(...).

    Stored values:
      - time axis: rest-frame phase (days since t0)
      - mags: absolute magnitudes (AB)

    data: list of templates; each template: {band: {"ph": array, "mag": array}}
    """

    def __init__(self, num_lightcurves=None, load_from=None, lightcurves=None, t_grid=None):
        if lightcurves is not None:
            self.data   = lightcurves
            self.t_grid = t_grid
        elif load_from:
            if not os.path.exists(load_from):
                raise FileNotFoundError(f"SLSN templates not found: {load_from}")
            with open(load_from, "rb") as f:
                obj = pickle.load(f)
            if "lightcurves" not in obj:
                raise ValueError("templates.pkl missing key 'lightcurves'")
            self.data   = obj["lightcurves"]
            self.t_grid = obj.get("t_grid", None)
        else:
            self.data, self.t_grid = [], None

        # for evaluate(); not strictly required but nice to have
        self.filts = ['u', 'g', 'r', 'i', 'z', 'y']

    def interp(self, t, filtername, lc_indx=0):
        """Interpolate absolute magnitude at rest-frame phase t (days) in `filtername`."""
        t   = np.asarray(t, float)
        ph  = np.asarray(self.data[lc_indx][filtername]["ph"],  float)
        mag = np.asarray(self.data[lc_indx][filtername]["mag"], float)

        if ph.size == 0:
            return np.full_like(t, np.nan, dtype=float)

        # Log-time interpolation if support is strictly post-peak (>0)
        if np.all(ph > 0):
            x  = np.log10(np.clip(t, ph.min(), ph.max()))
            xp = np.log10(ph)
            out = np.interp(x, xp, mag, left=np.nan, right=np.nan)
        else:
            out = np.interp(t, ph, mag, left=np.nan, right=np.nan)

        out[(t < ph.min()) | (t > ph.max())] = np.nan
        return out

    # --------- Builder: from catalog (fits GP, saves pickle, returns LC) ---------
    @classmethod
    def from_catalog(cls,
                     inputs: CatalogInputs,
                     *,
                     filename_pattern: str = "{name}.csv",
                     use_ul: bool = False,
                     tpad_pre_days: float = 5.0,
                     tpad_post_days: float = 160.0,
                     n_time: int = 220,
                     kernel: george.kernels.Kernel | None = None,
                     min_points_for_fit: int = 6,
                     save_to: Path | None = None) -> "LC":
        """
        Build SLSN templates from per-event photometry using a 2D GP and parameters.txt
        (redshift + optional peak MJD). Saves {'lightcurves','t_grid'} if save_to is set.
        """
        phot_dir = Path(inputs.photometry_dir)
        params = inputs.params_table.copy()
        params.columns = [str(c).strip() for c in params.columns]

        def resolve_col(df, key):
            if key in df.columns:
                return key
            # soft match by lowercase
            kl = key.lower()
            for c in df.columns:
                if str(c).lower() == kl:
                    return c
            return key  # may raise later if truly absent

        name_col = resolve_col(params, inputs.name_col)
        z_col    = resolve_col(params, inputs.z_col)
        peak_col = inputs.peak_mjd_col if inputs.peak_mjd_col in params.columns else None

        templates = []
        saved_t_grid = None

        for _, row in params.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue

            # redshift
            try:
                z = float(row[z_col])
            except Exception:
                warnings.warn(f"[{name}] missing/invalid redshift; skipping.")
                continue
            if not np.isfinite(z) or z <= 0:
                warnings.warn(f"[{name}] non-positive redshift; skipping.")
                continue

            # locate per-event file
            f_csv  = phot_dir / filename_pattern.format(name=name)
            f_parq = f_csv.with_suffix(".parquet")
            path = f_csv if f_csv.exists() else (f_parq if f_parq.exists() else None)
            if path is None:
                warnings.warn(f"[{name}] no photometry file found.")
                continue

            # load photometry
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            cols = {c.lower().strip(): c for c in df.columns}
            def colget(k): return df[cols[k]]

            try:
                mjd = np.asarray(colget("mjd"), float)
                mag = np.asarray(colget("mag"), float)
                mag_err = np.asarray(colget("mag_err"), float) if "mag_err" in cols else np.full_like(mag, np.nan)
                filt = np.asarray(colget("filter")).astype(str)
                ul   = np.asarray(colget("ul"), bool) if ("ul" in cols and not use_ul) else np.zeros_like(mag, bool)
            except Exception as e:
                warnings.warn(f"[{name}] bad columns: {e!r}")
                continue

            # drop U/L if requested
            keep = ~ul
            mjd, mag, mag_err, filt = mjd[keep], mag[keep], mag_err[keep], filt[keep]

            # drop non-finite rows
            good = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(mag_err)
            if good.sum() < min_points_for_fit:
                warnings.warn(f"[{name}] too few finite points after cuts; skipping.")
                continue
            mjd, mag, mag_err, filt = mjd[good], mag[good], mag_err[good], filt[good]

            # map training bands → frequency (Hz)
            filt_lc = [s.lower().strip() for s in filt]
            nu = np.array([angstrom_to_hz(ZTF_EFF_LAMBDA.get(f, np.nan)) for f in filt_lc], float)
            ok = np.isfinite(nu)
            if ok.sum() < min_points_for_fit:
                warnings.warn(f"[{name}] unsupported filters / no frequencies; skipping.")
                continue
            mjd, mag, mag_err, nu = mjd[ok], mag[ok], mag_err[ok], nu[ok]

            # choose reference epoch t0 (observed frame)
            t0 = None
            if peak_col is not None:
                try:
                    t0 = float(row[peak_col])
                except Exception:
                    t0 = None
            if (t0 is None) or (not np.isfinite(t0)):
                # fallback: time of min mag among ZTF gri
                sel = np.isin(np.array(filt_lc)[ok], ["ztfg", "ztfr", "ztfi"])
                if sel.sum() >= 1:
                    idx_local = np.nanargmin(mag[sel])
                    idx = np.flatnonzero(sel)[idx_local]
                    t0 = mjd[idx]
                else:
                    t0 = mjd[np.nanargmin(mag)]

            # prepare flux domain for GP (fit in observed MJD)
            flux    = mag_to_flux_jy(mag)
            fluxerr = magerr_to_fluxerr_jy(mag, mag_err)

            # fit GP
            try:
                gp_predict = fit_2d_gp(mjd, nu, flux, fluxerr, kernel=kernel)
            except Exception as e:
                warnings.warn(f"[{name}] GP fit failed: {e!r}")
                continue

            # evaluate window around t0 (observed frame), then convert to rest-frame phase
            t_eval_obs = np.linspace(t0 - tpad_pre_days, t0 + tpad_post_days, n_time)
            pred = gp_predict_surface(gp_predict, t_eval_obs, LSST_EFF_FREQ)

            # rest-frame phase & absolute magnitudes
            DM = dm_from_z(z)
            phase = (t_eval_obs - t0) / (1.0 + z)   # days since t0 (rest frame)

            tpl = {}
            for band, (f_mu, _f_sig) in pred.items():
                m_app = flux_jy_to_mag(f_mu)
                M_abs = m_app - DM

                goodp = np.isfinite(phase) & np.isfinite(M_abs)
                ph_b = phase[goodp]
                mag_b = M_abs[goodp]

                # require minimal support
                if ph_b.size < 5:
                    tpl[band] = {"ph": np.array([], float), "mag": np.array([], float)}
                    continue

                # enforce post-peak positive support for log-time interpolation
                floor = 1e-4
                keep_pos = ph_b >= floor
                if keep_pos.sum() >= 3:
                    ph_b  = ph_b[keep_pos]
                    mag_b = mag_b[keep_pos]

                order = np.argsort(ph_b)
                tpl[band] = {"ph": ph_b[order], "mag": mag_b[order]}

            templates.append(tpl)
            saved_t_grid = phase.tolist() if saved_t_grid is None else saved_t_grid

        model = cls(lightcurves=templates, t_grid=saved_t_grid)

        if save_to is not None:
            atomic_save_pickle({"lightcurves": model.data, "t_grid": model.t_grid}, save_to)

        return model

    # ---- optional augmentation ----
    def augment(self,
                n_draws: int,
                mag_shift_range: tuple[float, float] = (-0.5, 0.5),
                time_stretch_range: tuple[float, float] = (0.8, 1.2),
                rng: np.random.Generator | None = None) -> "LC":
        """Simple magnitude shift and time-stretch in rest-frame phase."""
        if rng is None:
            rng = np.random.default_rng(1234)
        new_data = []
        for _ in range(n_draws):
            base = self.data[rng.integers(0, len(self.data))]
            dmag = rng.uniform(*mag_shift_range)
            stretch = rng.uniform(*time_stretch_range)
            tpl = {}
            for band, comp in base.items():
                ph  = np.asarray(comp["ph"], float)
                mag = np.asarray(comp["mag"], float)
                if ph.size < 3:
                    tpl[band] = {"ph": ph.copy(), "mag": mag.copy()}
                    continue
                ph_new  = ph / max(stretch, 1e-6)
                mag_new = mag + dmag
                order = np.argsort(ph_new)
                tpl[band] = {"ph": ph_new[order], "mag": mag_new[order]}
            new_data.append(tpl)
        return LC(lightcurves=(self.data + new_data), t_grid=self.t_grid)


# ----------------------------
# Atomic pickle (safe writes)
# ----------------------------

def atomic_save_pickle(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(path))
        print(f"[INFO] Saved atomically to {path}")
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

