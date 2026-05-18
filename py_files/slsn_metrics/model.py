"""
model.py — LC class for SLSN templates with mag-grid synthesis.

Stores absolute rest-frame templates and provides fast apparent magnitude
synthesis via pre-computed grids or on-the-fly SED integration.
"""
from __future__ import annotations
import os
import pickle
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from rubin_sim.phot_utils import Sed
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor  
from functools import partial  

from .constants import (
    C_MS, C_CM_S, A_TO_CM, JY_TO_CGS, PHASE_BIN_STEP,
    get_lsst_bands, dm_from_z, angstrom_to_hz,
    mag_to_flux_jy, magerr_to_fluxerr_jy, flux_jy_to_mag  
)

from .gp_build import fit_2d_gp, gp_predict_surface, pick_t0_hybrid
from .export_slsne_photometry import canonical_filter

# =============================================================================
# method to map catalog bands to LSST:
# =============================================================================

def map_catalog_to_lsst_band(catalog_band):
    """
    Map catalog photometric band to nearest LSST band.
    
    Common SLSN catalog bands → LSST:
    - B, b → g (both blue, ~440 nm)
    - V, v → r (both visual, ~550 nm)  
    - R, r_catalog → i (both red, ~650 nm)
    - g_catalog → g (same)
    - i_catalog → i (same)
    - z_catalog → z (same)
    """
    mapping = {
        'B': 'g', 'b': 'g',
        'V': 'r', 'v': 'r',
        'R': 'i', 'r': 'i',  # Catalog R is closer to LSST i
        'g': 'g', 'G': 'g',
        'i': 'i', 'I': 'i',
        'z': 'z', 'Z': 'z',
        'u': 'u', 'U': 'u',
        'y': 'y', 'Y': 'y',
    }
    return mapping.get(catalog_band, None)

def get_color_offset(catalog_band, lsst_band):
    """
    Empirical color corrections for catalog → LSST band mapping.
    
    Based on typical SLSN SEDs (hot, blue continuum).
    Returns magnitude offset: m_LSST = m_catalog + offset
    """
    offsets = {
        ('B', 'g'): -0.1,   # B slightly bluer than g
        ('V', 'r'): 0.0,    # V and r are very close
        ('R', 'i'): 0.15,   # R bluer than i, so i is fainter
    }
    return offsets.get((catalog_band, lsst_band), 0.0)

# =============================================================================
# SED synthesis from rest-frame grid
# =============================================================================

def _interp_Fnu_abs_at_phase(sed_grid: dict, phase_rest: float):
    """
    Interpolate stored absolute rest-frame SED at given phase.
    
    sed_grid contains:
      'phase'      : 1D array (days, rest)
      'lam_rest_A' : 1D array (Å, rest)
      'Fnu_abs'    : 2D array [N_phase, N_lambda] in Jy at 10 pc
    
    Returns
    -------
    lam_rest_A, Fnu_abs : arrays or (None, None) if out of range
    """
    ph = np.asarray(sed_grid["phase"], float)
    lam_rest_A = np.asarray(sed_grid["lam_rest_A"], float)
    Fnu_abs_grid = np.asarray(sed_grid["Fnu_abs"], float)
    
    if not (np.isfinite(phase_rest) and ph.min() <= phase_rest <= ph.max()):
        return None, None
    
    Fnu_abs = np.empty_like(lam_rest_A, dtype=float)
    for j in range(lam_rest_A.size):
        Fnu_abs[j] = np.interp(phase_rest, ph, Fnu_abs_grid[:, j], 
                               left=np.nan, right=np.nan)
    
    return lam_rest_A, Fnu_abs

def synthesize_mag_at_z(sed_grid: dict, phase_rest: float, z: float, filt: str) -> float:
    """
    Compute apparent AB magnitude in LSST filter at redshift z.
    
    Uses absolute rest-frame SED (Fν at 10 pc), transforms to observed
    frame Fλ at Earth, and integrates with LSST bandpass.
    
    Parameters
    ----------
    sed_grid : dict
        SED grid with 'phase', 'lam_rest_A', 'Fnu_abs'
    phase_rest : float
        Rest-frame phase (days)
    z : float
        Redshift
    filt : str
        LSST filter name ('u', 'g', 'r', 'i', 'z', 'y')
    
    Returns
    -------
    mag : float
        Apparent AB magnitude (or np.nan if out of support)
    """
    bands = get_lsst_bands()
    if filt not in bands:
        return np.nan
    
    lam_rest_A, Fnu_abs_10pc = _interp_Fnu_abs_at_phase(sed_grid, phase_rest)
    if lam_rest_A is None:
        return np.nan
    
    # Observed frame λ
    lam_obs_A = lam_rest_A * (1.0 + z)
    lam_obs_nm = lam_obs_A / 10.0
    lam_obs_cm = lam_obs_A * A_TO_CM
    
    # Distance scaling
    DL_pc = cosmo.luminosity_distance(float(z)).to_value(u.pc)
    scale = (DL_pc / 10.0)**2 * (1.0 + z)
    Fnu_obs_Jy = Fnu_abs_10pc / scale
    Fnu_obs_cgs = Fnu_obs_Jy * JY_TO_CGS
    Flambda_obs = Fnu_obs_cgs * (C_CM_S / (lam_obs_cm**2))
    
    # Check overlap
    bp = bands[filt]
    overlap_mask = (lam_obs_nm >= bp.wavelen.min()) & (lam_obs_nm <= bp.wavelen.max())
    if np.sum(overlap_mask) < 20:
        return np.nan
    
    # Calculate magnitude
    sed = Sed(wavelen=lam_obs_nm, flambda=Flambda_obs)
    try:
        mag = float(sed.calc_mag(bp))
        return mag if np.isfinite(mag) else np.nan
    except Exception:
        return np.nan

# =============================================================================
# Helper function for parallel grid computation
# =============================================================================

def _compute_grid_slice(i_tpl, sed_grid, z_grid, phase_grid, filters):
    """
    Compute one template's contribution to mag grid.

    This function is at module level (not in class) so it can be pickled
    for multiprocessing.

    Parameters
    ----------
    i_tpl : int
        Template index
    sed_grid : list
        List of SED grids (one per template)
    z_grid : array
        Redshift grid
    phase_grid : array
        Phase grid (rest-frame days)
    filters : list
        Filter names

    Returns
    -------
    i_tpl : int
        Template index (for ordering)
    result : dict
        {filter: grid_slice} for this template
    """
    result = {}
    sed = sed_grid[i_tpl]
    
    for filt in filters:
        grid_slice = np.full((len(z_grid), len(phase_grid)), np.nan, dtype=np.float32)
        for i_z, z in enumerate(z_grid):
            m_app = np.array([synthesize_mag_at_z(sed, ph, z, filt) 
                             for ph in phase_grid])
            # Store apparent mag directly — synthesize_mag_at_z()
            # already includes full luminosity distance scaling.
            # evaluate_slsn() fast path adds only A_filt, same as slow path.
            grid_slice[i_z, :] = m_app
        result[filt] = grid_slice
    
    return i_tpl, result


# =============================================================================
# CatalogInputs dataclass
# =============================================================================

@dataclass
class CatalogInputs:
    """Configuration for building templates from catalog photometry."""
    photometry_dir: Path
    params_table: pd.DataFrame
    name_col: str = "name"
    z_col: str = "redshift_med"
    peak_mjd_col: str = "Peak_MJD_med"

# =============================================================================
# LC class
# =============================================================================

class LC:
    """
    SLSN light-curve model with rest-frame absolute magnitude templates.
    
    Stored format:
      - time axis: rest-frame phase (days since t0)
      - mags: absolute magnitudes (AB)
    
    Provides fast apparent magnitude synthesis via pre-computed magnitude
    grids or on-the-fly SED integration.
    
    Attributes
    ----------
    data : list of dict
        Each template: {band: {"ph": array, "mag": array}}
    t_grid : array or None
        Common phase grid (if applicable)
    names : list of str
        Template names (event IDs)
    sed_grid : list of dict
        Per-template SED grids for synthesis
    mag_grid : dict or None
        Pre-computed magnitude grids {filter: array[tpl, z, phase]}
    """
    
    def __init__(self, num_lightcurves=None, load_from=None,
                 lightcurves=None, t_grid=None, names=None):
        if lightcurves is not None:
            self.data = lightcurves
            self.t_grid = t_grid
            self.names = names if names is not None else [f"tpl_{i}" for i in range(len(self.data))]
            self.template_file = None
            self.sed_grid = None
        elif load_from:
            if not os.path.exists(load_from):
                raise FileNotFoundError(f"Templates not found: {load_from}")
            import joblib
            try:
                obj = joblib.load(load_from)
            except Exception:
                # Fallback for plain pickle files (e.g. legacy GP templates)
                with open(load_from, "rb") as f:
                    obj = pickle.load(f)
            if "lightcurves" not in obj:
                raise ValueError("templates.pkl missing 'lightcurves'")
            self.data = obj["lightcurves"]
            self.t_grid = obj.get("t_grid", None)
            self.names = obj.get("names", [f"tpl_{i}" for i in range(len(self.data))])
            self.template_file = load_from
            self.sed_grid = obj.get("sed_grid", None)
            self.median_cenwave_by_band = obj.get("median_cenwave_by_band", None)
        else:
            self.data, self.t_grid = [], None
            self.names = []
            self.template_file = None
            self.sed_grid = None
    
    def interp(self, t, filtername, lc_indx=0):
        """
        Interpolate magnitude at time(s) t for a given filter.
        
        Handles both scalar and array inputs correctly.
        """
        ph = np.asarray(self.data[lc_indx][filtername]["ph"], dtype=float)
        mag = np.asarray(self.data[lc_indx][filtername]["mag"], dtype=float)
        
        # Always work with arrays internally
        t_arr = np.atleast_1d(t)
        scalar_input = np.ndim(t) == 0  # Check if input was scalar
        
        # Check model type
        if hasattr(self, 'model_type') and self.model_type == 'grb':
            # Log-space interpolation for GRB power-law
            ph_pos = np.clip(ph, 1e-5, None)
            t_pos = np.clip(t_arr, 1e-5, None)
            xp = np.log10(ph_pos)
            x = np.log10(t_pos)
            out = np.interp(x, xp, mag, left=np.nan, right=np.nan)
            out[t_arr < ph_pos.min()] = np.nan
        else:
            # Linear interpolation for SLSN
            out = np.interp(t_arr, ph, mag, left=np.nan, right=np.nan)
            # Set out-of-range values to NaN
            out[(t_arr < ph.min()) | (t_arr > ph.max())] = np.nan
        
        # Return scalar if input was scalar
        return float(out[0]) if scalar_input else out
    
    # =========================================================================
    # Builder: from catalog with GP fitting
    # =========================================================================
    
    @classmethod
    def from_catalog(cls,
                     inputs: CatalogInputs,
                     *,
                     filename_pattern: str = "{name}.csv",
                     use_ul: bool = False,
                     tpad_pre_days: float = 5.0,
                     tpad_post_days: float = 160.0,
                     n_time: int = 220,
                     kernel=None,
                     min_points_for_fit: int = 6,
                     save_to: Path | None = None,
                     **kwargs) -> "LC":
        """
        Build SLSN templates from per-event photometry via 2D GP fitting.
        
        Requires per-row Cenwave (Å) column from {name}_cenwave.csv.
        Band labels preserved from CSV 'filter' column.
        
        Parameters
        ----------
        inputs : CatalogInputs
            Catalog configuration
        filename_pattern : str
            Pattern for per-event files (must contain {name})
        use_ul : bool
            Include upper limits in fitting
        tpad_pre_days, tpad_post_days : float
            Template time padding (observed frame)
        n_time : int
            Number of time samples
        kernel : george.kernels.Kernel, optional
            Custom GP kernel
        min_points_for_fit : int
            Minimum points required per event
        save_to : Path, optional
            Save templates to this file
        
        Returns
        -------
        model : LC
            Template model instance
        """
        
        phot_dir = Path(inputs.photometry_dir)
        params = inputs.params_table.copy()
        params.columns = [str(c).strip() for c in params.columns]
        
        def resolve_col(df, key):
            if key in df.columns:
                return key
            kl = key.lower()
            for c in df.columns:
                if str(c).lower() == kl:
                    return c
            return key
        
        name_col = resolve_col(params, inputs.name_col)
        z_col = resolve_col(params, inputs.z_col)
        peak_col = inputs.peak_mjd_col if inputs.peak_mjd_col in params.columns else None
        
        templates, names, meta, sed_grid = [], [], [], []
        median_cenwave_by_band = {}
        t0_cat_list, t0_data_list, z_used, dm_list = [], [], [], []
        saved_t_grid = None
        
        for _, row in params.iterrows():
            name = str(row[name_col]).strip()
            if not name or name.lower() in {"nan", "none"}:
                continue
            
            try:
                z = float(row[z_col])
            except Exception:
                warnings.warn(f"[{name}] missing/invalid redshift; skipping.")
                continue
            if not np.isfinite(z) or z <= 0:
                warnings.warn(f"[{name}] non-positive redshift; skipping.")
                continue
            
            # Find per-event file with Cenwave
            path = _pick_event_file_with_cenwave(phot_dir, name, filename_pattern)
            if path is None:
                warnings.warn(f"[{name}] no file with 'Cenwave' found; skipping.")
                continue
            
            # Load photometry
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            cols = {c.lower().strip(): c for c in df.columns}
            need = ["mjd", "mag", "filter", "cenwave"]
            miss = [k for k in need if k not in cols]
            if miss:
                warnings.warn(f"[{name}] missing {miss}; skipping.")
                continue
            
            def colget(k): return df[cols[k]]
            
            mjd = pd.to_numeric(colget("mjd"), errors="coerce").to_numpy(float)
            mag = pd.to_numeric(colget("mag"), errors="coerce").to_numpy(float)
            merr = (pd.to_numeric(colget("mag_err"), errors="coerce").to_numpy(float)
                    if "mag_err" in cols else np.full_like(mag, np.nan))
            band = colget("filter").astype(str).to_numpy()
            lamA = pd.to_numeric(colget("cenwave"), errors="coerce").to_numpy(float)
            
            if "ul" in cols and not use_ul:
                ul = colget("ul").astype(bool).to_numpy()
                keep = ~ul
            else:
                keep = np.ones_like(mjd, dtype=bool)
            
            mjd, mag, merr, band, lamA = mjd[keep], mag[keep], merr[keep], band[keep], lamA[keep]
            
            good = np.isfinite(mjd) & np.isfinite(mag) & np.isfinite(merr) & np.isfinite(lamA) & (lamA > 0)
            if good.sum() < min_points_for_fit:
                warnings.warn(f"[{name}] too few points after cuts; skipping.")
                continue
            
            mjd, mag, merr, band, lamA = mjd[good], mag[good], merr[good], band[good], lamA[good]
            
            # Fit 2D GP
            nu = angstrom_to_hz(lamA)
            flux = mag_to_flux_jy(mag)
            fluxerr = magerr_to_fluxerr_jy(mag, merr)
            
            try:
                gp_predict = fit_2d_gp(mjd, nu, flux, fluxerr, kernel=kernel)
            except Exception as e:
                warnings.warn(f"[{name}] GP fit failed: {e!r}")
                continue
            
            # Choose t0
            t0_cat = None
            if peak_col is not None:
                try:
                    t0_val = float(row[peak_col])
                    if np.isfinite(t0_val):
                        t0_cat = t0_val
                except Exception:
                    pass
            
            t0_used, t0_info = pick_t0_hybrid(
                mjd, band, lamA, z=z, gp_predict=gp_predict,
                t0_catalog=t0_cat, window_days=15.0,
                min_pts_in_window=5, agree_days=4.0, mag=mag
            )
            
            t0_cat_list.append(float(t0_info.get("t0_cat", np.nan)))
            t0_data_list.append(float(t0_used))
            z_used.append(float(z))
            dm_list.append(float(dm_from_z(z)))
            
            # Evaluation grid
            t_eval_obs = np.linspace(t0_used - tpad_pre_days, 
                                     t0_used + tpad_post_days, n_time)
            
            # Per-band median Cenwave
            uniq_bands, inv_b = np.unique(band, return_inverse=True)
            med_vals = np.array([np.nanmedian(lamA[inv_b == i]) 
                                 for i in range(uniq_bands.size)], dtype=float)
            med_lam_by_band = dict(zip(uniq_bands.tolist(), med_vals.tolist()))
            median_cenwave_by_band[name] = {str(b): float(L) 
                                             for b, L in med_lam_by_band.items()}
            
            target_freq_hz = {b: float(angstrom_to_hz(L)) 
                              for b, L in med_lam_by_band.items()}
            
            # Predict GP surface
            pred = gp_predict_surface(gp_predict, t_eval_obs, target_freq_hz)
            
            # Rest-frame phase & absolute mags
            DM = dm_from_z(z)
            phase = (t_eval_obs - t0_used) / (1.0 + z)
            
            tpl = {}
            for b, (f_mu, _f_sig) in pred.items():
                m_app = flux_jy_to_mag(f_mu)
                M_abs = m_app - DM
                goodp = np.isfinite(phase) & np.isfinite(M_abs)
                ph_b = phase[goodp]
                mag_b = M_abs[goodp]
                
                if ph_b.size < 5:
                    tpl[b] = {"ph": np.array([], float), "mag": np.array([], float)}
                    continue
                
                # Enforce post-peak positive support
                floor = 1e-4
                keep_pos = ph_b >= floor
                if keep_pos.sum() >= 3:
                    ph_b = ph_b[keep_pos]
                    mag_b = mag_b[keep_pos]
                
                order = np.argsort(ph_b)
                tpl[b] = {"ph": ph_b[order], "mag": mag_b[order]}
            
            meta.append({
                "name": name, "z": float(z), "t0_used": float(t0_used), **t0_info
            })
            
            # Build SED grid
            lam_obs_min = np.nanpercentile(lamA, 5)
            lam_obs_max = np.nanpercentile(lamA, 95)
            lam_rest_min = lam_obs_min / (1.0 + z)
            lam_rest_max = lam_obs_max / (1.0 + z)
            pad = 0.05 * (lam_rest_max - lam_rest_min)
            lam_rest = np.linspace(max(1000.0, lam_rest_min - pad),
                                   min(25000.0, lam_rest_max + pad), 180).astype(float)
            
            lam_obs_for_eval = (1.0 + z) * lam_rest
            nu_obs_for_eval = angstrom_to_hz(lam_obs_for_eval)
            nu_rest_for_eval = angstrom_to_hz(lam_rest)  # Use REST frequencies

            
            F_app = np.empty((t_eval_obs.size, lam_rest.size), dtype=float)
            Var_app = np.empty_like(F_app)
            
            for j, nu_rest_j in enumerate(nu_rest_for_eval):
                # Convert to observer frequency for THIS catalog object
                nu_obs_j = nu_rest_j / (1.0 + z)
                Xp = np.vstack([t_eval_obs, np.full_like(t_eval_obs, nu_obs_j)]).T
                mu_j, var_j = gp_predict(Xp, return_var=True)
                F_app[:, j] = mu_j
            
            DL_pc = cosmo.luminosity_distance(z).to_value(u.pc)
            scale_abs = (DL_pc / 10.0)**2 * (1.0 + z)
            Fnu_abs = scale_abs * F_app
            
            nu_obs_min = np.nanmin(angstrom_to_hz(lamA))
            nu_obs_max = np.nanmax(angstrom_to_hz(lamA))
            coverage = (
                (nu_obs_for_eval[None, :] >= 0.95*nu_obs_min) &
                (nu_obs_for_eval[None, :] <= 1.05*nu_obs_max) &
                np.isfinite(Fnu_abs)
            )


            
            sed_entry = {
                "phase": phase.astype(float),
                "lam_rest_A": lam_rest.astype(float),
                "Fnu_abs": Fnu_abs.astype(float),
                "coverage": coverage.astype(bool),
            }
            
            templates.append(tpl)
            sed_grid.append(sed_entry)
            names.append(name)
            
            if saved_t_grid is None:
                saved_t_grid = phase.tolist()
        
        # Build model
        model = cls(lightcurves=templates, t_grid=saved_t_grid, names=names)
        model.sed_grid = sed_grid
        model.median_cenwave_by_band = median_cenwave_by_band
        
        if save_to is not None:
            tpl_hash = _template_hash(templates, sed_grid, names, saved_t_grid)
            grid_meta = {
                "phase_grid": None, "z_grid": None,
                "grid_version": 1, "phase_bin_step": float(PHASE_BIN_STEP),
            }
            payload = {
                "lightcurves": templates, "t_grid": saved_t_grid, "names": names,
                "sed_grid": sed_grid, "median_cenwave_by_band": median_cenwave_by_band,
                "meta": {
                    "z": z_used, "dm": dm_list, "t0_cat": t0_cat_list,
                    "t0_data": t0_data_list, "template_hash": tpl_hash,
                    "grid_meta": grid_meta,
                },
            }
            _atomic_joblib_dump(payload, save_to)
            model.template_file = str(save_to)
        
        return model
    
    # =========================================================================
    # Magnitude grid building (optional fast path)
    # =========================================================================
    
    def build_magnitude_grid(self, z_grid=None, phase_grid=None,
                             filters='ugrizy', save_to=None,
                             checkpoint_every=50):
        """
        Pre-compute LSST apparent magnitudes on (template, z, phase, filter) grid.
        Enables 100x+ speedup by avoiding repeated SED syntheses.
        
        Parameters
        ----------
        z_grid : array, optional
            Redshift grid for interpolation
        phase_grid : array, optional  
            Phase grid (rest-frame days)
        filters : str or list
            LSST filters to compute
        save_to : Path, optional
            Save final grid to this file
        checkpoint_every : int
            Save checkpoint every N templates (default 50 for parallel batching)
        
        Returns
        -------
        self : LC
            Returns self for chaining
        """
        print("[build_magnitude_grid] Starting...")
        
        if z_grid is None:
            z_grid = np.linspace(0.02, 2.0, 50)
        
        # AUTO-DETECT phase range from SED grid if not specified
        if phase_grid is None:
            if not hasattr(self, 'sed_grid') or not self.sed_grid:
                raise RuntimeError("No SED grid available to auto-detect phase range. "
                                 "Provide phase_grid explicitly or build templates first.")
            
            print("[build_magnitude_grid] Auto-detecting phase range from SED grids...")
            
            all_phase_min = []
            all_phase_max = []
            
            for sed in self.sed_grid:
                if 'phase' in sed:
                    phases = np.asarray(sed['phase'], float)
                    all_phase_min.append(phases.min())
                    all_phase_max.append(phases.max())
            
            if not all_phase_min:
                raise RuntimeError("No phase data found in SED grids")
            
            global_phase_min = min(all_phase_min)
            global_phase_max = max(all_phase_max)
            
            print(f"  Detected SED phase range: [{global_phase_min:.1f}, {global_phase_max:.1f}] days")
            
            # Add 10% padding on each side
            phase_min_padded = global_phase_min - abs(global_phase_min) * 0.1 - 2
            phase_max_padded = global_phase_max + global_phase_max * 0.1 + 10
            
            # Build appropriate grid based on whether we have negative phases
            if phase_min_padded < 0:
                # Split into negative and positive regions
                n_neg = max(30, int(-phase_min_padded))  # At least 30 points for negatives
                n_pos = max(180, int(phase_max_padded))  # At least 180 for positives
                
                phase_grid = np.concatenate([
                    np.linspace(phase_min_padded, -0.1, n_neg),
                    np.geomspace(0.1, phase_max_padded, n_pos)
                ])
                print(f"  Created hybrid grid (linear + geomspace) with {len(phase_grid)} points")
            else:
                # All positive - use geomspace
                phase_grid = np.geomspace(
                    max(0.1, phase_min_padded), 
                    phase_max_padded, 
                    250
                )
                print(f"  Created geomspace grid with {len(phase_grid)} points")
            
            print(f"  Final phase grid: [{phase_grid.min():.1f}, {phase_grid.max():.1f}] days")
        
        # Continue with the rest (NOTICE: back to base indentation level)
        z_grid = np.asarray(z_grid, float)
        phase_grid = np.asarray(phase_grid, float)
    
        if isinstance(filters, str):
            filters = list(filters)
        
        n_templates = len(self.sed_grid)
        n_z = len(z_grid)
        n_phase = len(phase_grid)
        
        # Checkpoint setup
        checkpoint_file = save_to.with_suffix('.checkpoint.pkl') if save_to else None
        start_idx = 0
        
        # Try to resume from checkpoint
        if checkpoint_file and checkpoint_file.exists():
            print(f"[checkpoint] Found existing checkpoint, loading...")
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Validate checkpoint matches current request
            if (checkpoint['filters'] == filters and 
                np.allclose(checkpoint['z_grid'], z_grid) and
                np.allclose(checkpoint['phase_grid'], phase_grid)):
                
                self.mag_grid = checkpoint['mag_grid']
                start_idx = checkpoint['last_completed'] + 1
                print(f"[checkpoint] Resuming from template {start_idx}/{n_templates}")
            else:
                print(f"[checkpoint] Grid parameters changed, starting fresh")
                checkpoint_file.unlink()
                start_idx = 0
        
        # Initialize grid if starting fresh
        if start_idx == 0:
            self.mag_grid = {}
            for filt in filters:
                self.mag_grid[filt] = np.full((n_templates, n_z, n_phase), 
                                              np.nan, dtype=np.float32)
        
        # DMs no longer needed — _compute_grid_slice now stores
        # apparent mags directly (synthesize_mag_at_z includes DL scaling)
        
        # Process in batches with checkpointing
        print(f"[build_magnitude_grid] Processing {n_templates - start_idx} templates")
        print(f"[build_magnitude_grid] Using {min(4, n_templates)} parallel workers")
        print(f"[build_magnitude_grid] Checkpointing every {checkpoint_every} templates")
        
        # Create batches
        batch_size = checkpoint_every
        remaining = list(range(start_idx, n_templates))
        
        for batch_start in range(0, len(remaining), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining))
            batch_indices = remaining[batch_start:batch_end]
            
            print(f"[batch] Processing templates {batch_indices[0]}-{batch_indices[-1]}")
            
            with ProcessPoolExecutor(max_workers=min(4, len(batch_indices))) as executor:
                # Create worker function with fixed parameters
                worker = partial(
                    _compute_grid_slice,
                    sed_grid=self.sed_grid,
                    z_grid=z_grid,
                    phase_grid=phase_grid,
                    filters=filters,
                            )
                
                # Process batch in parallel with progress bar
                results = list(tqdm(
                    executor.map(worker, batch_indices),
                    total=len(batch_indices),
                    desc=f"Batch {batch_start//batch_size + 1}",
                    unit="tpl"
                ))
                
                # Collect batch results
                for i_tpl, result in results:
                    for filt in filters:
                        self.mag_grid[filt][i_tpl, :, :] = result[filt]
            
            # Checkpoint after batch completion
            if checkpoint_file:
                checkpoint = {
                    'mag_grid': self.mag_grid,
                    'last_completed': batch_indices[-1],
                    'z_grid': z_grid,
                    'phase_grid': phase_grid,
                    'filters': filters,
                    'n_templates': n_templates
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"[checkpoint] Saved at template {batch_indices[-1]}/{n_templates}")
        
        self.mag_grid_axes = {'z': z_grid, 'phase': phase_grid}
        print(f"[build_magnitude_grid] Complete. Grid shape: {self.mag_grid[filters[0]].shape}")

        self._build_interpolators()

        if save_to:
            grid_data = {
                'mag_grid': self.mag_grid,
                'mag_grid_axes': self.mag_grid_axes,
                'filters': filters
            }
            atomic_save_pickle(grid_data, save_to)
            print(f"[build_magnitude_grid] Saved to {save_to}")

        # Clean up checkpoint only after successful save
        if checkpoint_file and checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"[checkpoint] Removed (build complete, grid saved)")
        
        return self
    
    def load_magnitude_grid(self, grid_file):
        """Load pre-computed magnitude grid from disk."""
        if not os.path.exists(grid_file):
            raise FileNotFoundError(f"Grid file not found: {grid_file}")
        with open(grid_file, 'rb') as f:
            grid_data = pickle.load(f)
        self.mag_grid = grid_data['mag_grid']
        self.mag_grid_axes = grid_data['mag_grid_axes']
        print(f"[load_magnitude_grid] Loaded with filters: {list(self.mag_grid.keys())}")
        self._build_interpolators()
        return self
    
    def _build_interpolators(self):
        """Build RegularGridInterpolators (once per process, not serialized)."""
        if not hasattr(self, "mag_grid") or not self.mag_grid:
            raise RuntimeError("mag_grid not available. Build or load first.")
        
        z_axis = self.mag_grid_axes['z']
        # Ensure phase grid is strictly ascending (no duplicates at segment junctions)
        ph_axis = np.unique(self.mag_grid_axes['phase'])
        
        interps = {}
        for filt, cube in self.mag_grid.items():
            n_tpl = cube.shape[0]
            interps[filt] = [
                RegularGridInterpolator(
                    (z_axis, ph_axis), cube[i_tpl, :, :],
                    bounds_error=False, fill_value=np.nan, method='linear'
                )
                for i_tpl in range(n_tpl)
            ]
        self._interps = interps

# =============================================================================
# Template utilities
# =============================================================================

def _load_names(templates_or_path):
    """Extract template names from LC instance or pickle file."""
    if hasattr(templates_or_path, "names"):
        return list(getattr(templates_or_path, "names", []))
    with open(templates_or_path, "rb") as f:
        obj = pickle.load(f)
    return list(obj.get("names", []))

def template_index_for_event(templates_or_path, event_name: str) -> int:
    """Get template index for named event."""
    names = _load_names(templates_or_path)
    if not names:
        raise RuntimeError("No template names available.")
    try:
        return names.index(event_name)
    except ValueError as e:
        raise ValueError(f"'{event_name}' not found. Available: {len(names)} templates.") from e

def template_name_for_index(templates_or_path, idx: int) -> str:
    """Get event name for template index."""
    names = _load_names(templates_or_path)
    if not names:
        return f"tpl_{idx}"
    if not (0 <= idx < len(names)):
        raise IndexError(f"Index {idx} out of range (0..{len(names)-1}).")
    return names[idx]

def get_median_cenwave_map(templates_or_path, event_name):
    """Return {band: Cenwave_A} for an event if cached; else {}."""
    if hasattr(templates_or_path, "median_cenwave_by_band"):
        m = getattr(templates_or_path, "median_cenwave_by_band", None)
        return m.get(event_name, {}) if isinstance(m, dict) else {}
    if isinstance(templates_or_path, (str, os.PathLike)):
        with open(templates_or_path, "rb") as f:
            obj = pickle.load(f)
        m = obj.get("median_cenwave_by_band", {})
        return m.get(event_name, {}) if isinstance(m, dict) else {}
    return {}

def list_t0_meta(templates_file):
    """Extract t0 selection metadata from templates file."""
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    
    meta = obj.get("meta", {})
    names = obj.get("names", [])
    
    # Handle different metadata formats
    if isinstance(meta, dict) and not isinstance(meta, list):
        # Format: {'z': [...], 'dm': [...], 't0_cat': [...]}
        # Convert to list of dicts
        
        n_templates = len(names) if names else len(meta.get('z', []))
        
        rows = []
        for i in range(n_templates):
            row = {'name': names[i] if i < len(names) else f'tpl_{i}'}
            
            # Extract each field for this template
            for key in ['z', 'dm', 't0_cat', 't0_data']:
                if key in meta and i < len(meta[key]):
                    row[key] = meta[key][i]
                else:
                    row[key] = np.nan
            
            # Determine t0_used (prefer t0_data, fallback to t0_cat)
            if 't0_data' in row and np.isfinite(row['t0_data']):
                row['t0_used'] = row['t0_data']
            elif 't0_cat' in row and np.isfinite(row['t0_cat']):
                row['t0_used'] = row['t0_cat']
            else:
                row['t0_used'] = np.nan
            
            row['source'] = 'catalog' if np.isfinite(row.get('t0_cat')) else 'unknown'
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
    
    elif isinstance(meta, list):
        # Format: [{'name': ..., 'z': ..., ...}, ...]
        if not meta:
            # Create minimal fallback
            meta = [{"name": name, "z": np.nan, "t0_used": np.nan, "source": "unknown"} 
                    for name in names]
        df = pd.DataFrame(meta)
    
    else:
        # Empty or unknown format - create minimal DataFrame
        df = pd.DataFrame([{"name": name, "z": np.nan, "t0_used": np.nan, "source": "unknown"} 
                           for name in names])
    
    # Sort by name if column exists
    if "name" in df.columns and not df.empty:
        df = df.sort_values("name").reset_index(drop=True)
    
    return df

# =============================================================================
# Helper file with Cenwave support
# =============================================================================

def _pick_event_file_with_cenwave(phot_dir: Path, name: str, 
                                   filename_pattern: str) -> Path | None:
    """
    Find per-event file with 'Cenwave' column.
    Priority: {name}_cenwave.csv > {name}_cenwave.parquet > pattern match with Cenwave.
    """
    phot_dir = Path(phot_dir)
    
    def has_cenwave(p: Path) -> bool:
        if not p.exists():
            return False
        try:
            df = (pd.read_parquet(p) if p.suffix == ".parquet" 
                  else pd.read_csv(p, nrows=5))
            return any(c.lower() == "cenwave" for c in map(str, df.columns))
        except Exception:
            return False
    
    direct = phot_dir / filename_pattern.format(name=name)
    if has_cenwave(direct):
        return direct
    
    c1 = phot_dir / f"{name}_cenwave.csv"
    c2 = phot_dir / f"{name}_cenwave.parquet"
    if has_cenwave(c1): return c1
    if has_cenwave(c2): return c2
    
    return direct if has_cenwave(direct) else None

# =============================================================================
# Safe persistence
# =============================================================================

def atomic_save_pickle(obj, path: Path):
    """Atomic write to avoid corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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

def _atomic_joblib_dump(obj, path, compress=("zstd", 3)):
    """Prefer joblib+zstd for speed; fallback to pickle."""
    try:
        import joblib
        from pathlib import Path
        tmp = str(Path(path).with_suffix(Path(path).suffix + ".tmp"))
        joblib.dump(obj, tmp, compress=compress)
        os.replace(tmp, str(path))
        print(f"[INFO] Saved (joblib,zstd) to {path}")
    except Exception:
        atomic_save_pickle(obj, path)

def _template_hash(lightcurves, sed_grid, names, t_grid, 
                   *, algo="blake2b", digest_size=16):
    """Generate hash of template payload for versioning."""
    import hashlib
    payload = pickle.dumps(
        {"lcs": lightcurves, "sed": sed_grid, "names": names, "t_grid": t_grid},
        protocol=pickle.HIGHEST_PROTOCOL
    )
    h = (hashlib.blake2b(payload, digest_size=digest_size) if algo == "blake2b" 
         else hashlib.sha256(payload))
    return h.hexdigest()
