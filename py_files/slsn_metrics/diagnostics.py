"""
diagnostics.py — QA checks and plotting for SLSN templates.

Provides residual analysis, GP refitting diagnostics, and visualization tools.
Heavy GP refits are disabled by default to avoid O(N³) cost during production.
"""
import pickle
import warnings
from pathlib import Path
import numpy as np
import os
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt
from .constants import angstrom_to_hz, dm_from_z, mag_to_flux_jy, magerr_to_fluxerr_jy, flux_jy_to_mag
from .model import get_median_cenwave_map
from scipy.stats import linregress
from .gp_build import fit_2d_gp
    


# Policy: keep heavy GP refits off the hot path
ALLOW_GPFIT_IN_DIAGNOSTICS = False

__all__ = [
    "diagnose_abs_from_templates",
    "diagnose_internal_consistency",
    "plot_residuals",
    "plot_event_obs",
    "plot_event_model",
    "plot_event_obs_vs_model",
    "list_event_bands",
    "list_template_bands",
    "characterize_template_coverage",
    "compute_slsn_properties",
    "assess_literature_coverage",
    "find_missing_archetypes",
    "plot_rate_evolution_comparison",
    "compare_simulated_vs_observed_rates",
    "plot_population_rate_vs_redshift",
    "plot_detection_diagnostics",
    "plot_sky_detection",
    "plot_mc_rate_uncertainty",
    "plot_mc_rate_uncertainty_panel",
]

# =============================================================================
# Helper functions
# =============================================================================

def list_event_bands(per_event_dir: Path, event_name: str) -> list[str]:
    """List distinct filter strings in {event}_cenwave.csv."""
    p = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(p)
    cols = {c.lower(): c for c in df.columns}
    key = cols.get("filter")
    if key is None:
        raise KeyError(f"{p.name} has no 'filter' column (found {list(df.columns)})")
    return sorted(df[key].astype(str).unique().tolist())

def list_template_bands(templates_file: Path, template_idx: int) -> list[str]:
    """List band keys present in a saved template."""
    obj = pickle.load(open(templates_file, "rb"))
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]
    return sorted([k for k, v in tpl.items() 
                   if isinstance(v, dict) and {"ph", "mag"} <= set(v)])

#=============================================================================
# Diagnostic 0: Validate Template Quality
#=============================================================================
def validate_template_quality(templates_file: Path, min_bands=3, min_phase_span=20):
    """
    Flag low-quality templates.
    
    Parameters
    ----------
    templates_file : Path
        Templates pickle file
    min_bands : int
        Minimum number of bands required
    min_phase_span : float
        Minimum phase coverage (days)
    
    Returns
    -------
    issues : DataFrame
        Templates failing quality checks
    """
    df = characterize_template_coverage(templates_file, save_summary=False)
    
    issues = []
    for _, row in df.iterrows():
        problems = []
        if row['n_bands'] < min_bands:
            problems.append(f"only {row['n_bands']} bands")
        if row.get('phase_span', 0) < min_phase_span:
            problems.append(f"phase span {row.get('phase_span', 0):.1f}d")
        
        if problems:
            issues.append({
                'name': row['name'],
                'template_idx': row['tpl_idx'],
                'issues': '; '.join(problems)
            })
    
    return pd.DataFrame(issues) if issues else pd.DataFrame()

# =============================================================================
# Diagnostic 1: Observed vs. (template + DM)
# =============================================================================

def diagnose_abs_from_templates(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    *,
    z: float,
    t0: float,
    bands: list[str] | None = None,
    case_sensitive: bool = True,
    use_log_time_for_model: bool = False,
    return_with_phase: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute residual = m_obs - (M_template(phase) + DM(z)).
    
    Expect ~0 mag if absolute magnitude construction is correct.
    
    Parameters
    ----------
    event_name : str
        Event identifier
    per_event_dir : Path
        Directory with {event}_cenwave.csv
    templates_file : Path
        Saved templates pickle
    template_idx : int
        Template index to compare
    z : float
        Redshift
    t0 : float
        Reference time (MJD)
    bands : list, optional
        Bands to compare (default: all shared)
    case_sensitive : bool
        Match band names case-sensitively
    use_log_time_for_model : bool
        Use log-time interpolation for model
    return_with_phase : bool
        Include phase column for vs-phase plotting
    verbose : bool
        Print summary statistics
    
    Returns
    -------
    df : DataFrame
        Columns: band, resid_mag, [phase]
    """
    # Load model
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]
    
    # Load observations
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    for need in ("mjd", "mag", "filter"):
        if need not in cols:
            raise ValueError(f"{csv.name} missing column '{need}'")
    
    mjd = df[cols["mjd"]].to_numpy(float)
    mobs = df[cols["mag"]].to_numpy(float)
    fobs = df[cols["filter"]].astype(str)
    phase_obs = (mjd - float(t0)) / (1.0 + float(z))
    
    # Bands to compare
    model_bands = [k for k, v in tpl.items() 
                   if isinstance(v, dict) and {"ph", "mag"} <= set(v)]
    want = bands or sorted(set(model_bands) | set(np.unique(fobs)))
    
    DM = dm_from_z(z)
    rows = []
    
    for b in want:
        # Model component
        key = (b if case_sensitive else 
               next((k for k in model_bands if k.lower() == str(b).lower()), None))
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph", "mag"} <= set(comp)):
            continue
        ph = np.asarray(comp["ph"], float)
        Ma = np.asarray(comp["mag"], float)
        okm = np.isfinite(ph) & np.isfinite(Ma)
        if okm.sum() < 2:
            continue
        
        # Obs in this band
        mask = (fobs == b) if case_sensitive else (fobs.str.lower() == str(b).lower())
        if not mask.any():
            continue
        ph_i = phase_obs[mask]
        m_i = mobs[mask]
        
        # Interpolate model
        if use_log_time_for_model:
            pos = okm & (ph > 0)
            if pos.sum() < 2:
                continue
            xp = np.log10(ph[pos])
            fp = Ma[pos]
            sel = ph_i > 0
            Mi = np.full_like(ph_i, np.nan, float)
            Mi[sel] = np.interp(
                np.log10(np.clip(ph_i[sel], 10**xp.min(), 10**xp.max())), 
                xp, fp
            )
        else:
            xp, fp = ph[okm], Ma[okm]
            Mi = np.interp(np.clip(ph_i, xp.min(), xp.max()), xp, fp, 
                          left=np.nan, right=np.nan)
        
        resid = m_i - (Mi + DM)
        for rr, phv in zip(resid, ph_i):
            rows.append((b, float(rr), float(phv)))
    
    if not rows:
        if verbose:
            print("[diagnose_abs_from_templates] No overlapping points.")
        return pd.DataFrame(columns=["band", "resid_mag", "phase"])
    
    out = pd.DataFrame(rows, columns=["band", "resid_mag", "phase"])
    if not return_with_phase:
        out = out[["band", "resid_mag"]]
    
    if verbose:
        g = out.groupby("band")["resid_mag"]
        summ = pd.DataFrame({
            "N": g.size(), "median": g.median(), 
            "mean": g.mean(), "std": g.std()
        })
        print("[abs-from-templates] m_obs - (M_abs + DM) [mag]:")
        print(summ.sort_index())
        print("Global median:", np.nanmedian(out["resid_mag"]))
    
    return out

# =============================================================================
# Diagnostic 2: GP refit consistency (DISABLED by default)
# =============================================================================

def diagnose_internal_consistency(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    *,
    z: float,
    t0: float,
    n_time: int = 220,
    verbose: bool = True,
    refit_gp: bool | None = None,
) -> pd.DataFrame:
    """
    Re-fit GP and compare m_GP vs (M_template + DM).
    
    DISABLED by default to avoid O(N³) fits during production.
    Use diagnose_abs_from_templates for lightweight checks.
    
    Parameters
    ----------
    refit_gp : bool, optional
        If None, uses global ALLOW_GPFIT_IN_DIAGNOSTICS setting
    
    Returns
    -------
    df : DataFrame
        Residuals by band and phase
    """
    if refit_gp is None:
        refit_gp = ALLOW_GPFIT_IN_DIAGNOSTICS
    
    if not refit_gp:
        if verbose:
            print("[diagnose_internal_consistency] Skipped (refit_gp=False). "
                  "Use diagnose_abs_from_templates for lightweight check.")
        return pd.DataFrame(columns=["band", "resid_mag", "phase"])
    
    # Load template
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs = obj.get("lightcurves", [])
    if not (0 <= template_idx < len(lcs)):
        raise IndexError("template_idx out of range")
    tpl = lcs[template_idx]
    
    # Load event data
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    cols = {c.lower(): c for c in df.columns}
    need = ("mjd", "mag", "mag_err", "filter", "cenwave")
    for k in need:
        if k not in cols:
            raise ValueError(f"{csv.name} missing column '{k}'")
    
    mjd = df[cols["mjd"]].to_numpy(float)
    mag = df[cols["mag"]].to_numpy(float)
    merr = df[cols["mag_err"]].to_numpy(float)
    band = df[cols["filter"]].astype(str)
    lamA = df[cols["cenwave"]].to_numpy(float)
    
    # Flux domain
    flux = mag_to_flux_jy(mag)
    fluxerr = magerr_to_fluxerr_jy(mag, merr)
    nu = angstrom_to_hz(lamA)
    
    # Fit GP
    good = (np.isfinite(mjd) & np.isfinite(nu) & 
            np.isfinite(flux) & np.isfinite(fluxerr) & (fluxerr > 0))
    mjd_, nu_, y_, yerr_ = mjd[good], nu[good], flux[good], fluxerr[good]
    
    try:
        gp_predict = fit_2d_gp(mjd_, nu_, y_, yerr_)
    except Exception as e:
        warnings.warn(f"GP fit failed: {e!r}")
        return pd.DataFrame(columns=["band", "resid_mag", "phase"])
    
    # Evaluation grid
    t_eval = np.linspace(mjd_.min(), mjd_.max(), int(n_time))
    
    # Per-band median Cenwave
    med_map = get_median_cenwave_map(templates_file, event_name)
    if med_map:
        med_lam = pd.Series(med_map)
    else:
        med_lam = (pd.DataFrame({"b": band, "lam": lamA})
                   .groupby("b", sort=False)["lam"].median())
    
    rows = []
    DM = dm_from_z(z)
    
    for b, L in med_lam.items():
        nu0 = angstrom_to_hz(L)
        Xp = np.vstack([t_eval, np.full_like(t_eval, nu0)]).T
        mu, var = gp_predict(Xp, return_var=True)
        
        # Flux -> mag
        m_gp = np.full_like(mu, np.nan, float)
        ok = mu > 0
        m_gp[ok] = flux_jy_to_mag(mu[ok])
        
        # Template abs mags
        comp = tpl.get(b) or next((tpl[k] for k in tpl 
                                    if k.lower() == str(b).lower()), None)
        if not (isinstance(comp, dict) and {"ph", "mag"} <= set(comp)):
            continue
        ph = np.asarray(comp["ph"], float)
        Ma = np.asarray(comp["mag"], float)
        okm = np.isfinite(ph) & np.isfinite(Ma)
        if okm.sum() < 2:
            continue
        
        phase_eval = (t_eval - float(t0)) / (1.0 + float(z))
        M_on_grid = np.interp(
            np.clip(phase_eval, ph[okm].min(), ph[okm].max()), 
            ph[okm], Ma[okm], left=np.nan, right=np.nan
        )
        resid = m_gp - (M_on_grid + DM)
        
        for r, phv in zip(resid, phase_eval):
            if np.isfinite(r):
                rows.append((b, float(r), float(phv)))
    
    out = pd.DataFrame(rows, columns=["band", "resid_mag", "phase"])
    
    if verbose and not out.empty:
        g = out.groupby("band")["resid_mag"]
        summ = pd.DataFrame({"N": g.size(), "median": g.median(), "std": g.std()})
        print("[internal] m_GP - (M_abs + DM) [mag]:")
        print(summ.sort_index())
        print("Global median:", np.nanmedian(out["resid_mag"]))
    
    return out

# =============================================================================
# Residual plotting
# =============================================================================

def plot_residuals(df: pd.DataFrame, *, by_band=True, vs_phase=False, 
                   title=None):
    """
    Plot residuals from diagnostic functions.
    
    Parameters
    ----------
    df : DataFrame
        Output from diagnose_abs_from_templates or diagnose_internal_consistency
    by_band : bool
        Show per-band summary
    vs_phase : bool
        Plot residual vs phase (requires 'phase' column)
    title : str, optional
        Plot title
    """
    if df is None or df.empty:
        print("[plot_residuals] nothing to plot.")
        return
    
    if vs_phase and "phase" not in df.columns:
        print("[plot_residuals] 'phase' column not present; plotting by band only.")
        vs_phase = False
    
    if vs_phase:
        bands = sorted(df["band"].unique())
        cmap = plt.get_cmap("tab20")
        color_map = {b: cmap(i % 20) for i, b in enumerate(bands)}
        
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for b in bands:
            sub = df[df["band"] == b]
            ax.scatter(sub["phase"], sub["resid_mag"], s=12, alpha=0.6, 
                       label=b, facecolors="none",
                       edgecolors=color_map[b], linewidths=0.8)
            
            # Running median
            if len(sub) >= 8:
                q = sub.sort_values("phase")
                k = max(5, len(q) // 12)
                med = q["resid_mag"].rolling(
                    window=k, center=True, min_periods=max(3, k//2)
                ).median()
                ax.plot(q["phase"], med, alpha=0.9, lw=1.5, color=color_map[b])
        
        ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
        ax.set_xlabel("Rest-frame Phase [days]")
        ax.set_ylabel("Residual  m_obs − (M + DM)  [mag]")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncol=min(8, len(bands)))
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
        return
    
    if by_band:
        g = df.groupby("band")["resid_mag"]
        summ = pd.DataFrame({
            "N": g.size(), "median": g.median(), "std": g.std()
        }).sort_index()
        
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.errorbar(summ.index, summ["median"], yerr=summ["std"], 
                    fmt="o", capsize=3)
        ax.axhline(0, ls="--", lw=1, color="k", alpha=0.5)
        ax.set_ylabel("Residual  m_obs − (M + DM)  [mag]")
        ax.set_xlabel("Band")
        ax.grid(True, axis="y", alpha=0.25)
        if title:
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

# =============================================================================
# Observation/Model plotting
# =============================================================================

def plot_event_obs(
    event_name: str,
    per_event_dir: Path,
    output_dir=None,
    *,
    use_phase: bool = False,
    z: float | None = None,
    t0: float | None = None,
    bands: list[str] | None = None,
    case_sensitive: bool = True,
    scale_to_peak: bool = False,
    s_obs: float = 14.0,
    alpha_obs: float = 0.8,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
):
    """Plot observations only from {event}_cenwave.csv."""
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    
    # Create case-insensitive column mapping
    cols = {c.lower(): c for c in df.columns}
    
    # Validate required columns
    need = {"mjd", "mag", "filter"}
    if not need.issubset(cols):
        raise ValueError(f"{csv.name} missing {need - set(cols)}. Found: {list(df.columns)}")
    
    # Access columns using the mapping
    x = df[cols["mjd"]].to_numpy(float)
    y = df[cols["mag"]].to_numpy(float)
    f = df[cols["filter"]].astype(str)
    
    if use_phase:
        if z is None or t0 is None:
            raise ValueError("use_phase=True requires z and t0.")
        x = (x - float(t0)) / (1.0 + float(z))
    
    if scale_to_peak and np.isfinite(y).any():
        y = y - np.nanmin(y)
    
    band_vals = np.unique(f) if bands is None else np.array(bands, dtype=str)
    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(band_vals)}
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for b in band_vals:
        mask = (f == b) if case_sensitive else (f.str.lower() == str(b).lower())
        if not mask.any():
            continue
        xi, yi = x[mask], y[mask]
        ok = np.isfinite(xi) & np.isfinite(yi)
        if ok.sum() == 0:
            continue
        ax.scatter(xi[ok], yi[ok], s=s_obs, alpha=alpha_obs, 
                   facecolors="none", edgecolors=color_map[b], 
                   linewidths=0.9, label=b)
    
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Rest-frame Phase [days]" if use_phase else "MJD")
    ax.set_ylabel("Scaled Apparent Mag" if scale_to_peak else "Apparent Mag")
    ax.legend(frameon=False, ncol=min(8, len(band_vals)))
    ax.set_title(title or event_name)
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"Catalog_{title or event_name}.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_event_model(
    templates_file: Path,
    template_idx: int,
    output_dir=None,
    *,
    bands: list[str] | None = None,
    case_sensitive: bool = True,
    use_log_time: bool = False,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    lw: float = 1.6,
    alpha: float = 0.9,
):
    """Plot template curves (absolute mag vs rest-frame phase)."""
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    
    lcs = obj.get("lightcurves", [])
    names = obj.get("names", [f"tpl_{i}" for i in range(len(lcs))])
    
    if not (0 <= template_idx < len(lcs)):
        raise IndexError(f"template_idx out of range (0..{len(lcs)-1}).")
    
    tpl = lcs[template_idx]
    
    def keys_for_plot():
        ks = []
        for k, comp in tpl.items():
            if isinstance(comp, dict) and {"ph", "mag"} <= set(comp):
                ks.append(k)
        return ks
    
    draw_keys = bands if bands is not None else keys_for_plot()
    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(draw_keys)}
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for b in draw_keys:
        key = (b if case_sensitive else 
               next((k for k in tpl if str(k).lower() == str(b).lower()), None))
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph", "mag"} <= set(comp)):
            continue
        t = np.asarray(comp["ph"], float)
        m = np.asarray(comp["mag"], float)
        ok = np.isfinite(t) & np.isfinite(m)
        if use_log_time:
            ok &= (t > 0)
        if ok.sum() < 2:
            continue
        x = np.log10(t[ok]) if use_log_time else t[ok]
        y = m[ok]
        ax.plot(x, y, color=color_map[b], lw=lw, alpha=alpha, label=b)
    
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("log10(Phase [days])" if use_log_time else "Rest-frame Phase [days]")
    ax.set_ylabel("Absolute Magnitude (AB)")
    ax.legend(frameon=False, ncol=min(8, len(draw_keys)))
    ax.set_title(title or f"{names[template_idx]} (template #{template_idx})")
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"model.png"), dpi=150, bbox_inches="tight")
    plt.show()

def plot_event_obs_vs_model(
    event_name: str,
    per_event_dir: Path,
    templates_file: Path,
    template_idx: int,
    output_dir=None,
    *,
    use_phase: bool = True,
    z: float | None = None,
    t0: float | None = None,
    bands: list[str] | None = None,
    case_sensitive: bool = True,
    scale_to_peak: bool = False,
    use_log_time_for_model: bool = False,
    ylim: tuple[float, float] | None = None,
    s_obs: float = 12.0,
    lw_model: float = 1.6,
):
    """Overlay observations (points) with template curves (lines)."""
    # Load model
    with open(templates_file, "rb") as f:
        obj = pickle.load(f)
    lcs = obj.get("lightcurves", [])
    tpl = lcs[template_idx]
    
    # Load observations
    csv = Path(per_event_dir) / f"{event_name}_cenwave.csv"
    df = pd.read_csv(csv)
    
    # Create case-insensitive column mapping
    cols = {c.lower(): c for c in df.columns}
    
    # Validate columns
    need = {"mjd", "mag", "filter"}
    if not need.issubset(cols):
        raise ValueError(f"{csv.name} missing {need - set(cols)}")
    
    # Access using mapping
    x_obs = df[cols["mjd"]].to_numpy(float)
    y_obs = df[cols["mag"]].to_numpy(float)
    f_obs = df[cols["filter"]].astype(str)
    
    if use_phase:
        if z is None or t0 is None:
            raise ValueError("use_phase=True requires z and t0.")
        x_obs = (x_obs - float(t0)) / (1.0 + float(z))
    
    if scale_to_peak and np.isfinite(y_obs).any():
        y_obs = y_obs - np.nanmin(y_obs)
    
    model_keys = [k for k, comp in tpl.items() 
                  if isinstance(comp, dict) and {"ph", "mag"} <= set(comp)]
    want = bands or sorted(set(model_keys) | set(np.unique(f_obs)))
    cmap = plt.get_cmap("tab20")
    color_map = {b: cmap(i % 20) for i, b in enumerate(want)}
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot obs
    for b in want:
        mask = (f_obs == b) if case_sensitive else (f_obs.str.lower() == str(b).lower())
        if mask.any():
            xi, yi = x_obs[mask], y_obs[mask]
            ok = np.isfinite(xi) & np.isfinite(yi)
            if ok.sum():
                ax.scatter(xi[ok], yi[ok], s=s_obs, alpha=0.75,
                           facecolors="none", edgecolors=color_map[b],
                           linewidths=0.9, label=f"{b} (obs)")
    
    # Plot model
    for b in want:
        key = (b if case_sensitive else 
               next((k for k in tpl if str(k).lower() == str(b).lower()), None))
        comp = tpl.get(key)
        if not (isinstance(comp, dict) and {"ph", "mag"} <= set(comp)):
            continue
        t = np.asarray(comp["ph"], float)
        m = np.asarray(comp["mag"], float)
        ok = np.isfinite(t) & np.isfinite(m)
        if use_log_time_for_model:
            ok &= (t > 0)
        if ok.sum() == 0:
            continue
        x = np.log10(t[ok]) if use_log_time_for_model else t[ok]
        y = m[ok]
        ax.plot(x, y, color=color_map[b], lw=lw_model, alpha=0.9, 
                label=f"{b} (model)")
    
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Rest-frame Phase [days]" if use_phase else "MJD")
    ax.set_ylabel("Scaled Apparent Mag" if scale_to_peak else "Magnitude")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.set_title(f"{event_name}  —  template #{template_idx}")
    plt.tight_layout()
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"combined_plot.png"), dpi=150, bbox_inches="tight")
    plt.show()

# =============================================================================
# Template coverage analysis
# =============================================================================

def characterize_template_coverage(templates_file, save_summary=True):
    """Fast summary of template characteristics."""
    with open(templates_file, 'rb') as f:
        obj = pickle.load(f)
    
    lcs = obj['lightcurves']
    names = obj.get('names', [f'tpl_{i}' for i in range(len(lcs))])
    meta = obj.get('meta', {})
    
    z_arr = meta.get('z', [])
    dm_arr = meta.get('dm', [])
    t0_cat_arr = meta.get('t0_cat', [])
    t0_data_arr = meta.get('t0_data', [])
    
    rows = []
    for i, (name, tpl) in enumerate(zip(names, lcs)):
        row = {
            'tpl_idx': i, 'name': name,
            'z': z_arr[i] if i < len(z_arr) else np.nan,
            'dm': dm_arr[i] if i < len(dm_arr) else np.nan,
            't0_cat': t0_cat_arr[i] if i < len(t0_cat_arr) else np.nan,
            't0_data': t0_data_arr[i] if i < len(t0_data_arr) else np.nan,
        }
        
        bands = [b for b, d in tpl.items() 
                 if isinstance(d, dict) and 'ph' in d and 'mag' in d and len(d['ph']) > 0]
        row['n_bands'] = len(bands)
        row['bands'] = ','.join(sorted(bands))
        
        for check_band in ['r', 'g', 'i']:
            if check_band in tpl and isinstance(tpl[check_band], dict):
                ph = np.asarray(tpl[check_band].get('ph', []), float)
                mag = np.asarray(tpl[check_band].get('mag', []), float)
                if len(ph) > 0:
                    row['ref_band'] = check_band
                    row['phase_min'] = float(ph.min())
                    row['phase_max'] = float(ph.max())
                    row['phase_span'] = float(ph.max() - ph.min())
                    row['n_phases'] = len(ph)
                    row['mag_peak'] = float(mag[np.argmin(np.abs(ph))])
                    break
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if save_summary:
        summary_path = templates_file.parent / f"{templates_file.stem}_coverage_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"Saved to {summary_path}")
    
    return df

def compute_slsn_properties(df_cov, templates_file):
    """Compute SLSN-specific observable properties from templates."""
    with open(templates_file, 'rb') as f:
        obj = pickle.load(f)
    
    lcs = obj['lightcurves']
    rows = []
    
    for i, tpl in enumerate(lcs):
        row = {'tpl_idx': i}
        
        if 'r' in tpl and isinstance(tpl['r'], dict):
            ph = np.asarray(tpl['r']['ph'], float)
            mag = np.asarray(tpl['r']['mag'], float)
            
            if len(ph) > 5:
                peak_idx = np.argmin(mag)
                row['M_peak_r'] = float(mag[peak_idx])
                
                post = (ph > 0) & (ph >= 15) & (ph <= 50)
                if post.sum() >= 3:
                    slope, _, _, _, _ = linregress(ph[post], mag[post])
                    row['decline_rate_r'] = float(slope)
                
                thresh = row['M_peak_r'] + 1.0
                above = mag < thresh
                if above.sum() >= 2:
                    row['width_r'] = float(ph[above].max() - ph[above].min())
        
        if 'g' in tpl and 'r' in tpl:
            g_mag = np.asarray(tpl['g']['mag'], float)
            g_ph = np.asarray(tpl['g']['ph'], float)
            r_mag = np.asarray(tpl['r']['mag'], float)
            r_ph = np.asarray(tpl['r']['ph'], float)
            
            if len(g_ph) > 0 and len(r_ph) > 0:
                g_near_peak = g_mag[np.argmin(np.abs(g_ph))]
                r_near_peak = r_mag[np.argmin(np.abs(r_ph))]
                row['g_minus_r'] = float(g_near_peak - r_near_peak)
        
        rows.append(row)
    
    df_props = pd.DataFrame(rows)
    return df_cov.merge(df_props, on='tpl_idx', how='left')

def assess_literature_coverage(df_cov, output_dir=None):
    """Compare template distributions to published SLSN samples."""
    lit_ranges = {
        'M_peak_r': (-23.0, -19.5),
        'decline_rate_r': (0.005, 0.08),
        'g_minus_r': (-0.3, 0.5),
        'width_r': (20, 80),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    gaps = {}
    
    for ax, (param, (lit_min, lit_max)) in zip(axes, lit_ranges.items()):
        if param not in df_cov.columns:
            ax.text(0.5, 0.5, f'{param}\nNo Data', ha='center', va='center')
            gaps[param] = "no_data"
            continue
        
        vals = df_cov[param].dropna()
        if len(vals) == 0:
            gaps[param] = "no_data"
            continue
        
        ax.hist(vals, bins=20, alpha=0.7, edgecolor='black', label='Your templates')
        ax.axvline(lit_min, color='red', ls='--', lw=2, label='Lit range')
        ax.axvline(lit_max, color='red', ls='--', lw=2)
        ax.axvspan(lit_min, lit_max, alpha=0.2, color='red')
        
        coverage_frac = np.sum((vals >= lit_min) & (vals <= lit_max)) / len(vals)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(f'{param}\nCoverage: {coverage_frac*100:.0f}% in lit range')
        ax.legend()
        
        if coverage_frac < 0.5:
            gaps[param] = "poor_coverage"
        else:
            gaps[param] = "good"
    
    plt.tight_layout()
    _out = (Path(output_dir) / "SLSN_template_coverage_vs_literature.png"
            if output_dir else Path("SLSN_template_coverage_vs_literature.png"))
    plt.savefig(_out, dpi=150)
    plt.show()
    
    print("\n=== Coverage Assessment ===")
    for param, status in gaps.items():
        print(f"{param:20s}: {status}")
    
    return gaps

def find_missing_archetypes(df_cov):
    """Identify specific SLSN types not well-represented."""
    missing = []
    
    if 'decline_rate_r' in df_cov.columns:
        vals = df_cov['decline_rate_r'].dropna()
        fast = vals > 0.05
        slow = vals < 0.02
        print(f"Fast evolvers (>0.05 mag/day): {fast.sum()}")
        print(f"Slow evolvers (<0.02 mag/day): {slow.sum()}")
        if fast.sum() < 10:
            missing.append("fast_declining")
        if slow.sum() < 10:
            missing.append("slow_declining")
    
    if 'M_peak_r' in df_cov.columns:
        vals = df_cov['M_peak_r'].dropna()
        bright = vals < -22
        faint = vals > -20.5
        print(f"Very bright (M_r < -22): {bright.sum()}")
        print(f"Faint (M_r > -20.5): {faint.sum()}")
        if bright.sum() < 10:
            missing.append("super_luminous")
        if faint.sum() < 10:
            missing.append("faint_end")
    
    if 'g_minus_r' in df_cov.columns:
        vals = df_cov['g_minus_r'].dropna()
        blue = vals < -0.1
        red = vals > 0.3
        print(f"Blue (g-r < -0.1): {blue.sum()}")
        print(f"Red (g-r > 0.3): {red.sum()}")
        if blue.sum() < 5:
            missing.append("very_blue")
        if red.sum() < 5:
            missing.append("red_events")
    
    if missing:
        print(f"\nMissing archetypes: {', '.join(missing)}")
        print("Consider: (1) adding more events, (2) restricting z_max")
    else:
        print("\n✓ Good coverage across major SLSN types")
    
    return missing

def _axes_array(fig, ax):
    """Return a flattened array of axes whether ax is scalar or ndarray."""
    return np.atleast_1d(ax).ravel()

# -----------------------------
# Population-generation plots
# -----------------------------

def plot_population_diagnostics(
    *, ra_rad, dec_rad, peak_times, distances_mpc, z_vals,
    ebv=None, gall=None, galb=None, outdir=None, prefix="population"
):
    """
    Standard set of QA plots for a generated population (used by population.py).
    All angular inputs must be radians.
    """
    
    import os 

    ra_rad  = np.asarray(ra_rad, float)
    dec_rad = np.asarray(dec_rad, float)

    # auto-detect if values look like degrees; convert to radians if so
    if np.nanmax(np.abs(ra_rad)) > 2*np.pi + 0.1 or np.nanmax(np.abs(dec_rad)) > np.pi + 0.1:
        ra_rad  = np.radians(ra_rad)
        dec_rad = np.radians(dec_rad)

    # whenever you need degrees for a histogram/label:
    ra_deg  = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    os.makedirs(outdir, exist_ok=True) if outdir else None

    # 0) Sky distribution (deg)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(np.degrees(ra_rad), np.degrees(dec_rad), s=1, alpha=0.5)
    ax.set_xlabel('RA (deg)'); ax.set_ylabel('Dec (deg)')
    ax.set_title('Sky Distribution'); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_sky.png"), dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)

    # 1) RA, 2) Dec (radians)
    for dat, label in [(ra_rad, "RA [rad]"), (dec_rad, "Dec [rad]")]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(dat, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(label); ax.set_ylabel("N Events"); ax.grid(True, alpha=0.3)
        ax.set_title(f"Injected Population — {label.split()[0]} Distribution")
        if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_{label.split()[0].lower()}_hist.png"), dpi=150)
        plt.show(); plt.close(fig)

    # 3) Peak time (days)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(peak_times, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Peak Time [days since survey start]")
    ax.set_ylabel("N Events"); ax.grid(True, alpha=0.3)
    ax.set_title("Peak Time Distribution")
    if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_peaktime_hist.png"), dpi=150)
    plt.show(); plt.close(fig)

    # 4) Distance (Mpc)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(distances_mpc, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Comoving Distance [Mpc]"); ax.set_ylabel("N Events"); ax.grid(True, alpha=0.3)
    ax.set_title("Distance Distribution")
    if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_distance_hist.png"), dpi=150)
    plt.show(); plt.close(fig)

    # 5) Redshift
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(z_vals, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Redshift"); ax.set_ylabel("N Events"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Redshift Distribution (z = {np.nanmin(z_vals):.3f} – {np.nanmax(z_vals):.3f})")
    if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_z_hist.png"), dpi=150)
    plt.show(); plt.close(fig)

    # 6) Galactic (optional)
    if gall is not None and galb is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes = _axes_array(fig, axes)
        axes[0].hist(gall, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel("Galactic Longitude [deg]"); axes[0].set_ylabel("N Events")
        axes[0].set_title("Galactic Longitude Distribution"); axes[0].grid(True, alpha=0.3)
        axes[1].hist(galb, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Galactic Latitude [deg]"); axes[1].set_ylabel("N Events")
        axes[1].set_title("Galactic Latitude Distribution"); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_galactic_hist.png"), dpi=150)
        plt.show(); plt.close(fig)

    # 7) EBV (optional)
    if ebv is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(ebv, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("E(B−V) [mag]"); ax.set_ylabel("N Events")
        ax.set_title("Extinction Distribution"); ax.grid(True, alpha=0.3)
        if outdir: fig.savefig(os.path.join(outdir, f"{prefix}_ebv_hist.png"), dpi=150)
        plt.show(); plt.close(fig)

# --------------------------------
# --------------------------------
# Detection/efficiency diagnostics
# --------------------------------
# --------------------------------


def plot_healpix_efficiency(bundle_metric_values, ra_rad, dec_rad, nside=64, outpath=None, title="Detection Efficiency", figsize=(7, 4)):
    """Mollweide map of detection efficiency (injected vs detected)."""
    npix = hp.nside2npix(nside)
    injected_map = np.zeros(npix)
    detected_map = np.zeros(npix)

    theta = 0.5*np.pi - dec_rad
    phi = ra_rad
    pix = hp.ang2pix(nside, theta, phi)

    for i, p in enumerate(pix):
        injected_map[p] += 1
        if int(bundle_metric_values[i]) == 1:
            detected_map[p] += 1

    eff = np.full(npix, hp.UNSEEN)
    mask = injected_map > 0
    eff[mask] = (detected_map[mask] > 0).astype(float)

    hp.mollview(eff, title=title, unit='Efficiency', cmap='viridis',
                min=0, max=1, hold=False, xsize=800)
    plt.gcf().set_size_inches(figsize)
    hp.graticule()
    plt.show()
    if outpath:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        plt.close()

# ==============================
# Plot population LCs
# ===============================

# ---------------------------------------------------------------------
#  Helper: index lookup for a given sid in slicer.slice_points and finding avaialble bands 
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
def _sp_index_for_sid(slice_points, sid):
    """Find index for sid in slicer.slice_points."""
    sids = np.asarray(slice_points['sid']).astype(int)
    sid = int(sid)
    match = np.where(sids == sid)[0]
    if match.size == 0:
        raise KeyError(f"sid={sid} not found in slicer.slice_points")
    return int(match[0])

# ---------------------------------------------------------------------

def _get_available_bands(df_obs, slicer, templates=None):
    """Return a sorted list of available bands across slicer + df_obs + templates."""
    bands = set()

    # from extinction map keys
    sp_keys = slicer.slice_points.keys()
    bands.update([k.split("_", 1)[1] for k in sp_keys if k.startswith("A_")])

    # from observed data
    if "filter" in df_obs.columns:
        try:
            all_filters = np.unique(np.concatenate(df_obs["filter"].values))
            bands.update(all_filters)
        except Exception:
            pass

    # from template grid
    if templates is not None and hasattr(templates, "grid"):
        bands.update(list(templates.grid.keys()))

    # clean up and sort
    bands = [b for b in sorted(set(str(b).lower() for b in bands)) if len(b) == 1]
    return bands




# ---------------------------------------------------------------------
#  Single-event LC: model @ obs epochs + optional measured points
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
def plot_population_lc_at_obs(
    sid,
    df_obs,
    slicer,
    templates,
    bands="auto",
    apply_time_dilation=True,
    overlay_observations=True,
    show_smooth=False,
    smooth_step_days=0.5,
    fast_peaks=False,
    title_prefix="Population LC @ observed epochs",
    colors=None,
    connect_points=True,       #  toggle 1
    smooth_visual=False,       #  toggle 2
    ax=None
):
    """
    Plot true population light curves at actual observed epochs.
    Faithfully reconstructs apparent magnitudes using stored DM, A_band, and z.
    Provides optional line connections and smoothed visual guides for clarity.

    Parameters
    ----------
    sid : int
        Event identifier in both df_obs and slicer.slice_points.
    df_obs : pandas.DataFrame
        Table containing mjd_obs, mag_obs, filter, peak_mjd, etc.
    slicer : object with .slice_points
        Population slice_points dict holding distance_modulus, A_<band>, etc.
    templates : object
        Light-curve model providing .interp(phase, filter, file_indx).
    bands : tuple or 'auto'
        Filters to plot; 'auto' will detect from slicer + df_obs + templates.
    apply_time_dilation : bool
        Apply (1+z) stretch to time axis for observed-frame alignment.
    overlay_observations : bool
        Scatter measured photometry alongside the model.
    connect_points : bool
        Draw lines connecting model@obs points for each band.
    smooth_visual : bool
        Add gentle smoothed line (savgol-filtered) for visual clarity.
    fast_peaks : bool
        If True, scatter stored `peak_app_mag_ebv_*` only.
    """

    sp = slicer.slice_points
    sid = int(sid)
    # --- index lookup ---
    sids = np.asarray(sp['sid']).astype(int)
    ix = np.where(sids == sid)[0]
    if ix.size == 0:
        raise KeyError(f"sid={sid} not found in slicer.slice_points")
    ix = int(ix[0])

    # --- auto band detection ---
    if bands == "auto":
        bands = _get_available_bands(df_obs, slicer, templates)

    # --- color scheme ---
    if colors is None:
        colors = {'u': 'k', 'g': 'b', 'r': 'g', 'i': 'r', 'z': 'magenta', 'y': 'gold'}

    # --- get event row ---
    row = df_obs.loc[df_obs['sid'] == sid]
    if row.empty:
        raise ValueError(f"sid={sid} not found in df_obs")
    row = row.iloc[0]

    peak_mjd = float(row['peak_mjd'])
    z        = float(row.get('z', sp['z'][ix]))
    DM       = float(row.get('distance_modulus', sp['distance_modulus'][ix]))
    tmpl_idx = int(sp['file_indx'][ix])
    dist_Mpc = float(sp['distance'][ix])
    ebv      = float(sp['ebv'][ix])

    # --- fast peaks only ---
    if fast_peaks:
        fig, ax = plt.subplots(figsize=(8,5))
        for f in bands:
            key = f'peak_app_mag_ebv_{f}'
            if key not in sp:
                continue
            m_peak = float(sp[key][ix])
            ax.scatter([0], [m_peak], s=40,
                       color=colors.get(f,'gray'), label=f)
        ax.set_title(f"{title_prefix} (peaks only) · sid={sid} · idx={tmpl_idx} · "
                     f"d={dist_Mpc:.0f} Mpc · E(B−V)={ebv:.3f}")
        ax.set_xlabel("Days from peak")
        ax.set_ylabel("Apparent magnitude")
        ax.invert_yaxis()
        ax.legend(ncol=6, fontsize=9, frameon=False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        return {"sid": sid, "mode": "peaks"}

    # --- observed data arrays ---
    T = np.asarray(row['mjd_obs'], float)
    M = np.asarray(row['mag_obs'], float)
    F = np.asarray(row['filter'],  object)



    # choose axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        created_fig = True

    model_mag = {}
    obs_data  = {}

    # --- per-band reconstruction ---
    for f in bands:
        f = str(f).lower()
        A_key = f"A_{f}"
        if A_key not in sp:
            continue
        A_f = float(sp[A_key][ix])

        mask = (F == f) & np.isfinite(T)
        t_band = T[mask] if np.any(mask) else np.array([])
        phase_band = (t_band - peak_mjd) / (1.0 + z) if apply_time_dilation else (t_band - peak_mjd)

        try:
            if t_band.size > 0:
                        m_abs = np.asarray(templates.interp(phase_band, f, tmpl_idx), float)
                        m_app = m_abs + DM + A_f
            
                        # keep only finite (inside template support)
                        finite = np.isfinite(m_app)
                        if not np.any(finite):
                            continue
                        t_f = t_band[finite]
                        m_f = m_app[finite]
            
                        model_mag[f] = (t_f, m_f)
            
                        ax.scatter(t_f - peak_mjd, m_f, s=28,
                                   color=colors.get(f, 'gray'), alpha=0.9,
                                   edgecolors='k', linewidths=0.4, label=f"{f} model@obs")
            
                        if connect_points and t_f.size > 1:
                            order = np.argsort(t_f)
                            ax.plot(t_f[order] - peak_mjd, m_f[order],
                                    color=colors.get(f, 'gray'), lw=1.2, alpha=0.7, zorder=1)
            
                        if smooth_visual and t_f.size > 3:
                            from scipy.signal import savgol_filter
                            order = np.argsort(t_f)
                            t_sorted = t_f[order] - peak_mjd
                            m_smooth = savgol_filter(m_f[order], 5, 2, mode='interp')
                            ax.plot(t_sorted, m_smooth,
                                    color=colors.get(f,'gray'), lw=1.0, alpha=0.5, linestyle='--', zorder=0)

            # optional smooth guide (for completeness)
            if show_smooth:
                tmin = (t_band.min() if t_band.size > 0 else peak_mjd - 10)
                tmax = (t_band.max() if t_band.size > 0 else peak_mjd + 40)
                t_smooth = np.arange(tmin, tmax + smooth_step_days, smooth_step_days)
                phase_s = (t_smooth - peak_mjd) / (1.0 + z) if apply_time_dilation else (t_smooth - peak_mjd)
                m_abs_s = np.asarray(templates.interp(phase_s, f, tmpl_idx), float)
                m_app_s = m_abs_s + DM + A_f
                ax.plot(t_smooth - peak_mjd, m_app_s,
                        color=colors.get(f,'gray'), lw=1.0, alpha=0.5)

        except Exception:
            continue

        # overlay observed photometry
        if overlay_observations and t_band.size > 0:
            obs_finite = np.isfinite(M[mask])
            if np.any(obs_finite):
                ax.scatter(t_band[obs_finite] - peak_mjd, M[mask][obs_finite], s=25,
                           facecolors='none', edgecolors='k', linewidths=0.6,
                           label=f"{f} obs")
                obs_data[f] = (t_band[obs_finite], M[mask][obs_finite])

    # labels/title once
    ax.set_xlabel("Days from peak (observed frame)")
    ax.set_ylabel("Apparent magnitude")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    title = (f"{title_prefix} · sid={sid} · idx={tmpl_idx} · "
             f"d={dist_Mpc:.0f} Mpc · z={z:.3f} · E(B−V)={ebv:.3f}")
    ax.set_title(title, fontsize=11)

    # only manage legend/show if we own the figure
    if created_fig:
        ax.legend(ncol=min(3, len(ax.get_legend_handles_labels()[0])), frameon=False)
        plt.tight_layout()
        plt.show()

    return {"sid": sid, "model_at_obs": model_mag, "obs": obs_data}





# ---------------------------------------------------------------------
#  Multi-event convenience wrapper
# ---------------------------------------------------------------------
def plot_population_lc_multi_at_obs(
    sids,
    df_obs,
    slicer,
    templates,
    overlap=False,
    **kwargs
):
    results = {}

    if overlap:
        fig, ax = plt.subplots(figsize=(9,5.5))
        # turn off per-call observations to reduce clutter (can flip if you want)
        kwargs.setdefault("overlay_observations", False)
        for sid in sids:
            try:
                res = plot_population_lc_at_obs(
                    sid, df_obs, slicer, templates, ax=ax, **kwargs
                )
                results[int(sid)] = res
            except Exception as e:
                print(f"[warn] sid={sid}: {e}")

        # Legend once; avoids the UserWarning & duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), ncol=6, fontsize=9, frameon=False)
        plt.tight_layout()
        plt.show()
        return results

    # sequential (non-overlap) path unchanged
    for sid in sids:
        try:
            res = plot_population_lc_at_obs(sid, df_obs, slicer, templates, **kwargs)
            results[int(sid)] = res
        except Exception as e:
            print(f"[warn] sid={sid}: {e}")
    return results


# ---------------------------------------------------------------------
# Plotting detected vs not detected through each metric on a mosaic 
# ---------------------------------------------------------------------

def plot_metrics_mosaic_grid(
    df_detect,
    df_char,
    df_spec,
    *,
    n_examples=6,
    snr_threshold=5.0,
    figsize=(18, 14),
    outpath=None,
    seed=42,
    filter_colors=None,
    marker_size=30,
    upperlim_marker='v',
    title_prefix="SLSN Metrics Comparison"
):
    """
    Create a grid mosaic with individual event subplots for Pass vs Fail.
    
    Layout: 3 rows (Detection, Characterization, Spec Trigger) × 2 columns (Pass, Fail)
    Each panel shows multiple individual event light curves in a grid.
    
    Parameters
    ----------
    df_detect : pd.DataFrame
        Detection results with 'detected' column and observation arrays
    df_char : pd.DataFrame
        Characterization results with 'characterized' column
    df_spec : pd.DataFrame
        Spec trigger results with 'spec_trigger' column
    n_examples : int
        Number of example events per Pass/Fail panel (will create grid)
    snr_threshold : float
        SNR threshold for detections vs upper limits
    figsize : tuple
        Figure size
    outpath : str or Path
        Save path
    seed : int
        Random seed for sampling
    filter_colors : dict
        Custom filter colors
    marker_size : float
        Size of observation markers
    upperlim_marker : str
        Marker for upper limits (default: 'v' for downward triangle)
    title_prefix : str
        Overall title prefix
    
    Returns
    -------
    fig : matplotlib.Figure
    
    Notes
    -----
    Add to diagnostics.py after existing plot functions.
    
    This version creates individual subplots for each event, organized in a grid.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Default filter colors
    if filter_colors is None:
        filter_colors = {
            'u': '#56108C', 'g': '#0077BB', 'r': '#33A02C',
            'i': '#E31A1C', 'z': '#FF7F00', 'y': '#B15928'
        }
    
    rng = np.random.default_rng(seed)
    
    # Helper function to plot single event in a subplot
    def plot_single_event(ax, event_row, show_legend=False):
        """Plot one event's observations with error bars and upper limits."""
        # Extract observation arrays
        mjd_obs = np.asarray(event_row.get('mjd_obs', []), dtype=float)
        mag_obs = np.asarray(event_row.get('mag_obs', []), dtype=float)
        snr_obs = np.asarray(event_row.get('snr_obs', []), dtype=float)
        filter_obs = np.asarray(event_row.get('filter', []), dtype=str)
        
        # Estimate magnitude errors from SNR: σ_m ≈ 1.0857 / SNR
        if 'magerr_obs' in event_row and event_row['magerr_obs'] is not None:
            magerr_obs = np.asarray(event_row['magerr_obs'], dtype=float)
        else:
            magerr_obs = np.where(snr_obs > 0, 1.0857 / snr_obs, np.nan)
        
        # Get event metadata
        peak_mjd = float(event_row.get('peak_mjd', 0))
        z = float(event_row.get('z', 0))
        sid = int(event_row.get('sid', -1))
        
        # Time relative to peak
        time_rel = mjd_obs - peak_mjd
        
        # Get unique filters
        filters_present = np.unique(filter_obs)
        
        # Plot each filter
        for filt in filters_present:
            mask = filter_obs == filt
            if not np.any(mask):
                continue
            
            t_filt = time_rel[mask]
            m_filt = mag_obs[mask]
            snr_filt = snr_obs[mask]
            merr_filt = magerr_obs[mask]
            
            color = filter_colors.get(filt, 'gray')
            
            # Separate detections and upper limits
            det_mask = snr_filt >= snr_threshold
            lim_mask = snr_filt < snr_threshold
            
            # Plot detections with error bars
            if np.any(det_mask):
                ax.errorbar(
                    t_filt[det_mask], m_filt[det_mask],
                    yerr=merr_filt[det_mask],
                    fmt='o', color=color, markersize=np.sqrt(marker_size),
                    alpha=0.7, capsize=2, capthick=1,
                    label=filt if show_legend else None,
                    zorder=3
                )
            
            # Plot upper limits
            if np.any(lim_mask):
                ax.scatter(
                    t_filt[lim_mask], m_filt[lim_mask],
                    marker=upperlim_marker, s=marker_size,
                    color=color, alpha=0.3, zorder=2
                )
        
        # Formatting
        ax.invert_yaxis()
        ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)
        
        # Compact title
        ax.set_title(f'sid={sid}, z={z:.2f}', fontsize=7, pad=2)
        
        if show_legend and len(filters_present) > 0:
            ax.legend(fontsize=6, ncol=3, framealpha=0.7, loc='best')
    
    # Define metrics
    metrics_data = [
        ('DETECTION', df_detect, 'detected'),
        ('CHARACTERIZATION', df_char, 'characterized'),
        ('SPEC TRIGGER', df_spec, 'spec_trigger')
    ]
    
    # Determine grid layout for events (e.g., 3x2 for 6 events)
    n_rows_per_panel = int(np.ceil(np.sqrt(n_examples)))
    n_cols_per_panel = int(np.ceil(n_examples / n_rows_per_panel))
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Main GridSpec: 3 rows (metrics) × 2 columns (pass/fail)
    gs_main = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                       left=0.08, right=0.96, top=0.93, bottom=0.05)
    
    # Plot each metric
    for row_idx, (metric_name, df, status_col) in enumerate(metrics_data):
        
        # Check if status column exists
        if status_col not in df.columns:
            print(f"⚠️  Warning: '{status_col}' not found in dataframe for {metric_name}")
            print(f"    Available columns: {df.columns.tolist()}")
            # Add the column as False if missing
            df[status_col] = False
        
        # Check for observation data
        required_cols = ['mjd_obs', 'mag_obs', 'filter', 'snr_obs']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing observation columns in {metric_name}: {missing_cols}")
            continue
        
        # Separate pass and fail
        pass_df = df[df[status_col] == True].copy()
        fail_df = df[df[status_col] == False].copy()
        
        n_pass = len(pass_df)
        n_fail = len(fail_df)
        
        print(f"\n{metric_name}:")
        print(f"  Pass: {n_pass} / {len(df)} ({100*n_pass/len(df):.1f}%)")
        print(f"  Fail: {n_fail} / {len(df)} ({100*n_fail/len(df):.1f}%)")
        
        # Sample examples
        n_pass_sample = min(n_examples, n_pass)
        n_fail_sample = min(n_examples, n_fail)
        
        if n_pass_sample > 0:
            pass_sample = pass_df.sample(n=n_pass_sample, random_state=seed)
        else:
            pass_sample = pass_df
        
        if n_fail_sample > 0:
            fail_sample = fail_df.sample(n=n_fail_sample, random_state=seed)
        else:
            fail_sample = fail_df
        
        # ==========================
        # PASS COLUMN (left)
        # ==========================
        gs_pass = gs_main[row_idx, 0].subgridspec(
            n_rows_per_panel, n_cols_per_panel,
            hspace=0.3, wspace=0.25
        )
        
        if len(pass_sample) > 0:
            for idx, (_, event_row) in enumerate(pass_sample.iterrows()):
                if idx >= n_examples:
                    break
                i = idx // n_cols_per_panel
                j = idx % n_cols_per_panel
                ax = fig.add_subplot(gs_pass[i, j])
                plot_single_event(ax, event_row, show_legend=(idx == 0))
                
                # Only label bottom row
                if i == n_rows_per_panel - 1:
                    ax.set_xlabel('Days from peak', fontsize=7)
                else:
                    ax.set_xlabel('')
                
                # Only label left column
                if j == 0:
                    ax.set_ylabel('App mag', fontsize=7)
                else:
                    ax.set_ylabel('')
            
            # Add Pass/Fail title at top of column
            if row_idx == 0:
                pass_rate = 100 * n_pass / len(df)
                ax_title = fig.add_subplot(gs_pass[0, :])
                ax_title.axis('off')
                ax_title.text(
                    0.5, 1.3, f'✓ PASS ({pass_rate:.1f}%)',
                    transform=ax_title.transAxes,
                    fontsize=14, fontweight='bold',
                    color='green', ha='center'
                )
        else:
            # No passing events
            ax = fig.add_subplot(gs_pass[:, :])
            ax.text(0.5, 0.5, f'No passing events\nfor {metric_name}',
                   ha='center', va='center', fontsize=11, color='gray')
            ax.axis('off')
        
        # Add row label on far left
        if row_idx == 0:
            ax_label = fig.add_subplot(gs_pass[n_rows_per_panel//2, 0])
            ax_label.text(
                -0.6, 0.5, 'DETECTION',
                transform=ax_label.transAxes,
                fontsize=13, fontweight='bold',
                rotation=90, va='center', ha='right'
            )
        elif row_idx == 1:
            ax_label = fig.add_subplot(gs_pass[n_rows_per_panel//2, 0])
            ax_label.text(
                -0.6, 0.5, 'CHARACTERIZATION',
                transform=ax_label.transAxes,
                fontsize=13, fontweight='bold',
                rotation=90, va='center', ha='right'
            )
        elif row_idx == 2:
            ax_label = fig.add_subplot(gs_pass[n_rows_per_panel//2, 0])
            ax_label.text(
                -0.6, 0.5, 'SPEC TRIGGER',
                transform=ax_label.transAxes,
                fontsize=13, fontweight='bold',
                rotation=90, va='center', ha='right'
            )
        
        # ==========================
        # FAIL COLUMN (right)
        # ==========================
        gs_fail = gs_main[row_idx, 1].subgridspec(
            n_rows_per_panel, n_cols_per_panel,
            hspace=0.3, wspace=0.25
        )
        
        if len(fail_sample) > 0:
            for idx, (_, event_row) in enumerate(fail_sample.iterrows()):
                if idx >= n_examples:
                    break
                i = idx // n_cols_per_panel
                j = idx % n_cols_per_panel
                ax = fig.add_subplot(gs_fail[i, j])
                plot_single_event(ax, event_row, show_legend=(idx == 0))
                
                # Only label bottom row
                if i == n_rows_per_panel - 1:
                    ax.set_xlabel('Days from peak', fontsize=7)
                else:
                    ax.set_xlabel('')
                
                # Only label left column
                if j == 0:
                    ax.set_ylabel('App mag', fontsize=7)
                else:
                    ax.set_ylabel('')
            
            # Add Pass/Fail title at top of column
            if row_idx == 0:
                fail_rate = 100 * n_fail / len(df)
                ax_title = fig.add_subplot(gs_fail[0, :])
                ax_title.axis('off')
                ax_title.text(
                    0.5, 1.3, f'✗ FAIL ({fail_rate:.1f}%)',
                    transform=ax_title.transAxes,
                    fontsize=14, fontweight='bold',
                    color='red', ha='center'
                )
        else:
            # No failing events
            ax = fig.add_subplot(gs_fail[:, :])
            ax.text(0.5, 0.5, f'No failing events\nfor {metric_name}',
                   ha='center', va='center', fontsize=11, color='gray')
            ax.axis('off')
    
    # Overall title
    fig.suptitle(
        f'{title_prefix}\nRubin Observations Only (○ = SNR≥5, {upperlim_marker} = upper limit)',
        fontsize=15, fontweight='bold', y=0.97
    )
    
    # Save if requested
    if outpath is not None:
        plt.savefig(outpath, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved mosaic to {outpath}")
    
    return fig

# =============================================================================
# Rate evolution theory plot (Figure 1 from abstract)
# =============================================================================

def plot_rate_evolution_comparison(z_grid=None, OH_max=8.3,
                                   R_ref=1e-7, z_ref=0.17,
                                   save_path=None):
    """
    Reproduce Figure 1 from abstract: SLSN rate evolution with/without
    metallicity dependence, compared to observed rate measurements.

    Two-panel layout:
      Top    — f(z): fraction of SF in low-metallicity galaxies
      Bottom — R(z): full rate evolution vs observed data points

    Physics
    -------
    - Madau & Dickinson 2014 CSFRD
    - Tremonti+04 MZR + Andrews & Martini 2013 redshift evolution
    - Leja+2020 stellar mass function, Leja+2022 main sequence SFR
    - Schulze+2021 metallicity threshold OH_max = 8.3

    Parameters
    ----------
    z_grid : array, optional
        Redshift grid for plotting. Default: np.linspace(0.0, 3.0, 100).
        Note: metallicity_fraction is evaluated on a coarser grid
        internally because it integrates over stellar mass at each z.
    OH_max : float
        Metallicity threshold 12 + log10(O/H)_max. Default 8.3.
    R_ref : float
        Reference rate at z_ref [Mpc^-3 yr^-1]. Default 1e-7 (Quimby+13).
    z_ref : float
        Reference redshift. Default 0.17 (Quimby+13).
    save_path : str or Path, optional
        If given, saves figure here.

    Returns
    -------
    fig : matplotlib.Figure
    """
    # Local imports to avoid circular dependency:
    # population.py imports diagnostics.py, so we cannot import at module level.
    from .population import (
        slsn_rate_evolution,
        cosmic_sfr_density_MD14,
        metallicity_fraction,
        OBSERVED_RATES,
    )

    if z_grid is None:
        z_grid = np.linspace(0.0, 3.0, 100)

    z_grid = np.asarray(z_grid)

    # --- Rate with metallicity evolution (full model) ---
    rate_with_metallicity = slsn_rate_evolution(z_grid, R_ref, z_ref, OH_max)

    # --- Rate WITHOUT metallicity (pure CSFRD, f(z) = constant = 1) ---
    psi_z   = cosmic_sfr_density_MD14(z_grid)
    psi_ref = cosmic_sfr_density_MD14(z_ref)
    rate_no_metallicity = R_ref * psi_z / psi_ref

    # --- Metallicity fraction f(z) ---
    # Coarser grid because metallicity_fraction integrates over stellar mass
    z_coarse = np.linspace(z_grid.min(), z_grid.max(), 40)
    f_z = metallicity_fraction(z_coarse, OH_max)

    fig, (ax_frac, ax_rate) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # ------------------------------------------------------------------
    # Top panel: f(z)
    # ------------------------------------------------------------------
    ax_frac.plot(z_coarse, f_z, 'r-', lw=2.5,
                 label=f'$f(z)$  [OH$_{{max}}$ = {OH_max}]')
    ax_frac.axvline(z_ref, color='gray', ls=':', lw=1.5,
                    alpha=0.7, label=f'$z_{{ref}}$ = {z_ref}')
    ax_frac.set_ylabel('Fraction $f(z)$', fontsize=13)
    ax_frac.set_ylim(0, 1)
    ax_frac.legend(fontsize=11)
    ax_frac.grid(alpha=0.3)
    ax_frac.set_title(
        'Fraction of SF in Low-Metallicity Galaxies\n'
        '(Tremonti+04 MZR + Andrews & Martini 2013)',
        fontsize=13
    )

    # ------------------------------------------------------------------
    # Bottom panel: R(z) vs observed measurements
    # ------------------------------------------------------------------
    ax_rate.plot(z_grid, rate_with_metallicity, 'r-', lw=2.5,
                 label='With metallicity evolution  $R(z) \\propto \\Psi(z)\\cdot f(z)$')
    ax_rate.plot(z_grid, rate_no_metallicity, 'b--', lw=2.5,
                 label='No metallicity (pure CSFRD)  $R(z) \\propto \\Psi(z)$')

    # Observed data points — each labeled by reference
    for obs in OBSERVED_RATES:
        ax_rate.errorbar(
            obs['z'], obs['rate'],
            yerr=[[obs['err_low']], [obs['err_high']]],
            fmt='ko', markersize=8, capsize=5, capthick=2,
            label=obs['ref']
        )

    ax_rate.axvline(z_ref, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax_rate.set_xlabel('Redshift $z$', fontsize=13)
    ax_rate.set_ylabel('Volumetric Rate  [Mpc$^{-3}$ yr$^{-1}$]', fontsize=13)
    ax_rate.set_yscale('log')
    ax_rate.set_ylim(1e-8, 1e-5)
    ax_rate.legend(fontsize=10)
    ax_rate.grid(alpha=0.3, which='both')
    ax_rate.set_title(
        f'SLSN Volumetric Rate Evolution\n'
        f'$R_{{ref}}$ = {R_ref:.0e} Mpc$^{{-3}}$ yr$^{{-1}}$ at $z_{{ref}}$ = {z_ref}',
        fontsize=13
    )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved rate evolution plot → {save_path}")

    plt.show()
    return fig

def compare_simulated_vs_observed_rates(population_slicer, z_bins=None):
    from .population import OBSERVED_RATES
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u_astropy

    sp = population_slicer.slice_points
    z_vals = sp['z']
    peak_times = sp['peak_time']
    t_survey_years = (peak_times.max() - peak_times.min()) / 365.25

    # Sky fraction from galactic latitude cut
    # |b| > 15° covers fraction = 1 - sin(15°) of full sky
    sky_fraction = 1.0 - np.sin(np.radians(15.0))  # ≈ 0.741

    if z_bins is None:
        # Bins matched to observed rate measurements + population range
        z_bins = np.array([0.1, 0.25, 0.4, 0.7, 1.0, 1.5, 2.0])

    results = []
    for i in range(len(z_bins) - 1):
        z_low, z_high = z_bins[i], z_bins[i + 1]
        z_mid = 0.5 * (z_low + z_high)

        mask = (z_vals >= z_low) & (z_vals < z_high)
        n_events = int(mask.sum())

        V_low  = cosmo.comoving_volume(z_low).to_value(u_astropy.Mpc**3)
        V_high = cosmo.comoving_volume(z_high).to_value(u_astropy.Mpc**3)
        # Correct for sky fraction — population doesn't cover full sky
        V_shell_effective = (V_high - V_low) * sky_fraction

        rate_sim = n_events / (V_shell_effective * t_survey_years) if V_shell_effective > 0 else 0.0

        obs_match = [obs for obs in OBSERVED_RATES if z_low <= obs['z'] < z_high]
        if obs_match:
            rate_obs = obs_match[0]['rate']
            ref = obs_match[0]['ref']
        else:
            rate_obs = np.nan
            ref = '—'

        results.append({
            'z_min': z_low, 'z_max': z_high, 'z_mid': z_mid,
            'n_events': n_events, 'volume_Mpc3': V_shell_effective,
            'rate_simulated': rate_sim, 'rate_observed': rate_obs,
            'ratio_sim/obs': rate_sim / rate_obs if np.isfinite(rate_obs) else np.nan,
            'reference': ref
        })

    df = pd.DataFrame(results)
    print("\n" + "=" * 85)
    print("SIMULATED VS OBSERVED RATE COMPARISON")
    print("=" * 85)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2e}'))
    print("=" * 85 + "\n")
    return df


def plot_population_rate_vs_redshift(population_slicer,
                                      rate_model='evolving',
                                      R_ref=1e-7, z_ref=0.17, OH_max=8.3,
                                      tabulated_csv=None, model_name=None,
                                      sky_fraction=None,
                                      z_bins=None, save_path=None):
    """
    Diagnostic: simulated event rate per redshift bin vs theoretical curve
    and observed measurements.

    Call AFTER generate_SLSN_PopSlicer to verify the population reproduces
    the intended rate evolution.

    Parameters
    ----------
    population_slicer : UserPointsSlicer
        Output of generate_SLSN_PopSlicer
    rate_model : str
        'evolving' or 'constant' — controls whether theory curve is plotted
    R_ref : float
        Reference rate [Mpc^-3 yr^-1]
    z_ref : float
        Reference redshift
    OH_max : float
        Metallicity threshold
    z_bins : array, optional
        Redshift bin edges for histogram. Default: 15 equal bins
    save_path : str or Path, optional
        Save path for figure

    Returns
    -------
    fig : matplotlib.Figure
    """
    from .population import slsn_rate_evolution, OBSERVED_RATES
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u_astropy

    sp = population_slicer.slice_points
    z_vals = sp['z']
    peak_times = sp['peak_time']
    t_survey_years = (peak_times.max() - peak_times.min()) / 365.25

    if z_bins is None:
        z_bins = np.linspace(z_vals.min(), z_vals.max(), 15)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulated rate per bin
    counts, edges = np.histogram(z_vals, bins=z_bins)
    z_centers = 0.5 * (edges[:-1] + edges[1:])

    rates_sim = []
    for i in range(len(edges) - 1):
        V = (cosmo.comoving_volume(edges[i + 1]).to_value(u_astropy.Mpc**3)
             - cosmo.comoving_volume(edges[i]).to_value(u_astropy.Mpc**3))
        rates_sim.append(counts[i] / (V * t_survey_years) if V > 0 else 0.0)

    ax.scatter(z_centers, rates_sim, s=100, alpha=0.7, c='steelblue',
               edgecolors='k', linewidths=1.5, zorder=3, label='Simulated')

    # Theoretical curve
    if rate_model == 'tabulated':
        # Ben's CSV-based theory curve — correct curve for tabulated populations
        from .population import load_tabulated_rate
        from .paths import get_rate_csv_path
        csv = tabulated_csv if tabulated_csv else get_rate_csv_path()
        mname = model_name if model_name else 'fe_dependent'
        rate_interp = load_tabulated_rate(csv, mname)
        z_theory = np.linspace(z_vals.min(), z_vals.max(), 100)
        rate_theory = rate_interp(z_theory)
        if sky_fraction is not None:
            rate_theory = rate_theory * sky_fraction
            sky_label = f' x {sky_fraction:.3f} sky'
        else:
            sky_label = ' (full sky)'
        ax.plot(z_theory, rate_theory, 'r-', lw=2.5,
                label=f'Theory R(z) [{mname}] (Frohmaier+2021){sky_label}', zorder=2)
    elif rate_model == 'evolving':
        z_theory = np.linspace(z_vals.min(), z_vals.max(), 100)
        rate_theory = slsn_rate_evolution(z_theory, R_ref, z_ref, OH_max)
        if sky_fraction is not None:
            rate_theory = rate_theory * sky_fraction
        ax.plot(z_theory, rate_theory, 'r-', lw=2.5,
                label=f'Theory R(z)  [OH_max={OH_max}]', zorder=2)

    # Observed points
    for i, obs in enumerate(OBSERVED_RATES):
        if z_vals.min() <= obs['z'] <= z_vals.max():
            ax.errorbar(obs['z'], obs['rate'],
                        yerr=[[obs['err_low']], [obs['err_high']]],
                        fmt='ko', markersize=10, capsize=5, capthick=2,
                        label=obs['ref'], zorder=4)

    ax.set_xlabel('Redshift', fontsize=13)
    ax.set_ylabel('Volumetric Rate  [Mpc$^{-3}$ yr$^{-1}$]', fontsize=13)
    ax.set_yscale('log')
    ax.set_title(f'Simulated SLSN Rate vs Redshift  ({rate_model} model)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which='both')

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved → {save_path}")

    plt.show()
    return fig


def plot_detection_diagnostics(
    df_obs,
    population_slicer=None,
    filtername="r",
    mjd0=60980.5,
    bins_gap=np.arange(-100, 200, 5),
    bins_mag=np.arange(18, 32, 0.25),
    show_residuals=True,
    save_dir=None,
):
    """
    Post-metric diagnostic plots for SLSN detection results.

    Ported from shared_utils_legacy.plot_population_diagnostics.
    Produces three plots:
      A) Time gap  : first detectable observation minus peak time
      B) Peak mag  : observed peak apparent magnitude histogram
      C) Inj vs obs: implanted vs cadence-observed peak magnitude scatter

    Parameters
    ----------
    df_obs : DataFrame
        Output of metric run.  Required columns: detected, peak_time,
        mjd_obs, mag_obs, snr_obs, filter, sid.
        Optional: first_det_mjd, per_filter_min_mag, peak_app_mag_ebv_{f}.
    population_slicer : UserPointsSlicer, optional
        Used to pull injected peak magnitudes by sid.
    filtername : str
        LSST filter for magnitude plots. Default r.
    mjd0 : float
        Survey start MJD (must match metric run).
    bins_gap : array
        Histogram bins for time-gap plot (days).
    bins_mag : array
        Histogram bins for magnitude plot.
    show_residuals : bool
        Show observed-minus-injected residual histogram.
    save_dir : str or Path, optional
        If given, save each figure here as a PNG.
    """
    df = df_obs.copy()
    detected = df["detected"].astype(bool).values if "detected" in df.columns                else np.zeros(len(df), bool)

    if save_dir is not None:
        save_dir = Path(save_dir)

    def _min_in_filter(filters, mags, snrs, f, snr_cut=5.0):
        fa = np.asarray(filters)
        ma = np.asarray(mags, dtype=float)
        sa = np.asarray(snrs, dtype=float)
        mask = (fa == f) & np.isfinite(ma) & (ma < 90) & np.isfinite(sa) & (sa >= snr_cut)
        return float(np.min(ma[mask])) if np.any(mask) else np.nan

    def _first_det_mjd(mjds, snrs):
        ma = np.asarray(mjds, dtype=float)
        sa = np.asarray(snrs, dtype=float)
        good = np.isfinite(sa) & (sa >= 5) & np.isfinite(ma)
        return float(np.min(ma[good])) if np.any(good) else np.nan

    if "first_det_mjd" not in df.columns:
        df["first_det_mjd"] = df.apply(
            lambda r: _first_det_mjd(r.get("mjd_obs", []), r.get("snr_obs", [])),
            axis=1
        )

    if "per_filter_min_mag" not in df.columns:
        df["per_filter_min_mag"] = df.apply(
            lambda r: {
                f: _min_in_filter(r.get("filter", []), r.get("mag_obs", []),
                                   r.get("snr_obs", []), f)
                for f in "ugrizy"
            }, axis=1
        )

    if "peak_time" in df.columns and "first_det_mjd" in df.columns:
        peak_mjd  = mjd0 + df["peak_time"].astype(float).values
        first_det = df["first_det_mjd"].astype(float).values
        dt        = first_det - peak_mjd

        vals_nd = dt[~detected & np.isfinite(dt)]
        vals_d  = dt[ detected & np.isfinite(dt)]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if vals_nd.size: ax.hist(vals_nd, bins=bins_gap, alpha=0.5, label="non-detected")
        if vals_d.size:  ax.hist(vals_d,  bins=bins_gap, alpha=0.8, label="detected")
        ax.axvline(0, ls="--", lw=1, color="k")
        ax.set_xlabel("First detectable observation - Peak (days)")
        ax.set_ylabel("Number of events")
        ax.set_title("Time gap to first detectable observation")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "diag_time_gap.png", dpi=150)
        plt.show()
    else:
        print("[diag] Missing peak_time or first_det_mjd - skipping time gap plot.")

    m_obs_peak = np.array(
        [row.get("per_filter_min_mag", {}).get(filtername, np.nan)
         if isinstance(row.get("per_filter_min_mag"), dict) else np.nan
         for _, row in df.iterrows()], dtype=float
    )

    if np.any(np.isfinite(m_obs_peak)):
        vals_nd = m_obs_peak[~detected & np.isfinite(m_obs_peak)]
        vals_d  = m_obs_peak[ detected & np.isfinite(m_obs_peak)]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if vals_nd.size: ax.hist(vals_nd, bins=bins_mag, alpha=0.5, label="non-detected")
        if vals_d.size:  ax.hist(vals_d,  bins=bins_mag, alpha=0.8, label="detected")
        ax.set_xlabel(f"Observed peak apparent mag ({filtername}-band)")
        ax.set_ylabel("Number of events")
        ax.set_title(f"Peak apparent magnitude - {filtername} band")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"diag_peak_mag_{filtername}.png", dpi=150)
        plt.show()
    else:
        print(f"[diag] No finite observed peak mags for {filtername} - skipping.")

    inj_col = f"peak_app_mag_ebv_{filtername}"
    m_inj = np.full(len(df), np.nan)

    if inj_col in df.columns:
        m_inj = df[inj_col].astype(float).values
    elif population_slicer is not None:
        sp = population_slicer.slice_points
        if inj_col in sp:
            inj_arr = np.asarray(sp[inj_col], float)
            sid_arr = np.asarray(sp["sid"])
            sid_to_inj = dict(zip(sid_arr, inj_arr))
            m_inj = np.array([sid_to_inj.get(int(s), np.nan)
                               for s in df["sid"].values], dtype=float)

    good = np.isfinite(m_inj) & np.isfinite(m_obs_peak)
    if np.any(good):
        fig, ax = plt.subplots(figsize=(5.6, 5.2))
        ax.scatter(m_inj[~detected & good], m_obs_peak[~detected & good],
                   s=10, alpha=0.4, label="non-detected")
        ax.scatter(m_inj[ detected & good], m_obs_peak[ detected & good],
                   s=18, alpha=0.8, label="detected")
        lo, hi = m_inj[good].min(), m_inj[good].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_xlabel(f"Injected peak mag ({filtername})")
        ax.set_ylabel(f"Observed peak mag ({filtername})")
        ax.set_title(f"Injected vs Observed peak - {filtername} band")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / f"diag_inj_vs_obs_{filtername}.png", dpi=150)
        plt.show()

        if show_residuals:
            res = m_obs_peak[good] - m_inj[good]
            vals_nd = res[~detected[good]]
            vals_d  = res[ detected[good]]
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            if vals_nd.size: ax.hist(vals_nd, bins=np.arange(-3, 3.05, 0.1),
                                      alpha=0.5, label="non-detected")
            if vals_d.size:  ax.hist(vals_d,  bins=np.arange(-3, 3.05, 0.1),
                                      alpha=0.8, label="detected")
            ax.axvline(0, color="k", ls="--", lw=1)
            ax.set_xlabel("Observed - Injected (mag)")
            ax.set_ylabel("Number of events")
            ax.set_title(f"Peak mag residuals - {filtername} band")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir:
                plt.savefig(save_dir / f"diag_residuals_{filtername}.png", dpi=150)
            plt.show()
    else:
        print("[diag] No finite injected peak mags - skipping injected vs observed plot.")
        print(f"       Tip: pass population_slicer with {inj_col} in slice_points.")


# =============================================================================

def plot_sky_detection(
    df_obs,
    population_slicer,
    mjd0=60980.5,
    filtername="r",
    cadence_name="cadence",
    save_dir=None,
):
    """
    Spatial and temporal diagnostic plots for SLSN detection results.

    Ported from shared_utils_legacy.run_detect plot block.
    Produces five plots:
      1) RA  vs apparent peak magnitude
      2) Dec vs apparent peak magnitude
      3) Peak time distribution by survey year
      4) Detections per survey year
      5) Declination distribution injected vs detected
    """
    sp = population_slicer.slice_points
    detected = df_obs["detected"].astype(bool).values if "detected" in df_obs.columns                else np.zeros(len(df_obs), bool)

    if save_dir is not None:
        save_dir = Path(save_dir)

    sp_sids = np.asarray(sp["sid"])
    sp_ra   = np.asarray(sp["ra"])
    sp_dec  = np.asarray(sp["dec"])

    mag_col = f"peak_app_mag_ebv_{filtername}"
    sp_mag  = np.asarray(sp[mag_col], float) if mag_col in sp               else np.full(len(sp_sids), np.nan)

    sid_to_idx = {int(s): i for i, s in enumerate(sp_sids)}

    df_sids   = df_obs["sid"].values.astype(int)
    ras       = np.array([np.degrees(sp_ra [sid_to_idx[s]]) if s in sid_to_idx else np.nan for s in df_sids])
    decs      = np.array([np.degrees(sp_dec[sid_to_idx[s]]) if s in sid_to_idx else np.nan for s in df_sids])
    peak_mags = np.array([sp_mag[sid_to_idx[s]]             if s in sid_to_idx else np.nan for s in df_sids])

    finite = np.isfinite(ras) & np.isfinite(peak_mags)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(ras[finite], peak_mags[finite], c="black", s=8, alpha=0.4, label="Injected")
    ax.scatter(ras[finite & detected], peak_mags[finite & detected],
               c="red", s=18, alpha=0.9, edgecolors="black", label="Detected")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel(f"Apparent Peak Mag ({filtername}-band)")
    ax.set_title(f"{cadence_name} - Apparent Mag vs RA")
    ax.invert_yaxis(); ax.grid(True); ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"diag_ra_vs_mag_{filtername}.png", dpi=150)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(decs[finite], peak_mags[finite], c="black", s=8, alpha=0.4, label="Injected")
    ax.scatter(decs[finite & detected], peak_mags[finite & detected],
               c="red", s=18, alpha=0.9, edgecolors="black", label="Detected")
    ax.set_xlabel("Dec [deg]")
    ax.set_ylabel(f"Apparent Peak Mag ({filtername}-band)")
    ax.set_title(f"{cadence_name} - Apparent Mag vs Dec")
    ax.invert_yaxis(); ax.grid(True); ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / f"diag_dec_vs_mag_{filtername}.png", dpi=150)
    plt.show()

    if "peak_time" in df_obs.columns:
        years_all = (df_obs["peak_time"].astype(float).values / 365.25).astype(int) + 1
        years_det = years_all[detected]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(years_all, bins=np.arange(0.5, 11.5, 1), edgecolor="black", alpha=0.7, label="All events")
        ax.set_xticks(np.arange(1, 11))
        ax.set_xticklabels([f"Year {i}" for i in range(1, 11)])
        ax.set_xlabel("Survey Year")
        ax.set_ylabel("Number of Events")
        ax.set_title("Distribution of Peak Times (all injected events)")
        ax.grid(True); ax.legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "diag_peak_time_dist.png", dpi=150)
        plt.show()

        from collections import Counter
        det_counts = Counter(years_det.tolist())
        yr_bins  = np.arange(1, 11)
        det_vals = [det_counts.get(int(y), 0) for y in yr_bins]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(yr_bins, det_vals, width=0.7, align="center", edgecolor="black", color="steelblue")
        ax.set_xticks(yr_bins)
        ax.set_xticklabels([f"Year {i}" for i in yr_bins])
        ax.set_xlabel("Survey Year")
        ax.set_ylabel("Number of Detections")
        ax.set_title(f"{cadence_name} - Detections per Survey Year")
        ax.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(save_dir / "diag_detections_per_year.png", dpi=150)
        plt.show()

    all_decs = np.degrees(sp_dec)
    det_decs = decs[finite & detected]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_decs, bins=50, alpha=0.5, label="Injected (full pop)")
    if det_decs.size:
        ax.hist(det_decs, bins=50, alpha=0.8, color="red", label="Detected")
    ax.set_xlabel("Declination [deg]")
    ax.set_ylabel("Number of Events")
    ax.set_title(f"{cadence_name} - Declination Distribution")
    ax.legend(); ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "diag_dec_distribution.png", dpi=150)
    plt.show()


# =============================================================================
# MC Rate Uncertainty
# =============================================================================

def _sample_R_ref(n_samples, R_mode=35.0, sig_hi=25.0, sig_lo=13.0, seed=None):
    """
    Sample R_ref from Frohmaier+2021 asymmetric errors.
    35 +25/-13 Gpc^-3 yr^-1

    Uses a split normal: positive draws use sig_hi, negative use sig_lo.
    Clips to minimum of 1.0 to avoid unphysical negatives.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n_samples)
    samples = np.where(u >= 0,
                       R_mode + sig_hi * u,
                       R_mode + sig_lo * u)
    return np.clip(samples, 1.0, None)


def plot_mc_rate_uncertainty(
    detect_vals,
    peak_times,
    N_injected_nominal,
    R_ref_nominal=35.0,
    n_realizations=1000,
    survey_years=None,
    model_label='fe_dependent',
    cadence_label='baseline_v5.1.1_10yrs',
    metric_label='Detections',
    comparison_detect_vals=None,
    comparison_peak_times=None,
    comparison_N_injected=None,
    comparison_label='naive',
    comparison_R_ref=None,
    seed=42,
    save_dir=None,
):
    """
    Monte Carlo rate uncertainty on cumulative SLSN detections vs survey length.

    Parameters
    ----------
    detect_vals : np.ndarray, shape (N_events,)
        Per-event 0/1 detection flags from .npy metric output.
    peak_times : np.ndarray, shape (N_events,)
        Per-event peak time in relative days (1-3652) from population pickle.
    N_injected_nominal : int
        Total number of injected events in the population.
    R_ref_nominal : float
        The R_ref value (Gpc^-3 yr^-1) used when generating the population.
    n_realizations : int
        Number of MC draws. 1000 is sufficient; runs in seconds.
    survey_years : list of float, optional
        Survey durations to evaluate. Defaults to [1, 2, ..., 10].
    model_label : str
        Label for the primary model (used in plot titles and filenames).
    cadence_label : str
        Cadence name (used in plot titles and filenames).
    comparison_detect_vals : np.ndarray, optional
        detect_vals for a second model (e.g. naive). If provided, Plot 2
        (significance vs survey time) is also produced.
    comparison_peak_times : np.ndarray, optional
        peak_times for the second model.
    comparison_N_injected : int, optional
        N_injected for the second model.
    comparison_label : str
        Label for the second model.
    comparison_R_ref : float, optional
        R_ref used to generate the comparison population. Defaults to R_ref_nominal.
    seed : int
        Random seed for reproducibility.
    save_dir : Path or str, optional
        If provided, saves figures here.

    Notes
    -----
    Core logic:
      For each realization i:
        1. Draw R_ref_i from Frohmaier+2021 split-normal distribution
        2. scale_factor = R_ref_i / R_ref_nominal
        3. For each survey year t:
             N_det(t, i) = sum(detect_vals[peak_times <= t*365.25]) * scale_factor

    N_detected scales linearly with R_ref because: more events injected at the
    same efficiency yields proportionally more detections. No new MAF runs needed.

    Outputs
    -------
    Plot 1 : N(SLSNe) vs survey length with 68% MC uncertainty band.
    Plot 2 : Significance vs survey time (only if comparison model provided).
             significance(t) = |N_fe - N_naive| / sqrt(sigma_fe^2 + sigma_naive^2)
             Horizontal lines at 3-sigma and 5-sigma answer Adam's question directly.
    """
    if survey_years is None:
        survey_years = list(range(1, 11))
    survey_years = np.asarray(survey_years, dtype=float)

    # --- Draw R_ref samples and compute scale factors ---
    R_samples    = _sample_R_ref(n_realizations, seed=seed)
    scale_factors = R_samples / R_ref_nominal   # shape: (n_realizations,)

    # --- Cumulative detections at each survey year (nominal) ---
    # peak_times is in relative days; t years = t*365.25 days
    base_cumulative = np.array([
        detect_vals[peak_times <= yr * 365.25].sum()
        for yr in survey_years
    ], dtype=float)   # shape: (n_years,)

    # --- Scale across all realizations ---
    # mc_matrix shape: (n_realizations, n_years)
    mc_matrix = scale_factors[:, None] * base_cumulative[None, :]

    med   = np.median(mc_matrix, axis=0)
    lo_16 = np.percentile(mc_matrix, 16, axis=0)
    hi_84 = np.percentile(mc_matrix, 84, axis=0)

    # --- Plot 1: N(SLSNe) vs survey year ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(survey_years, lo_16, hi_84, alpha=0.25,
                    label=f'{model_label} 68% CI')
    ax.plot(survey_years, med, lw=2, label=f'{model_label} median')

    if comparison_detect_vals is not None:
        comp_R = comparison_R_ref if comparison_R_ref is not None else R_ref_nominal
        comp_base = np.array([
            comparison_detect_vals[comparison_peak_times <= yr * 365.25].sum()
            for yr in survey_years
        ], dtype=float)
        comp_scale  = _sample_R_ref(n_realizations, seed=seed + 1) / comp_R
        comp_matrix = comp_scale[:, None] * comp_base[None, :]
        comp_med    = np.median(comp_matrix, axis=0)
        comp_lo     = np.percentile(comp_matrix, 16, axis=0)
        comp_hi     = np.percentile(comp_matrix, 84, axis=0)
        ax.fill_between(survey_years, comp_lo, comp_hi, alpha=0.20,
                        color='orange', label=f'{comparison_label} 68% CI')
        ax.plot(survey_years, comp_med, lw=2, color='orange',
                label=f'{comparison_label} median')

    ax.set_xlabel('Survey Duration [years]')
    ax.set_ylabel(f'Cumulative SLSN {metric_label}')
    ax.set_title(f'MC Rate Uncertainty ({metric_label}): {model_label} | {cadence_label}\n'
                 f'R_ref sampled from Frohmaier+2021: 35 +25/-13 Gpc⁻³ yr⁻¹  '
                 f'(n={n_realizations} realizations)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / f'mc_rate_N_vs_year_{model_label}_{cadence_label}.png',
                    dpi=150)
    plt.show()

    # --- Plot 2: Significance vs survey year (only if comparison provided) ---
    if comparison_detect_vals is not None:
        sigma_fe   = mc_matrix.std(axis=0)
        sigma_comp = comp_matrix.std(axis=0)
        denom      = np.sqrt(sigma_fe**2 + sigma_comp**2)
        denom      = np.where(denom == 0, np.nan, denom)
        significance = np.abs(med - comp_med) / denom

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(survey_years, significance, lw=2, color='purple',
                 label=f'{model_label} vs {comparison_label}')
        ax2.axhline(3.0, ls='--', color='red',     label='3σ threshold')
        ax2.axhline(5.0, ls=':',  color='darkred', label='5σ threshold')
        ax2.set_xlabel('Survey Duration [years]')
        ax2.set_ylabel('Significance (σ)')
        ax2.set_title(f'Model Separation ({metric_label}): {model_label} vs {comparison_label} | {cadence_label}')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        if save_dir:
            plt.savefig(
                Path(save_dir) / f'mc_significance_{model_label}_vs_{comparison_label}_{cadence_label}.png',
                dpi=150)
        plt.show()


def plot_mc_rate_uncertainty_panel(
    pop_data,
    cadence,
    metric_key='detect',
    metric_label='Detections',
    R_ref_nominal=35.0,
    n_realizations=1000,
    survey_years=None,
    seed=42,
    save_dir=None,
):
    """
    One figure per cadence showing all 3 rate models + 3 significance curves.

    Layout (2 panels, stacked vertically):
      Top    : N(SLSNe) vs survey year — fe_dependent, o_dependent, naive
               each with 68% MC uncertainty band
      Bottom : Significance vs survey year — 3 curves:
               fe vs naive, o vs naive, fe vs o
               with 3-sigma and 5-sigma threshold lines

    Parameters
    ----------
    pop_data : dict
        Keyed by model name. Each entry must have:
          'detect' / 'characterize' / 'spectrigger' : dict keyed by cadence -> np.ndarray
          'peak_time' : np.ndarray of relative days (1-3652)
          'z'         : np.ndarray (used for N_injected)
    cadence : str
        Cadence name to plot.
    metric_key : str
        One of 'detect', 'characterize', 'spectrigger'.
    metric_label : str
        Human-readable label for y-axis and title.
    R_ref_nominal : float
        R_ref used to generate populations (Gpc^-3 yr^-1).
    n_realizations : int
        Number of MC draws.
    survey_years : list of float, optional
        Defaults to [1, 2, ..., 10].
    seed : int
        Random seed for reproducibility.
    save_dir : Path or str, optional
        Directory to save figure.

    Notes
    -----
    Colors and linestyles match notebook convention:
      fe_dependent : #4C72B0, solid
      o_dependent  : #DD8452, dashed
      naive        : #55A868, dotted
    Significance curves:
      fe vs naive  : #4C72B0 (blue,   solid)
      o  vs naive  : #DD8452 (orange, dashed)
      fe vs o      : #9B59B6 (purple, dash-dot)
    """
    if survey_years is None:
        survey_years = list(range(1, 11))
    survey_years = np.asarray(survey_years, dtype=float)

    COLORS = {
        'fe_dependent': '#4C72B0',
        'o_dependent':  '#DD8452',
        'naive':        '#55A868',
    }
    LS = {
        'fe_dependent': '-',
        'o_dependent':  '--',
        'naive':        ':',
    }
    MODELS = ['fe_dependent', 'o_dependent', 'naive']

    # --- Compute MC matrix for each model ---
    # mc_matrices[model] shape: (n_realizations, n_years)
    mc_matrices = {}
    medians     = {}
    lo16s       = {}
    hi84s       = {}
    sigmas      = {}

    for model in MODELS:
        detect_vals = pop_data[model][metric_key][cadence]
        peak_times  = pop_data[model]['peak_time']

        if detect_vals is None:
            print(f'  WARNING: {model} × {cadence} × {metric_key} is None — skipping')
            mc_matrices[model] = None
            continue

        R_samples     = _sample_R_ref(n_realizations, seed=seed)

        n_injected      = len(detect_vals)
        V_ref           = n_injected / R_ref_nominal
        base_cumulative = np.array([
            detect_vals[peak_times <= yr * 365.25].sum()
            for yr in survey_years
        ], dtype=float)

        efficiency_t = base_cumulative / n_injected
        mc_mat = efficiency_t[None, :] * R_samples[:, None] * V_ref
        mc_matrices[model] = mc_mat
        medians[model]     = np.median(mc_mat, axis=0)
        lo16s[model]       = np.percentile(mc_mat, 16, axis=0)
        hi84s[model]       = np.percentile(mc_mat, 84, axis=0)
        sigmas[model]      = mc_mat.std(axis=0)

    # --- Build figure: 3 panels stacked ---
    fig, (ax_n, ax_sig, ax_ratio) = plt.subplots(
        3, 1, figsize=(10, 11),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 2, 2], 'hspace': 0.25}
    )

    # --- Top panel: N(SLSNe) vs survey year ---
    for model in MODELS:
        if mc_matrices[model] is None:
            continue
        ax_n.fill_between(
            survey_years, lo16s[model], hi84s[model],
            alpha=0.20, color=COLORS[model]
        )
        ax_n.plot(
            survey_years, medians[model],
            color=COLORS[model], ls=LS[model], lw=2,
            label=f'{model} median'
        )

    ax_n.set_ylabel(f'Cumulative SLSN {metric_label}')
    ax_n.set_title(
        f'MC Rate Uncertainty ({metric_label}) | {cadence}\n'
        f'R_ref ~ Frohmaier+2021: 35 +25/−13 Gpc⁻³ yr⁻¹  (n={n_realizations}, σ=MC+Poisson)'
    )
    ax_n.legend(loc='upper left', fontsize=9)
    ax_n.grid(True, alpha=0.4)

    # --- Bottom panel: significance vs survey year ---
    SIG_PAIRS = [
        ('fe_dependent', 'naive',        '#4C72B0', '-',   'fe vs naive'),
        ('o_dependent',  'naive',        '#DD8452', '--',  'o vs naive'),
        ('fe_dependent', 'o_dependent',  '#9B59B6', '-.',  'fe vs o'),
    ]

    for m1, m2, color, ls, label in SIG_PAIRS:
        if mc_matrices.get(m1) is None or mc_matrices.get(m2) is None:
            continue
        poisson_var = medians[m1] + medians[m2]
        denom = np.sqrt(sigmas[m1]**2 + sigmas[m2]**2 + poisson_var)
        denom = np.where(denom == 0, np.nan, denom)
        sig   = np.abs(medians[m1] - medians[m2]) / denom
        ax_sig.plot(survey_years, sig, color=color, ls=ls, lw=2, label=label)

    ax_sig.axhline(3.0, ls='--', color='red',     lw=1.2, label='3σ')
    ax_sig.axhline(5.0, ls=':',  color='darkred', lw=1.2, label='5σ')
    ax_sig.set_xlabel('Survey Duration [years]')
    ax_sig.set_ylabel('Significance (σ)')
    ax_sig.legend(loc='upper left', fontsize=9)
    ax_sig.grid(True, alpha=0.4)
    ax_sig.set_xlim(survey_years[0], survey_years[-1])

    # --- Ratio panel: R_ref cancels, only Poisson uncertainty remains ---
    # For each realization: ratio = N_m1 / N_m2 = (eff_m1 * R_ref * V_m1) / (eff_m2 * R_ref * V_m2)
    # R_ref_i cancels exactly. Remaining uncertainty is Poisson only.
    # Significance via delta-method on ln(ratio):
    #   sigma_ln_ratio = sqrt(1/N_m1 + 1/N_m2)
    #   sig = |ln(ratio)| / sigma_ln_ratio
    RATIO_PAIRS = [
        ('fe_dependent', 'naive',        '#4C72B0', '-',   'fe / naive'),
        ('o_dependent',  'naive',        '#DD8452', '--',  'o / naive'),
        ('fe_dependent', 'o_dependent',  '#9B59B6', '-.',  'fe / o'),
    ]

    for m1, m2, color, ls, label in RATIO_PAIRS:
        if mc_matrices.get(m1) is None or mc_matrices.get(m2) is None:
            continue
        N1 = medians[m1]
        N2 = medians[m2]
        # Guard: skip time steps where either model has zero detections
        valid = (N1 > 0) & (N2 > 0)
        ratio      = np.where(valid, N1 / N2,       np.nan)
        ln_ratio   = np.where(valid, np.log(ratio), np.nan)
        sigma_frac = np.where(valid, np.sqrt(1.0/N1 + 1.0/N2), np.nan)
        sig_ratio  = np.abs(ln_ratio) / sigma_frac
        ax_ratio.plot(survey_years, sig_ratio, color=color, ls=ls, lw=2, label=label)

    ax_ratio.axhline(3.0, ls='--', color='red',     lw=1.2, label='3σ')
    ax_ratio.axhline(5.0, ls=':',  color='darkred', lw=1.2, label='5σ')
    ax_ratio.set_xlabel('Survey Duration [years]')
    ax_ratio.set_ylabel('Ratio Significance (σ)')
    ax_ratio.set_title('Ratio significance — R_ref cancels, Poisson only')
    ax_ratio.legend(loc='upper left', fontsize=9)
    ax_ratio.grid(True, alpha=0.4)
    ax_ratio.set_xlim(survey_years[0], survey_years[-1])

    plt.tight_layout()
    if save_dir:
        fname = f'mc_panel_3panel_{metric_key}_{cadence}.png'
        plt.savefig(Path(save_dir) / fname, dpi=150)
        print(f'  Saved: {fname}')
    plt.show()
