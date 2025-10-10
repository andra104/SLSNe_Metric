"""
runners.py — MAF execution wrappers for SLSN metrics.

Clean runners without GRB-specific assumptions.
"""
import os
import numpy as np
import pandas as pd
import healpy as hp
from rubin_sim.phot_utils import DustValues
import matplotlib.pyplot as plt
from collections import OrderedDict
import rubin_sim.maf.db as db
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from rubin_sim.maf.metric_bundles import MetricBundle, MetricBundleGroup
from .metrics import SLSN_Detect_Metric, SLSN_CharacterizeMetric, SLSN_SpecTriggerMetric
from .diagnostics import plot_healpix_efficiency


# --------------------------------------------
# Helper function to apply either redshift or distance (directly)
# --------------------------------------------

def get_distance_bounds(d_min=None, d_max=None, z_min=None, z_max=None):
    """
    Return distance bounds in Mpc from either distance or redshift input.

    Parameters
    ----------
    d_min, d_max : float or None
        Distance bounds in Mpc.
    z_min, z_max : float or None
        Redshift bounds.

    Returns
    -------
    (d_min_Mpc, d_max_Mpc) : tuple of floats
    """

    # If distances are provided, return them directly
    if d_min is not None and d_max is not None:
        return d_min, d_max

    # Else convert redshifts to distances
    if z_min is not None and z_max is not None:
        d_min = cosmo.comoving_distance(z_min).to_value(u.Mpc)
        d_max = cosmo.comoving_distance(z_max).to_value(u.Mpc)
        print(f"[INFO] z_min = {z_min:.5f} -> d_min = {d_min:.15f} Mpc")
        print(f"[INFO] z_max = {z_max:.5f} -> d_max = {d_max:.15f} Mpc")
        return d_min, d_max

    raise ValueError("You must provide either (d_min, d_max) or (z_min, z_max)")

def build_filenames(rate_density, 
                        z_min, 
                        z_max,
                        d_min, 
                        d_max,
                        science_case, #"GRBafterglows" for instance
                        testname=None,
                        testname_metric_only=None,
                        ignore_triples=None,                   
                        use_extinction=None,
                        use_kcorrect=None,
                        base_dir=None):
    """
    Construct filenames for saving templates, filename, output dataframe, storage_dir, summary_filename

    Parameters
    ----------


    Returns
    -------
    four strs : path for templates, filename, output dataframe, storage_dir, summary_filename
    """

    if ignore_triples==True:
        testname_metric_only = str(testname_metric_only)+"_it_"+str(ignore_triples)

    #shar
    if base_dir==None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
        
    label = (f"{science_case}_den_{rate_density}"
             f"_d_{d_min}-{d_max}_Mpc"
             f"_z_{z_min}-{z_max}"
             f"_ext_{use_extinction}_kcor_{use_kcorrect}_{testname}")
    print(label)

    storage_dir = os.path.join(base_dir, science_case)
    templates_file = os.path.join(storage_dir, label+"_templates.pkl")
    pop_file = os.path.join(storage_dir, label+"_population.pkl")
    df_file = os.path.join(storage_dir, label+f"_{testname_metric_only}_obs_record")
    summary_filename = os.path.join(storage_dir, label+f"_{testname_metric_only}_multi_summary.csv")
    
    return templates_file, pop_file, df_file, storage_dir, summary_filename


def run_slsn_detect(
    templates,
    population,
    cadences,
    db_dir,
    output_dir,
    store_obs_mode="meta",
    mjd0=60980.5,
    ignore_triples=True,
    save_csv=True,
    make_plots=True,
    clean_temp=True,  # ← ADD THIS
    verbose=True
):
    """
    Run SLSN detection metric across multiple cadences.
    
    Parameters
    ----------
    templates : LC
        Template model with mag_grid
    population : UserPointsSlicer
        Population slicer with slice_points
    cadences : list
        OpSim database names
    db_dir : str
        Directory containing .db files
    output_dir : str
        Where to save results
    store_obs_mode : str
        "none", "meta", "diag", or "full"
    mjd0 : float
        Survey start MJD
    ignore_triples : bool
        Exclude triple visits
    save_csv : bool
        Save detection DataFrame
    make_plots : bool
        Generate diagnostic plots
    clean_temp : bool
        Remove temporary .npz files after completion
    verbose : bool
        Print progress
    
    Returns
    -------
    dict
        {cadence: detection_dataframe}
    """
    import rubin_sim.maf as maf
    from rubin_sim.maf import metric_bundles, db
    import os
    import shutil
    from collections import OrderedDict
    
    results = {}
    n_events = len(population.slice_points['distance'])
    
    for cadence in cadences:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running cadence: {cadence}")
            print(f"{'='*60}")
        
        # Setup temporary directory
        temp_dir = os.path.join(output_dir, f"Metric_temp_{cadence}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Database and results setup
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        results_db = db.ResultsDb(out_dir=temp_dir)
        
        # Create metric
        metric = SLSN_Detect_Metric(
            lc_model=templates,
            mjd0=mjd0,
            use_extinction=True,
            use_kcorrect=False,
            store_obs_mode=store_obs_mode
        )
        
        # Build metric bundle
        constraint = "scheduler_note not like 'long%'" if ignore_triples else ""
        bundle = metric_bundles.MetricBundle(
            metric, population, constraint
        )
        
        # Run metric
        group = metric_bundles.MetricBundleGroup(
            {"Detect_all": bundle}, opsdb,
            out_dir=temp_dir, results_db=results_db
        )
        group.run_all()
        
        # Extract results
        obs_records = dict(metric.obs_records)
        df_obs = _build_detection_dataframe(obs_records, population, templates, mjd0)
        
        # Add detection statistics
        n_detected = df_obs['detected'].sum()
        det_df = df_obs[df_obs['detected'] == 1]
        
        if verbose:
            print(f"\nResults for {cadence}:")
            print(f"  Events simulated: {n_events}")
            print(f"  Events detected:  {n_detected} ({100*n_detected/n_events:.1f}%)")
            if len(det_df) > 0:
                print(f"  Avg filters/detection: {det_df['n_filters_detected'].mean():.1f}")
                print(f"  Avg observations/detection: {det_df['n_observations_detected'].mean():.1f}")
        
        # Save CSV
        if save_csv:
            csv_file = os.path.join(output_dir, f"slsn_detect_{cadence}.csv")
            df_obs.to_csv(csv_file, index=False)
            
            # Also save local efficiency CSV (like shared_utils)
            eff_file = os.path.join(output_dir, f"local_efficiency_{cadence}.csv")
            with open(eff_file, "w") as f:
                f.write("sid,n_filters_detected\n")
                for _, row in df_obs.iterrows():
                    f.write(f"{int(row['sid'])},{int(row['n_filters_detected'])}\n")
            
            if verbose:
                print(f"  Saved: {csv_file}")
                print(f"  Saved: {eff_file}")

        # --- cadence-aware injected peak per filter (only for filters actually seen) ---
        ax1 = DustValues().ax1
        
        def _cadence_peak_for_sid(sid: int, f: str):
            rec = obs_records.get(sid, {})
            F = np.asarray(rec.get('filter', []))
            mask = (F == f)
            if not np.any(mask):
                return np.nan
        
            T = np.asarray(rec.get('mjd_obs', []), float)[mask]
            if T.size == 0:
                return np.nan
        
            tt  = T - mjd0 - population.slice_points['peak_time'][sid]
            idx = int(population.slice_points['file_indx'][sid])
        
            # guard: template must actually have this band
            tpl = templates.data[idx]
            if (f not in tpl) or ('mag' not in tpl[f]) or len(tpl[f]['mag']) == 0:
                return np.nan
        
            mags = templates.interp(tt, f, idx)
            good = np.isfinite(mags)
            if not np.any(good):
                return np.nan
            mags = mags[good]
        
            dm  = float(np.asarray(population.slice_points['distance_modulus'])[sid])
            # prefer precomputed A_f per event; fallback to ax1 * EBV
            A_vec = population.slice_points.get(f'A_{f}', None)
            if A_vec is not None:
                A_f = float(np.asarray(A_vec)[sid])
            else:
                ebv = float(np.asarray(population.slice_points['ebv'])[sid])
                A_f = float(ax1[f] * ebv)
        
            return float(np.nanmin(mags + dm + A_f))
        
        # compute for only the filters that actually appear in obs_records
        filters_seen = sorted({
            flt for rec in obs_records.values()
            for flt in np.unique(rec.get('filter', []))
            if isinstance(rec.get('filter', []), (list, np.ndarray))
        })
        all_sids = df_obs['sid'].dropna().astype(int).tolist()
        for f in filters_seen:
            col = f'injected_peak_ebv_mag_{f}_cadence'
            df_obs[col] = [_cadence_peak_for_sid(sid, f) for sid in all_sids]

        print("Filter testing.....")
        for f in 'ugrizy':
            for suffix in ('_cadence',''):
                col = f'injected_peak_ebv_mag_{f}{suffix}'
                if col in df_obs.columns:
                    print(f"[diag] {col}: finite={np.isfinite(df_obs[col].values).sum()}")

        # Generate all diagnostic plots
        if make_plots:
            _make_all_detection_plots(
                df_obs, population, bundle, cadence, output_dir, verbose=verbose, show_plots = True
            )
        
        results[cadence] = df_obs
        
        # Cleanup temporary files
        if clean_temp:
            if verbose:
                print(f"[CLEANUP] Removing temp directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results, metric


def _make_all_detection_plots(df_obs, population, bundle, cadence, output_dir, verbose=True, show_plots = True):
    """Generate comprehensive diagnostic plots (matching shared_utils functionality)."""
    import healpy as hp
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    def _choose_band_with_data(df, preferred=('r','g','i','z','y','u')):
        # 1) prefer cadence-aware injected columns
        for f in preferred:
            c = f'injected_peak_ebv_mag_{f}_cadence'
            if c in df.columns and np.isfinite(df[c].values).sum() > 20:  # > some small threshold
                return f
        # 2) else try global population columns
        for f in preferred:
            c = f'injected_peak_ebv_mag_{f}'
            if c in df.columns and np.isfinite(df[c].values).sum() > 20:
                return f
        # 3) else fall back to the band with most observed peaks from per_filter_min_mag
        counts = {}
        if 'per_filter_min_mag' in df.columns:
            for f in preferred:
                vals = df['per_filter_min_mag'].apply(lambda d: (d or {}).get(f, np.nan)).astype(float).values
                counts[f] = np.isfinite(vals).sum()
            if counts and max(counts.values()) > 0:
                return max(counts, key=counts.get)
        return 'r'


    if verbose:
        print("   Generating diagnostic plots...")

    os.makedirs(output_dir, exist_ok=True)
    
    # ---- extras used by the 4 extra panels ----
    snr_thresh = 5.0
    
    # ensure an absolute peak time exists; use the metric's mjd0 if available
    if 'peak_mjd' not in df_obs.columns and 'peak_time' in df_obs.columns:
        try:
            mjd0_here = getattr(bundle.metric, 'mjd0', 0.0)
        except Exception:
            mjd0_here = 0.0
        df_obs['peak_mjd'] = mjd0_here + pd.to_numeric(df_obs['peak_time'], errors='coerce')


    # Convenience
    det_mask = df_obs['detected'].astype(bool).values

        # ---- observed peak in a chosen band (r) ----
    filtername = _choose_band_with_data(df_obs)
    print(f"[diag] using band for injected/observed panels: {filtername}")

    det_mask = df_obs['detected'].astype(bool).values
    
    have_dict = ('per_filter_min_mag' in df_obs.columns) and df_obs['per_filter_min_mag'].notna().any()
    
    if have_dict:
        def _pull(d):
            return d.get(filtername, np.nan) if isinstance(d, dict) else np.nan
        m_obs_peak = df_obs['per_filter_min_mag'].apply(_pull).astype(float).values
    else:
        # fallback: recompute from stored visit lists
        m_obs_peak = np.full(len(df_obs), np.nan, float)
        if all(c in df_obs.columns for c in ('filter', 'mag_obs', 'snr_obs')):
            for i, (F, M, S) in enumerate(zip(df_obs['filter'], df_obs['mag_obs'], df_obs['snr_obs'])):
                F = np.asarray(F)
                M = np.asarray(M, float)
                S = np.asarray(S, float)
                sel = (F == filtername) & np.isfinite(M) & (M < 90) & np.isfinite(S) & (S >= snr_thresh)
                m_obs_peak[i] = float(np.min(M[sel])) if np.any(sel) else np.nan


    # 1) HEALPix Efficiency Map
    nside = getattr(population, 'nside', 64)
    npix = hp.nside2npix(nside)

    injected_map = np.zeros(npix)
    detected_map = np.zeros(npix)

    ra_rad = population.slice_points['ra']
    dec_rad = population.slice_points['dec']
    theta = 0.5*np.pi - dec_rad
    phi = ra_rad
    pix_inds = hp.ang2pix(nside, theta, phi)

    for i, pix in enumerate(pix_inds):
        injected_map[pix] += 1
        if bundle.metric_values[i] == 1:
            detected_map[pix] += 1

    eff_map = np.full(npix, hp.UNSEEN, dtype=float)
    mask = injected_map > 0
    eff_map[mask] = detected_map[mask] / injected_map[mask]

    plt.figure(figsize=(12, 6))
    hp.mollview(eff_map, title=f"{cadence} — SLSN Detection Efficiency",
                unit='Efficiency', cmap='viridis', hold=True)
    hp.graticule()
    plt.savefig(os.path.join(output_dir, f"{cadence}_efficiency_healpix.png"),
                dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # 2) Detection Time Distribution (injected vs detected)
    detected_df = df_obs[det_mask]
    if len(detected_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        all_years = (population.slice_points['peak_time'] / 365.25).astype(int) + 1
        det_years = detected_df['year'].values
        bins = np.arange(1, 11)
        ax.hist(all_years, bins=bins, alpha=0.5, label='Injected', color='gray')
        ax.hist(det_years, bins=bins, alpha=0.8, label='Detected', color='green')
        ax.set_xlabel("Survey Year"); ax.set_ylabel("Number of Detections")
        ax.set_title(f"{cadence} — Detection Distribution Over Time")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cadence}_time_distribution.png"),
                    dpi=150, bbox_inches='tight')
        if show_plots:
            plt.show()
        plt.close()

    # 7) Detections per year (bar)
    if 'year' in df_obs.columns and det_mask.any():
        det_per_year = df_obs[det_mask].groupby('year').size()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(det_per_year.index, det_per_year.values, width=0.7,
               edgecolor='black', alpha=0.8)
        ax.set_xlabel('Survey Year'); ax.set_ylabel('Number of Detections')
        ax.set_title(f'{cadence} — Detections by Year')
        ax.set_xticks(range(1, 11)); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{cadence}_detections_per_year.png'), dpi=150)
        if show_plots:
            plt.show()
        plt.close()

    # 3) Declination Distribution (deg; injected vs detected)
    fig, ax = plt.subplots(figsize=(10, 5))
    all_dec_deg = np.degrees(population.slice_points['dec'])
    det_dec_deg = np.degrees(detected_df['dec'].values) if len(detected_df) > 0 else []
    ax.hist(all_dec_deg, bins=50, alpha=0.5, label='Injected', color='gray')
    if len(det_dec_deg) > 0:
        ax.hist(det_dec_deg, bins=50, alpha=0.8, label='Detected', color='red')
    ax.set_xlabel("Declination [deg]"); ax.set_ylabel("Number of Events")
    ax.set_title(f"{cadence} — Declination Distribution")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cadence}_dec_distribution.png"),
                dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # 4) Redshift Distribution + Efficiency vs z
    fig, ax = plt.subplots(figsize=(10, 5))
    all_z = population.slice_points['z']
    det_z = detected_df['z'].values if len(detected_df) > 0 else []

    ax.hist(all_z, bins=30, alpha=0.5, label='Injected', color='gray')
    if len(det_z) > 0:
        ax.hist(det_z, bins=30, alpha=0.8, label='Detected', color='blue')
    ax.set_xlabel('Redshift'); ax.set_ylabel('N Events')
    ax.set_title(f'{cadence} — Redshift Distribution'); ax.legend()
    ax.grid(alpha=0.3)

    z_bins = np.linspace(np.nanmin(all_z), np.nanmax(all_z), 15)
    z_centers = 0.5*(z_bins[:-1] + z_bins[1:])
    eff_vs_z = []
    
    for i in range(len(z_bins)-1):
        m = (df_obs['z'] >= z_bins[i]) & (df_obs['z'] < z_bins[i+1])
        eff_vs_z.append(df_obs.loc[m, 'detected'].mean() if m.any() else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(z_centers, eff_vs_z, 'o-', color='steelblue', markersize=8)
    ax.axhline(df_obs['detected'].mean(), color='red', linestyle='--',
                    label=f"Overall: {df_obs['detected'].mean():.1%}")
    ax.set_xlabel('Redshift'); ax.set_ylabel('Detection Efficiency')
    ax.set_ylim(0, 1); ax.set_title(f'{cadence} — Efficiency vs Redshift')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{cadence}_z_analysis.png"),
                dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # 5) Peak mag vs RA (cadence-aware if present)
    # 5) Peak mag vs RA - ALL FILTERS
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Use minimum across all filters (same as DEC plot)
    if 'per_filter_min_mag' in df_obs.columns and df_obs['per_filter_min_mag'].notna().any():
        def _get_min_across_all_filters(per_filt_dict):
            if not isinstance(per_filt_dict, dict):
                return np.nan
            valid_mags = [v for v in per_filt_dict.values() if np.isfinite(v) and v < 90]
            return float(np.min(valid_mags)) if len(valid_mags) > 0 else np.nan
        
        peak_mags = df_obs['per_filter_min_mag'].apply(_get_min_across_all_filters).values
        mag_label = 'Peak Apparent Magnitude (brightest filter)'
        
    else:
        # Fallback
        print("FALLBACK")

        filtername = _choose_band_with_data(df_obs)
        col_cad = f'injected_peak_ebv_mag_{filtername}_cadence'
        if col_cad in df_obs.columns:
            peak_mags = df_obs[col_cad].values
        elif f'injected_peak_ebv_mag_{filtername}' in df_obs.columns:
            peak_mags = df_obs[f'injected_peak_ebv_mag_{filtername}'].values
        else:
            peak_mags = df_obs['eval_peak_mag'].values
        mag_label = f'Peak Apparent Magnitude ({filtername})'
    
    ras = df_obs['ra'].values
    finite_mask = np.isfinite(peak_mags) & (peak_mags < 90)
    
    ax.scatter(ras[~det_mask & finite_mask], peak_mags[~det_mask & finite_mask], 
               c='gray', s=8, alpha=0.4, label='Not detected')
    ax.scatter(ras[det_mask & finite_mask], peak_mags[det_mask & finite_mask], 
               c='red', s=20, alpha=0.9, edgecolors='black', linewidths=0.5, label='Detected')
    
    ax.set_xlabel('RA (rad)')
    ax.set_ylabel(mag_label)
    ax.set_title(f'{cadence} — Peak Mag vs RA (all filters)')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{cadence}_mag_vs_ra.png'), dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    # 6) Peak mag vs DEC (cadence-aware if present)
    # 6) Peak mag vs DEC - ALL FILTERS (showing all detected events)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # OPTION A: Use minimum magnitude across ALL filters where event was detected
    # This ensures ALL detected events appear in the plot
    if 'per_filter_min_mag' in df_obs.columns and df_obs['per_filter_min_mag'].notna().any():
        def _get_min_across_all_filters(per_filt_dict):
            """Extract minimum magnitude across all filters for this event."""
            if not isinstance(per_filt_dict, dict):
                return np.nan
            valid_mags = [v for v in per_filt_dict.values() if np.isfinite(v) and v < 90]
            return float(np.min(valid_mags)) if len(valid_mags) > 0 else np.nan
        
        peak_mags = df_obs['per_filter_min_mag'].apply(_get_min_across_all_filters).values
        mag_label = 'Peak Apparent Magnitude (brightest filter)'
        print(f"[diag] Using minimum magnitude across all filters for each event")
    else:
        print("FALLBACK")
        # Fallback: use single-filter approach
        filtername = _choose_band_with_data(df_obs)
        print(f"[diag] Falling back to single band: {filtername}")
        
        col_cad = f'injected_peak_ebv_mag_{filtername}_cadence'
        if col_cad in df_obs.columns:
            peak_mags = df_obs[col_cad].values
        elif f'injected_peak_ebv_mag_{filtername}' in df_obs.columns:
            peak_mags = df_obs[f'injected_peak_ebv_mag_{filtername}'].values
        else:
            peak_mags = df_obs['eval_peak_mag'].values
        mag_label = f'Peak Apparent Magnitude ({filtername})'

    #decs = df_obs['dec'].values
    decs = df_obs['dec'].to_numpy(float)

    # Set appropriate axis limits based on units
    if np.nanmax(np.abs(decs)) > np.pi:  # looks like degrees
        xlabel = 'DEC (deg)'
    else:  # looks like radians
        xlabel = 'DEC (rad)'
    
    # Plot ALL events (detected and non-detected)
    # Use finite mask to exclude NaN magnitudes
    finite_mask = np.isfinite(peak_mags) & (peak_mags < 90)
    
    ax.scatter(decs[~det_mask & finite_mask], peak_mags[~det_mask & finite_mask], 
               c='gray', s=8, alpha=0.4, label='Not detected')
    ax.scatter(decs[det_mask & finite_mask], peak_mags[det_mask & finite_mask], 
               c='red', s=20, alpha=0.9, edgecolors='black', linewidths=0.5, label='Detected')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(mag_label)  # Use the dynamic label from above
    ax.set_title(f'{cadence} — Peak Mag vs DEC (all filters)')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{cadence}_mag_vs_dec.png'), dpi=150)
    if show_plots:
        plt.show()
    plt.close()

    def _injected_peak_series(df, population, filt='r'):
        base = f'injected_peak_ebv_mag_{filt}'
        cad  = f'{base}_cadence'
        if cad in df.columns:
            return pd.to_numeric(df[cad], errors='coerce').values, f'Injected peak ({filt}, cadence-aware)'
        if base in df.columns:
            return pd.to_numeric(df[base], errors='coerce').values, f'Injected peak ({filt}, global)'
    
        # slicer fallback: M_abs + DM + A_f * E(B-V)
        sp = population.slice_points
        key_abs = f'peak_mag_abs_{filt}'
        if all(k in sp for k in (key_abs, 'distance_modulus', 'ebv')):
            dm  = np.asarray(sp['distance_modulus'], float)
            ebv = np.asarray(sp['ebv'], float)
            A_vec = sp.get(f'A_{filt}', None)
            if A_vec is not None:
                A = np.asarray(A_vec, float)
            else:
                A = DustValues().ax1[filt] * ebv
            m_abs = np.asarray(sp[key_abs], float)
            if len(m_abs) == len(df):
                return (m_abs + dm + A), f'Injected peak ({filt}, fallback)'
    
        return np.full(len(df), np.nan, float), f'Injected peak ({filt}, unavailable)'

    m_inj, inj_label = _injected_peak_series(df_obs, population, filt=filtername)

    print(f"[diag] finite first_det_mjd: {np.isfinite(pd.to_numeric(df_obs.get('first_det_mjd', np.nan))).sum()}")
    print(f"[diag] per_filter_min_mag present: {'per_filter_min_mag' in df_obs.columns}")
    print(f"[diag] finite observed-peak (r): {np.isfinite(m_obs_peak).sum()}")
    print(f"[diag] finite injected-peak (r): {np.isfinite(m_inj).sum()}")

    # =========================
    # (5) Time gap to first detection (first_det_mjd − peak)
    # =========================
    if ("first_det_mjd" in df_obs.columns) and ("peak_time" in df_obs.columns):
        peak_mjd = (df_obs["peak_time"].astype(float).values + 0.0)  # relative days; add mjd0 externally if desired
        # If your peak_time is relative to mjd0, convert to absolute MJD by adding the metric's mjd0:
        # peak_mjd = detect_metric.mjd0 + df_obs["peak_time"].astype(float).values
        # but we don't have metric here; so we plot the *relative* gap in days, which is identical either way:
        first_det = df_obs["first_det_mjd"].astype(float).values
        # If first_det_mjd was absolute while peak_time is relative, subtract mjd0 outside before passing df_obs.

        # If your pipeline stored absolute first_det_mjd and relative peak_time,
        # you can inject mjd0 in the caller and precompute a column 'peak_mjd' = mjd0 + peak_time.

        dt = first_det - (getattr(df_obs, "peak_mjd", np.zeros_like(peak_mjd)) if hasattr(df_obs, "peak_mjd") else peak_mjd)
        det_mask = df_obs["detected"].astype(bool).values

        vals_nd = dt[(~det_mask) & np.isfinite(dt)]
        vals_d  = dt[( det_mask) & np.isfinite(dt)]

        if np.isfinite(dt).any():
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            if vals_nd.size: ax.hist(vals_nd, bins=np.arange(-5, 30.5, 0.5), alpha=0.5, label="non-detected")
            if vals_d.size:  ax.hist(vals_d,  bins=np.arange(-5, 30.5, 0.5), alpha=0.8, label="detected")
            ax.axvline(0, ls="--", lw=1, color="k")
            ax.set_xlabel("First detection − peak (days)")
            ax.set_ylabel("Number of events")
            ax.set_title(f"{cadence} — Time gap to first detection")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{cadence}_timegap_firstdet.png"), dpi=150, bbox_inches="tight")
            if show_plots: plt.show()
            plt.close()

    # =========================
    # (6) Observed peak magnitude histogram in a band
    # =========================
    # prefer df['per_filter_min_mag'][band]; else recompute from lists
    det_mask = df_obs["detected"].astype(bool).values
    have_dict = "per_filter_min_mag" in df_obs.columns
    if have_dict:
        m_obs_peak = df_obs["per_filter_min_mag"].apply(lambda d: (d or {}).get(filtername, np.nan)).astype(float).values
    else:
        # recompute minimal observed mag in this band at S/N>=snr_thresh
        m_obs_peak = np.full(len(df_obs), np.nan, float)
        if all(c in df_obs.columns for c in ("filter","mag_obs","snr_obs")):
            for i, (F, M, S) in enumerate(zip(df_obs["filter"], df_obs["mag_obs"], df_obs["snr_obs"])):
                F = np.asarray(F); M = np.asarray(M, float); S = np.asarray(S, float)
                sel = (F == filtername) & np.isfinite(M) & (M < 90) & np.isfinite(S) & (S >= snr_thresh)
                m_obs_peak[i] = float(np.min(M[sel])) if np.any(sel) else np.nan

    vals_nd = m_obs_peak[(~det_mask) & np.isfinite(m_obs_peak)]
    vals_d  = m_obs_peak[( det_mask) & np.isfinite(m_obs_peak)]
    if np.isfinite(m_obs_peak).any():
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if vals_nd.size: ax.hist(vals_nd, bins=np.arange(14, 28.5, 0.25), alpha=0.5, label="non-detected")
        if vals_d.size:  ax.hist(vals_d,  bins=np.arange(14, 28.5, 0.25), alpha=0.8, label="detected")
        ax.set_xlabel(f"Observed peak apparent mag ({filtername})")
        ax.set_ylabel("Number of events")
        ax.set_title(f"{cadence} — Observed peak magnitude in {filtername}")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cadence}_peakmag_hist_{filtername}.png"), dpi=150, bbox_inches="tight")
        if show_plots: plt.show()
        plt.close()

    # =========================
    # (7) Implanted vs observed peak (scatter) + residuals
    # =========================
    # Injected (apparent) peak preference: cadence-aware -> global -> fallback from slicer
    base = f"injected_peak_ebv_mag_{filtername}"
    col_cad = f"{base}_cadence"
    if col_cad in df_obs.columns:
        m_inj = df_obs[col_cad].astype(float).values
        inj_label = f"Injected peak ({filtername}, cadence-aware)"
    elif base in df_obs.columns:
        m_inj = df_obs[base].astype(float).values
        inj_label = f"Injected peak ({filtername}, global)"
    else:
        # fallback: use slicer's stored absolute peak + DM + EBV*A_f
        m_inj = np.full(len(df_obs), np.nan, float)
        sp = population.slice_points
        key_abs = f"peak_mag_abs_{filtername}"
        if all(k in sp for k in (key_abs, 'distance_modulus', 'ebv')):
            A_vec = sp.get(f"A_{filtername}", None)
            dm = np.asarray(sp['distance_modulus'], float)
            ebv = np.asarray(sp['ebv'], float)
            if A_vec is not None:
                A = np.asarray(A_vec, float)
            else:
                # rubin_sim extinction coefficients (DustValues.ax1)
                from rubin_sim.phot_utils import DustValues
                A = DustValues().ax1[filtername] * ebv
            m_abs = np.asarray(sp[key_abs], float)
            if len(m_abs) == len(df_obs):
                m_inj = m_abs + dm + A
        inj_label = f"Injected peak ({filtername}, fallback)"

    m_obs = m_obs_peak.copy()
    good = np.isfinite(m_inj) & np.isfinite(m_obs)



    if np.any(good):
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        ax.scatter(m_inj[~det_mask & good], m_obs[~det_mask & good], s=10, alpha=0.5, label="non-detected")
        ax.scatter(m_inj[ det_mask & good], m_obs[ det_mask & good], s=18, alpha=0.8, label="detected")
        lo, hi = np.nanmin(m_inj[good]), np.nanmax(m_inj[good])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.invert_xaxis(); ax.invert_yaxis()
        ax.set_xlabel(inj_label); ax.set_ylabel(f"Observed peak mag ({filtername})")
        ax.set_title(f"{cadence} — Implanted vs observed peaks ({filtername})")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cadence}_implanted_vs_observed_{filtername}.png"), dpi=150, bbox_inches="tight")
        if show_plots: plt.show()
        plt.close()

        # residuals
        res = m_obs[good] - m_inj[good]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.hist(res[~det_mask[good]], bins=np.arange(-3, 3.05, 0.05), alpha=0.5, label="non-detected")
        ax.hist(res[ det_mask[good]], bins=np.arange(-3, 3.05, 0.05), alpha=0.8, label="detected")
        ax.set_xlabel("Observed − Injected (mag)"); ax.set_ylabel("Number of events")
        ax.set_title(f"{cadence} — Peak mag residuals ({filtername})")
        ax.axvline(0, color='k', ls='--', lw=1); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cadence}_peak_residuals_{filtername}.png"), dpi=150, bbox_inches="tight")
        if show_plots: plt.show()
        plt.close()


    if verbose:
        print("   Saved diagnostic plots")


def _build_detection_dataframe(obs_records, population, templates, mjd0):
    """Build cleaned dataframe from obs_records."""
    df = pd.DataFrame.from_dict(obs_records, orient='index').reset_index()
    df.rename(columns={'index': 'sid'}, inplace=True)

    # If any duplicate columns exist, keep the first
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 1) sid: coerce safely to numeric (nullable Int64)
    df['sid'] = pd.to_numeric(df['sid'], errors='coerce')

    # If obs_records brought its own (non-scalar) peak_time, drop it.
    # We'll reattach from population (scalar per sid).
    if 'peak_time' in df.columns:
        del df['peak_time']

    # Map population metadata (guard NA sids / out-of-bounds)
    def _sp_map(key):
        arr = population.slice_points.get(key, None)
        if arr is None:
            return lambda _: np.nan
        n = len(arr)
        return lambda sid: arr[int(sid)] if (pd.notna(sid) and 0 <= int(sid) < n) else np.nan

    for key in ['distance', 'distance_modulus', 'peak_time', 'file_indx', 'ebv', 'ra', 'dec']:
        if key in population.slice_points:
            df[key] = df['sid'].map(_sp_map(key))


    # Coerce list-like obs columns once (robust for plotting/analysis)
    def _as_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, list):
            return x
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return []
        return [x]

    for col in ('filter', 'mjd_obs', 'mag_obs', 'snr_obs'):
        if col in df.columns:
            df[col] = df[col].apply(_as_list)

    # Initialize per-event stats
    df['n_observations_detected'] = 0
    df['n_filters_detected'] = 0
    df['eval_peak_mag'] = np.nan
    df['first_det_mjd'] = np.nan
    df['per_filter_min_mag'] = [{} for _ in range(len(df))]

    # Single pass over rows
    for idx, row in df.iterrows():
        f = np.asarray(row.get('filter', []))
        s = np.asarray(row.get('snr_obs', []), float)
        m = np.asarray(row.get('mag_obs', []), float)
        t = np.asarray(row.get('mjd_obs', []), float)

        good = np.isfinite(s) & (s >= 5) & np.isfinite(m) & (m < 90) & np.isfinite(t)

        # counts
        df.at[idx, 'n_observations_detected'] = int(good.sum())
        df.at[idx, 'n_filters_detected'] = int(len(np.unique(f[good]))) if good.any() else 0

        # cadence-evaluated peak mag (over any band)
        finite_m = m[np.isfinite(m) & (m < 90)]
        if finite_m.size:
            df.at[idx, 'eval_peak_mag'] = float(np.min(finite_m))

        # first detectable visit time
        t_det = t[good]
        if t_det.size:
            df.at[idx, 'first_det_mjd'] = float(np.min(t_det))

        # min mag per LSST band at S/N≥5
        per_band = {}
        if f.size:
            for band in 'ugrizy':
                sel = good & (f == band)
                per_band[band] = float(np.min(m[sel])) if np.any(sel) else np.nan
        df.at[idx, 'per_filter_min_mag'] = per_band

    # 2) year: force numeric first, then nullable integer
    peak_time_num = pd.to_numeric(df['peak_time'], errors='coerce')
    year_float = np.floor(peak_time_num / 365.25)
    df['year'] = pd.Series(year_float, index=df.index).astype('Int64') + 1

    # Finally, convert sid to nullable Int64 at the very end
    df['sid'] = df['sid'].astype('Int64')

    # (Optional) convenience for other plots
    df['peak_mjd'] = (mjd0 + peak_time_num).where(peak_time_num.notna(), np.nan)

    return df

def run_slsn_multi_metrics(
    templates,
    population,
    cadences,
    db_dir,
    output_dir,
    *,
    metrics_list=None,
    mjd0=60980.5,
    ignore_triples=True,
    save_summary=True,
    make_plots=False,
    verbose=True
):
    """
    Run multiple SLSN metrics on cadences and summarize results.
    
    Parameters
    ----------
    templates : LC
        Template model
    population : UserPointsSlicer
        Population slicer
    cadences : list of str
        OpSim database names
    db_dir : str or Path
        Database directory
    output_dir : str or Path
        Output directory
    metrics_list : list of Metric instances, optional
        If None, runs [Detect, Characterize, SpecTrigger]
    mjd0 : float
        Survey start MJD
    ignore_triples : bool
        Exclude triple visits
    save_summary : bool
        Save summary CSV
    make_plots : bool
        Generate HEALPix efficiency maps
    verbose : bool
        Print progress
    
    Returns
    -------
    summary_df : DataFrame
        Summary results across all metrics and cadences
    """
    if metrics_list is None:
        metrics_list = [
            SLSN_Detect_Metric(lc_model=templates, mjd0=mjd0, 
                               store_obs_mode="none"),
            SLSN_CharacterizeMetric(lc_model=templates, mjd0=mjd0, 
                                     store_obs_mode="none"),
            SLSN_SpecTriggerMetric(lc_model=templates, mjd0=mjd0, 
                                    store_obs_mode="none")
        ]
    
    os.makedirs(output_dir, exist_ok=True)
    n_events = len(population.slice_points['distance'])
    note = "scheduler_note not like 'long%'" if ignore_triples else ""
    
    summary_rows = []
    
    for cadence in cadences:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running cadence: {cadence}")
            print(f"{'='*60}")
        
        opsdb = os.path.join(db_dir, f"{cadence}.db")
        temp_dir = os.path.join(output_dir, f"_temp_{cadence}")
        os.makedirs(temp_dir, exist_ok=True)
        resultsDb = db.ResultsDb(out_dir=temp_dir)
        
        for metric in metrics_list:
            metric_name = metric.__class__.__name__
            if verbose:
                print(f"  Running {metric_name}...")
            
            bundle = MetricBundle(metric, population, note)
            group = MetricBundleGroup({metric_name: bundle}, opsdb,
                                       out_dir=temp_dir, results_db=resultsDb)
            group.run_all()
            
            # Summary stats
            n_success = int(bundle.metric_values.sum())
            efficiency = n_success / n_events
            
            summary_rows.append({
                'cadence': cadence,
                'metric': metric_name,
                'n_events': n_events,
                'n_success': n_success,
                'efficiency': efficiency
            })
            
            if verbose:
                print(f"    Efficiency: {100*efficiency:.1f}% ({n_success}/{n_events})")
            
            # Optional HEALPix plot
            if make_plots:
                _make_healpix_efficiency_map(
                    bundle, population, cadence, metric_name, output_dir
                )
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    summary_df = pd.DataFrame(summary_rows)
    
    if save_summary:
        summary_file = os.path.join(output_dir, "slsn_multi_metrics_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        if verbose:
            print(f"\nSummary saved: {summary_file}")
    
    return summary_df


def _make_healpix_efficiency_map(bundle, population, cadence, metric_name, output_dir):
    """Generate HEALPix efficiency map."""
    import healpy as hp
    
    nside = getattr(population, 'nside', 64)
    npix = hp.nside2npix(nside)
    
    injected_map = np.zeros(npix)
    success_map = np.zeros(npix)
    
    ra_rad = population.slice_points['ra']
    dec_rad = population.slice_points['dec']
    theta = 0.5 * np.pi - dec_rad
    phi = ra_rad
    pix_inds = hp.ang2pix(nside, theta, phi)
    
    for i, pix in enumerate(pix_inds):
        injected_map[pix] += 1
        if bundle.metric_values[i] == 1:
            success_map[pix] += 1
    
    eff_map = np.zeros(npix)
    mask = injected_map > 0
    eff_map[mask] = success_map[mask] / injected_map[mask]
    eff_map[~mask] = hp.UNSEEN
    
    hp.mollview(eff_map, title=f'{cadence} — {metric_name} Efficiency',
                unit='Efficiency', cmap='viridis')
    hp.graticule()
    plt.savefig(os.path.join(output_dir, f'{cadence}_{metric_name}_healpix.png'), dpi=150)
    plt.show()
    plt.close()