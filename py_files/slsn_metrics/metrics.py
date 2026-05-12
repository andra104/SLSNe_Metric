"""
metrics.py — MAF metrics for SLSN detection, characterization, and spectroscopic triggers.

Core evaluator with cached SED synthesis, plus three metric classes following
Firth+2015 and Inserra+2024 criteria.
"""
import numpy as np
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.phot_utils import DustValues
from .constants import PHASE_BIN_STEP, phase_bucket_vec
from .model import synthesize_mag_at_z, map_catalog_to_lsst_band, get_color_offset


# =============================================================================
# Detection logic
# =============================================================================

def detect_slsn(filters, snr, times, mags, obs_record):
    """
    SLSN detection criterion (Firth+2015).
    
    Returns True if:
    1. ≥2 filters with SNR≥5 detections
    2. Rising light curve observed (consecutive brightening)
    3. Observations span ≥15 days
    
    Parameters
    ----------
    filters : array
        Filter names
    snr : array
        Signal-to-noise ratios
    times : array
        MJD times
    mags : array
        Apparent magnitudes
    obs_record : dict
        Observation metadata
    
    Returns
    -------
    detected : bool
    """
    # 1) Multi-band detection
    detected_filters = []
    for f in np.unique(filters):
        mask = (filters == f) & (snr >= 5)
        if np.sum(mask) >= 1:
            detected_filters.append(f)
    
    if len(detected_filters) < 2:
        return False
    
    # 2) Observe rising light curve
    rising_observed = False
    for f in detected_filters:
        mask = (filters == f) & (snr >= 5)
        t_filt = times[mask]
        m_filt = mags[mask]
        
        if len(t_filt) >= 2:
            order = np.argsort(t_filt)
            t_sorted = t_filt[order]
            m_sorted = m_filt[order]
            
            for i in range(len(m_sorted) - 1):
                dt = t_sorted[i+1] - t_sorted[i]
                dm = m_sorted[i+1] - m_sorted[i]
                
                # Rising: dm < -0.1 mag over 0.5 < dt < 30 days
                if dm < -0.1 and dt > 0.5 and dt < 30:
                    rising_observed = True
                    break
        
        if rising_observed:
            break
    
    if not rising_observed:
        return False
    
    # 3) Temporal baseline
    detected_mask = snr >= 5
    if np.sum(detected_mask) < 2:
        return False
    
    duration = np.ptp(times[detected_mask])
    if duration < 15:
        return False
    
    return True

# def detect_slsn(filters, snr, times, mags, obs_record):
#     """
#     TEMPORARY SIMPLE DETECTION - Just for testing!
    
#     Returns True if:
#     - At least 2 observations with SNR >= 5
#     - In at least 2 different filters
    
#     (This removes rising, duration, and all other complex checks)
#     """
#     # Just check for multi-band detections
#     good = snr >= 5
    
#     if np.sum(good) < 2:
#         return False
    
#     detected_filters = np.unique(filters[good])
    
#     if len(detected_filters) < 2:
#         return False
    
#     return True  # That's it! Super simple.
# =============================================================================
# Core evaluator with cached synthesis
# =============================================================================

def synthesize_mag_at_z_cached(cache, sed_grid, phase_bin_idx, z, filt, 
                                *, step=PHASE_BIN_STEP):
    """
    Cached SED synthesis using phase bin indices as keys.
    
    Cache key: (id(sed_grid), filt, bin_idx, round(z, 5))
    """
    key = (id(sed_grid), str(filt), int(phase_bin_idx), round(float(z), 5))
    if key in cache:
        return cache[key]
    phase_center = float(phase_bin_idx) * float(step)
    val = synthesize_mag_at_z(sed_grid, phase_center, z, filt)
    cache[key] = val
    return val

def _m52snr(mag: np.ndarray, m5: np.ndarray) -> np.ndarray:
    """LSST single-visit SNR from 5σ depth: SNR ≈ 5 * 10^{0.4(m5 - m)}."""
    snr = np.full_like(mag, np.nan, dtype=float)
    finite = np.isfinite(mag) & np.isfinite(m5)
    snr[finite] = 5.0 * (10.0 ** (0.4 * (m5[finite] - mag[finite])))
    return snr

def evaluate_slsn(self, dataSlice, slice_point, return_full_obs=True):
    """
    Direct magnitude evaluation - NO SED synthesis.
    
    Uses template magnitudes in catalog bands, maps to LSST filters,
    and applies distance modulus + extinction.
    
    Returns (snr, filters, mjds, obs_record|None).
    
    Parameters
    ----------
    self : SLSN_Base_Metric
        Metric instance with lc_model, _mag_cache, etc.
    dataSlice : structured array
        OpSim observations
    slice_point : dict
        Event properties (z, peak_time, file_indx, ebv, A_f, etc.)
    return_full_obs : bool
        If True, return full observation record
    
    Returns
    -------
    snr : array
        Signal-to-noise ratios
    filters : array
        Filter names
    mjds : array
        Observation times
    obs_record : dict or None
        Full observation metadata (if requested)
    """
    # Get event parameters
    tpl_idx = int(slice_point['file_indx'])
    z = float(slice_point['z'])
    peak_time = float(slice_point['peak_time'])
    dm = float(slice_point.get('distance_modulus', 0.0))
    ebv = float(slice_point['ebv'])
    
    # Get template data structure
    template = self.lc_model.data[tpl_idx]
    
    # Find available catalog bands in this template
    available_bands = [b for b in template.keys() 
                      if isinstance(template[b], dict) and 'ph' in template[b]]
    
    # Detect physical template track: .data is empty, .sed_grid is populated
    # MOSFiT posteriors are constrained by post-peak photometry only (PHASE_GRID = 1..400).
    # Pre-peak observations (time_rel < 1.0) return np.nan from synthesize_mag_at_z.
    # Detection efficiency from the physical path is therefore a lower bound on the
    # true high-z detection rate — this is intentional and scientifically correct.
    is_physical = (
        len(available_bands) == 0
        and hasattr(self.lc_model, 'sed_grid')
        and bool(self.lc_model.sed_grid)
    )

    if not available_bands and not is_physical:
        if return_full_obs:
            return np.array([]), np.array([]), np.array([]), None
        return np.array([]), np.array([]), np.array([])

    # Process observations
    mjds = dataSlice[self.mjdCol]
    filts = dataSlice[self.filterCol]
    m5 = dataSlice[self.m5Col]

    # Rest-frame time relative to peak
    time_rel = (mjds - self.mjd0 - peak_time) / (1.0 + z)

    mags = np.full_like(mjds, np.nan, dtype=float)

    # For each observation
    if is_physical:
        # Physical template path: full SED synthesis via synthesize_mag_at_z().
        # Takes LSST filter directly — no catalog band lookup or color offset needed.
        # synthesize_mag_at_z() applies luminosity distance internally — do NOT add dm.
        # Pre-peak phases (time_rel < 1.0) return np.nan — intentional lower bound.
        sed_entry = self.lc_model.sed_grid[tpl_idx]
        for i, (t, filt) in enumerate(zip(time_rel, filts)):
            # Quantize phase to 0.2-day bins for cache key — coarser than
            # physical PHASE_GRID spacing but negligible science impact.
            phase_bin_idx = int(phase_bucket_vec(np.array([t]), return_index=True)[0])
            m_app = synthesize_mag_at_z_cached(
                self._mag_cache, sed_entry, phase_bin_idx, z, filt
            )
            if not np.isfinite(m_app):
                continue
            A_filt = float(slice_point.get(f'A_{filt}', 0.0))
            if A_filt == 0.0:
                dust_model = DustValues()
                A_filt = dust_model.ax1[filt] * ebv
            mags[i] = m_app + A_filt
    else:
        # GP template path: catalog band interpolation — unchanged.
        for i, (t, filt) in enumerate(zip(time_rel, filts)):
            # Find best catalog band for this LSST filter
            catalog_band = None
            for cat_b in available_bands:
                lsst_b = map_catalog_to_lsst_band(cat_b)
                if lsst_b == filt:
                    catalog_band = cat_b
                    break

            if catalog_band is None:
                # Try to find any usable band and apply color correction
                for cat_b in available_bands:
                    lsst_b = map_catalog_to_lsst_band(cat_b)
                    if lsst_b:  # Any valid mapping
                        catalog_band = cat_b
                        break

            if catalog_band is None:
                continue  # No usable band, leave as NaN

            # Interpolate absolute magnitude from template
            M_abs = self.lc_model.interp(t, catalog_band, tpl_idx)

            if not np.isfinite(M_abs):
                continue

            # Apply color offset if bands don't match exactly
            lsst_mapped = map_catalog_to_lsst_band(catalog_band)
            color_offset = get_color_offset(catalog_band, filt) if lsst_mapped != filt else 0.0

            # Apparent magnitude = Absolute + DM + Extinction + Color
            m_app = M_abs + dm + color_offset

            # Add extinction (per-filter if available, else generic)
            A_filt = slice_point.get(f'A_{filt}', 0.0)
            if A_filt == 0.0:
                # Fallback: use dust model
                dust_model = DustValues()
                A_filt = dust_model.ax1[filt] * ebv

            m_app += A_filt

            mags[i] = m_app
    
    # Calculate SNR
    snr = _m52snr(mags, m5)
    
    # Build obs record
    if return_full_obs:
        obs_record = {
            'mjd_obs': mjds,
            'mag_obs': mags,
            'snr_obs': snr,
            'filter': filts,
            'available_bands': ','.join(available_bands)
        }
        return snr, filts, mjds, obs_record
    
    return snr, filts, mjds

# =============================================================================
# Base metric class
# =============================================================================

DEFAULT_STORE_MODE = "meta"

class SLSN_Base_Metric(BaseMetric):
    """
    Base metric for SLSN simulations with cached SED synthesis.
    
    Parameters
    ----------
    lc_model : LC
        Template model instance
    mjdCol, m5Col, filterCol, nightCol : str
        OpSim column names
    mjd0 : float
        Survey start MJD
    use_extinction : bool
        Apply Galactic extinction
    use_kcorrect : bool
        Apply K-corrections (experimental)
    store_obs_mode : str
        Storage mode: "none" | "meta" | "diag" | "full"
    diag_store : bool
        Enable diagnostic sampling
    """
    
    def __init__(self, *, metricName='BaseSLSNMetric',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night', mjd0=60980.5,
                 lc_model=None, use_extinction=True, use_kcorrect=False,
                 k_correct_type=None, k_correct_arg=None,
                 store_obs_mode=DEFAULT_STORE_MODE, diag_store=False,
                 **kwargs):
        
        if lc_model is None:
            raise ValueError("lc_model required")
        
        self.store_obs_mode = store_obs_mode
        self.diag_store = bool(diag_store)
        
        if self.store_obs_mode == "diag" and not self.diag_store:
            self.diag_store = True
        
        self._mag_cache = {}
        self.lc_model = lc_model
        self.ax1 = DustValues().ax1
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.mjd0 = mjd0
        self.use_extinction = use_extinction
        self.use_kcorrect = use_kcorrect
        self.k_correct_type = k_correct_type
        self.k_correct_arg = k_correct_arg
        
        cols = [mjdCol, m5Col, filterCol, nightCol]
        super().__init__(col=cols, metric_name=metricName,
                         units='Detection Efficiency', **kwargs)
    
    def detect(self, filters, snr, times, obs_record):
        """SLSN detection logic (override in subclasses if needed)."""
        mags = np.asarray(obs_record.get('mag_obs', []))
        return detect_slsn(filters, snr, times, mags, obs_record)

# =============================================================================
# Detection metric
# =============================================================================

class SLSN_Detect_Metric(SLSN_Base_Metric):
    """
    Binary detection metric (Firth+2015 criteria).
    
    Returns 1.0 if detected, 0.0 otherwise.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = kwargs.get('metricName', 'SLSN_Detect')
        self.obs_records = {}
    
    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate_slsn(
            self, dataSlice, slice_point, return_full_obs=True
        )
        if obs_record is None or snr.size == 0:
            return self.badval
        
        detected = self.detect(filters, snr, times, obs_record)
        
        # Always build minimal meta
        meta = {
            'detected': bool(detected),
            'sid': int(slice_point['sid']),
            'file_indx': int(slice_point['file_indx']),
            'z': float(slice_point['z']),
            'ra': float(slice_point['ra']),
            'dec': float(slice_point['dec']),
            'distance_Mpc': float(slice_point['distance']),
            'ebv': float(slice_point['ebv']),
            'peak_time': float(slice_point['peak_time']),
        }
        
        mode = getattr(self, "store_obs_mode", DEFAULT_STORE_MODE)
        
        if mode == "none":
            self.obs_records[slice_point['sid']] = meta
        elif mode == "meta":
            self.obs_records[slice_point['sid']] = meta
        elif mode == "diag":
            diag_keys = ("diag_sample_mjd", "diag_sample_mag", 
                         "diag_sample_snr", "diag_sample_filter")
            diag = {k: obs_record.get(k, []) for k in diag_keys}
            rec = {**meta, **diag}
            self.obs_records[slice_point['sid']] = rec
        elif mode == "full":
            full = dict(obs_record)
            full.update(meta)
            self.obs_records[slice_point['sid']] = full
        else:
            self.obs_records[slice_point['sid']] = meta
        
        return 1.0 if detected else 0.0

# =============================================================================
# Characterization metric
# =============================================================================

class SLSN_CharacterizeMetric(SLSN_Base_Metric):
    """
    Characterization metric (Inserra+2024 criteria).
    
    Requires detection plus:
    - ≥5 epochs with SNR≥5
    - ≥3 different filters
    - ≥2 epochs within ±10 days of peak
    - ≥1 epoch beyond +30 days post-peak
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = 'SLSN_Characterize'
        self.obs_records = {}
    
    def run(self, dataSlice, slice_point=None):
            snr, filters, times, obs_record = evaluate_slsn(
                self, dataSlice, slice_point, return_full_obs=True
            )
            
            # Initialize as failed
            characterized = False
            n_epochs = 0
            n_filters_char = 0
            
            # If no observations at all
            if obs_record is None or len(snr) == 0:
                if self.store_obs_mode == "full":
                    # Store minimal record
                    self.obs_records[slice_point['sid']] = {
                        'characterized': False,
                        'sid': int(slice_point['sid']),
                        'z': float(slice_point['z']),
                        'peak_time': float(slice_point['peak_time']),
                        'peak_mjd': self.mjd0 + float(slice_point['peak_time']),
                        'mjd_obs': [],
                        'mag_obs': [],
                        'snr_obs': [],
                        'filter': []
                    }
                return 0.0
            
            # Must pass detection first
            detected = detect_slsn(filters, snr, times,
                                   np.asarray(obs_record['mag_obs'], float),
                                   obs_record)
            
            if detected:
                # Characterization criteria
                good = snr >= 5
                n_epochs = int(np.sum(good))
                n_filters_char = np.unique(filters[good]).size if np.any(good) else 0
                
                peak_mjd = self.mjd0 + float(slice_point['peak_time'])
                mjd_obs = np.asarray(obs_record['mjd_obs'], float)
                
                # Check all criteria
                has_enough_epochs = n_epochs >= 5
                has_enough_filters = n_filters_char >= 3
                
                near_peak = good & (np.abs(mjd_obs - peak_mjd) <= 10.0)
                has_near_peak = np.sum(near_peak) >= 2
                
                post_peak = good & (mjd_obs > (peak_mjd + 30.0))
                has_post_peak = np.sum(post_peak) >= 1
                
                # All criteria must pass
                characterized = (has_enough_epochs and has_enough_filters and 
                                has_near_peak and has_post_peak)
            
            # ALWAYS store the record (pass or fail)
            if self.store_obs_mode == "full":
                obs_record.update({
                    'characterized': characterized,
                    'n_epochs': n_epochs,
                    'n_filters_char': n_filters_char,
                    'sid': int(slice_point['sid']),
                    'z': float(slice_point['z']),
                    'ra': float(slice_point['ra']),
                    'dec': float(slice_point['dec']),
                    'distance_Mpc': float(slice_point['distance']),
                    'ebv': float(slice_point['ebv']),
                    'peak_time': float(slice_point['peak_time']),
                })
                self.obs_records[slice_point['sid']] = obs_record
            
            return 1.0 if characterized else 0.0

# =============================================================================
# Villar+2018 light curve quality metrics
# =============================================================================

class SLSN_VillarMetric(SLSN_Base_Metric):
    """
    Light curve quality metrics from Villar, Nicholl & Berger 2018
    (ApJ 869, 166), Table 4.

    Implements three of their 19 metrics that best predict parameter
    recoverability. Does NOT require detection (Firth+2015) — these are
    independent photometric quality cuts on the raw cadence sampling,
    exactly as defined in Villar+2018.

    Sub-criteria (each stored independently):
      M1: > n_det_total SNR>=5 detections across all filters
          (their baseline "discovery" criterion, ~9600/yr in WFD)
      M2: > n_det_peak SNR>=5 detections within 1 mag of peak brightness
          ("during peak", ~2690/yr in WFD)
      M3: Measurable duration in r-band — both rise AND decline by
          1 magnitude observed (their highest information-content metric,
          ~960/yr in WFD)

    The metric returns 1.0 if ANY sub-criterion passes. All three
    sub-criterion results are stored in obs_records for comparison.

    Parameters
    ----------
    n_det_total : int
        Minimum total SNR>=5 detections. Default 10 (Villar+2018 Table 4).
    n_det_peak : int
        Minimum detections within 1 mag of peak. Default 20 (Villar+2018).
    peak_window_mag : float
        "Near peak" defined as within this many mags of peak. Default 1.0.

    Notes
    -----
    obs_records storage: only in "full" or "meta" mode (not "none").
    Unlike SLSN_Detect_Metric, per-event records are not needed for
    the production count metric — summary metric_values are sufficient.

    References
    ----------
    Villar, Nicholl & Berger 2018, ApJ 869, 166, Table 4.
    """

    def __init__(self, n_det_total=10, n_det_peak=20,
                 peak_window_mag=1.0, **kwargs):
        super().__init__(**kwargs)
        self.metricName = 'SLSN_Villar'
        self.n_det_total = n_det_total
        self.n_det_peak = n_det_peak
        self.peak_window_mag = peak_window_mag
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate_slsn(
            self, dataSlice, slice_point, return_full_obs=True
        )

        m1_pass = False  # >10 total detections
        m2_pass = False  # >20 during peak
        m3_pass = False  # measurable duration in r

        if obs_record is None or snr.size == 0:
            if self.store_obs_mode == "full":
                self.obs_records[slice_point['sid']] = {
                    'villar_pass': False,
                    'm1_total_det': False,
                    'm2_peak_det': False,
                    'm3_duration_r': False,
                    'sid': int(slice_point['sid']),
                    'z': float(slice_point['z']),
                    'peak_time': float(slice_point['peak_time']),
                }
            return 0.0

        mags    = np.asarray(obs_record['mag_obs'], float)
        mjd_obs = np.asarray(obs_record['mjd_obs'], float)
        good    = snr >= 5

        # --- M1: >n_det_total SNR>=5 detections ---
        m1_pass = int(np.sum(good)) > self.n_det_total

        # --- M2: >n_det_peak detections within 1 mag of peak ---
        peak_mjd = self.mjd0 + float(slice_point['peak_time'])
        if np.any(good):
            peak_mag_obs = np.nanmin(mags[good]) if np.any(good) else np.nan
            if np.isfinite(peak_mag_obs):
                near_peak = good & (mags <= peak_mag_obs + self.peak_window_mag)
                m2_pass = int(np.sum(near_peak)) > self.n_det_peak

        # --- M3: Measurable duration in r-band ---
        # Both rise (pre-peak brightening by >=1 mag) and decline
        # (post-peak fading by >=1 mag) must be observed in r-band.
        r_good = good & (filters == 'r')
        if np.sum(r_good) >= 2:
            t_r = mjd_obs[r_good]
            m_r = mags[r_good]
            order = np.argsort(t_r)
            t_r = t_r[order]
            m_r = m_r[order]

            peak_mag_r = np.nanmin(m_r)
            peak_t_r   = t_r[np.argmin(m_r)]

            # Rise: at least one obs >=1 mag fainter than peak BEFORE peak
            pre  = t_r < peak_t_r
            post = t_r > peak_t_r

            rise_measured     = np.any(pre)  and np.any(m_r[pre]  >= peak_mag_r + 1.0)
            decline_measured  = np.any(post) and np.any(m_r[post] >= peak_mag_r + 1.0)
            m3_pass = rise_measured and decline_measured

        villar_pass = m1_pass or m2_pass or m3_pass

        # Deliberate: only store per-event records in full/meta mode.
        # Unlike SLSN_Detect_Metric (which stores in all modes including "none"),
        # VillarMetric sub-criteria are aggregated counts — per-event records
        # are only needed for diagnostic analysis, not routine summary runs.
        if self.store_obs_mode in ("full", "meta"):
            obs_record.update({
                'villar_pass':    villar_pass,
                'm1_total_det':   m1_pass,
                'm2_peak_det':    m2_pass,
                'm3_duration_r':  m3_pass,
                'sid':  int(slice_point['sid']),
                'z':    float(slice_point['z']),
                'ra':   float(slice_point['ra']),
                'dec':  float(slice_point['dec']),
                'distance_Mpc': float(slice_point['distance']),
                'ebv':  float(slice_point['ebv']),
                'peak_time': float(slice_point['peak_time']),
            })
            self.obs_records[slice_point['sid']] = obs_record

        return 1.0 if villar_pass else 0.0


# =============================================================================
# Spectroscopic trigger metric (redesigned)
# =============================================================================


# =============================================================================
# PLAsTiCC / ELAsTiCC alert-pipeline trigger metric
# =============================================================================

class SLSN_ELAsTiCC_Metric(SLSN_Base_Metric):
    """
    Alert-pipeline trigger criterion from PLAsTiCC/ELAsTiCC
    (Kessler+2019, PASP 131, 094501, Section 6.3).

    Confirmed by Ved Shah (private communication) as identical
    for ELAsTiCC. Cite both Kessler+2019 and Shah+2024/2025.

    Asks: would the Rubin alert pipeline write an alert for this event?
    This is a necessary but not sufficient condition for any science
    follow-up. Much less strict than SLSN_Detect_Metric (Firth+2015).

    Criterion:
      >= 2 observations with |S/N| > 3, separated by >= 30 minutes.
      Absolute value of S/N is used — both flux increases and decreases
      count, permissive by design to include all variable transients.

    Note: PLAsTiCC used S/N_true rather than measured S/N, flagged as
    a known mistake in Kessler+2019 footnote 57. We use our simulated
    SNR which is the equivalent quantity in this pipeline.

    References
    ----------
    Kessler+2019 : PASP 131, 094501, Section 6.3 (PLAsTiCC trigger model)
    Shah+2024    : 2024MNRAS.528.1109S (ELAsTiCC, confirmed same criteria)
    """

    def __init__(self, min_snr=3.0, min_sep_minutes=30.0, **kwargs):
        super().__init__(**kwargs)
        self.metricName   = 'SLSN_ELAsTiCC'
        self.min_snr      = min_snr
        self.min_sep_days = min_sep_minutes / 1440.0   # convert minutes -> days
        self.obs_records  = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate_slsn(
            self, dataSlice, slice_point, return_full_obs=True
        )

        elasticc_pass = False

        if obs_record is None or snr.size == 0:
            if self.store_obs_mode in ("full", "meta"):
                self.obs_records[slice_point['sid']] = {
                    'elasticc_pass': False,
                    'sid':       int(slice_point['sid']),
                    'z':         float(slice_point['z']),
                    'ra':        float(slice_point['ra']),
                    'dec':       float(slice_point['dec']),
                    'peak_time': float(slice_point['peak_time']),
                }
            return 0.0

        # |S/N| > min_snr — absolute value, both flux directions count
        good      = np.abs(snr) > self.min_snr
        good_times = times[good]

        # Need >= 2 such detections separated by >= min_sep_days (30 min default)
        if np.sum(good) >= 2:
            t_sorted = np.sort(good_times)
            for i in range(len(t_sorted) - 1):
                if t_sorted[i + 1] - t_sorted[i] >= self.min_sep_days:
                    elasticc_pass = True
                    break

        if self.store_obs_mode in ("full", "meta"):
            self.obs_records[slice_point['sid']] = {
                'elasticc_pass': elasticc_pass,
                'n_good_det':    int(np.sum(good)),
                'sid':       int(slice_point['sid']),
                'z':         float(slice_point['z']),
                'ra':        float(slice_point['ra']),
                'dec':       float(slice_point['dec']),
                'peak_time': float(slice_point['peak_time']),
            }

        return 1.0 if elasticc_pass else 0.0

class SLSN_SpecTriggerMetric(SLSN_Base_Metric):
    """
    Spectroscopic trigger metric — redesigned based on SLSN physics.

    Asks whether Rubin detected enough of the SLSN near peak to
    substantiate a spectroscopic follow-up request. This requires
    catching the event at its highest energy output — when temperature
    and brightness best distinguish it from contaminants — with enough
    observations to confirm the slow evolution characteristic of SLSNe.

    Distinct from CharacterizeMetric, which requires coverage of the
    full light curve before and after peak. SpecTrigger only asks
    about the peak window when spectroscopy would be actionable.

    Requires detection first (Firth+2015), then:
      1. >=n_near_peak SNR>=5 detections within ±peak_window days of peak
         (must catch it near maximum, not just on decline)
      2. >=2 filters detected near peak
         (color information to distinguish from contaminants)
      3. Peak apparent magnitude brighter than mag_limit
         (spectrograph feasibility — 23.0 is educated guess for 4-8m ToO;
          pending confirmation from instrumentation team)
      4. Slow evolution: Δmag < decline_limit over any 30-day window
         near peak (distinguishes SLSNe from faster transients)

    Parameters
    ----------
    mag_limit : float
        Faintest apparent magnitude for spectroscopic follow-up.
        Default 23.0 — educated estimate for 4-8m class telescope ToO.
        ⚠ Pending confirmation from instrumentation collaborators.
    peak_window : float
        Days around peak to require detections. Default 20.0.
        Justified by 15-50 day rise times (Nicholl+2021).
    n_near_peak : int
        Minimum detections within peak_window. Default 2.
    decline_limit : float
        Max Δmag over 30 days near peak. Default 1.0.
        SLSNe have tdur > 50 days (Nicholl+2021).

    References
    ----------
    Nicholl+2021 : rise timescales 15-50 days, T=12,000-15,000K at peak
    Aamer+2025   : temperature evolution, blue continuum at peak
    Villar+2018  : peak magnitude distribution 19-23 mag in WFD
    """

    def __init__(self, mag_limit=23.0, peak_window=20.0,
                 n_near_peak=2, decline_limit=1.0, **kwargs):
        super().__init__(**kwargs)
        self.metricName = 'SLSN_SpecTrigger'
        self.mag_limit    = mag_limit
        self.peak_window  = peak_window
        self.n_near_peak  = n_near_peak
        self.decline_limit = decline_limit
        self.obs_records  = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate_slsn(
            self, dataSlice, slice_point, return_full_obs=True
        )

        spec_triggered      = False
        near_peak_pass      = False
        multifilter_pass    = False
        brightness_pass     = False
        slow_evolution_pass = False

        if obs_record is None or snr.size == 0:
            if self.store_obs_mode == "full":
                self.obs_records[slice_point['sid']] = {
                    'spec_trigger': False,
                    'sid': int(slice_point['sid']),
                    'z': float(slice_point['z']),
                    'peak_time': float(slice_point['peak_time']),
                }
            return 0.0

        mags    = np.asarray(obs_record['mag_obs'], float)
        mjd_obs = np.asarray(obs_record['mjd_obs'], float)

        # Must pass detection first (Firth+2015)
        detected = detect_slsn(filters, snr, mjd_obs, mags, obs_record)

        if detected:
            peak_mjd = self.mjd0 + float(slice_point['peak_time'])
            good     = snr >= 5
            near     = good & (np.abs(mjd_obs - peak_mjd) <= self.peak_window)

            # --- Criterion 1: >=n_near_peak detections within ±peak_window ---
            near_peak_pass = int(np.sum(near)) >= self.n_near_peak

            # --- Criterion 2: >=2 filters near peak ---
            if np.any(near):
                n_filters_near = len(np.unique(filters[near]))
                multifilter_pass = n_filters_near >= 2

            # --- Criterion 3: Brightness ---
            # Use apparent mag from slice_point if available (injected),
            # fall back to observed peak near window
            inj_col = 'peak_app_mag_ebv_r'
            if inj_col in slice_point:
                peak_mag = float(slice_point[inj_col])
            elif np.any(near):
                peak_mag = float(np.nanmin(mags[near]))
            else:
                peak_mag = np.nan

            if np.isfinite(peak_mag):
                brightness_pass = peak_mag <= self.mag_limit

            # --- Criterion 4: Slow evolution near peak ---
            # Any 30-day window near peak with Δmag < decline_limit
            window = good & (np.abs(mjd_obs - peak_mjd) <= 45.0)
            if np.sum(window) >= 2:
                t_w = mjd_obs[window]
                m_w = mags[window]
                order = np.argsort(t_w)
                t_w = t_w[order]
                m_w = m_w[order]
                for j in range(len(t_w) - 1):
                    if (t_w[j+1] - t_w[j]) <= 35.0:
                        dm = abs(m_w[j+1] - m_w[j])
                        if dm < self.decline_limit:
                            slow_evolution_pass = True
                            break

            spec_triggered = (near_peak_pass and multifilter_pass
                              and brightness_pass and slow_evolution_pass)

        if self.store_obs_mode in ("full", "meta"):
            obs_record.update({
                'spec_trigger':       spec_triggered,
                'near_peak_pass':     near_peak_pass,
                'multifilter_pass':   multifilter_pass,
                'brightness_pass':    brightness_pass,
                'slow_evolution_pass': slow_evolution_pass,
                'sid':  int(slice_point['sid']),
                'z':    float(slice_point['z']),
                'ra':   float(slice_point['ra']),
                'dec':  float(slice_point['dec']),
                'distance_Mpc': float(slice_point['distance']),
                'ebv':  float(slice_point['ebv']),
                'peak_time': float(slice_point['peak_time']),
            })
            self.obs_records[slice_point['sid']] = obs_record

        return 1.0 if spec_triggered else 0.0


# Alias for backward compatibility
Detect_Metric = SLSN_Detect_Metric
