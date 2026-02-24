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
    
    if not available_bands:
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
# Spectroscopic trigger metric
# =============================================================================

class SLSN_SpecTriggerMetric(SLSN_Base_Metric):
    """
    Spectroscopic trigger metric (near-peak brightness + color).
    
    Requires detection plus:
    - ≥1 epoch with SNR≥5 within ±5 days of peak
    - min(mag) near peak < 21.0
    - Optional: (g-r) < 0.3 near peak
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metricName = 'SLSN_SpecTrigger'
        self.obs_records = {}

    def run(self, dataSlice, slice_point=None):
        snr, filters, times, obs_record = evaluate_slsn(
            self, dataSlice, slice_point, return_full_obs=True
        )
        
        # Initialize as failed
        spec_triggered = False
        min_mag_near_peak = np.nan
        has_g_near_peak = False
        has_r_near_peak = False
        
        # If no observations
        if obs_record is None or snr.size == 0:
            if self.store_obs_mode == "full":
                self.obs_records[slice_point['sid']] = {
                    'spec_trigger': False,
                    'min_mag_near_peak': np.nan,
                    'has_g_near_peak': False,
                    'has_r_near_peak': False,
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
        
        mags = np.asarray(obs_record['mag_obs'], float)
        mjd_obs = np.asarray(obs_record['mjd_obs'], float)
        
        # Must pass detection
        detected = detect_slsn(filters, snr, mjd_obs, mags, obs_record)
        
        if detected:
            peak_mjd = self.mjd0 + float(slice_point['peak_time'])
            near_peak = (snr >= 5) & (np.abs(mjd_obs - peak_mjd) <= 5.0)
            
            if np.any(near_peak):
                min_mag_near_peak = float(np.min(mags[near_peak]))
                
                # Check brightness requirement
                if min_mag_near_peak <= 21.0:
                    # Check color (optional)
                    has_g_near_peak = np.any(near_peak & (filters == 'g'))
                    has_r_near_peak = np.any(near_peak & (filters == 'r'))
                    
                    passes_color = True
                    if has_g_near_peak and has_r_near_peak:
                        g_mag = np.min(mags[near_peak & (filters == 'g')])
                        r_mag = np.min(mags[near_peak & (filters == 'r')])
                        passes_color = (g_mag - r_mag) <= 0.3
                    
                    spec_triggered = passes_color
        
        # ALWAYS store the record (pass or fail)
        if self.store_obs_mode == "full":
            obs_record.update({
                'spec_trigger': spec_triggered,
                'min_mag_near_peak': min_mag_near_peak,
                'has_g_near_peak': has_g_near_peak,
                'has_r_near_peak': has_r_near_peak,
                'sid': int(slice_point['sid']),
                'z': float(slice_point['z']),
                'ra': float(slice_point['ra']),
                'dec': float(slice_point['dec']),
                'distance_Mpc': float(slice_point['distance']),
                'ebv': float(slice_point['ebv']),
                'peak_time': float(slice_point['peak_time']),
            })
            self.obs_records[slice_point['sid']] = obs_record
        
        return 1.0 if spec_triggered else 0.0
    

# Alias for backward compatibility
Detect_Metric = SLSN_Detect_Metric
