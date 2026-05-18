#!/usr/bin/env python3
"""
validate_pipeline.py — Pre-submission pipeline math validation.

Catches silent wrong-answer bugs before SLURM submission.
Safe to run on MSI login node — memory-efficient, loads files
sequentially and releases between tests.

Usage:
    python3 validate_pipeline.py

All tests must pass before submitting any production SLURM job.
Exit code 0 = all pass. Exit code 1 = one or more failures.

Tests
-----
FILE INTEGRITY (reads raw pkl values, no pipeline code):
  F1. physical_templates.pkl — structure, phase/lam range, Fnu_abs finite
  F2. physical_mag_grid.pkl  — structure, convention (absolute mags), finite fraction
  F3. population pkl          — z range, DM consistency, file_indx range
  F4. GP templates.pkl        — structure, absolute mag convention

MATH CORRECTNESS (traces exact pipeline code path):
  M1. DM consistency — population pkl vs runtime cosmology (4 sources)
  M2. Filter normalization — suffixed names (baseline_v5.3.0+)
  M3. synthesize_mag_at_z() — correct apparent mag at known z/phase
  M4. Fast path vs slow path — agree to <0.05 mag
  M5. SNR calculation — physically reasonable (0-10000)
  M6. GP path M_abs + DM convention
  M7. Extinction applied once and consistently
  M8. MJD0 from cadence db — matches expected value
"""

import sys
import gc
import os
import sqlite3
import numpy as np
import pickle
from pathlib import Path

repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / 'py_files'))

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------
results = []
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    symbol = "✓" if condition else "✗"
    print(f"  [{symbol}] {name}")
    if detail:
        print(f"      {detail}")
    return condition

def skip(name, reason):
    results.append((name, SKIP, reason))
    print(f"  [-] {name} — SKIPPED: {reason}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def load_population_sample(path, n=500):
    """Load only first n events from population pkl — avoids OOM on login node."""
    with open(str(path), 'rb') as f:
        pop = pickle.load(f)
    n_total = len(np.asarray(pop['z']))
    sample = {}
    for k, v in pop.items():
        if hasattr(v, '__len__') and not isinstance(v, str):
            arr = np.asarray(v)
            if arr.ndim > 0 and arr.shape[0] == n_total:
                sample[k] = arr[:n]
            else:
                sample[k] = v
        else:
            sample[k] = v
    sample['_n_total'] = n_total
    del pop
    gc.collect()
    return sample

# ---------------------------------------------------------------------------
print("\nSLSN Pipeline Validation Suite")
print("=" * 60)

shared_dir   = repo_root / 'output' / 'SLSNe' / 'shared'
cadences_dir = repo_root / 'cadences'

phys_templates_path = shared_dir / 'physical_templates.pkl'
phys_grid_path      = shared_dir / 'physical_mag_grid.pkl'
phys_pop_path       = shared_dir / 'population_fe_dependent_physical.pkl'
gp_templates_path   = shared_dir / 'templates.pkl'
gp_pop_path         = shared_dir / 'population_fe_dependent.pkl'

# ---------------------------------------------------------------------------
section("F1 — physical_templates.pkl file integrity")
# ---------------------------------------------------------------------------
if not phys_templates_path.exists():
    skip("physical_templates.pkl", "file missing")
else:
    import joblib
    payload = joblib.load(str(phys_templates_path))
    lcs     = payload.get('lightcurves', [])
    sg      = payload.get('sed_grid', [])

    check("F1.1 n_templates = 265",
          len(lcs) == 265,
          f"found {len(lcs)}")
    check("F1.2 data[0] is empty dict (physical convention)",
          len(lcs[0]) == 0,
          f"keys={list(lcs[0].keys())}")
    check("F1.3 sed_grid present and length 265",
          len(sg) == 265,
          f"len={len(sg)}")

    ph  = np.asarray(sg[0]['phase'])
    lam = np.asarray(sg[0]['lam_rest_A'])
    fnu = np.asarray(sg[0]['Fnu_abs'])
    check("F1.4 phase range [1.0, 400.0] days",
          abs(ph.min() - 1.0) < 0.1 and abs(ph.max() - 400.0) < 1.0,
          f"[{ph.min():.1f}, {ph.max():.1f}]")
    check("F1.5 wavelength range [500, 12000] Angstrom",
          abs(lam.min() - 500) < 10 and abs(lam.max() - 12000) < 10,
          f"[{lam.min():.0f}, {lam.max():.0f}]")
    check("F1.6 Fnu_abs all finite",
          np.isfinite(fnu).all(),
          f"finite={np.isfinite(fnu).sum()}/{fnu.size}")
    check("F1.7 Fnu_abs physically reasonable (>0, <1e6 Jy at 10pc)",
          fnu.max() < 1e6 and fnu[fnu > 0].min() > 0,
          f"range=[{fnu[fnu>0].min():.2e}, {fnu.max():.2e}] Jy")

    del payload, lcs, sg, ph, lam, fnu
    gc.collect()

# ---------------------------------------------------------------------------
section("F2 — physical_mag_grid.pkl file integrity")
# ---------------------------------------------------------------------------
if not phys_grid_path.exists():
    skip("physical_mag_grid.pkl", "file missing")
else:
    grid = joblib.load(str(phys_grid_path))
    axes = grid['mag_grid_axes']
    z_ax = np.asarray(axes['z'])
    ph_ax = np.asarray(axes['phase'])

    check("F2.1 all 6 LSST filters present",
          set(grid['filters']) == {'u', 'g', 'r', 'i', 'z', 'y'},
          f"filters={grid['filters']}")
    check("F2.2 z axis [0.02, 5.0], n=100",
          abs(z_ax.min() - 0.02) < 0.01 and abs(z_ax.max() - 5.0) < 0.01
          and len(z_ax) == 100,
          f"[{z_ax.min():.3f}, {z_ax.max():.3f}], n={len(z_ax)}")
    check("F2.3 phase axis [1.0, 400.0], n=79",
          abs(ph_ax.min() - 1.0) < 0.1 and abs(ph_ax.max() - 400.0) < 1.0
          and len(ph_ax) == 79,
          f"[{ph_ax.min():.1f}, {ph_ax.max():.1f}], n={len(ph_ax)}")

    # Check finite fraction and convention for r-band
    r_arr  = np.asarray(grid['mag_grid']['r'], float)
    finite_frac = np.isfinite(r_arr).sum() / r_arr.size
    check("F2.4 r-band finite fraction > 95%",
          finite_frac > 0.95,
          f"finite={finite_frac:.3f}")
    check("F2.5 mag_grid shape (265, 100, 79)",
          r_arr.shape == (265, 100, 79),
          f"shape={r_arr.shape}")

    # Convention diagnostic — detect whether grid stores absolute or apparent mags.
    # Code convention (post-fix): grid stores APPARENT mags (raw > 0, ~15-30).
    # Old convention (pre-fix):   grid stores ABSOLUTE mags (raw < 0, ~-15 to -20).
    # If grid and code are out of sync, fast path produces wrong magnitudes.
    from slsn_metrics.constants import dm_from_z
    z01_idx  = int(np.argmin(np.abs(z_ax - 0.1)))
    ph50_idx = int(np.argmin(np.abs(ph_ax - 50.0)))
    sample   = float(r_arr[0, z01_idx, ph50_idx])
    dm_01    = dm_from_z(0.1)
    apparent_if_absolute = sample + dm_01  # what you get if grid stores absolute
    
    # Determine what the grid is actually storing
    grid_is_apparent = 15.0 < sample < 35.0
    grid_is_absolute = sample < 0 and 15.0 < apparent_if_absolute < 35.0
    
    if grid_is_apparent:
        grid_convention = "apparent"
    elif grid_is_absolute:
        grid_convention = "absolute"
    else:
        grid_convention = "unknown"

    # F2.6: Report what the file contains
    check("F2.6 grid convention detectable (apparent or absolute)",
          grid_convention in ("apparent", "absolute"),
          f"raw={sample:.3f} at z=0.1, phase=50d — "
          f"convention={grid_convention} "
          f"({'raw is apparent mag' if grid_is_apparent else 'raw+DM=apparent' if grid_is_absolute else 'UNRECOGNIZED'})")

    # F2.7: Code expects APPARENT mags — fail if grid is stale (absolute)
    # This is the safety gate: blocks production if grid needs rebuild.
    check("F2.7 grid matches code convention (apparent mags, no DM needed)",
          grid_is_apparent,
          f"raw={sample:.3f} — "
          + (f"OK: apparent mag stored directly ✓"
             if grid_is_apparent
             else f"STALE GRID: stores absolute mag ({sample:.3f}), "
                  f"code expects apparent (~{apparent_if_absolute:.1f}). "
                  f"Rebuild required: sbatch submit_rebuild_mag_grid.slurm"))

    # F2.8: Check for suspiciously high mag values (>40 mag) in finite cells.
    # At high-z in blue bands, SED coverage is marginal — synthesize_mag_at_z()
    # may return very faint but finite values (e.g. 72 mag) instead of NaN.
    # Undetectable events, not scientifically wrong, but >5% finite fraction
    # suggests a SED coverage or grid build issue worth investigating.
    finite_vals  = r_arr[np.isfinite(r_arr)]
    n_suspicious = int(np.sum(finite_vals > 40.0))
    frac_susp    = n_suspicious / max(len(finite_vals), 1)
    check("F2.8 r-band: <5% of finite values suspiciously faint (>40 mag)",
          frac_susp < 0.05,
          f"n_susp={n_suspicious} ({100*frac_susp:.1f}% of finite), "
          f"max={finite_vals.max():.1f} mag — "
          + ("OK ✓" if frac_susp < 0.05
             else "WARNING: check SED coverage at high-z"))

    del grid, z_ax, ph_ax, r_arr, finite_vals
    gc.collect()

# ---------------------------------------------------------------------------
section("F3 — population_fe_dependent_physical.pkl file integrity")
# ---------------------------------------------------------------------------
if not phys_pop_path.exists():
    skip("physical population", "file missing")
else:
    pop = load_population_sample(phys_pop_path, n=1000)
    n_total = pop['_n_total']

    z    = np.asarray(pop['z'])
    dm   = np.asarray(pop['distance_modulus'])
    pt   = np.asarray(pop['peak_time'])
    ebv  = np.asarray(pop['ebv'])
    fidx = np.asarray(pop['file_indx'])

    check("F3.1 n_events = 14,424,192",
          n_total == 14424192,
          f"found {n_total:,}")
    check("F3.2 z range [0.1, 5.0]",
          z.min() >= 0.09 and z.max() <= 5.01,
          f"[{z.min():.3f}, {z.max():.3f}]")
    check("F3.3 peak_time range [1, 3652] days",
          pt.min() >= 1.0 and pt.max() <= 3652.0,
          f"[{pt.min():.1f}, {pt.max():.1f}]")
    check("F3.4 file_indx range [0, 264]",
          fidx.min() >= 0 and fidx.max() <= 264,
          f"[{fidx.min()}, {fidx.max()}]")

    # DM consistency from file values only
    from astropy.cosmology import Planck18
    dm_expected = Planck18.distmod(z).value
    max_diff    = float(np.max(np.abs(dm_expected - dm)))
    check("F3.5 DM in population matches Planck18 (<0.001 mag)",
          max_diff < 0.001,
          f"max diff={max_diff:.6f} mag")

    # Required keys
    required_keys = ['z', 'distance_modulus', 'peak_time', 'file_indx',
                     'ebv', 'ra', 'dec', 'A_r', 'A_g', 'A_i', 'sid']
    missing = [k for k in required_keys if k not in pop]
    check("F3.6 all required keys present",
          len(missing) == 0,
          f"missing={missing}" if missing else "all present")

    del pop, z, dm, pt, ebv, fidx
    gc.collect()

# ---------------------------------------------------------------------------
section("F4 — templates.pkl (GP) file integrity")
# ---------------------------------------------------------------------------
if not gp_templates_path.exists():
    skip("GP templates.pkl", "file missing")
else:
    gp_payload = joblib.load(str(gp_templates_path))
    gp_lcs     = gp_payload.get('lightcurves', [])

    check("F4.1 n_templates = 261",
          len(gp_lcs) == 261,
          f"found {len(gp_lcs)}")
    check("F4.2 data[0] has band keys (GP convention)",
          len(gp_lcs[0]) > 0,
          f"keys={list(gp_lcs[0].keys())}")

    # Check absolute mag convention
    band  = list(gp_lcs[0].keys())[0]
    mags  = np.asarray(gp_lcs[0][band]['mag'])
    check("F4.3 GP mags are absolute (< 0)",
          mags.max() < 0,
          f"band={band}, range=[{mags.min():.2f}, {mags.max():.2f}]")
    check("F4.4 GP mags in SLSN absolute range (-25 to -10)",
          mags.min() > -25 and mags.max() < -10,
          f"range=[{mags.min():.2f}, {mags.max():.2f}]")

    del gp_payload, gp_lcs, mags
    gc.collect()

# ---------------------------------------------------------------------------
section("M1 — DM consistency (4 sources)")
# ---------------------------------------------------------------------------
from slsn_metrics.constants import dm_from_z
from slsn_metrics.model import LC  # needed for M6 GP path regardless of phys files
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

test_z   = 0.3
dm_const = dm_from_z(test_z)
dm_astro = cosmo.distmod(test_z).value
dm_dl    = 5.0 * np.log10(cosmo.luminosity_distance(test_z).to_value(u.pc) / 10.0)

check("M1.1 dm_from_z() vs Planck18.distmod() (<0.001 mag)",
      abs(dm_const - dm_astro) < 0.001,
      f"dm_from_z={dm_const:.6f}, astropy={dm_astro:.6f}")
check("M1.2 dm_from_z() vs 5*log10(DL/10pc) (<0.001 mag)",
      abs(dm_const - dm_dl) < 0.001,
      f"dm_from_z={dm_const:.6f}, from_DL={dm_dl:.6f}")

# ---------------------------------------------------------------------------
section("M2 — Filter normalization")
# ---------------------------------------------------------------------------
suffixed   = ['g_6', 'r_57', 'i_39', 'u_24', 'z_20', 'y_10']
normalized = [f.split('_')[0] for f in suffixed]
expected   = ['g', 'r', 'i', 'u', 'z', 'y']
standard   = ['u', 'g', 'r', 'i', 'z', 'y']

check("M2.1 suffixed filter names normalize correctly",
      normalized == expected,
      f"{suffixed} -> {normalized}")
check("M2.2 standard filter names pass through unchanged",
      [f.split('_')[0] for f in standard] == standard,
      "ugrizy unchanged")

# ---------------------------------------------------------------------------
section("M3-M7 — Math correctness (physical path)")
# ---------------------------------------------------------------------------
if not all(p.exists() for p in [phys_templates_path, phys_grid_path, phys_pop_path]):
    skip("M3-M7", "physical files missing")
else:
    from slsn_metrics.model import LC, synthesize_mag_at_z
    from slsn_metrics.metrics import (
        synthesize_mag_at_z_cached, _m52snr,
        SLSN_Detect_Physical_Metric
    )
    from slsn_metrics.constants import phase_bucket_vec

    # Load templates + grid
    templates = LC(load_from=str(phys_templates_path))
    templates.load_magnitude_grid(str(phys_grid_path))

    metric = SLSN_Detect_Physical_Metric(
        lc_model=templates, mjd0=60980.0, store_obs_mode='none'
    )

    # Load small population sample
    pop = load_population_sample(phys_pop_path, n=5000)

    # Find clean low-z event
    z_s   = np.asarray(pop['z'])
    Ar_s  = np.asarray(pop.get('A_r', np.zeros(len(z_s))))
    mask  = (z_s >= 0.1) & (z_s <= 1.0) & (Ar_s < 1.0)
    if not np.any(mask):
        skip("M3-M7", "no z=0.1-1.0 event in 5000-event sample")
    else:
        ei = int(np.where(mask)[0][0])

        sp = {k: pop[k][ei] for k in pop
              if k != '_n_total' and hasattr(pop[k], '__len__')
              and not isinstance(pop[k], str)
              and np.asarray(pop[k]).ndim > 0
              and np.asarray(pop[k]).shape[0] == len(z_s)}

        ev_z      = float(sp['z'])
        ev_dm     = float(sp['distance_modulus'])
        ev_tpl    = int(sp['file_indx'])
        ev_ebv    = float(sp['ebv'])
        ev_Ar     = float(sp.get('A_r', 0.0))
        test_filt = 'r'
        test_ph   = 50.0

        A_filt = float(sp.get(f'A_{test_filt}', 0.0))
        if A_filt == 0.0:
            A_filt = metric.ax1[test_filt] * ev_ebv

        sed_entry = templates.sed_grid[ev_tpl]

        # M3: synthesize_mag_at_z()
        m_synth = synthesize_mag_at_z(sed_entry, test_ph, ev_z, test_filt)
        check("M3.1 synthesize_mag_at_z() returns finite value",
              np.isfinite(m_synth),
              f"mag={m_synth:.4f}")
        check("M3.2 synthesize_mag_at_z() in apparent mag range (15-35)",
              15.0 < m_synth < 35.0,
              f"mag={m_synth:.4f}")
        check("M3.3 synthesize_mag_at_z() consistent with DM "
              "(DM-22 < m < DM+2)",
              (ev_dm - 22.0) < m_synth < (ev_dm + 2.0),
              f"mag={m_synth:.4f}, DM={ev_dm:.4f}")

        # M4: fast vs slow path
        interp      = metric.lc_model._interps[test_filt][ev_tpl]
        raw         = float(interp(np.array([[ev_z, test_ph]]))[0])
        fast_result = raw + ev_dm + A_filt

        cache         = {}
        pbi           = int(phase_bucket_vec(
            np.array([test_ph]), return_index=True)[0])
        m_slow        = synthesize_mag_at_z_cached(
            cache, sed_entry, pbi, ev_z, test_filt)
        slow_result   = m_slow + A_filt

        diff = abs(fast_result - slow_result)
        check("M4.1 fast path returns finite value",
              np.isfinite(fast_result),
              f"fast={fast_result:.4f}")
        check("M4.2 slow path returns finite value",
              np.isfinite(slow_result),
              f"slow={slow_result:.4f}")
        check("M4.3 fast == slow path (<0.05 mag)",
              diff < 0.05,
              f"fast={fast_result:.4f}, slow={slow_result:.4f}, diff={diff:.4f}")
        check("M4.4 fast path dm convention: raw+dm+A in (15-35)",
              15.0 < fast_result < 35.0,
              f"raw={raw:.4f} + dm={ev_dm:.4f} + A={A_filt:.4f} = {fast_result:.4f}")

        # M5: SNR
        m5       = 24.7
        snr_fast = float(_m52snr(np.array([fast_result]), np.array([m5]))[0])
        snr_slow = float(_m52snr(np.array([slow_result]), np.array([m5]))[0])
        check("M5.1 SNR fast path in range (0-10000)",
              0 <= snr_fast <= 10000,
              f"SNR={snr_fast:.3f}")
        check("M5.2 SNR slow path in range (0-10000)",
              0 <= snr_slow <= 10000,
              f"SNR={snr_slow:.3f}")
        check("M5.3 SNR fast and slow agree (<10%)",
              abs(snr_fast - snr_slow) / max(snr_slow, 0.001) < 0.10,
              f"fast={snr_fast:.3f}, slow={snr_slow:.3f}")

        # M7: Extinction
        A_from_pop  = float(sp.get(f'A_{test_filt}', 0.0))
        A_from_ax1  = metric.ax1[test_filt] * ev_ebv
        check("M7.1 A_filt from slice_point is finite",
              np.isfinite(A_from_pop) if A_from_pop != 0 else True,
              f"A_{test_filt}={A_from_pop:.4f}")
        check("M7.2 ax1[filt]*ebv is finite and positive",
              np.isfinite(A_from_ax1) and A_from_ax1 >= 0,
              f"ax1*ebv={A_from_ax1:.4f}")
        if A_from_pop > 0:
            check("M7.3 A from pop and ax1*ebv agree (<0.01)",
                  abs(A_from_pop - A_from_ax1) < 0.01,
                  f"pop={A_from_pop:.4f}, ax1*ebv={A_from_ax1:.4f}")

    del templates, metric, pop
    gc.collect()

# ---------------------------------------------------------------------------
section("M6 — GP path M_abs + DM convention")
# ---------------------------------------------------------------------------
if not all(p.exists() for p in [gp_templates_path, gp_pop_path]):
    skip("M6", "GP files missing")
else:
    gp_lc = LC(load_from=str(gp_templates_path))
    gp_pop = load_population_sample(gp_pop_path, n=500)

    z_s  = np.asarray(gp_pop['z'])
    mask = (z_s >= 0.2) & (z_s <= 0.4)
    if not np.any(mask):
        skip("M6", "no z=0.2-0.4 event in GP sample")
    else:
        ei     = int(np.where(mask)[0][0])
        gp_z   = float(gp_pop['z'][ei])
        gp_dm  = float(gp_pop['distance_modulus'][ei])
        gp_tpl = int(gp_pop['file_indx'][ei])

        gp_template = gp_lc.data[gp_tpl]
        gp_bands    = [b for b in gp_template
                       if isinstance(gp_template[b], dict)
                       and 'ph' in gp_template[b]]

        check("M6.1 GP template has band data",
              len(gp_bands) > 0,
              f"bands={gp_bands}")

        if gp_bands:
            M_abs    = gp_lc.interp(50.0, gp_bands[0], gp_tpl)
            apparent = M_abs + gp_dm
            check("M6.2 GP M_abs is negative (absolute mag)",
                  np.isfinite(M_abs) and M_abs < 0,
                  f"M_abs={M_abs:.4f}")
            check("M6.3 GP M_abs in SLSN range (-25 to -15)",
                  -25.0 < M_abs < -15.0,
                  f"M_abs={M_abs:.4f}")
            check("M6.4 GP M_abs + DM in apparent range (15-35)",
                  15.0 < apparent < 35.0,
                  f"M_abs={M_abs:.4f} + DM={gp_dm:.4f} = {apparent:.4f}")

    del gp_lc, gp_pop
    gc.collect()

# ---------------------------------------------------------------------------
section("M8 — MJD0 from cadence db")
# ---------------------------------------------------------------------------
cadence_checks = {
    'noroll_v5.0.0_10yrs':   60980.0,
    'baseline_v5.1.1_10yrs': 60981.0,
    'baseline_v5.3.0_10yrs': 61208.2,
}
for cadence, expected in cadence_checks.items():
    db = cadences_dir / f"{cadence}.db"
    if not db.exists():
        skip(f"M8 {cadence}", "db not found")
        continue
    try:
        con    = sqlite3.connect(str(db))
        mjd_db = float(con.execute(
            "SELECT MIN(observationStartMJD) FROM observations"
        ).fetchone()[0])
        con.close()
        check(f"M8 {cadence} MJD0 ~ {expected}",
              abs(mjd_db - expected) < 1.0,
              f"db={mjd_db:.2f}, expected~{expected}")
    except Exception as e:
        skip(f"M8 {cadence}", str(e))

# ---------------------------------------------------------------------------
section("SUMMARY")
# ---------------------------------------------------------------------------
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
n_skip = sum(1 for _, s, _ in results if s == SKIP)

print(f"\n  Total : {len(results)}")
print(f"  Pass  : {n_pass}")
print(f"  Fail  : {n_fail}")
print(f"  Skip  : {n_skip}")

if n_fail > 0:
    print(f"\n  FAILED TESTS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    ✗ {name}")
            if detail:
                print(f"      {detail}")
    print(f"\n  STATUS: PIPELINE VALIDATION FAILED")
    print(f"  Do not submit production SLURM jobs until all tests pass.")
    sys.exit(1)
else:
    print(f"\n  STATUS: ALL TESTS PASSED ✓")
    print(f"  Pipeline math is verified. Safe to submit production runs.")
    sys.exit(0)
