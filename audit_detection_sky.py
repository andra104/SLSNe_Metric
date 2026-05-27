"""
audit_detection_sky.py — GP vs physical detection sky audit.

Loads populations and detect .npy arrays, prints a summary table,
and saves a 2x2 figure to output/SLSNe/audit_detection_sky.png.
"""

import sys, pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent

GP_PKL   = REPO / "output/SLSNe/shared/population_naive.pkl"
PH_PKL   = REPO / "output/SLSNe/shared/population_naive_physical.pkl"
GP_NPY   = REPO / "output/SLSNe/naive/metric_values_detect_naive_baseline_v5.1.1_10yrs_z0.1-2.0_260409_0257.npy"
PH_NPY   = REPO / "output/SLSNe/naive_physical/metric_values_detect_naive_physical_baseline_v5.1.1_10yrs_z0.1-1.5_adaptive.npy"
OUT_FIG  = REPO / "output/SLSNe/audit_detection_sky.png"
NSIDE    = 64

GP_Z  = (0.1, 2.0)
PH_Z  = (0.1, 1.5)

DEC_BINS = np.arange(-90, 100, 10)  # 10° bins -90 to +90
FALSE_DET_CUT = 10.0  # deg — above this is outside Rubin footprint

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

print("Loading populations and npy arrays...")
gp_pop = load_pkl(GP_PKL)
ph_pop = load_pkl(PH_PKL)
gp_det = np.load(GP_NPY)
ph_det = np.load(PH_NPY)

# ---------------------------------------------------------------------------
# Build masked arrays aligned to npy
# ---------------------------------------------------------------------------
def build_track(pop, det_arr, z_lo, z_hi, label):
    ra  = np.asarray(pop["ra"],  float)
    dec = np.asarray(pop["dec"], float)
    z   = np.asarray(pop["z"],   float)

    mask = (z >= z_lo) & (z <= z_hi)
    ra_m  = ra[mask]
    dec_m = dec[mask]
    z_m   = z[mask]

    n_pop = mask.sum()
    assert len(det_arr) == n_pop, (
        f"{label}: population z-filtered size {n_pop} != npy size {len(det_arr)}"
    )

    dec_deg = np.degrees(dec_m)

    detected   = det_arr == 1
    not_det    = det_arr == 0
    oof        = det_arr == -1   # out-of-footprint (physical only)

    n_det  = detected.sum()
    n_no   = not_det.sum()
    n_oof  = oof.sum()

    dec_det = dec_deg[detected]
    false_det = (dec_det > FALSE_DET_CUT).sum()
    legit_det = (dec_det <= FALSE_DET_CUT).sum()

    return {
        "label":     label,
        "ra":        ra_m,
        "dec_rad":   dec_m,
        "dec_deg":   dec_deg,
        "det_arr":   det_arr,
        "detected":  detected,
        "n_pop":     n_pop,
        "n_det":     n_det,
        "n_no":      n_no,
        "n_oof":     n_oof,
        "dec_det":   dec_det,
        "false_det": false_det,
        "legit_det": legit_det,
    }

gp = build_track(gp_pop, gp_det, *GP_Z, "GP (naive, z=0.1-2.0)")
ph = build_track(ph_pop, ph_det, *PH_Z, "Physical (naive, z=0.1-1.5)")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(t):
    n = t["n_pop"]
    nd = t["n_det"]
    rate = 100 * nd / n if n > 0 else 0.0
    false_rate = 100 * t["false_det"] / nd if nd > 0 else 0.0
    legit_rate = 100 * t["legit_det"] / nd if nd > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Track: {t['label']}")
    print(f"{'='*60}")
    print(f"  Total events (z-filtered):  {n:>10,}")
    print(f"  Detected   (==1):           {t['n_det']:>10,}  ({rate:.2f}%)")
    print(f"  Not det    (==0):           {t['n_no']:>10,}")
    print(f"  Out-of-footprint (==-1):    {t['n_oof']:>10,}")
    print(f"  Detections above Dec=+10°:  {t['false_det']:>10,}  ({false_rate:.2f}% of detections — should be ~0)")
    print(f"  Detections at/below Dec=+10°:{t['legit_det']:>9,}  ({legit_rate:.2f}% of detections)")
    print()
    print(f"  Dec histogram of detected events (10° bins):")
    counts, _ = np.histogram(t["dec_det"], bins=DEC_BINS)
    for i, c in enumerate(counts):
        lo, hi = DEC_BINS[i], DEC_BINS[i+1]
        bar = "█" * min(int(c / max(counts + [1]) * 40), 40)
        flag = " ← ABOVE RUBIN LIMIT" if lo >= FALSE_DET_CUT else ""
        print(f"    [{lo:+4.0f}, {hi:+4.0f}): {c:6,}  {bar}{flag}")

print_summary(gp)
print_summary(ph)

# ---------------------------------------------------------------------------
# HEALPix sky maps
# ---------------------------------------------------------------------------
def make_healpix_map(ra_rad, dec_rad, det_arr, nside=NSIDE):
    npix = hp.nside2npix(nside)
    theta = 0.5 * np.pi - dec_rad
    phi   = ra_rad
    pix   = hp.ang2pix(nside, theta, phi)

    inj_map  = np.zeros(npix, dtype=float)
    det_map  = np.zeros(npix, dtype=float)

    np.add.at(inj_map, pix, 1)
    np.add.at(det_map, pix[det_arr == 1], 1)

    return inj_map, det_map

gp_inj_map, gp_det_map  = make_healpix_map(gp["ra"], gp["dec_rad"], gp["det_arr"])
ph_inj_map, ph_det_map  = make_healpix_map(ph["ra"], ph["dec_rad"], ph["det_arr"])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

def plot_dec_hist(ax, t, color):
    counts, _ = np.histogram(t["dec_det"], bins=DEC_BINS)
    centers = 0.5 * (DEC_BINS[:-1] + DEC_BINS[1:])
    ax.bar(centers, counts, width=9, color=color, alpha=0.75, edgecolor="k", linewidth=0.4)
    ax.axvline(FALSE_DET_CUT, color="red", ls="--", lw=1.5, label=f"Dec=+{FALSE_DET_CUT:.0f}°")
    false_rate = 100 * t["false_det"] / t["n_det"] if t["n_det"] > 0 else 0.0
    ax.set_xlabel("Declination [deg]", fontsize=11)
    ax.set_ylabel("N detected events", fontsize=11)
    ax.set_title(
        f"{t['label']}\n"
        f"n_det={t['n_det']:,}  false_det_rate={false_rate:.2f}%",
        fontsize=10
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

def plot_healpix(ax, inj_map, det_map, title):
    nside_local = hp.npix2nside(len(inj_map))
    npix = len(inj_map)
    img = np.zeros(npix, dtype=float)
    has_inj = inj_map > 0
    has_det = det_map > 0
    img[has_inj] = 0.3    # purple-ish: injected
    img[has_det] = 1.0    # yellow: detected
    img[~has_inj] = np.nan

    # Mollweide via healpy cartview on axes
    # Use cartview for embedding in subplot
    lon = np.linspace(-180, 180, 720)
    lat = np.linspace(-90,  90, 360)
    LON, LAT = np.meshgrid(lon, lat)
    theta_g = np.radians(90 - LAT.ravel())
    phi_g   = np.radians(LON.ravel() % 360)
    pix_g   = hp.ang2pix(nside_local, theta_g, phi_g)
    grid    = img[pix_g].reshape(LAT.shape)

    im = ax.pcolormesh(LON, LAT, grid, cmap="plasma", vmin=0, vmax=1, shading="auto")
    ax.axhline(FALSE_DET_CUT, color="red", ls="--", lw=1.2, label=f"Dec=+{FALSE_DET_CUT:.0f}°")
    ax.axhline(34.4, color="white", ls=":", lw=1.0, alpha=0.7, label="Rubin limit (+34.4°)")
    ax.set_xlabel("RA [deg]", fontsize=10)
    ax.set_ylabel("Dec [deg]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="0.3=injected  1.0=detected")

plot_dec_hist(axes[0, 0], gp, color="#4C72B0")
plot_dec_hist(axes[0, 1], ph, color="#DD8452")
plot_healpix(axes[1, 0], gp_inj_map, gp_det_map,
             f"GP sky map  (injected=purple, detected=yellow)")
plot_healpix(axes[1, 1], ph_inj_map, ph_det_map,
             f"Physical sky map  (injected=purple, detected=yellow)")

plt.suptitle("Detection Sky Audit — GP vs Physical (naive rate model, baseline_v5.1.1)",
             fontsize=12, y=1.01)
plt.tight_layout()
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {OUT_FIG}")