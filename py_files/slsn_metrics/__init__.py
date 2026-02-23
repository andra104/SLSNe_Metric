"""
SLSN Metrics Package — Modular SLSN simulation tools for Rubin MAF.

Modules:
--------
constants             : Physical constants, cosmology, bandpasses
export_slsne_photometry : Read/clean/export per-event photometry
gp_build              : 2D GP fitting and t0 selection
model                 : LC class with template/grid synthesis
population            : SLSN population generation
metrics               : MAF metrics (detect, characterize, spec trigger)
diagnostics           : QA checks and plotting

Quick Start:
------------
# Build templates from catalog
from model import LC, CatalogInputs
import pandas as pd

params = pd.read_csv("all_parameters.txt", sep=r"\s+")
inputs = CatalogInputs(
    photometry_dir="data/per_event_cenwave/",
    params_table=params
)
templates = LC.from_catalog(inputs, save_to="slsn_templates.pkl")

# Generate population
from population import generate_SLSN_PopSlicer
pop = generate_SLSN_PopSlicer(
    templates, 
    rate_density=1e-7,
    save_to="slsn_population.pkl"
)

# Run metrics
from metrics import SLSN_Detect_Metric
metric = SLSN_Detect_Metric(lc_model=templates)
# ... use with MAF bundle
"""

__version__ = "0.1.0"

# =============================================================================
# Core imports for users
# =============================================================================

# Path configuration — import first so everything else can use it
from .paths import (
    get_repo_root,
    get_cadences_dir,
    get_output_dir,
    get_data_dir,
    get_per_event_dir,
    get_all_events_dir,
    get_shared_output_dir,
    get_cadence_path,
    get_cadence_output_dir,
    get_rubin_sim_data_dir,
    set_rubin_sim_data_dir,
    print_paths,
    quick_path_check,
)

# Constants & utilities
from .constants import (
    dm_from_z,
    dm_from_z_fast,
    z_from_comoving_fast,
    get_lsst_bands,
    angstrom_to_hz,
    hz_to_angstrom,
    mag_to_flux_jy,
    flux_jy_to_mag,
    magerr_to_fluxerr_jy,
    phase_bucket_vec,
    PHASE_BIN_STEP,
    C_MS,
    F0_JY,
)

# Photometry export
from .export_slsne_photometry import (
    canonical_filter,
    read_supernova_table_txt,
    to_export,
    process_one,
    process_all_events,
    per_filter_cenwave,
    attach_cenwave_to_perevent_csv,
    audit_cenwave_mode,
    audit_cenwave_variability,
    load_allparams_robust,
)

# GP building
from .gp_build import (
    default_gp_kernel,
    fit_2d_gp,
    gp_predict_surface,
    pick_t0_hybrid,
)

# Model & templates
from .model import (
    _compute_grid_slice,
    LC,
    CatalogInputs,
    synthesize_mag_at_z,
    template_index_for_event,
    template_name_for_index,
    get_median_cenwave_map,
    list_t0_meta,
    atomic_save_pickle,
)

# Population
from .population import (
    generate_SLSN_PopSlicer,
    sample_rate_from_volume, 
    inject_uniform_healpix,
)

# Metrics
from .metrics import (
    SLSN_Base_Metric,
    SLSN_Detect_Metric,
    SLSN_CharacterizeMetric,
    SLSN_SpecTriggerMetric,
    Detect_Metric,  # alias
    detect_slsn,
    evaluate_slsn,
)

#Runners (detect and metrics)
from .runners import (
    get_distance_bounds, 
    build_filenames,
    run_slsn_detect,
    _build_detection_dataframe,
    _make_all_detection_plots,
    run_slsn_multi_metrics,
    _make_healpix_efficiency_map,
)

# Diagnostics
from .diagnostics import (
    plot_metrics_mosaic_grid,
    _get_available_bands,
    plot_population_lc_at_obs,
    plot_population_lc_multi_at_obs,
    validate_template_quality,
    diagnose_abs_from_templates,
    diagnose_internal_consistency,
    plot_residuals,
    plot_event_obs,
    plot_event_model,
    plot_event_obs_vs_model,
    list_event_bands,
    list_template_bands,
    characterize_template_coverage,
    compute_slsn_properties,
    assess_literature_coverage,
    find_missing_archetypes,
    plot_population_diagnostics,
    plot_detect_diagnostics,
    plot_healpix_efficiency,
    plot_population_lc_at_obs,
    plot_population_lc_multi_at_obs,
)

# =============================================================================
# Module organization
# =============================================================================

__all__ = [
    # Constants
    "dm_from_z",
    "z_from_comoving_fast",
    "get_lsst_bands",
    "angstrom_to_hz",
    "mag_to_flux_jy",
    "flux_jy_to_mag",
    
    # Export
    "canonical_filter",
    "process_all_events",
    "per_filter_cenwave",
    "attach_cenwave_to_perevent_csv",
    
    # GP
    "fit_2d_gp",
    "pick_t0_hybrid",
    
    # Model
    "LC",
    "CatalogInputs",
    "synthesize_mag_at_z",
    "template_index_for_event",
    
    # Population
    "generate_SLSN_PopSlicer",
    
    # Metrics
    "SLSN_Detect_Metric",
    "SLSN_CharacterizeMetric",
    "SLSN_SpecTriggerMetric",
    "detect_slsn",
    "evaluate_slsn",
    
    # Diagnostics
    "diagnose_abs_from_templates",
    "plot_residuals",
    "plot_event_obs",
    "plot_event_model",
    "characterize_template_coverage",
]