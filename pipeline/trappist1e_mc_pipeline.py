# ============================================================
# SELF-CONTAINED SINGLE-CELL NOTEBOOK
#
# TRAPPIST-1e ATMOS -> photochemistry -> petitRADTRANS -> JWST
#
# PUBLICATION-GRADE CHI-SQUARE / DELTA-CHI-SQUARE / BIC / SNR ANALYSIS
#
# This version fixes 6 discrepancies between the original code and
# Sections 3.2.2 / 3.5 / Eq. 3.4-3.5 / Objective 4 of the thesis
# proposal "Investigation of TRAPPIST-1e Atmosphere Using 1-D
# Photochem and petitRADTRANS Codes" (M. Waqas, ITB, 2026):
#
#   FIX 1:
#   FIX 2: added O2 line opacity, H2 Rayleigh scattering, and
#          N2-N2 / N2-H2 collision-induced-absorption (CIA)
#   FIX 3: reference pressure -> 0.1 bar (was 1.0 bar)
#   FIX 4: stellar radius -> 0.117 R_sun, Agol et al. 2021 (was 0.1192)
#   FIX 5: Gaussian line-spread-function convolution to R~100
#          (NIRSpec PRISM) applied to the model BEFORE JWST binning
#   FIX 6: Bayesian Information Criterion (BIC) and per-bin
#          signal-to-noise / detectability (N_transits needed)
#          added alongside chi2 / reduced-chi2 / delta-chi2
#
#   FIX 7 (this version): switched the comparison spectrum from the
#          5-column Canas001 (Espinoza et al. 2025) file to the
#          3-column (wavelength_um, depth_ppm, error_ppm)
#          "tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt"
#          spectrum -- the GP stellar-contamination-corrected,
#          all-four-visits-combined transmission spectrum. Bin edges
#          are not given in this file and are reconstructed from the
#          midpoints between adjacent wavelength centers. All outputs
#          are written to a NEW results directory / master CSV so
#          they never overwrite or mix with the original Canas-based
#          run.
#
# Workflow:
#
#   Monte Carlo atmospheric parameters
#              |
#              v
#        ATMOS climate
#              |
#              v
#    Photochemical steady state
#              |
#              v
#       petitRADTRANS (line-by-line)
#              |
#              v
#     Gaussian LSF convolution (R~100)      <- FIX 5
#              |
#              v
#       JWST binning
#              |
#              v
#    Weighted least-squares (offset fit)
#              |
#              v
#         chi2 / reduced chi2 / BIC          <- FIX 6
#              |
#              v
#       ALL MODELS FINISHED
#              |
#              v
#     chi2_min / BIC_min
#              |
#              v
#     delta_chi2 / delta_BIC
#              |
#              v
#     relative likelihood
#              |
#              v
#     per-bin SNR / N_transits needed        <- FIX 6
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d

from photochem import EvoAtmosphere
from photochem.clima import AdiabatClimate
from petitRADTRANS.radtrans import Radtrans

from threadpoolctl import threadpool_limits


# Limit BLAS/OpenMP threads to avoid excessive CPU usage
_ = threadpool_limits(limits=4)


import input_files
import utils


# ============================================================
# 2. ONE-TIME INPUT-FILE SETUP
# ============================================================

# Safe to re-run.
# These create the reaction file and TRAPPIST-1 stellar flux file.

input_files.create_zahnle_HNOC()
input_files.create_TRAPPIST1e_stellar_flux()


# ============================================================
# 3. USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# Input / output files
# ------------------------------------------------------------

INPUT_CSV = "MC_2000_atmosphere_sweep.csv"

# FIX 7: separate results directory / master CSV for the flat-spectrum
# comparison run, so this never overwrites the original Canas-based
# results.

RESULTS_DIR = Path("MC_2000_model_results_flatspec")

MASTER_CSV = RESULTS_DIR / "MC_2000_master_results_flatspec.csv"

# FIX 7: switched from the 5-column Canas001 (Espinoza et al. 2025)
# file to the 3-column decontaminated/corrected/flat spectrum.

JWST_FILE = (
    "/home/waqas/Downloads/"
    "TRAPPIST-1e-GTO-2025-main/"
    "espinoza+2025/figure3/"
    "tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt"
)


# ------------------------------------------------------------
# Model range
# ------------------------------------------------------------

# Testing:
#
# START_MODEL = 0
# END_MODEL = 2
#
# Production:
#
# START_MODEL = 0
# END_MODEL = None

START_MODEL = 1995

END_MODEL = 2000

SKIP_ALREADY_DONE = True


# ============================================================
# 4. PUBLICATION-GRADE JWST FIT SETTINGS
# ============================================================

# ------------------------------------------------------------
# Wavelength range
# ------------------------------------------------------------
#
# None = use complete available JWST spectrum.
#
# Keep None unless you have a scientific reason to restrict
# the wavelength range.

JWST_WAVELENGTH_MIN_UM = None

JWST_WAVELENGTH_MAX_UM = None


# ------------------------------------------------------------
# Minimum number of pRT wavelength points per JWST bin
# ------------------------------------------------------------

MIN_MODEL_POINTS_PER_BIN = 2


# ------------------------------------------------------------
# Fit a constant vertical offset?
# ------------------------------------------------------------
#
# True:
#   The model spectrum is shifted vertically to minimize chi2.
#   This accounts for the arbitrary reference radius / reference
#   pressure level in transmission spectroscopy.
#   The fitted offset counts as ONE fitted parameter.
#   Therefore: dof = N_data - 1
# ------------------------------------------------------------

FIT_VERTICAL_OFFSET = True


# ------------------------------------------------------------
# Require complete pRT coverage of every JWST bin?
# ------------------------------------------------------------

REQUIRE_FULL_BIN_COVERAGE = True


# ------------------------------------------------------------
# FIX 5: Instrument resolving power for LSF convolution
# ------------------------------------------------------------
#
# Section 3.5 of the proposal: "we will convolve the synthetic
# transmission spectra with a Gaussian line-spread function to
# match the relevant JWST mode ... NIRSpec PRISM resolution
# (R ~ 100 over 0.6-5.2 um)".
#
# NOTE: the "R10000" in the new JWST_FILE's filename refers to the
# resolution of that *reduced data product's wavelength grid*, not
# to the instrument LSF. It does not override JWST_RESOLVING_POWER,
# which must stay at the NIRSpec PRISM LSF resolution (R~100) used
# to convolve the pRT model before binning, per Section 3.5.
#
# JWST_LSF_OVERSAMPLE controls how finely the convolved spectrum
# is re-gridded relative to the LSF width, purely a numerical
# accuracy setting (higher = smoother, slower). It does not change
# the physical resolving power.
# ------------------------------------------------------------

JWST_RESOLVING_POWER = 100.0

JWST_LSF_OVERSAMPLE = 10


# ------------------------------------------------------------
# FIX 6: Detectability / SNR settings
# ------------------------------------------------------------
#
# Eq. 3.5 of the proposal:
#
#   SNR(lambda) = delta_depth(lambda) / (sigma_ph(lambda) / sqrt(N_tr))
#
# JWST_FILE is the GP stellar-contamination-corrected transmission
# spectrum combining all four TRAPPIST-1e visits (decontaminated,
# corrected, flattened; cf. the Espinoza et al. 2025 reduction, and
# the right-hand "all four visits" panels of the rejection-
# significance figure). We back out the single-transit photon noise
# from those combined error bars via
# sigma_ph = sigma_data * sqrt(N_TRANSITS_OBSERVED), then compute,
# for every bin, how many transits would be needed to reach
# SNR_TARGET on that bin's model-predicted feature amplitude.
# ------------------------------------------------------------

N_TRANSITS_OBSERVED = 4

SNR_TARGET = 5.0

# Number of fitted (nuisance) parameters used both in dof and in BIC.
# Currently this is just the constant vertical offset. If additional
# free parameters are ever fitted (e.g. a scattering slope), update
# this in one place.

N_FITTED_PARAMETERS = 1 if FIT_VERTICAL_OFFSET else 0


# ------------------------------------------------------------
# Numerical safety
# ------------------------------------------------------------

EPS = 1e-30


# ============================================================
# 5. RESULT COLUMNS
# ============================================================

RESULT_COLUMNS = [

    # Atmospheric parameters
    "model_id",
    "P_N2",
    "P_CO2",
    "P_CH4",
    "T_surf_input",
    "T_surf_climate",

    # Convergence
    "climate_converged",
    "photochemistry_converged",

    # JWST data information
    "n_data",
    "dof",
    "wavelength_min_um",
    "wavelength_max_um",
    "offset_ppm",

    # Statistical fit
    "chi2",
    "reduced_chi2",
    "delta_chi2",
    "relative_likelihood",
    "rms_residual_ppm",

    # FIX 6: information-criterion and detectability summary
    "bic",
    "delta_bic",
    "max_feature_snr",
    "n_transits_for_detection",

    # Spectrum properties
    "spectral_amplitude_ppm",
    "max_transit_depth_ppm",
    "min_transit_depth_ppm",
]


# ============================================================
# 6. CREATE RESULTS DIRECTORY
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 7. LOAD JWST DATA
# ============================================================

def load_jwst_data(jwst_file):
    """
    Load the GP-corrected, all-four-visits-combined TRAPPIST-1e
    transmission spectrum (decontaminated, corrected, flat;
    R~10000 wavelength grid, prior to instrument-LSF convolution
    of the *model*).

    File columns: wavelength_um, depth_ppm, error_ppm

    FIX 7: this file gives bin *centers* only, not lower/upper bin
    widths (unlike the original 5-column Canas001 file), so the
    edges used for binning the pRT model (bin_model_to_jwst) are
    reconstructed from the midpoints between adjacent wavelength
    centers. The first/last bin edges are mirrored from their
    nearest-neighbor spacing.
    """

    data = np.genfromtxt(jwst_file, comments="#")

    if data.ndim != 2:
        raise ValueError("JWST data file does not contain a valid 2D table.")

    if data.shape[1] < 3:
        raise ValueError(
            "File must contain at least 3 columns: wavelength, depth, error."
        )

    wavelength = data[:, 0]
    depth = data[:, 1]
    error = data[:, 2]

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    depth = depth[order]
    error = error[order]

    # Reconstruct bin edges from midpoints between adjacent centers.
    mid = 0.5 * (wavelength[:-1] + wavelength[1:])

    lower_edge = np.empty_like(wavelength)
    upper_edge = np.empty_like(wavelength)

    lower_edge[1:] = mid
    upper_edge[:-1] = mid

    lower_edge[0] = wavelength[0] - (mid[0] - wavelength[0])
    upper_edge[-1] = wavelength[-1] + (wavelength[-1] - mid[-1])

    valid = (
        np.isfinite(wavelength)
        & np.isfinite(depth)
        & np.isfinite(error)
        & (error > 0)
    )

    wavelength = wavelength[valid]
    lower_edge = lower_edge[valid]
    upper_edge = upper_edge[valid]
    depth = depth[valid]
    error = error[valid]

    if len(wavelength) == 0:
        raise RuntimeError("No valid JWST data points remain after filtering.")

    return {
        "wavelength": wavelength,
        "lower_edge": lower_edge,
        "upper_edge": upper_edge,
        "depth": depth,
        "error": error,
    }


jwst_data = load_jwst_data(JWST_FILE)

print("=" * 70)
print("JWST DATA LOADED")
print("=" * 70)
print(f"JWST file: {JWST_FILE}")
print(f"JWST data points available: {len(jwst_data['wavelength'])}")
print(
    f"Wavelength range: "
    f"{np.min(jwst_data['lower_edge']):.5f} - "
    f"{np.max(jwst_data['upper_edge']):.5f} um"
)
print("=" * 70)


# ============================================================
# 8. CONSTANTS
# ============================================================

MOLECULAR_WEIGHTS = {
    "N2": 28.014,
    "CO2": 44.0095,
    "H2O": 18.01528,
    "CH4": 16.043,
    "CO": 28.010,
    "O2": 31.998,
    "H2": 2.016,
}

SPECIES = list(MOLECULAR_WEIGHTS.keys())

# ------------------------------------------------------------
# Physical constants
# ------------------------------------------------------------

G = 6.67430e-11
M_EARTH = 5.9722e24
R_EARTH = 6.371e6
R_SUN_CGS = 6.957e10

# ------------------------------------------------------------
# TRAPPIST-1e parameters
# ------------------------------------------------------------

PLANET_MASS_EARTH = 0.692
PLANET_RADIUS_EARTH = 0.920

M_PLANET = PLANET_MASS_EARTH * M_EARTH
R_PLANET_M = PLANET_RADIUS_EARTH * R_EARTH
GRAVITY_M_S2 = G * M_PLANET / R_PLANET_M**2
PLANET_RADIUS_CGS = R_PLANET_M * 100.0
GRAVITY_CGS = GRAVITY_M_S2 * 100.0

# ------------------------------------------------------------
# Star
# ------------------------------------------------------------
#
# FIX 4: Section 3.2.2 of the proposal cites R_star = 0.117 R_sun
# (Agol et al. 2021). The original code used 0.1192, which does not
# match the cited source.

R_STAR_CGS = 0.117 * R_SUN_CGS

# ------------------------------------------------------------
# Transmission spectrum reference pressure
# ------------------------------------------------------------
#
# FIX 3: Section 3.2.2 of the proposal: "We used the reference
# radius at P_ref = 0.1 bar". The original code used 1.0 bar, which
# shifts the absolute transit depth (and hence the fitted offset).

REFERENCE_PRESSURE_BAR = 0.1

# ------------------------------------------------------------
# Eddy diffusion coefficient
# ------------------------------------------------------------

EDDY = 1e5


# ============================================================
# 9. EMPTY RESULT
# ============================================================

def empty_result(model_id, P_N2, P_CO2, P_CH4, T_surf_input):
    """
    Create a result row populated with NaN/False until
    individual stages succeed.
    """

    return {
        "model_id": int(model_id),
        "P_N2": float(P_N2),
        "P_CO2": float(P_CO2),
        "P_CH4": float(P_CH4),
        "T_surf_input": float(T_surf_input),

        "T_surf_climate": np.nan,
        "climate_converged": False,
        "photochemistry_converged": False,

        "n_data": 0,
        "dof": 0,
        "wavelength_min_um": np.nan,
        "wavelength_max_um": np.nan,
        "offset_ppm": np.nan,

        "chi2": np.nan,
        "reduced_chi2": np.nan,
        "delta_chi2": np.nan,
        "relative_likelihood": np.nan,
        "rms_residual_ppm": np.nan,

        # FIX 6
        "bic": np.nan,
        "delta_bic": np.nan,
        "max_feature_snr": np.nan,
        "n_transits_for_detection": np.nan,

        "spectral_amplitude_ppm": np.nan,
        "max_transit_depth_ppm": np.nan,
        "min_transit_depth_ppm": np.nan,
    }


# ============================================================
# 10. SAVE INDIVIDUAL RESULT
# ============================================================

def save_result(result):
    """
    Save one model CSV and safely update the master CSV.
    """

    row_df = pd.DataFrame([{k: result[k] for k in RESULT_COLUMNS}])

    model_file = RESULTS_DIR / f"model_{int(result['model_id']):04d}.csv"
    row_df.to_csv(model_file, index=False)

    if MASTER_CSV.exists():
        master = pd.read_csv(MASTER_CSV)
        if "model_id" in master.columns:
            master = master[master["model_id"] != int(result["model_id"])]
        master = pd.concat([master, row_df], ignore_index=True)
        master = master.sort_values("model_id").reset_index(drop=True)
    else:
        master = row_df

    master.to_csv(MASTER_CSV, index=False)


# ============================================================
# 11. FIX 5: CONVOLVE MODEL SPECTRUM TO INSTRUMENT RESOLUTION
# ============================================================

def convolve_to_resolution(
    model_wavelength_um,
    model_depth_ppm,
    resolving_power=JWST_RESOLVING_POWER,
    oversample=JWST_LSF_OVERSAMPLE,
):
    """
    Convolve a high-resolution model transmission spectrum with a
    Gaussian line-spread function (LSF) of constant resolving power R,
    as required by Section 3.5 of the proposal before binning onto the
    NIRSpec PRISM (R ~ 100) wavelength grid.

    Because R = lambda / delta_lambda is constant with wavelength, the
    LSF has a FIXED width when expressed in ln(wavelength) space. The
    spectrum is therefore:

        1. resampled onto a uniform ln(wavelength) grid
           (oversampled relative to the LSF width for numerical
           accuracy), then
        2. Gaussian-filtered on that uniform grid, then
        3. returned on the same (still finer-than-PRISM) log-uniform
           grid, ready for binning onto the observed JWST bins.

    Returns
    -------
    wavelength_um_conv : ndarray
    depth_ppm_conv      : ndarray
    """

    finite = np.isfinite(model_wavelength_um) & np.isfinite(model_depth_ppm)

    wl = model_wavelength_um[finite]
    dp = model_depth_ppm[finite]

    order = np.argsort(wl)
    wl = wl[order]
    dp = dp[order]

    if len(wl) < 3:
        raise RuntimeError(
            "Not enough finite model points to perform LSF convolution."
        )

    ln_wl = np.log(wl)

    # FWHM of the LSF in ln(wavelength) space is 1/R; convert to sigma.
    fwhm_ln = 1.0 / resolving_power
    sigma_ln = fwhm_ln / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # Build a uniform ln(wavelength) grid, oversampled relative to the
    # LSF width.
    d_ln_wl_grid = sigma_ln / oversample

    ln_wl_uniform = np.arange(ln_wl.min(), ln_wl.max(), d_ln_wl_grid)

    if len(ln_wl_uniform) < 3:
        raise RuntimeError(
            "Uniform log-wavelength grid for LSF convolution is degenerate; "
            "check resolving_power/oversample settings."
        )

    dp_uniform = np.interp(ln_wl_uniform, ln_wl, dp)

    sigma_pixels = sigma_ln / d_ln_wl_grid

    dp_convolved = gaussian_filter1d(dp_uniform, sigma=sigma_pixels, mode="nearest")

    wl_uniform = np.exp(ln_wl_uniform)

    return wl_uniform, dp_convolved


# ============================================================
# 12. BIN pRT MODEL TO JWST
# ============================================================

def bin_model_to_jwst(
    model_wavelength_um,
    model_depth_ppm,
    jwst_data,
    wavelength_min_um=None,
    wavelength_max_um=None,
):
    """
    Bin the (LSF-convolved) petitRADTRANS spectrum into the JWST
    observational bins. Each JWST bin is represented by the
    arithmetic mean of the model points falling inside that bin.
    """

    jwst_wavelength = jwst_data["wavelength"]
    jwst_lower = jwst_data["lower_edge"]
    jwst_upper = jwst_data["upper_edge"]
    jwst_depth = jwst_data["depth"]
    jwst_error = jwst_data["error"]

    wavelength_selection = np.ones(len(jwst_wavelength), dtype=bool)

    if wavelength_min_um is not None:
        wavelength_selection &= (jwst_upper >= wavelength_min_um)

    if wavelength_max_um is not None:
        wavelength_selection &= (jwst_lower <= wavelength_max_um)

    indices = np.where(wavelength_selection)[0]

    model_binned = []
    obs = []
    err = []
    centers = []
    lower_edges = []
    upper_edges = []
    n_model_points = []

    finite_model = np.isfinite(model_wavelength_um) & np.isfinite(model_depth_ppm)

    if not np.any(finite_model):
        raise RuntimeError("No finite pRT spectral points available.")

    model_min = np.min(model_wavelength_um[finite_model])
    model_max = np.max(model_wavelength_um[finite_model])

    for i in indices:

        lo = jwst_lower[i]
        hi = jwst_upper[i]

        if REQUIRE_FULL_BIN_COVERAGE:
            if lo < model_min or hi > model_max:
                continue

        in_bin = (
            finite_model
            & (model_wavelength_um >= lo)
            & (model_wavelength_um <= hi)
        )

        n_points = np.count_nonzero(in_bin)

        if n_points < MIN_MODEL_POINTS_PER_BIN:
            continue

        model_value = np.mean(model_depth_ppm[in_bin])

        if not np.isfinite(model_value):
            continue
        if not np.isfinite(jwst_depth[i]):
            continue
        if not np.isfinite(jwst_error[i]):
            continue
        if jwst_error[i] <= 0:
            continue

        model_binned.append(model_value)
        obs.append(jwst_depth[i])
        err.append(jwst_error[i])
        centers.append(jwst_wavelength[i])
        lower_edges.append(lo)
        upper_edges.append(hi)
        n_model_points.append(n_points)

    if len(obs) < 2:
        raise RuntimeError(
            "Fewer than 2 valid JWST/model bins remain "
            "after wavelength and coverage filtering."
        )

    return {
        "wavelength": np.asarray(centers),
        "lower_edge": np.asarray(lower_edges),
        "upper_edge": np.asarray(upper_edges),
        "obs": np.asarray(obs),
        "err": np.asarray(err),
        "model": np.asarray(model_binned),
        "n_model_points": np.asarray(n_model_points),
    }


# ============================================================
# 13. CALCULATE JWST CHI-SQUARE (+ FIX 6: BIC)
# ============================================================

def calculate_jwst_chi2(fit_data):
    """
    Calculate the weighted chi-square statistic and the Bayesian
    Information Criterion (BIC).

    Model:        M_i
    Observation:  D_i
    Uncertainty:  sigma_i
    Residual:     r_i = D_i - (M_i + offset)
    Chi-square:   chi2 = sum[(r_i / sigma_i)^2]

    If FIT_VERTICAL_OFFSET=True, one nuisance parameter (the vertical
    offset) is fitted, so dof = N_data - 1.

    FIX 6: BIC = chi2 + k * ln(N_data), where k = N_FITTED_PARAMETERS.
    This follows the standard Gaussian-likelihood BIC approximation
    (Eq. 3.4 discussion in the proposal, Objective 4). Because k is
    currently the same (=1, the offset) for every model in the grid,
    BIC and chi2 will rank models identically for this stage; BIC only
    becomes discriminating once models with different numbers of free
    parameters are compared (e.g. against a flat-line null model with
    k=1 offset-only, or a future model with additional free parameters
    such as a cloud-deck pressure).
    """

    obs = np.asarray(fit_data["obs"], dtype=float)
    err = np.asarray(fit_data["err"], dtype=float)
    model = np.asarray(fit_data["model"], dtype=float)

    valid = (
        np.isfinite(obs)
        & np.isfinite(err)
        & np.isfinite(model)
        & (err > 0)
    )

    obs = obs[valid]
    err = err[valid]
    model = model[valid]

    n_data = len(obs)

    if n_data < 2:
        raise RuntimeError(f"Only {n_data} valid JWST bins remain.")

    if FIT_VERTICAL_OFFSET:
        weights = 1.0 / np.maximum(err**2, EPS)
        offset_ppm = np.sum(weights * (obs - model)) / np.sum(weights)
    else:
        offset_ppm = 0.0

    model_shifted = model + offset_ppm
    residuals = obs - model_shifted

    chi2 = np.sum((residuals / err) ** 2)

    n_fitted_parameters = N_FITTED_PARAMETERS

    dof = n_data - n_fitted_parameters

    if dof <= 0:
        raise RuntimeError(
            f"Invalid degrees of freedom: N={n_data}, "
            f"parameters={n_fitted_parameters}"
        )

    reduced_chi2 = chi2 / dof

    rms_residual_ppm = np.sqrt(np.mean(residuals**2))

    # FIX 6: Bayesian Information Criterion
    bic = chi2 + n_fitted_parameters * np.log(n_data)

    return {
        "chi2": float(chi2),
        "reduced_chi2": float(reduced_chi2),
        "bic": float(bic),
        "n_data": int(n_data),
        "dof": int(dof),
        "offset_ppm": float(offset_ppm),
        "residuals": residuals,
        "model_shifted": model_shifted,
        "obs": obs,
        "err": err,
        "rms_residual_ppm": float(rms_residual_ppm),
    }


# ============================================================
# 14. FIX 6: DETECTABILITY / PER-BIN SNR
# ============================================================

def calculate_detectability(
    fit_data,
    stats,
    n_transits_observed=N_TRANSITS_OBSERVED,
    snr_target=SNR_TARGET,
):
    """
    Per-bin signal-to-noise and required-transit-count calculation,
    following Eq. 3.5 of the proposal:

        SNR(lambda) = delta_depth(lambda) / (sigma_ph(lambda) / sqrt(N_tr))

    Definitions used here:

    - delta_depth(lambda): the model's compositional "feature"
      amplitude in each bin, defined as the deviation of the
      (offset-shifted) model from its own error-weighted mean. A
      perfectly flat model has delta_depth = 0 everywhere; a model
      with strong absorption bands has large |delta_depth| in those
      bins. This isolates spectral shape from the arbitrary reference-
      radius offset.

    - sigma_ph(lambda): the single-transit photon-noise-limited error.
      The published JWST error bars already combine
      N_TRANSITS_OBSERVED transits (all four TRAPPIST-1e visits
      combined in this GP-corrected reduction), so the single-transit
      photon noise is recovered via
      sigma_ph = sigma_data * sqrt(N_TRANSITS_OBSERVED), assuming
      photon-noise-limited (1/sqrt(N)) scaling.

    - N_transits_for_detection: for the bin with the largest feature
      amplitude, the number of transits required to reach snr_target
      (default 5-sigma), computed by inverting the SNR equation:
      N_tr_required = N_TRANSITS_OBSERVED * (snr_target / SNR_current)^2,
      where SNR_current is the SNR already achieved with
      N_TRANSITS_OBSERVED transits at the current data noise level.

    Returns a dict with the per-bin arrays (for the detailed CSV) and
    two scalar summaries: max_feature_snr and n_transits_for_detection.
    """

    obs = np.asarray(fit_data["obs"], dtype=float)
    err = np.asarray(fit_data["err"], dtype=float)
    model_shifted = np.asarray(stats["model_shifted"], dtype=float)

    valid = np.isfinite(obs) & np.isfinite(err) & np.isfinite(model_shifted) & (err > 0)

    model_shifted = model_shifted[valid]
    err = err[valid]

    if len(model_shifted) < 1:
        return {
            "feature_amplitude_ppm": np.array([]),
            "photon_noise_single_transit_ppm": np.array([]),
            "snr_per_bin": np.array([]),
            "max_feature_snr": np.nan,
            "n_transits_for_detection": np.nan,
        }

    # Error-weighted mean model depth (proxy for the flat-spectrum
    # baseline set by the reference radius).
    weights = 1.0 / np.maximum(err**2, EPS)
    model_mean = np.sum(weights * model_shifted) / np.sum(weights)

    feature_amplitude_ppm = model_shifted - model_mean

    # Single-transit photon noise, backed out from the combined-transit
    # published error bars.
    photon_noise_single_transit_ppm = err * np.sqrt(n_transits_observed)

    # SNR already achieved in the existing (N_TRANSITS_OBSERVED-transit)
    # dataset for each bin's feature.
    snr_per_bin = np.abs(feature_amplitude_ppm) / err

    max_feature_snr = float(np.nanmax(snr_per_bin)) if len(snr_per_bin) else np.nan

    if np.isfinite(max_feature_snr) and max_feature_snr > 0:
        n_transits_for_detection = float(
            n_transits_observed * (snr_target / max_feature_snr) ** 2
        )
    else:
        n_transits_for_detection = np.nan

    return {
        "feature_amplitude_ppm": feature_amplitude_ppm,
        "photon_noise_single_transit_ppm": photon_noise_single_transit_ppm,
        "snr_per_bin": snr_per_bin,
        "max_feature_snr": max_feature_snr,
        "n_transits_for_detection": n_transits_for_detection,
    }


# ============================================================
# 15. RUN ONE COMPLETE ATMOSPHERIC MODEL
# ============================================================

def run_one_model(model_id, P_N2, P_CO2, P_CH4, T_surf_input):
    """
    Run one complete atmospheric model:
        climate -> photochemistry -> petitRADTRANS (lbl) ->
        LSF convolution -> JWST binning -> chi2/BIC/SNR
    """

    result = empty_result(model_id, P_N2, P_CO2, P_CH4, T_surf_input)

    # ========================================================
    # 1. CLIMATE
    # ========================================================

    c = AdiabatClimate(
        "input/species_climate.yaml",
        "input/settings_climate.yaml",
        "TRAPPIST1e_stellar_flux.txt",
    )

    P_i = np.ones(len(c.species_names)) * 1e-10
    P_i[c.species_names.index("H2O")] = 270.0
    P_i[c.species_names.index("N2")] = P_N2
    P_i[c.species_names.index("CO2")] = P_CO2
    P_i[c.species_names.index("CH4")] = P_CH4
    P_i *= 1e6  # bar -> dyn/cm^2

    c.RH = np.ones(len(c.species_names)) * 0.5
    c.P_top = 1e-5 * 1e6
    c.max_rc_iters = 30
    c.convective_newton_step_size = 0.05
    c.solve_for_T_trop = True
    c.T_trop = c.rad.skin_temperature(0.3)

    c.surface_temperature(P_i, T_guess=300)
    climate_return = c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below)

    result["T_surf_climate"] = float(c.T_surf)
    result["climate_converged"] = bool(climate_return)

    if not result["climate_converged"]:
        return result

    # ========================================================
    # 2. CLIMATE -> PHOTOCHEMISTRY
    # ========================================================

    atmosphere_file = RESULTS_DIR / f"atmosphere_init_{model_id:04d}.txt"

    c.out2atmosphere_txt(
        str(atmosphere_file),
        eddy=np.ones(c.z.shape[0]) * EDDY,
        overwrite=True,
    )

    # ========================================================
    # 3. PHOTOCHEMISTRY
    # ========================================================

    pc = EvoAtmosphere(
        "zahnle_HNOC.yaml",
        "input/settings_photochemistry.yaml",
        "TRAPPIST1e_stellar_flux.txt",
        str(atmosphere_file),
    )

    pc.var.verbose = 0
    pc.var.atol = 1e-23
    pc.var.upwind_molec_diff = True

    pc.set_lower_bc("N2", bc_type="press", press=P_N2 * 1e6)
    pc.set_lower_bc("CO2", bc_type="press", press=P_CO2 * 1e6)
    pc.set_lower_bc("CH4", bc_type="press", press=P_CH4 * 1e6)

    pc.update_vertical_grid(TOA_pressure=1e-7 * 1e6)

    utils.find_steady_state(
        pc,
        plot_species=["H2O", "N2", "O2", "CO2", "CH4", "CO", "H2"],
        xlim=(1e-10, 1),
        plot_freq=10**9,
    )

    result["photochemistry_converged"] = True

    # ========================================================
    # 4. PHOTOCHEMISTRY -> petitRADTRANS
    # ========================================================

    mf = pc.mole_fraction_dict()

    pressure_bar = np.asarray(mf["pressure"]) / 1e6
    temperature = np.asarray(mf["temp"])

    mole_fractions = {sp: np.asarray(mf[sp]) for sp in SPECIES}

    denominator = np.zeros_like(pressure_bar, dtype=float)
    for sp in SPECIES:
        denominator += mole_fractions[sp] * MOLECULAR_WEIGHTS[sp]

    mass_fractions = {
        sp: mole_fractions[sp] * MOLECULAR_WEIGHTS[sp] / denominator
        for sp in SPECIES
    }

    mean_molar_mass = np.zeros_like(pressure_bar, dtype=float)
    for sp in SPECIES:
        mean_molar_mass += mole_fractions[sp] * MOLECULAR_WEIGHTS[sp]

    # ========================================================
    # 5. REVERSE PRESSURE ORDER FOR pRT
    # ========================================================

    prt_pressures = pressure_bar[::-1]
    prt_temperature = temperature[::-1]
    prt_mass_fractions = {sp: mass_fractions[sp][::-1] for sp in SPECIES}
    prt_mean_molar_masses = mean_molar_mass[::-1]

    # ========================================================
    # 6. petitRADTRANS
    # ========================================================
    #
    # FIX 2: added O2 line opacity, H2 Rayleigh scattering, and
    # N2-N2 / N2-H2 collision-induced absorption (CIA), matching
    # Section 3.2.2:
    #   "Opacity sources include N2-N2 and N2-H2 collision-induced
    #   absorption (CIA), molecular line opacities for H2O, CO2, CH4,
    #   CO, and O2. Rayleigh scattering from N2 and H2 is also
    #   considered."
    #
    # NOTE: verify the exact keyword name for continuum opacities
    # against the installed petitRADTRANS version's API docs
    # (it has been named `continuum_opacities` in older pRT2 releases
    # and `gas_continuum_contributors` in pRT3-style releases; the
    # `radtrans` import path used here matches the pRT3 convention).

    rt = Radtrans(
        pressures=prt_pressures,
        line_species=["H2O", "CO2", "CH4", "CO", "O2"],
        rayleigh_species=["N2", "H2"],
        gas_continuum_contributors=["N2-N2", "N2-H2"],
        line_opacity_mode="c-k",
    )

    wavelengths_cm, transit_radii_cm, _ = rt.calculate_transit_radii(
        temperatures=prt_temperature,
        mass_fractions=prt_mass_fractions,
        mean_molar_masses=prt_mean_molar_masses,
        reference_gravity=GRAVITY_CGS,
        reference_pressure=REFERENCE_PRESSURE_BAR,
        planet_radius=PLANET_RADIUS_CGS,
        variable_gravity=True,
    )

    model_wavelength_um = wavelengths_cm * 1e4

    transit_depth_ppm = (transit_radii_cm / R_STAR_CGS) ** 2 * 1e6

    finite_spectrum = np.isfinite(model_wavelength_um) & np.isfinite(transit_depth_ppm)

    if not np.any(finite_spectrum):
        raise RuntimeError("pRT produced no finite spectrum.")

    result["min_transit_depth_ppm"] = float(np.nanmin(transit_depth_ppm))
    result["max_transit_depth_ppm"] = float(np.nanmax(transit_depth_ppm))
    result["spectral_amplitude_ppm"] = float(
        result["max_transit_depth_ppm"] - result["min_transit_depth_ppm"]
    )

    # ========================================================
    # 7. FIX 5: CONVOLVE TO INSTRUMENT RESOLUTION (R~100) BEFORE BINNING
    # ========================================================

    model_wavelength_um_conv, transit_depth_ppm_conv = convolve_to_resolution(
        model_wavelength_um,
        transit_depth_ppm,
        resolving_power=JWST_RESOLVING_POWER,
        oversample=JWST_LSF_OVERSAMPLE,
    )

    # ========================================================
    # 8. BIN CONVOLVED pRT MODEL TO JWST
    # ========================================================

    fit_data = bin_model_to_jwst(
        model_wavelength_um=model_wavelength_um_conv,
        model_depth_ppm=transit_depth_ppm_conv,
        jwst_data=jwst_data,
        wavelength_min_um=JWST_WAVELENGTH_MIN_UM,
        wavelength_max_um=JWST_WAVELENGTH_MAX_UM,
    )

    # ========================================================
    # 9. CALCULATE CHI-SQUARE / BIC
    # ========================================================

    stats = calculate_jwst_chi2(fit_data)

    result["n_data"] = stats["n_data"]
    result["dof"] = stats["dof"]
    result["wavelength_min_um"] = float(np.min(fit_data["wavelength"]))
    result["wavelength_max_um"] = float(np.max(fit_data["wavelength"]))
    result["offset_ppm"] = stats["offset_ppm"]
    result["chi2"] = stats["chi2"]
    result["reduced_chi2"] = stats["reduced_chi2"]
    result["bic"] = stats["bic"]
    result["rms_residual_ppm"] = stats["rms_residual_ppm"]

    # ========================================================
    # 10. FIX 6: DETECTABILITY / SNR
    # ========================================================

    detect = calculate_detectability(
        fit_data,
        stats,
        n_transits_observed=N_TRANSITS_OBSERVED,
        snr_target=SNR_TARGET,
    )

    result["max_feature_snr"] = detect["max_feature_snr"]
    result["n_transits_for_detection"] = detect["n_transits_for_detection"]

    # ========================================================
    # 11. SAVE DETAILED JWST FIT (per-bin reconstruction)
    # ========================================================

    jwst_fit_file = RESULTS_DIR / f"model_{int(model_id):04d}_jwst_fit.csv"

    jwst_fit_table = pd.DataFrame({
        "wavelength_um": fit_data["wavelength"],
        "lower_edge_um": fit_data["lower_edge"],
        "upper_edge_um": fit_data["upper_edge"],
        "observed_depth_ppm": stats["obs"],
        "observational_error_ppm": stats["err"],
        "model_depth_ppm": fit_data["model"],
        "model_shifted_depth_ppm": stats["model_shifted"],
        "residual_ppm": stats["residuals"],
        "n_model_points": fit_data["n_model_points"],
        "feature_amplitude_ppm": detect["feature_amplitude_ppm"],
        "photon_noise_single_transit_ppm": detect["photon_noise_single_transit_ppm"],
        "snr_per_bin": detect["snr_per_bin"],
    })

    jwst_fit_table.to_csv(jwst_fit_file, index=False)

    return result


# ============================================================
# 16. RUN MONTE CARLO SWEEP
# ============================================================

def run_mc_sweep(input_csv=INPUT_CSV, start_model=START_MODEL, end_model=END_MODEL):
    """
    Run all requested atmospheric models.
    """

    models = pd.read_csv(input_csv)

    required = {"P_N2", "P_CO2", "P_CH4", "T_surf"}
    missing = required.difference(models.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if end_model is None:
        end_model = len(models)
    end_model = min(end_model, len(models))

    already_done = set()
    if SKIP_ALREADY_DONE and MASTER_CSV.exists():
        old = pd.read_csv(MASTER_CSV)
        if "model_id" in old.columns:
            already_done = set(old["model_id"].astype(int).tolist())

    print()
    print("=" * 70)
    print("TRAPPIST-1e ATMOSPHERIC MONTE CARLO SWEEP (flat-spectrum comparison)")
    print("=" * 70)
    print(f"Input models: {len(models)}")
    print(f"Running model_id {start_model} through {end_model - 1}")
    print(f"JWST comparison file: {JWST_FILE}")
    print(f"Master output: {MASTER_CSV}")
    print("=" * 70)

    for model_id in range(start_model, end_model):

        if model_id in already_done:
            print(f"[{model_id:04d}] already saved -> skipping")
            continue

        row = models.iloc[model_id]

        P_N2 = float(row["P_N2"])
        P_CO2 = float(row["P_CO2"])
        P_CH4 = float(row["P_CH4"])
        T_surf_input = float(row["T_surf"])

        print()
        print(
            f"[{model_id:04d}] "
            f"N2={P_N2:.6g} bar, CO2={P_CO2:.6g} bar, "
            f"CH4={P_CH4:.6g} bar, Tsurf={T_surf_input:.3f} K"
        )

        result = empty_result(model_id, P_N2, P_CO2, P_CH4, T_surf_input)

        try:
            result = run_one_model(
                model_id=model_id,
                P_N2=P_N2,
                P_CO2=P_CO2,
                P_CH4=P_CH4,
                T_surf_input=T_surf_input,
            )
        except Exception as exc:
            print(f"[{model_id:04d}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
        finally:
            save_result(result)
            print(
                f"[{model_id:04d}] saved | "
                f"Tclim={result['T_surf_climate']} | "
                f"climate={result['climate_converged']} | "
                f"photo={result['photochemistry_converged']} | "
                f"N={result['n_data']} | "
                f"dof={result['dof']} | "
                f"chi2={result['chi2']} | "
                f"red_chi2={result['reduced_chi2']} | "
                f"BIC={result['bic']} | "
                f"max_SNR={result['max_feature_snr']} | "
                f"N_tr_for_5sig={result['n_transits_for_detection']}"
            )

    print()
    print("=" * 70)
    print("SWEEP FINISHED")
    print("=" * 70)
    print(f"Master results: {MASTER_CSV}")
    print("=" * 70)

    return pd.read_csv(MASTER_CSV)


# ============================================================
# 17. CALCULATE GLOBAL DELTA-CHI-SQUARE / DELTA-BIC
# ============================================================

def calculate_delta_chi2(results_df):
    """
    Calculate:
        Delta chi-square: dchi2 = chi2 - chi2_min
        Delta BIC:        dBIC  = BIC  - BIC_min          (FIX 6)
        Relative likelihood: L/L_max = exp(-dchi2 / 2)

    chi2_min and BIC_min are determined only from models that:
        1. converged in climate
        2. converged in photochemistry
        3. have finite chi2 (and, for BIC, finite BIC)
    """

    results = results_df.copy()

    valid = (
        results["climate_converged"].astype(bool)
        & results["photochemistry_converged"].astype(bool)
        & np.isfinite(results["chi2"])
    )

    if not np.any(valid):
        raise RuntimeError("No successful models with finite chi-square values.")

    chi2_min = results.loc[valid, "chi2"].min()

    results["delta_chi2"] = np.nan
    results["relative_likelihood"] = np.nan

    results.loc[valid, "delta_chi2"] = results.loc[valid, "chi2"] - chi2_min
    results.loc[valid, "relative_likelihood"] = np.exp(-0.5 * results.loc[valid, "delta_chi2"])

    # FIX 6: delta-BIC, analogous to delta-chi2
    valid_bic = valid & np.isfinite(results["bic"])
    results["delta_bic"] = np.nan
    if np.any(valid_bic):
        bic_min = results.loc[valid_bic, "bic"].min()
        results.loc[valid_bic, "delta_bic"] = results.loc[valid_bic, "bic"] - bic_min
    else:
        bic_min = np.nan

    best_idx = results.loc[valid, "chi2"].idxmin()
    best_model_id = int(results.loc[best_idx, "model_id"])

    print()
    print("=" * 70)
    print("GLOBAL JWST FIT SUMMARY (flat-spectrum comparison)")
    print("=" * 70)
    print(f"Best model ID       : {best_model_id}")
    print(f"Minimum chi2        : {chi2_min:.6f}")
    print(f"Reduced chi2        : {results.loc[best_idx, 'reduced_chi2']:.6f}")
    print(f"Minimum BIC         : {bic_min if np.isfinite(bic_min) else float('nan'):.6f}")
    print(f"N data points       : {int(results.loc[best_idx, 'n_data'])}")
    print(f"Degrees of freedom  : {int(results.loc[best_idx, 'dof'])}")
    print(
        f"Wavelength range    : "
        f"{results.loc[best_idx, 'wavelength_min_um']:.5f} - "
        f"{results.loc[best_idx, 'wavelength_max_um']:.5f} um"
    )
    print(f"Offset              : {results.loc[best_idx, 'offset_ppm']:.3f} ppm")
    print(f"RMS residual        : {results.loc[best_idx, 'rms_residual_ppm']:.3f} ppm")
    print(f"Max feature SNR     : {results.loc[best_idx, 'max_feature_snr']:.3f}")
    print(
        f"N transits for 5-sig: "
        f"{results.loc[best_idx, 'n_transits_for_detection']:.2f}"
    )

    print()
    print("BEST-FIT ATMOSPHERE")
    print("-" * 70)
    print(f"N2              = {results.loc[best_idx, 'P_N2']:.8g} bar")
    print(f"CO2             = {results.loc[best_idx, 'P_CO2']:.8g} bar")
    print(f"CH4             = {results.loc[best_idx, 'P_CH4']:.8g} bar")
    print(f"Tsurf input     = {results.loc[best_idx, 'T_surf_input']:.6f} K")
    print(f"Tsurf climate   = {results.loc[best_idx, 'T_surf_climate']:.6f} K")
    print("=" * 70)

    return results, chi2_min, best_model_id


# ============================================================
# 18. SAVE FINAL ANALYSIS TABLE
# ============================================================

def save_final_analysis(results, chi2_min):
    """
    Save: complete master table, successful models,
    delta-chi2-ranked models, and the best-fit model.
    All written into RESULTS_DIR (the flat-spectrum-comparison
    results directory), separate from any earlier Canas-based run.
    """

    results = results.sort_values("model_id").reset_index(drop=True)
    results.to_csv(MASTER_CSV, index=False)

    successful = results[
        results["climate_converged"].astype(bool)
        & results["photochemistry_converged"].astype(bool)
        & np.isfinite(results["chi2"])
    ].copy()

    successful = successful.sort_values("chi2").reset_index(drop=True)

    SUCCESSFUL_CSV = RESULTS_DIR / "successful_jwst_fits_flatspec.csv"
    successful.to_csv(SUCCESSFUL_CSV, index=False)

    delta_sorted = successful.sort_values("delta_chi2").reset_index(drop=True)

    DELTA_CSV = RESULTS_DIR / "jwst_delta_chi2_ranked_flatspec.csv"
    delta_sorted.to_csv(DELTA_CSV, index=False)

    if len(delta_sorted) > 0:
        best_model = delta_sorted.iloc[0].to_frame().T
        BEST_CSV = RESULTS_DIR / "best_fit_model_flatspec.csv"
        best_model.to_csv(BEST_CSV, index=False)

    print()
    print("=" * 70)
    print("FINAL FILES (flat-spectrum comparison)")
    print("=" * 70)
    print(f"Master results:\n  {MASTER_CSV}")
    print(f"\nSuccessful models:\n  {SUCCESSFUL_CSV}")
    print(f"\nDelta-chi2-ranked models:\n  {DELTA_CSV}")
    if len(delta_sorted) > 0:
        print(f"\nBest-fit model:\n  {BEST_CSV}")
    print("=" * 70)

    return results, successful, delta_sorted


# ============================================================
# 19. MAIN EXECUTION
# ============================================================

print()
print("=" * 70)
print("STARTING TRAPPIST-1e MODEL PIPELINE (flat-spectrum comparison)")
print("=" * 70)
print()
print("Workflow:")
print("ATMOS climate")
print("    v")
print("Photochemical steady state")
print("    v")
print("petitRADTRANS (line-by-line)")
print("    v")
print("Gaussian LSF convolution (R~100)")
print("    v")
print("JWST binning")
print("    v")
print("Weighted chi2 / BIC")
print("    v")
print("Global delta-chi2 / delta-BIC")
print("    v")
print("Relative likelihood")
print("    v")
print("Per-bin SNR / N_transits needed")
print()

results = run_mc_sweep()

results, chi2_min, best_model_id = calculate_delta_chi2(results)

(results, successful_results, delta_sorted) = save_final_analysis(results, chi2_min)

print()
print("=" * 70)
print("TOP 10 JWST-FAVORED MODELS (flat-spectrum comparison)")
print("=" * 70)

display_columns = [
    "model_id",
    "P_N2",
    "P_CO2",
    "P_CH4",
    "T_surf_input",
    "T_surf_climate",
    "n_data",
    "dof",
    "chi2",
    "reduced_chi2",
    "delta_chi2",
    "relative_likelihood",
    "bic",
    "delta_bic",
    "max_feature_snr",
    "n_transits_for_detection",
    "offset_ppm",
    "rms_residual_ppm",
]

if len(delta_sorted) > 0:
    display(delta_sorted[display_columns].head(10))
else:
    print("No successful JWST fits available.")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Minimum chi2 = {chi2_min:.6f}")
print(f"Best model ID = {best_model_id}")
print(f"Successful models = {len(successful_results)}")
print()
print(
    "Delta-chi2, delta-BIC, and relative likelihood have been calculated "
    "relative to the global minimum."
)
print()
print("For production:")
print("    START_MODEL = 0")
print("    END_MODEL = None")
print("=" * 70)