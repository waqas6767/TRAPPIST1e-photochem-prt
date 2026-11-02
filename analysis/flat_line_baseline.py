# ============================================================
# FLAT-LINE (FEATURELESS) BASELINE FOR THE TRAPPIST-1e PAPER
#
# Run this as a NEW, separate notebook cell in the same folder as your
# pipeline. It does NOT run Photochem or petitRADTRANS and does NOT
# touch your existing results; it only reads them.
#
# What it does
#   1. Takes the observed depths and error bars for EXACTLY the bins
#      your models were fitted to, from the per-bin file the pipeline
#      saved for the best model (model_XXXX_jwst_fit.csv).
#      -> same 67 bins, same errors, same wavelength range.
#   2. Re-computes that model's chi2 from the file and checks it equals
#      the value in the master CSV (proves the bins are identical).
#   3. Fits a featureless spectrum: a constant depth, using the SAME
#      inverse-variance weighted offset as calculate_jwst_chi2().
#   4. Compares the flat line with every converged atmosphere model:
#      delta-chi2, BIC (flat: k=1; atmosphere: k=1 offset + 3 gases),
#      and how many atmosphere models fit better than a flat line.
#   5. Prints the sentences to paste into the paper and saves a CSV.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist

# ------------------------------------------------------------
# Settings (same names/paths as the pipeline)
# ------------------------------------------------------------

RESULTS_DIR = Path("MC_2000_model_results_flatspec")
MASTER_CSV = RESULTS_DIR / "MC_2000_master_results_flatspec.csv"

JWST_FILE = (
    "/home/waqas/Downloads/"
    "TRAPPIST-1e-GTO-2025-main/"
    "espinoza+2025/figure3/"
    "tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt"
)

N_ATMOSPHERE_PARAMETERS = 3     # P_N2, P_CO2, P_CH4 (on top of the offset)
EPS = 1e-30

# ------------------------------------------------------------
# 1. Load master results and find the best converged model
# ------------------------------------------------------------

master = pd.read_csv(MASTER_CSV)
conv = (master["climate_converged"].astype(str).str.lower().isin(["true", "1"])
        & master["photochemistry_converged"].astype(str).str.lower().isin(["true", "1"])
        & np.isfinite(master["chi2"]))
models = master[conv].copy()
best = models.loc[models["chi2"].idxmin()]
best_id = int(best["model_id"])

print("=" * 70)
print("FLAT-LINE BASELINE")
print("=" * 70)
print(f"Converged atmosphere models : {len(models)}")
print(f"Best model                  : {best_id}  (chi2 = {best['chi2']:.4f})")

# ------------------------------------------------------------
# 2. Observed spectrum on exactly the fitted bins
# ------------------------------------------------------------

fit_file = RESULTS_DIR / f"model_{best_id:04d}_jwst_fit.csv"

if fit_file.exists():
    fit = pd.read_csv(fit_file)
    wl = fit["wavelength_um"].to_numpy(float)
    obs = fit["observed_depth_ppm"].to_numpy(float)
    err = fit["observational_error_ppm"].to_numpy(float)
    source = f"{fit_file} (exact bins used by the models)"

    # Check: recompute the best model's chi2 from this file
    mod = fit["model_depth_ppm"].to_numpy(float)
    w = 1.0 / np.maximum(err**2, EPS)
    off = np.sum(w * (obs - mod)) / np.sum(w)
    chi2_check = np.sum(((obs - mod - off) / err) ** 2)
    print(f"Check: chi2 of model {best_id} recomputed = {chi2_check:.4f} "
          f"(master CSV: {best['chi2']:.4f})")
    if abs(chi2_check - best["chi2"]) > 1e-3:
        raise RuntimeError("Recomputed chi2 does not match the master CSV - stop and check the files.")
else:
    # Fallback: read the spectrum file directly and keep the same wavelength range
    print(f"WARNING: {fit_file} not found - using the spectrum file and the fitted wavelength range.")
    data = np.genfromtxt(JWST_FILE, comments="#")
    data = data[np.argsort(data[:, 0])]
    wl, obs, err = data[:, 0], data[:, 1], data[:, 2]
    keep = (np.isfinite(wl) & np.isfinite(obs) & np.isfinite(err) & (err > 0)
            & (wl >= best["wavelength_min_um"] - 1e-6) & (wl <= best["wavelength_max_um"] + 1e-6))
    wl, obs, err = wl[keep], obs[keep], err[keep]
    source = f"{JWST_FILE} (wavelength cut {best['wavelength_min_um']:.3f}-{best['wavelength_max_um']:.3f} um)"

n = len(obs)
print(f"Spectrum source             : {source}")
print(f"Bins                        : {n}  ({wl.min():.3f}-{wl.max():.3f} um)")
if n != int(best["n_data"]):
    print(f"WARNING: {n} bins here but the models used {int(best['n_data'])}.")

# ------------------------------------------------------------
# 3. Flat line = constant depth (same weighted offset as the pipeline)
# ------------------------------------------------------------

w = 1.0 / np.maximum(err**2, EPS)
flat_level = np.sum(w * obs) / np.sum(w)
resid_flat = obs - flat_level
chi2_flat = float(np.sum((resid_flat / err) ** 2))
dof_flat = n - 1
p_flat = float(chi2_dist.sf(chi2_flat, dof_flat))
rms_flat = float(np.sqrt(np.mean(resid_flat**2)))

# ------------------------------------------------------------
# 4. Compare with the atmosphere models
# ------------------------------------------------------------

k_flat = 1
k_atm = 1 + N_ATMOSPHERE_PARAMETERS
bic_flat = chi2_flat + k_flat * np.log(n)
bic_best = best["chi2"] + k_atm * np.log(n)

dchi2 = chi2_flat - best["chi2"]                 # > 0 means the best atmosphere fits better
dbic = bic_flat - bic_best                       # > 0 means BIC prefers the atmosphere
n_better = int((models["chi2"] < chi2_flat).sum())
frac_better = n_better / len(models)

print()
print("-" * 70)
print("RESULTS")
print("-" * 70)
print(f"Flat-line depth             : {flat_level:.2f} ppm")
print(f"chi2_flat                   : {chi2_flat:.3f}   (dof = {dof_flat}, reduced = {chi2_flat/dof_flat:.4f}, p = {p_flat:.3f})")
print(f"rms residual (flat)         : {rms_flat:.2f} ppm")
print(f"chi2_best atmosphere        : {best['chi2']:.3f}")
print(f"Delta chi2 (flat - best)    : {dchi2:+.3f}")
print(f"BIC penalty, 3 extra params : {N_ATMOSPHERE_PARAMETERS*np.log(n):.2f}")
print(f"Delta BIC (flat - best)     : {dbic:+.3f}   (> 0 favours the atmosphere)")
print(f"Atmosphere models with chi2 < chi2_flat : {n_better} of {len(models)} ({100*frac_better:.1f}%)")

if dbic > 10:
    verdict = "BIC strongly prefers the best atmosphere model over a flat line."
elif dbic > 2:
    verdict = "BIC mildly prefers the best atmosphere model over a flat line."
elif dbic > -2:
    verdict = "BIC cannot distinguish the best atmosphere model from a flat line."
else:
    verdict = "BIC prefers the flat (featureless) spectrum over the best atmosphere model."
print(f"Verdict                     : {verdict}")

# ------------------------------------------------------------
# 5. Text for the paper + saved summary
# ------------------------------------------------------------

print()
print("-" * 70)
print("PASTE INTO SECTION 3.2")
print("-" * 70)
print(f"A constant-depth model gives $\\chi^2_{{\\rm flat}}={chi2_flat:.2f}$ "
      f"($\\chi^2_\\nu={chi2_flat/dof_flat:.3f}$, $P={p_flat:.2f}$), i.e., "
      f"$\\Delta\\chi^2=\\chi^2_{{\\rm flat}}-\\chi^2_{{\\min}}={dchi2:.2f}$ and "
      f"$\\Delta{{\\rm BIC}}={dbic:.2f}$. {n_better} of the {len(models)} coupled models "
      f"({100*frac_better:.0f}\\%) fit better than the flat line.")

summary = pd.DataFrame([{
    "n_bins": n, "wl_min_um": wl.min(), "wl_max_um": wl.max(),
    "flat_level_ppm": flat_level, "chi2_flat": chi2_flat, "dof_flat": dof_flat,
    "reduced_chi2_flat": chi2_flat / dof_flat, "p_flat": p_flat, "rms_flat_ppm": rms_flat,
    "best_model_id": best_id, "chi2_best": best["chi2"], "delta_chi2": dchi2,
    "bic_flat": bic_flat, "bic_best": bic_best, "delta_bic": dbic,
    "n_models_better_than_flat": n_better, "n_models": len(models), "verdict": verdict,
}])
out = RESULTS_DIR / "flat_line_baseline_summary.csv"
summary.to_csv(out, index=False)
print()
print(f"Saved: {out}")
print("=" * 70)
