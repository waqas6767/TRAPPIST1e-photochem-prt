# ============================================================
# REFIT ALL MODELS TO THE UNCORRECTED, QUIET VISITS 1 + 2
#
# Why: the spectrum used so far
#   (tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt)
# was decontaminated with a stellar-contamination model fitted while
# ASSUMING A FLAT PLANET (bf_model = 'flat' in Espinoza et al.'s
# plot_combined_corrected_folder.py). That correction can absorb real
# planetary features, so a flat line fitting best is partly built in.
#
# Visits 1 and 2 are the two quiet visits (Glidden et al. 2025 analysed
# them WITHOUT correction). Refitting to them is an independent test.
#
# This cell needs NO Photochem / petitRADTRANS run: it re-uses the binned
# model spectrum your pipeline already saved for every model
# (MC_2000_model_results_flatspec/model_XXXX_jwst_fit.csv, column
# model_depth_ppm). The visit files use the same 67 bins (bin widths
# agree to 0.3% median), so the saved models can be used directly.
#
# Fit: each visit gets its OWN free offset (2 offsets), chi2 summed over
# both visits: 134 points, dof = 132. Flat line: 2 offsets only.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2 as chi2_dist, spearmanr

RESULTS_DIR = Path("MC_2000_model_results_flatspec")
MASTER_CSV = RESULTS_DIR / "MC_2000_master_results_flatspec.csv"
VISIT_DIR = Path("/home/waqas/Downloads/TRAPPIST-1e-GTO-2025-main/espinoza+2025/figure3/data/jwst")
VISITS = [1, 2]
N_ATM_PARAMS = 3
EPS = 1e-30

# ---------------- observed spectra ----------------
obs, err, wl = [], [], None
for v in VISITS:
    w, d, e = np.loadtxt(VISIT_DIR / f"visit{v}.txt", unpack=True, usecols=(0, 3, 4))
    obs.append(d); err.append(e)
    wl = w if wl is None else wl
obs, err = np.array(obs), np.array(err)          # shape (n_visits, 67)
n_vis, n_bin = obs.shape
n_tot = n_vis * n_bin

def chi2_offsets(model):
    """chi2 for one binned model (67 values) with a free offset per visit."""
    total = 0.0
    for k in range(n_vis):
        wgt = 1.0 / np.maximum(err[k] ** 2, EPS)
        off = np.sum(wgt * (obs[k] - model)) / np.sum(wgt)
        total += np.sum(((obs[k] - model - off) / err[k]) ** 2)
    return total

chi2_flat = chi2_offsets(np.zeros(n_bin))
dof_flat = n_tot - n_vis

# ---------------- all converged models ----------------
master = pd.read_csv(MASTER_CSV)
ok = (master["climate_converged"].astype(str).str.lower().isin(["true", "1"])
      & master["photochemistry_converged"].astype(str).str.lower().isin(["true", "1"])
      & np.isfinite(master["chi2"]))
models = master[ok].copy()

chi2_v12, missing, bad_grid = [], 0, 0
for mid in models["model_id"].astype(int):
    f = RESULTS_DIR / f"model_{mid:04d}_jwst_fit.csv"
    if not f.exists():
        chi2_v12.append(np.nan); missing += 1; continue
    fit = pd.read_csv(f)
    if len(fit) != n_bin or np.abs(fit["wavelength_um"].to_numpy() - wl).max() > 2e-3:
        chi2_v12.append(np.nan); bad_grid += 1; continue
    chi2_v12.append(chi2_offsets(fit["model_depth_ppm"].to_numpy(float)))
models["chi2_v12"] = chi2_v12
models = models[np.isfinite(models["chi2_v12"])].copy()

print("=" * 70)
print("REFIT TO UNCORRECTED VISITS", VISITS)
print("=" * 70)
print(f"Data points: {n_tot} ({n_vis} visits x {n_bin} bins), offsets: {n_vis}")
print(f"Models refitted: {len(models)}   (missing fit files: {missing}, grid mismatch: {bad_grid})")

best = models.loc[models["chi2_v12"].idxmin()]
cmin = best["chi2_v12"]
dof_atm = n_tot - n_vis
models["dchi2_v12"] = models["chi2_v12"] - cmin

# ---------------- flat line vs atmospheres ----------------
bic_flat = chi2_flat + n_vis * np.log(n_tot)
bic_best = cmin + (n_vis + N_ATM_PARAMS) * np.log(n_tot)
L = np.exp(-0.5 * (models["chi2_v12"] - chi2_flat))
bayes = L.mean()                                   # prior-averaged likelihood ratio atm / flat
n_better = int((models["chi2_v12"] < chi2_flat).sum())

print()
print(f"chi2_flat            : {chi2_flat:.2f}  (dof {dof_flat}, reduced {chi2_flat/dof_flat:.3f}, P = {chi2_dist.sf(chi2_flat, dof_flat):.3f})")
print(f"best model           : {int(best['model_id'])}  chi2 = {cmin:.2f}  (reduced {cmin/dof_atm:.3f}, P = {chi2_dist.sf(cmin, dof_atm):.3f})")
print(f"worst model          : chi2 = {models['chi2_v12'].max():.2f}")
print(f"Delta chi2 flat-best : {chi2_flat - cmin:+.2f}   (> 0: best atmosphere fits better)")
print(f"Delta BIC flat-best  : {bic_flat - bic_best:+.2f}   (> 0 favours the atmosphere)")
print(f"Bayes factor atm/flat: {bayes:.3f}   (2 ln B = {2*np.log(bayes):+.2f})")
print(f"models better than flat: {n_better} of {len(models)}")
for t in (6.18, 11.83):
    print(f"models worse than flat by dchi2 > {t}: {(models['chi2_v12'] - chi2_flat > t).mean()*100:.1f}%")

# ---------------- does the parameter picture change? ----------------
w = np.exp(-0.5 * models["dchi2_v12"]); w /= w.sum()
def wq(x, q):
    i = np.argsort(x.values); c = np.cumsum(w.values[i]); return np.interp(q, c / c[-1], x.values[i])
print()
print("Posterior (visits 1+2)  [16, 50, 84 percentiles]:")
print(f"  log10 P_CO2 : {[round(wq(np.log10(models.P_CO2), q), 2) for q in (.16, .5, .84)]}   95% upper P_CO2 = {10**wq(np.log10(models.P_CO2), .95):.3f} bar")
print(f"  log10 P_CH4 : {[round(wq(np.log10(models.P_CH4), q), 2) for q in (.16, .5, .84)]}")
print(f"  P_N2        : {[round(wq(models.P_N2, q), 1) for q in (.16, .5, .84)]}")
print(f"  T_surf      : {[round(wq(models.T_surf_climate, q), 1) for q in (.16, .5, .84)]}")
print(f"  ESS         : {1/np.sum(w**2):.0f}")
print()
print("Agreement with the decontaminated-spectrum fit:")
print(f"  Spearman(chi2_decont, chi2_v12)       = {spearmanr(models['chi2'], models['chi2_v12'])[0]:+.3f}")
print(f"  Spearman(chi2_v12, max_feature_snr)   = {spearmanr(models['chi2_v12'], models['max_feature_snr'])[0]:+.3f}")
print(f"  Spearman(chi2_v12, log P_CO2)         = {spearmanr(models['chi2_v12'], np.log10(models['P_CO2']))[0]:+.3f}")
print(f"  Spearman(chi2_v12, log P_CH4)         = {spearmanr(models['chi2_v12'], np.log10(models['P_CH4']))[0]:+.3f}")
print(f"  Spearman(chi2_v12, P_N2)              = {spearmanr(models['chi2_v12'], models['P_N2'])[0]:+.3f}")
print()
print("Top 10 models (visits 1+2):")
print(models.nsmallest(10, "chi2_v12")[["model_id", "P_N2", "P_CO2", "P_CH4", "T_surf_climate", "chi2_v12", "chi2"]].to_string(index=False))

out = RESULTS_DIR / "refit_visits12.csv"
models[["model_id", "P_N2", "P_CO2", "P_CH4", "T_surf_climate", "chi2", "chi2_v12", "max_feature_snr"]].to_csv(out, index=False)
pd.DataFrame([{"chi2_flat": chi2_flat, "dof": dof_flat, "best_id": int(best["model_id"]), "chi2_best": cmin,
               "dchi2_flat_best": chi2_flat - cmin, "dbic_flat_best": bic_flat - bic_best,
               "bayes_atm_over_flat": bayes, "n_better_than_flat": n_better, "n_models": len(models)}]
             ).to_csv(RESULTS_DIR / "refit_visits12_summary.csv", index=False)
print(f"\nSaved: {out} and refit_visits12_summary.csv")
print("=" * 70)
