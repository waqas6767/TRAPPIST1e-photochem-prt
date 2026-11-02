"""
Observed TRAPPIST-1e spectrum compared with the ten best-fitting models, the
flat line and the worst-fitting converged model, with residuals from the
best fit.

Each model file is checked against the master table (chi2 and offset) before
plotting.

Inputs : data/MC_2000_master_results_flatspec_21.csv, data/model_fits/
Output : figures/fig_spectra_top10.pdf/.png
"""
import glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, FIGURES
D = ""
res = pd.read_csv(DATA + "MC_2000_master_results_flatspec_21.csv").set_index("model_id")
ok = res[res.climate_converged & res.photochemistry_converged]
top = list(ok.nsmallest(10, "chi2").index)
worst = int(ok.chi2.idxmax())
files = {int(re.search(r"model_(\d+)_jwst", f).group(1)): f for f in glob.glob(DATA + "model_fits/model_*_jwst_fit.csv")}
assert set(top + [worst]) <= set(files), set(top + [worst]) - set(files)


def load(mid):
    d = pd.read_csv(files[mid])
    w = 1 / d.observational_error_ppm**2
    off = np.sum(w * (d.observed_depth_ppm - d.model_depth_ppm)) / w.sum()
    chi = np.sum(((d.observed_depth_ppm - d.model_depth_ppm - off) / d.observational_error_ppm) ** 2)
    assert abs(chi - res.loc[mid, "chi2"]) < 1e-3 and abs(off - res.loc[mid, "offset_ppm"]) < 1e-2, mid
    return d, d.model_depth_ppm + off


d0, _ = load(top[0])
wl, obs, err = d0.wavelength_um, d0.observed_depth_ppm, d0.observational_error_ppm
w = 1 / err**2
flat = np.sum(w * obs) / w.sum()
chi_flat = np.sum(((obs - flat) / err) ** 2)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})

# excluded model for contrast
dw, mw = load(worst)
rw = res.loc[worst]
ax.plot(wl, mw, color="k", lw=1.8, ls=":", zorder=1,
        label=f"Disfavored: CO$_2$ = {rw.P_CO2:.2f} bar, Tsurf = {rw.T_surf_climate:.0f} K ($\\chi^2$ = {rw.chi2:.1f})")
ax.axhline(flat, color="k", lw=1.4, ls="--", zorder=2, label=f"Flat line ($\\chi^2$ = {chi_flat:.2f})")
for rank, mid in enumerate(top, 1):
    _, m = load(mid)
    ax.plot(wl, m, lw=2, zorder=3, label=f"Best Fit {rank} — $\\chi^2$ = {res.loc[mid, 'chi2']:.2f}")
ax.errorbar(wl, obs, yerr=err, fmt="o", ms=4, color="k", ecolor="k", elinewidth=1, capsize=2, zorder=4,
            label="JWST NIRSpec/PRISM (Espinoza et al. 2025)")

# absorption bands
for x, s in [(1.15, "CH$_4$"), (1.4, "H$_2$O"), (1.7, "CH$_4$"), (2.3, "CH$_4$"), (2.7, "CO$_2$"), (3.3, "CH$_4$"), (4.3, "CO$_2$")]:
    ax.axvline(x, color="0.85", lw=0.8, zorder=0)
    ax.text(x, 365, s, ha="center", va="top", fontsize=9, color="0.35")

ax.set_ylim(-400, 400)
ax.set_ylabel("Relative transit depth (ppm)", fontsize=13)
ax.set_title("TRAPPIST-1e — Observed Spectrum vs. Best-Fit Cloud-Free Models", fontsize=15)
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend(loc="lower right", fontsize=8, ncol=2)

_, mbest = load(top[0])
axr.axhline(0, color="k", lw=1)
axr.errorbar(wl, obs - mbest, yerr=err, fmt="o", ms=4, color="k", elinewidth=1, capsize=2)
axr.set_ylim(-400, 400)
axr.set_ylabel("Data − Best Fit 1\n(ppm)", fontsize=11)
axr.set_xlabel("Wavelength (μm)", fontsize=13)
axr.grid(True, which="both", ls=":", alpha=0.6)
axr.set_xlim(0.55, 5.3)

fig.subplots_adjust(left=0.1, right=0.97, top=0.93, bottom=0.08)
for ext in ("pdf", "png"):
    fig.savefig(FIGURES + f"fig_spectra_top10.{ext}", dpi=300)
spread = np.ptp(np.array([load(m)[1] for m in top]), axis=0)
print("worst", worst, "flat", round(flat, 2), round(chi_flat, 3), "max spread among top10 per bin", round(spread.max(), 1),
      "median spread", round(np.median(spread), 1), "median err", round(np.median(err), 1),
      "n bins |y|>400", int(((obs - err) < -400).sum() + ((obs + err) > 400).sum()))
mb = mbest.values; print("best model range", round(mb.min(), 1), round(mb.max(), 1), "ptp", round(np.ptp(mb), 1))
i43 = (wl > 4.1) & (wl < 4.5); print("4.3um worst-minus-flat mean", round((mw[i43] - flat).mean(), 1), "best", round((mb[i43] - flat).mean(), 1))
