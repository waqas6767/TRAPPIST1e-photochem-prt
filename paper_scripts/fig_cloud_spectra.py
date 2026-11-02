"""
Effect of an opaque grey cloud deck on the transmission spectrum of the
best-fitting model (Best Fit 1) and of the worst-fitting converged model,
compared with the observed JWST NIRSpec/PRISM spectrum. Each curve is the
binned, offset-fitted model spectrum for one cloud-top pressure.

Inputs : data/cloud_test/cloud_test_results.csv, data/cloud_test/spectra/
Output : figures/fig_cloud_spectra.pdf/.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, FIGURES
D = DATA + "cloud_test/"
r = pd.read_csv(D + "cloud_test_results.csv")
levels = ["clear", "1bar", "0.3bar", "0.1bar", "0.03bar", "0.01bar"]
labels = ["clear", "1 bar", "0.3 bar", "0.1 bar", "0.03 bar", "0.01 bar"]
colors = ["k"] + list(plt.cm.viridis(np.linspace(0.15, 0.9, 5)))

fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"hspace": 0.08})
for ax, mid, name in zip(axes, [1007, 1301], ["(a) Best Fit 1", "(b) Worst-fitting model"]):
    row = r[(r.model_id == mid) & (r.cloud == "clear")].iloc[0]
    for lev, lab, col in zip(levels, labels, colors):
        d = pd.read_csv(D + f"spectra/model_{mid:04d}_{lev}.csv")
        dchi = r[(r.model_id == mid) & (r.cloud == lev)].dchi2_vs_flat.iloc[0]
        ax.plot(d.wavelength_um, d.model_shifted_depth_ppm, color=col, lw=2.4 if lev == "clear" else 1.6,
                zorder=3, label=f"{lab} ($\\Delta\\chi^2$ = {dchi:+.1f})")
    w = 1 / d.observational_error_ppm**2
    flat = np.sum(w * d.observed_depth_ppm) / w.sum()
    ax.axhline(flat, color="0.4", ls="--", lw=1.2, zorder=2, label="Flat line")
    ax.errorbar(d.wavelength_um, d.observed_depth_ppm, yerr=d.observational_error_ppm, fmt="o", ms=3.5,
                color="0.25", elinewidth=0.9, capsize=1.5, zorder=4, label="JWST NIRSpec/PRISM")
    ax.set_ylim(-330, 250)
    ax.set_ylabel("Relative transit depth (ppm)", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.set_title(f"{name}: CO$_2$ = {row.P_CO2:.3g} bar, CH$_4$ = {row.P_CH4:.2g} bar, "
                 f"N$_2$ = {row.P_N2:.1f} bar, Tsurf = {row.T_surf:.0f} K", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8, ncol=2, title="Cloud top ($\\Delta\\chi^2$ vs. flat line)",
              title_fontsize=8)
axes[1].set_xlabel("Wavelength (μm)", fontsize=12)
axes[1].set_xlim(0.55, 5.3)
fig.suptitle("TRAPPIST-1e — Effect of an Opaque Grey Cloud Deck", fontsize=14, y=0.94)
fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.07)
for ext in ("pdf", "png"):
    fig.savefig(FIGURES + f"fig_cloud_spectra.{ext}", dpi=300)
