import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paths import DATA, DERIVED, MODELS, FIGURES
D = DERIVED
F = FIGURES
import os; os.makedirs(F, exist_ok=True)
R = json.load(open(D + "results.json"))
g = pd.read_csv(D + "converged_with_weights.csv")
a = pd.read_csv(D + "all_runs.csv"); fail = a[~a.conv]
u = pd.read_csv(D + "emu_unseen.csv")
M = np.load(D + "emu_maps.npz")

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#0d3a73", "#2a78d6", "#9cc3f0", "#eef4fc"])  # dark = good fit
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.fontset": "dejavuserif"})
chimin = g.chi2.min()
lab = {"P_N2": r"$p_{\rm N_2}$ (bar)", "P_CO2": r"$\log_{10}\,p_{\rm CO_2}$ (bar)",
       "P_CH4": r"$\log_{10}\,p_{\rm CH_4}$ (bar)", "T_surf_climate": r"$T_{\rm surf}$ (K)"}
def xv(df, c): return np.log10(df[c]) if c in ("P_CO2", "P_CH4") else df[c]

# ---- Fig 1: delta chi2 vs each parameter
fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.0), sharey=True)
for k, c in enumerate(["P_CO2", "P_CH4", "P_N2", "T_surf_climate"]):
    ax[k].scatter(xv(g, c), g.dchi, s=3, c=BLUE, alpha=0.35, lw=0, rasterized=True)
    for t, ls in [(2.30, "--"), (6.18, ":")]:
        ax[k].axhline(t, color=MUTED, lw=0.6, ls=ls)
    ax[k].set_xlabel(lab[c])
    if c != "T_surf_climate":
        ax[k].plot(xv(fail, c), np.full(len(fail), -1.2), "|", color=ORANGE, ms=4, mew=0.6, alpha=0.7)
ax[0].set_ylabel(r"$\Delta\chi^2$"); ax[0].set_ylim(-2.2, 24)
fig.subplots_adjust(wspace=0.08)
fig.savefig(F + "fig_dchi2_params.pdf"); plt.close(fig)

# ---- Fig 2: CO2-CH4 plane (MC models coloured by delta chi2) + failures
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))
o = g.sort_values("dchi", ascending=False)
sc = ax[0].scatter(np.log10(o.P_CO2), np.log10(o.P_CH4), c=o.dchi, cmap=SEQ.reversed().reversed(), vmin=0, vmax=15, s=7, lw=0, rasterized=True)
ax[0].scatter(np.log10(fail.P_CO2), np.log10(fail.P_CH4), marker="x", s=10, c=ORANGE, lw=0.7, label="non-converged")
ax[0].set_xlabel(lab["P_CO2"]); ax[0].set_ylabel(lab["P_CH4"]); ax[0].legend(loc="lower right", fontsize=7)
cb = fig.colorbar(sc, ax=ax[0], pad=0.02); cb.set_label(r"$\Delta\chi^2$ (grid model)")
ax[0].set_title("(a) Coupled-model grid", loc="left")
chi10 = M["chi10"] - chimin
im = ax[1].pcolormesh(M["gc"], M["gm"], chi10, cmap=SEQ, vmin=0, vmax=15, shading="auto", rasterized=True)
cs = ax[1].contour(M["gc"], M["gm"], chi10, levels=[2.30, 6.18], colors=[INK, INK], linewidths=[0.9, 0.6], linestyles=["-", "--"])
ax[1].clabel(cs, fmt={2.30: "68%", 6.18: "95%"}, fontsize=6.5)
ax[1].set_xlabel(lab["P_CO2"]); ax[1].set_ylabel(lab["P_CH4"])
cb = fig.colorbar(im, ax=ax[1], pad=0.02); cb.set_label(r"$\Delta\chi^2$ (emulator)")
ax[1].set_title(r"(b) Emulator, $p_{\rm N_2}=10$ bar", loc="left")
fig.subplots_adjust(wspace=0.35)
fig.savefig(F + "fig_co2_ch4_plane.pdf"); plt.close(fig)

# ---- Fig 3: posterior marginals
def wh(ax, x, w, bins, color, label, ls="-"):
    h, e = np.histogram(x, bins=bins, weights=w, density=True)
    ax.step(np.r_[e[:-1], e[-1]], np.r_[h, h[-1]], where="post", color=color, lw=1.4, ls=ls, label=label)
fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.9))
for k, c in enumerate(["P_CO2", "P_CH4", "P_N2", "T_surf_climate"]):
    x = xv(g, c); bins = np.linspace(x.min(), x.max(), 21)
    ax[k].hist(x, bins=bins, density=True, color=GRAY, alpha=0.35, lw=0, label="prior (converged)")
    wh(ax[k], x, g.w, bins, BLUE, "posterior")
    wh(ax[k], x, g.w_resc, bins, ORANGE, r"rescaled $\sigma$", ls="--")
    ax[k].set_xlabel(lab[c]); ax[k].set_yticks([])
ax[3].axvspan(273, 373, color=AQUA, alpha=0.10, lw=0)
ax[0].legend(fontsize=6.5, loc="upper right")
ax[0].set_ylabel("density")
fig.subplots_adjust(wspace=0.12)
fig.savefig(F + "fig_posteriors.pdf"); plt.close(fig)

# ---- Fig 4: emulator validation on unseen models
fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3))
for k, (t, p, nm, key) in enumerate([("chi2", "pc", r"$\chi^2$", "chi2"), ("N", "pn", r"$N_{5\sigma}$ (transits)", "n_transits_for_detection"), ("T", "pT", r"$T_{\rm surf}$ (K)", None)]):
    ax[k].scatter(u[t], u[p], s=3, c=BLUE, alpha=0.35, lw=0, rasterized=True)
    lo, hi = min(u[t].min(), u[p].min()), max(u[t].max(), u[p].max())
    ax[k].plot([lo, hi], [lo, hi], color=MUTED, lw=0.7)
    ax[k].set_xlabel("pipeline " + nm); ax[k].set_ylabel("emulator " + nm)
    m = R["emu_det"][key] if key else R["emu_clim"]
    unit = " K" if key is None else ""
    ax[k].text(0.04, 0.92, rf"$R^2={m['r2']:.3f}$" + f"\nMAE = {m['mae']:.2f}{unit}", transform=ax[k].transAxes, va="top", fontsize=7)
fig.subplots_adjust(wspace=0.38)
fig.savefig(F + "fig_emulator_validation.pdf"); plt.close(fig)

# ---- Fig 5: detectability map
fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.0))
NSEQ = LinearSegmentedColormap.from_list("n", ["#0d3a73", "#2a78d6", "#9cc3f0", "#f4f8fd"])
for k, (key, n2) in enumerate([("N1", 1), ("N10", 10)]):
    N3 = M[key] * 9 / 25
    im = ax[k].pcolormesh(M["gc"], M["gm"], M[key], cmap=NSEQ, vmin=25, vmax=100, shading="auto", rasterized=True)
    c1 = ax[k].contour(M["gc"], M["gm"], N3, levels=[19], colors=[ORANGE], linewidths=1.1)
    c2 = ax[k].contour(M["gc"], M["gm"], M["chi" + key[1:]] - chimin, levels=[2.30], colors=[INK], linewidths=0.8, linestyles="--")
    ax[k].set_xlabel(lab["P_CO2"]); ax[k].set_ylabel(lab["P_CH4"])
    ax[k].set_title(rf"({'ab'[k]}) $p_{{\rm N_2}}={n2}$ bar", loc="left")
cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label(r"$N_{5\sigma}$: transits for a 5$\sigma$ feature")
from matplotlib.lines import Line2D
ax[0].legend([Line2D([], [], color=ORANGE, lw=1.1), Line2D([], [], color=INK, lw=0.8, ls="--")],
             [r"$N_{3\sigma}=19$ (4 + 15 transits)", r"$\Delta\chi^2=2.3$"], loc="lower right", fontsize=6.5)
for k in range(2):
    ax[k].annotate("3$\\sigma$ reachable", xy=(-0.75, -5.2), color=INK, fontsize=7, ha="center")
fig.savefig(F + "fig_detectability.pdf"); plt.close(fig)
print("ok")
