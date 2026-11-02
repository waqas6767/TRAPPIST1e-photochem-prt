import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paths import DATA, FIGURES
D = DATA + "analysis_tables/"
C = pd.read_csv(D + "11_spearman_correlation_matrix.csv", index_col=0)
P = pd.read_csv(D + "12_spearman_pvalues.csv", index_col=0)

# drop exact duplicates: reduced chi2 == chi2 (rank-identical); N_5sigma == -SNR_max (deterministic)
keep = ["P_N2", "P_CO2", "P_CH4", "T_surf_input", "T_surf_climate", "chi2",
        "rms_residual_ppm", "max_feature_snr", "spectral_amplitude_ppm"]
lab = [r"$p_{\rm N_2}$", r"$p_{\rm CO_2}$", r"$p_{\rm CH_4}$", r"$T_{\rm surf}$ (stage 1)",
       r"$T_{\rm surf}$", r"$\chi^2$", "rms residual", r"SNR$_{\max}$", "amplitude"]
M = C.loc[keep, keep].to_numpy()
F = pd.read_csv(D + "13_correlations_FDR_all.csv")
fdr = {}
for _, r in F.iterrows():
    fdr[(r["Variable 1"], r["Variable 2"])] = fdr[(r["Variable 2"], r["Variable 1"])] = r["FDR_p"]
Pm = np.array([[fdr.get((a, b), 0.0) for b in keep] for a in keep])

INK, MUTED = "#0b0b0b", "#52514e"
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f6c3ab", "#f2f1ee", "#a9c9f0", "#2a78d6"])
plt.rcParams.update({"font.family": "serif", "font.size": 8, "mathtext.fontset": "dejavuserif",
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
n = len(keep)
fig, ax = plt.subplots(figsize=(5.6, 4.9))
mask = np.triu(np.ones_like(M, bool), k=1)
im = ax.imshow(np.ma.array(M, mask=mask), cmap=DIV, vmin=-1, vmax=1)
for i in range(n):
    for j in range(i + 1):
        v = M[i, j]
        txt = f"{v:.2f}" + ("" if Pm[i, j] < 0.05 or i == j else "*")
        ax.text(j, i, txt, ha="center", va="center", fontsize=6.8,
                color="white" if abs(v) > 0.75 else INK)
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(lab, rotation=45, ha="right"); ax.set_yticklabels(lab)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(r"Spearman $\rho$")
cb.outline.set_visible(False)
fig.savefig(FIGURES + "fig_correlation_matrix.pdf")
print("ok", P.loc[keep, keep].where(~mask).max().max())
