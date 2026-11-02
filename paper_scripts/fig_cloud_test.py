"""
Grey-cloud robustness test: fit quality against cloud-top pressure.

For the ten best-fitting models and ten excluded models (delta-chi2 > 11.83
relative to a flat line), plots delta-chi2 relative to the flat line for a
clear sky and for opaque grey cloud decks at 1, 0.3, 0.1, 0.03 and 0.01 bar.

Input : data/cloud_test/cloud_test_results.csv  (written by analysis/cloud_test.py)
Output: figures/fig_cloud_test.pdf/.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import DATA, FIGURES

r = pd.read_csv(DATA + "cloud_test/cloud_test_results.csv")
order = ["clear", "1bar", "0.3bar", "0.1bar", "0.03bar", "0.01bar"]
x = np.arange(len(order))

fig, ax = plt.subplots(figsize=(8, 5.5))
for mid, g in r.groupby("model_id"):
    g = g.set_index("cloud").loc[order]
    if g.group.iloc[0] == "best":
        ax.plot(x, g.dchi2_vs_flat, color="#2a78d6", lw=1.3, alpha=0.8)
    else:
        ax.plot(x, g.dchi2_vs_flat, color="#eb6834", lw=1.5, marker="o", ms=3)
for y, ls, lab in [(6.18, "--", "Δχ² = 6.18"), (11.83, ":", "Δχ² = 11.83")]:
    ax.axhline(y, color="k", ls=ls, lw=1, label=lab)
ax.axhline(0, color="0.6", lw=0.8)
ax.axvline(0.5, color="0.75", lw=0.8)
ax.plot([], [], color="#2a78d6", label="10 best-fitting models")
ax.plot([], [], color="#eb6834", marker="o", label="10 excluded models (CO₂ 0.007–0.69 bar)")
ax.set_xticks(x, ["clear\n(no cloud)", "1", "0.3", "0.1", "0.03", "0.01"])
ax.set_xlabel("Opaque cloud-top pressure (bar)", fontsize=12)
ax.set_ylabel("Δχ² relative to the flat line", fontsize=12)
ax.grid(True, ls=":", alpha=0.6)
ax.legend(fontsize=9)
ax.set_title("TRAPPIST-1e — Grey-Cloud Test", fontsize=13)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGURES + f"fig_cloud_test.{ext}", dpi=300)
