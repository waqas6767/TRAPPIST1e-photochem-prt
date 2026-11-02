"""T-P profiles of the ten best-fitting models (style matched to the author's figure).
Every file is checked against the results CSV before plotting."""
import glob, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import DATA, DERIVED, MODELS, FIGURES
D = ""
res = pd.read_csv(DATA + "MC_2000_master_results_flatspec_21.csv").set_index("model_id")
ok = res[res.climate_converged & res.photochemistry_converged]
top = list(ok.nsmallest(10, "chi2").index)
files = {int(re.search(r"init_(\d+)", f).group(1)): f for f in glob.glob(DATA + "atmosphere_profiles/atmosphere_init_*.txt")}
assert set(top) <= set(files), f"missing: {set(top) - set(files)}"

fig, ax = plt.subplots(figsize=(10, 8))
for rank, mid in enumerate(top, 1):
    a = pd.read_csv(files[mid], sep=r"\s+")
    z, P = a.alt.values, a.press.values
    Ps = np.exp(np.log(P[0]) - (np.log(P[1]) - np.log(P[0])) / (z[1] - z[0]) * z[0])
    r = res.loc[mid]
    err = [abs(a[s].values[0] * Ps / r[c] - 1) for s, c in (("N2", "P_N2"), ("CO2", "P_CO2"), ("CH4", "P_CH4"))]
    assert max(err) < 0.05, f"model {mid}: file does not match CSV"
    print(f"Best Fit {rank}: model {mid} verified ({100*max(err):.1f}%), Tsurf {r.T_surf_climate:.2f} K, chi2 {r.chi2:.3f}")
    ax.plot(a.temp, a.press, lw=2, label=f"Best Fit {rank} — Tsurf = {r.T_surf_climate:.2f} K")
ax.set_yscale("log"); ax.invert_yaxis()
ax.set_xlabel("Temperature (K)", fontsize=13); ax.set_ylabel("Pressure (bar)", fontsize=13)
ax.set_title("TRAPPIST-1e — Temperature–Pressure Profiles", fontsize=15)
ax.grid(True, which="both", ls=":", alpha=0.6)
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.89), fontsize=9)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIGURES + f"fig_tp_profiles.{ext}", dpi=300)
print("saved")
