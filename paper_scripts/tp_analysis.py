"""
Thermal-structure diagnostics of the ten best-fitting models: cold-trap
temperature and pressure, upper-atmosphere warm layer, and their
correlation with the CH4 and CO2 partial pressures.

Inputs : data/MC_2000_master_results_flatspec_21.csv, data/atmosphere_profiles/
Outputs: derived/tp_diagnostics.csv, derived/tp_stats.json
"""
import glob, re, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from paths import DATA, DERIVED, MODELS, FIGURES
D = ""
res = pd.read_csv(DATA + "MC_2000_master_results_flatspec_21.csv").set_index("model_id")
ok = res[res.climate_converged & res.photochemistry_converged]
top = list(ok.nsmallest(10, "chi2").index)
files = {int(re.search(r"init_(\d+)", f).group(1)): f for f in glob.glob(DATA + "atmosphere_profiles/atmosphere_init_*.txt")}
kB, amu, G, Mp, Rp = 1.380649e-23, 1.66054e-27, 6.6743e-11, 0.692*5.9722e24, 0.920*6.371e6
rows = []
for rank, m in enumerate(top, 1):
    a = pd.read_csv(files[m], sep=r"\s+"); r = res.loc[m]
    trop = a[(a.press < 1) & (a.press > 1e-2)]; i_min = trop.temp.idxmin()
    strat = a[(a.press < 1e-2) & (a.press > 1e-4)]; i_max = strat.temp.idxmax()
    phot = a[(a.press <= 1e-1) & (a.press >= 1e-3)]
    mu = (a[["N2","CO2","H2O","CH4"]].iloc[0] * [28.014, 44.01, 18.015, 16.043]).sum()
    g = G * Mp / Rp**2
    Tm = phot.temp.mean()
    rows.append(dict(rank=rank, id=m, N2=r.P_N2, CO2=r.P_CO2, CH4=r.P_CH4, Ts=r.T_surf_climate, chi2=r.chi2,
        snr=r.max_feature_snr, T0=a.temp.iloc[0], dTs=r.T_surf_climate - a.temp.iloc[0],
        Tmin=a.temp[i_min], Pmin=a.press[i_min], Tmax=a.temp[i_max], Pmax=a.press[i_max],
        dTinv=a.temp[i_max] - a.temp[i_min], fH2O_surf=a.H2O.iloc[0], fH2O_strat=a.H2O[a.press < a.press[i_min]].iloc[-1],
        Tphot=Tm, H_km=kB * Tm / (mu * amu * g) / 1e3, mu=mu))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
rng = np.random.default_rng(0)
def boot(x, y, n=10000):
    x, y = np.asarray(x), np.asarray(y); v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(i)) > 3: v.append(spearmanr(x[i], y[i])[0])
    return np.nanpercentile(v, [2.5, 97.5])
def pcor(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); Z = np.column_stack([np.ones(len(x)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return np.corrcoef(ex, ey)[0, 1]
out = {}
for a_, b_ in [("CH4","Tmax"),("CH4","Ts"),("CH4","Tmin"),("CH4","dTinv"),("CH4","Pmax"),("N2","Ts"),("CO2","Ts"),("N2","Tmax"),
               ("Tmin","fH2O_strat"),("CH4","fH2O_strat"),("CH4","snr"),("Tphot","snr"),("Tmax","snr"),("CH4","chi2"),("CH4","Tphot"),("Ts","fH2O_surf")]:
    rho, p = spearmanr(t[a_], t[b_]); ci = boot(t[a_], t[b_])
    out[f"{a_}-{b_}"] = dict(rho=rho, p=p, ci=list(ci)); print(f"{a_:5s}-{b_:10s} rho={rho:+.2f} p={p:.2g} 95%CI=[{ci[0]:+.2f},{ci[1]:+.2f}]")
for a_, b_, z_ in [("CH4","Ts","N2"),("CH4","Tmax","N2"),("CH4","Ts","CO2")]:
    v = pcor(t[a_], t[b_], t[z_]); out[f"{a_}-{b_}|{z_}"] = v; print(f"partial {a_}-{b_} | {z_}: {v:+.2f}")
# ensemble check: CH4 effect on Ts at fixed CO2 (low-CO2 slice) across all converged models
s = ok[ok.P_CO2 < 0.003]
r1 = spearmanr(np.log10(s.P_CH4), s.T_surf_climate)[0]
r2 = pcor(np.log10(s.P_CH4).values, s.T_surf_climate.values, s.P_N2.values)
print("ensemble low-CO2 slice n=", len(s), "rho CH4-Ts", r1, "partial|N2", r2)
out["slice"] = dict(n=len(s), rho=r1, partial_N2=r2)
for c in ["Tmin","Pmin","Tmax","Pmax","dTinv","fH2O_strat","fH2O_surf","dTs","H_km","Tphot","mu"]:
    out[c+"_range"] = [t[c].min(), t[c].max()]
t.to_csv(DERIVED + "tp_diagnostics.csv", index=False)
json.dump(out, open(DERIVED + "tp_stats.json", "w"), indent=1, default=float)
