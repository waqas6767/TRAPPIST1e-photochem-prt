"""All numbers for the TRAPPIST-1e paper. Writes results.json + derived tables."""
import json, pickle, numpy as np, pandas as pd
from scipy.stats import spearmanr, chi2 as CHI2
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

from paths import DATA, DERIVED, MODELS, FIGURES
OUT = DERIVED
SAFE = ("sklearn", "numpy", "pandas", "scipy", "collections", "builtins", "joblib", "_codecs", "copyreg")
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "__main__": return lambda *a, **k: None
        if m.split(".")[0] not in SAFE or (m == "builtins" and n in {"eval","exec","compile","open","__import__","getattr","system"}):
            raise pickle.UnpicklingError(f"{m}.{n}")
        return super().find_class(m, n)
def load(p):
    with open(p, "rb") as f: return U(f).load()

R = {}
d = pd.read_csv(DATA + "MC_2000_master_results_flatspec_21.csv")
d["conv"] = d.climate_converged & d.photochemistry_converged
g = d[d.conv].copy()
R["n_rows"] = len(d); R["n_conv"] = int(d.conv.sum()); R["n_fail"] = int((~d.conv).sum())
R["conv_frac"] = d.conv.mean()
R["n_data"] = int(g.n_data.iloc[0]); R["dof"] = int(g.dof.iloc[0])
R["prior"] = {"N2": [d.P_N2.min(), d.P_N2.max()], "CO2": [d.P_CO2.min(), d.P_CO2.max()], "CH4": [d.P_CH4.min(), d.P_CH4.max()]}
R["T_range"] = [g.T_surf_climate.min(), g.T_surf_climate.max()]
R["offset_mean"] = g.offset_ppm.mean(); R["offset_sd"] = g.offset_ppm.std()
R["rms_resid_range"] = [g.rms_residual_ppm.min(), g.rms_residual_ppm.max()]
R["amp_median"] = g.spectral_amplitude_ppm.median(); R["amp_range"] = [g.spectral_amplitude_ppm.min(), g.spectral_amplitude_ppm.max()]
R["snr_median"] = g.max_feature_snr.median(); R["snr_range"] = [g.max_feature_snr.min(), g.max_feature_snr.max()]
R["ntr_median"] = g.n_transits_for_detection.median(); R["ntr_range"] = [g.n_transits_for_detection.min(), g.n_transits_for_detection.max()]
# detection metric: N = N0 (k/SNR)^2 ?
ratio = g.n_transits_for_detection * g.max_feature_snr**2
R["N_SNR2_const"] = [ratio.min(), ratio.max()]  # expect 100 = 4*25

# --- chi2 summary
g["dchi"] = g.chi2 - g.chi2.min()
b = g.loc[g.chi2.idxmin()]
R["best"] = {k: float(b[k]) for k in ["model_id","P_N2","P_CO2","P_CH4","T_surf_climate","chi2","reduced_chi2","max_feature_snr","n_transits_for_detection","spectral_amplitude_ppm","offset_ppm"]}
R["chi2_range"] = [g.chi2.min(), g.chi2.max()]; R["redchi_range"] = [g.reduced_chi2.min(), g.reduced_chi2.max()]
R["dchi_max"] = g.dchi.max()
R["pval_best"] = CHI2.sf(g.chi2.min(), R["dof"]); R["pval_worst"] = CHI2.sf(g.chi2.max(), R["dof"])
R["delta_col_missing"] = int(g.delta_chi2.isna().sum())
R["spearman"] = {}
for c, lg in [("P_N2",0),("P_CO2",1),("P_CH4",1),("T_surf_climate",0)]:
    x = np.log10(g[c]) if lg else g[c]
    r, p = spearmanr(x, g.chi2); R["spearman"][c] = [r, p]
R["spearman_T_CO2"] = spearmanr(np.log10(g.P_CO2), g.T_surf_climate)[0]
R["spearman_T_CH4"] = spearmanr(np.log10(g.P_CH4), g.T_surf_climate)[0]
R["spearman_T_N2"] = spearmanr(g.P_N2, g.T_surf_climate)[0]
for lab, t in [("1s", 2.30), ("2s", 6.18), ("3s", 11.83)]:
    s = g[g.dchi < t]
    R[f"region_{lab}"] = {"n": len(s), "CO2_max": s.P_CO2.max(), "CH4_min": s.P_CH4.min(), "CH4_max": s.P_CH4.max(),
                          "T_min": s.T_surf_climate.min(), "T_max": s.T_surf_climate.max(), "N2_min": s.P_N2.min(), "N2_max": s.P_N2.max()}

# --- prior-weighted posterior (samples drawn from prior: N2 U, logCO2 U, logCH4 U)
def wq(x, w, qs):
    i = np.argsort(x); x, w = np.asarray(x)[i], np.asarray(w)[i]
    c = np.cumsum(w) / w.sum(); return [float(np.interp(q, c, x)) for q in qs]
def posterior(chi, tag, frame):
    w = np.exp(-0.5 * (chi - chi.min())); w = w / w.sum()
    ess = 1 / np.sum(w**2)
    P = {"ess": ess}
    for c, lg in [("P_N2",0),("P_CO2",1),("P_CH4",1),("T_surf_climate",0)]:
        x = np.log10(frame[c]) if lg else frame[c]
        P[c] = wq(x, w, [0.025, 0.16, 0.5, 0.84, 0.975])
        P[c + "_prior"] = list(np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975]))
    T = frame.T_surf_climate.to_numpy()
    P["T_frac_273_373"] = float(w[(T >= 273) & (T <= 373)].sum())
    P["T_frac_lt_300"] = float(w[T < 300].sum()); P["T_frac_gt_373"] = float(w[T > 373].sum())
    P["prior_T_frac_273_373"] = float(((T >= 273) & (T <= 373)).mean()); P["prior_T_frac_gt_373"] = float((T > 373).mean())
    P["CO2_95_upper"] = 10 ** wq(np.log10(frame.P_CO2), w, [0.95])[0]
    P["CH4_05_lower"] = 10 ** wq(np.log10(frame.P_CH4), w, [0.05])[0]
    Ntr = frame.n_transits_for_detection.to_numpy()
    P["Ntr_post_median"] = wq(Ntr, w, [0.5])[0]
    P["Ntr_post_16_84"] = wq(Ntr, w, [0.16, 0.84])
    R["post_" + tag] = P
    return w
w = posterior(g.chi2.to_numpy(), "nominal", g)
g["w"] = w
s = g.reduced_chi2.min()
R["rescale_factor"] = 1 / s
g["w_resc"] = posterior(g.chi2.to_numpy() / s, "rescaled", g)

# --- failures
R["fail_med"] = d[~d.conv][["P_N2","P_CO2","P_CH4"]].median().to_dict()
R["conv_med"] = g[["P_N2","P_CO2","P_CH4"]].median().to_dict()
bins = [-6, -5, -4, -3, -2]
d["lch4b"] = pd.cut(np.log10(d.P_CH4), bins, include_lowest=True)
R["fail_frac_by_CH4"] = {str(k): float(v) for k, v in (1 - d.groupby("lch4b", observed=True).conv.mean()).items()}
d["lco2b"] = pd.cut(np.log10(d.P_CO2), [-3, -2, -1, 0], include_lowest=True)
R["fail_frac_by_CO2"] = {str(k): float(v) for k, v in (1 - d.groupby("lco2b", observed=True).conv.mean()).items()}
d["n2b"] = pd.cut(d.P_N2, [0, 1, 5, 10, 20])
R["fail_frac_by_N2"] = {str(k): float(v) for k, v in (1 - d.groupby("n2b", observed=True).conv.mean()).items()}

# --- detectability (total transits for a 5-sigma single-feature detection)
N_have, N_prog = 4, 15
tot = N_have + N_prog
R["N_total_with_program"] = tot
Ntr = g.n_transits_for_detection
R["frac_models_N5_le_tot"] = float((Ntr <= tot).mean())
R["frac_models_N3_le_tot"] = float((Ntr * 9 / 25 <= tot).mean())
R["post_mass_N3_le_tot"] = float(g.w[Ntr * 9 / 25 <= tot].sum())
R["post_mass_N5_le_50"] = float(g.w[Ntr <= 50].sum())
R["frac_models_N5_le_50"] = float((Ntr <= 50).mean())
R["spearman_Ntr"] = {c: spearmanr(np.log10(g[c]) if c != "P_N2" else g[c], Ntr)[0] for c in ["P_N2","P_CO2","P_CH4"]}
R["spearman_Ntr_T"] = spearmanr(g.T_surf_climate, Ntr)[0]
R["spearman_amp_chi2"] = spearmanr(g.spectral_amplitude_ppm, g.chi2)[0]

# --- emulator
C = load(MODELS + "TRAPPIST1e_Climate_Detection_AI_Combined.pkl")
P = load(MODELS + "TRAPPIST1e_Detection_AI_COMPLETE.pkl")
clim, det = C["climate_model"], C["detection_model"]
R["emu_hyper"] = {"n_estimators": det.n_estimators, "max_depth": det.max_depth, "min_samples_leaf": det.min_samples_leaf,
                  "max_features": det.max_features, "clim_n_estimators": clim.n_estimators}
Xtr = pd.concat([P["X_train"], P["X_test"]])
R["emu_train_n"] = len(Xtr); R["emu_stored_metrics"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in P["metrics"].items()}
tr = set(map(tuple, Xtr[["P_N2","P_CO2","P_CH4"]].round(8).to_numpy()))
g["seen"] = [tuple(r) in tr for r in g[["P_N2","P_CO2","P_CH4"]].round(8).to_numpy()]
h = g[~g.seen].copy(); R["emu_unseen_n"] = len(h)
R["emu_train_id_max"] = int(g[g.seen].model_id.max())
G3 = h[["P_N2","P_CO2","P_CH4"]].to_numpy(float)
Tp = clim.predict(G3)
R["emu_clim"] = {"r2": r2_score(h.T_surf_climate, Tp), "mae": mean_absolute_error(h.T_surf_climate, Tp), "q90": float(np.quantile(abs(h.T_surf_climate - Tp), .9))}
Yp = det.predict(np.column_stack([G3, Tp]))
Yt = det.predict(np.column_stack([G3, h.T_surf_climate.to_numpy(float)]))
R["emu_det"] = {}; R["emu_det_trueT"] = {}
for i, c in enumerate(["chi2","reduced_chi2","n_transits_for_detection"]):
    t = h[c].to_numpy(float)
    for key, Y in [("emu_det", Yp), ("emu_det_trueT", Yt)]:
        e = abs(t - Y[:, i])
        R[key][c] = {"r2": r2_score(t, Y[:, i]), "mae": e.mean(), "q90": float(np.quantile(e, .9)), "max": e.max(), "rho": spearmanr(t, Y[:, i])[0]}
h["pc"], h["pn"], h["pT"] = Yp[:, 0], Yp[:, 2], Tp
lo = h[h.dchi < 3]
R["emu_lowchi"] = {"n": len(lo), "mae": float(abs(lo.chi2 - lo.pc).mean()), "bias": float((lo.pc - lo.chi2).mean()), "rho": spearmanr(lo.chi2, lo.pc)[0]}
hi = h[h.n_transits_for_detection > 80]
R["emu_highN"] = {"n": len(hi), "bias": float((hi.pn - hi.n_transits_for_detection).mean()), "pred_max": float(h.pn.max()), "true_max": float(h.n_transits_for_detection.max())}
R["emu_importance"] = dict(zip(C["detection_feature_columns"], map(float, det.feature_importances_)))
top_t = set(g.nsmallest(20, "chi2").model_id)
gp = det.predict(np.column_stack([g[["P_N2","P_CO2","P_CH4"]].to_numpy(float), clim.predict(g[["P_N2","P_CO2","P_CH4"]].to_numpy(float))]))
g["emu_chi2"] = gp[:, 0]
R["emu_top20_overlap"] = len(top_t & set(g.nsmallest(20, "emu_chi2").model_id))
pd.DataFrame({"model_id": h.model_id, "chi2": h.chi2, "pc": h.pc, "N": h.n_transits_for_detection, "pn": h.pn,
              "T": h.T_surf_climate, "pT": h.pT}).to_csv(OUT + "emu_unseen.csv", index=False)

# failed runs through emulator (sensitivity only)
f = d[~d.conv].copy()
F3 = f[["P_N2","P_CO2","P_CH4"]].to_numpy(float)
pf = det.predict(np.column_stack([F3, clim.predict(F3)]))
f["emu_chi2"] = pf[:, 0]
R["fail_emu_chi2"] = {"median": float(np.median(pf[:, 0])), "min": float(pf[:, 0].min()), "frac_dchi_lt_2.3": float((pf[:, 0] - g.chi2.min() < 2.3).mean())}
allchi = np.concatenate([g.chi2.to_numpy(), pf[:, 0]])
allf = pd.concat([g[["P_N2","P_CO2","P_CH4"]], f[["P_N2","P_CO2","P_CH4"]]])
allf["T_surf_climate"] = np.concatenate([g.T_surf_climate.to_numpy(), clim.predict(F3)])
allf["n_transits_for_detection"] = np.concatenate([Ntr.to_numpy(), pf[:, 2]])
posterior(allchi, "with_failed_emulated", allf)

# --- retrain on all converged models (10-fold CV)
Xall = g[["P_N2","P_CO2","P_CH4","T_surf_climate"]].to_numpy(float)
Yall = g[["chi2","reduced_chi2","n_transits_for_detection"]].to_numpy(float)
kf = KFold(10, shuffle=True, random_state=0)
cvp = cross_val_predict(clone(det), Xall, Yall, cv=kf, n_jobs=-1)
R["emu_retrain_cv"] = {c: {"r2": r2_score(Yall[:, i], cvp[:, i]), "mae": mean_absolute_error(Yall[:, i], cvp[:, i])} for i, c in enumerate(["chi2","reduced_chi2","n_transits_for_detection"])}
cvT = cross_val_predict(clone(clim), Xall[:, :3], Xall[:, 3], cv=kf, n_jobs=-1)
R["emu_retrain_cv_T"] = {"r2": r2_score(Xall[:, 3], cvT), "mae": mean_absolute_error(Xall[:, 3], cvT)}
det_all = clone(det).fit(Xall, Yall); clim_all = clone(clim).fit(Xall[:, :3], Xall[:, 3])

# --- dense emulator posterior over the prior box (retrained emulator)
rng = np.random.default_rng(1)
M = 200000
lo_, hi_ = R["prior"]["N2"]
S = np.column_stack([rng.uniform(lo_, hi_, M), 10 ** rng.uniform(-3, np.log10(0.7), M), 10 ** rng.uniform(-6, -2, M)])
TS = clim_all.predict(S); YS = det_all.predict(np.column_stack([S, TS]))
we = np.exp(-0.5 * (YS[:, 0] - YS[:, 0].min())); we /= we.sum()
E = {"ess": 1 / np.sum(we**2)}
for j, (c, lg) in enumerate([("P_N2",0),("P_CO2",1),("P_CH4",1)]):
    x = np.log10(S[:, j]) if lg else S[:, j]; E[c] = wq(x, we, [0.025, 0.16, 0.5, 0.84, 0.975])
E["T_surf_climate"] = wq(TS, we, [0.025, 0.16, 0.5, 0.84, 0.975])
E["CO2_95_upper"] = 10 ** wq(np.log10(S[:, 1]), we, [0.95])[0]
E["T_frac_273_373"] = float(we[(TS >= 273) & (TS <= 373)].sum())
R["post_emulator_dense"] = E
np.savez_compressed(OUT + "emu_dense.npz", S=S[:40000], T=TS[:40000], chi=YS[:40000, 0], N=YS[:40000, 2])
# map over CO2-CH4 at posterior-median N2 (and N2 marginalised)
gc, gm = np.linspace(-3, np.log10(0.7), 60), np.linspace(-6, -2, 60)
AA, BB = np.meshgrid(gc, gm)
maps = {}
for n2 in [1.0, 10.0]:
    X = np.column_stack([np.full(AA.size, n2), 10 ** AA.ravel(), 10 ** BB.ravel()])
    T = clim_all.predict(X); Y = det_all.predict(np.column_stack([X, T]))
    maps[n2] = (Y[:, 0].reshape(AA.shape), Y[:, 2].reshape(AA.shape), T.reshape(AA.shape))
np.savez_compressed(OUT + "emu_maps.npz", gc=gc, gm=gm, chi1=maps[1.0][0], N1=maps[1.0][1], T1=maps[1.0][2],
                    chi10=maps[10.0][0], N10=maps[10.0][1], T10=maps[10.0][2])

# --- top models table
top = g.nsmallest(8, "chi2")[["model_id","P_N2","P_CO2","P_CH4","T_surf_climate","chi2","reduced_chi2","max_feature_snr","n_transits_for_detection"]]
top.to_csv(OUT + "top_models.csv", index=False)
g.to_csv(OUT + "converged_with_weights.csv", index=False)
d[["model_id","P_N2","P_CO2","P_CH4","conv"]].to_csv(OUT + "all_runs.csv", index=False)

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
json.dump(conv(R), open(OUT + "results.json", "w"), indent=1)
print(json.dumps(conv(R), indent=1))
