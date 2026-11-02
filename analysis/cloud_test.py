"""
Grey-cloud robustness test.

Tests whether an opaque (grey) cloud deck changes the fit of the
cloud-free models, in particular whether it can make the atmospheres
excluded by the data (delta-chi2 > 11.83 relative to a flat line)
compatible with the observed spectrum.

Models: the ten best-fitting models and ten excluded models chosen evenly
in CO2 partial pressure (always including the worst-fitting model).

For each model the climate and photochemistry are re-run with the pipeline's
settings; the transmission spectrum is then computed with petitRADTRANS for
a clear sky and for cloud-top pressures of 1, 0.3, 0.1, 0.03 and 0.01 bar
(`opaque_cloud_top_pressure`), convolved, binned and fitted exactly as in the
pipeline. The clear-sky chi2 is compared with the master table as a check
that the re-run reproduces the original model.

Run from the pipeline folder (input/, zahnle_HNOC.yaml,
TRAPPIST1e_stellar_flux.txt, MC_2000_model_results_flatspec/).
Outputs:
    cloud_test/cloud_test_results.csv          one row per model and cloud level
    cloud_test/spectra/model_XXXX_<cloud>.csv  binned spectra
Finished models are skipped when the script is restarted.
"""

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from photochem import EvoAtmosphere
from photochem.clima import AdiabatClimate
from petitRADTRANS.radtrans import Radtrans
from threadpoolctl import threadpool_limits

_ = threadpool_limits(limits=4)

# ---------------- settings ----------------
MASTER_CSV = Path("MC_2000_model_results_flatspec/MC_2000_master_results_flatspec.csv")
JWST_FILE = "tspectra_decontaminated_corrected_flat_JWST-CLR-R10000.txt"   # observed spectrum; set to your copy
OUT_DIR = Path("cloud_test")
OUT_CSV = OUT_DIR / "cloud_test_results.csv"
SPEC_DIR = OUT_DIR / "spectra"
CLOUD_TOPS_BAR = [None, 1.0, 0.3, 0.1, 0.03, 0.01]     # None = clear sky
N_BEST, N_EXCLUDED = 10, 10
EXCLUDE_THRESHOLD = 11.83                                # Delta chi2 vs flat line
MAX_STEPS = 50000

# pipeline settings (identical to run_one_model)
EDDY = 1e5
JWST_RESOLVING_POWER = 100.0
JWST_LSF_OVERSAMPLE = 10
MIN_MODEL_POINTS_PER_BIN = 2
EPS = 1e-30
MOLECULAR_WEIGHTS = {"N2": 28.014, "CO2": 44.0095, "H2O": 18.01528, "CH4": 16.043,
                     "CO": 28.010, "O2": 31.998, "H2": 2.016}
SPECIES = list(MOLECULAR_WEIGHTS)
G, M_EARTH, R_EARTH, R_SUN_CGS = 6.67430e-11, 5.9722e24, 6.371e6, 6.957e10
M_PLANET = 0.692 * M_EARTH
R_PLANET_M = 0.920 * R_EARTH
GRAVITY_CGS = G * M_PLANET / R_PLANET_M**2 * 100.0
PLANET_RADIUS_CGS = R_PLANET_M * 100.0
R_STAR_CGS = 0.117 * R_SUN_CGS
REFERENCE_PRESSURE_BAR = 0.1

OUT_DIR.mkdir(exist_ok=True)
SPEC_DIR.mkdir(exist_ok=True)


def as_bool(s):
    return s.astype(str).str.lower().isin(["true", "1"])


def cloud_label(p):
    return "clear" if p is None else f"{p:g}bar"


# ---------------- data (same as load_jwst_data) ----------------
data = np.genfromtxt(JWST_FILE, comments="#")
data = data[np.argsort(data[:, 0])]
jw, jd, je = data[:, 0], data[:, 1], data[:, 2]
mid = 0.5 * (jw[:-1] + jw[1:])
lo = np.empty_like(jw); hi = np.empty_like(jw)
lo[1:] = mid; hi[:-1] = mid
lo[0] = jw[0] - (mid[0] - jw[0]); hi[-1] = jw[-1] + (jw[-1] - mid[-1])
good = np.isfinite(jw) & np.isfinite(jd) & np.isfinite(je) & (je > 0)
jw, jd, je, lo, hi = jw[good], jd[good], je[good], lo[good], hi[good]


def convolve(wl, dp):
    f = np.isfinite(wl) & np.isfinite(dp)
    wl, dp = wl[f], dp[f]
    o = np.argsort(wl); wl, dp = wl[o], dp[o]
    lnw = np.log(wl)
    sig = (1.0 / JWST_RESOLVING_POWER) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    step = sig / JWST_LSF_OVERSAMPLE
    grid = np.arange(lnw.min(), lnw.max(), step)
    return np.exp(grid), gaussian_filter1d(np.interp(grid, lnw, dp), sigma=sig / step, mode="nearest")


def bin_and_fit(wl, dp):
    fin = np.isfinite(wl) & np.isfinite(dp)
    wmin, wmax = wl[fin].min(), wl[fin].max()
    rows = []
    for i in range(len(jw)):
        if lo[i] < wmin or hi[i] > wmax:
            continue
        m = fin & (wl >= lo[i]) & (wl <= hi[i])
        if m.sum() < MIN_MODEL_POINTS_PER_BIN:
            continue
        rows.append((jw[i], jd[i], je[i], np.mean(dp[m])))
    w_, o_, e_, m_ = map(np.array, zip(*rows))
    wt = 1.0 / np.maximum(e_**2, EPS)
    off = np.sum(wt * (o_ - m_)) / np.sum(wt)
    chi2 = float(np.sum(((o_ - m_ - off) / e_) ** 2))
    return chi2, off, pd.DataFrame({"wavelength_um": w_, "observed_depth_ppm": o_,
                                    "observational_error_ppm": e_, "model_depth_ppm": m_,
                                    "model_shifted_depth_ppm": m_ + off})


wt_all = 1.0 / je**2
CHI2_FLAT = float(np.sum(((jd - np.sum(wt_all * jd) / wt_all.sum()) / je) ** 2))

# ---------------- choose the models ----------------
master = pd.read_csv(MASTER_CSV)
ok = master[as_bool(master["climate_converged"]) & as_bool(master["photochemistry_converged"])
            & np.isfinite(master["chi2"])].copy()
best = ok.nsmallest(N_BEST, "chi2").assign(group="best")
exc = ok[ok["chi2"] - CHI2_FLAT > EXCLUDE_THRESHOLD].sort_values("P_CO2")
worst_id = int(ok.loc[ok["chi2"].idxmax(), "model_id"])
idx = np.unique(np.linspace(0, len(exc) - 1, N_EXCLUDED).round().astype(int))
pick = exc.iloc[idx]
if worst_id not in set(pick.model_id):
    pick = pd.concat([pick.iloc[:-1], exc[exc.model_id == worst_id]])
sample = pd.concat([best, pick.assign(group="excluded")]).reset_index(drop=True)

done = set()
if OUT_CSV.exists():
    done = set(pd.read_csv(OUT_CSV)["model_id"].astype(int))

print("=" * 70)
print("GREY-CLOUD ROBUSTNESS TEST")
print("=" * 70)
print(f"Flat-line chi2                : {CHI2_FLAT:.3f}")
print(f"Excluded models available     : {len(exc)} (Delta chi2 > {EXCLUDE_THRESHOLD})")
print(f"Models in test                : {len(sample)} ({N_BEST} best + {len(pick)} excluded)")
print(f"Cloud tops (bar)              : {[cloud_label(p) for p in CLOUD_TOPS_BAR]}")
print(f"Already done (skipped)        : {len(done & set(sample.model_id))}")
print("=" * 70)


# ---------------- one model ----------------
def run_model(row):
    mid_ = int(row.model_id)
    P_N2, P_CO2, P_CH4 = float(row.P_N2), float(row.P_CO2), float(row.P_CH4)
    t0 = time.time()

    c = AdiabatClimate("input/species_climate.yaml", "input/settings_climate.yaml",
                       "TRAPPIST1e_stellar_flux.txt")
    P_i = np.ones(len(c.species_names)) * 1e-10
    P_i[c.species_names.index("H2O")] = 270.0
    P_i[c.species_names.index("N2")] = P_N2
    P_i[c.species_names.index("CO2")] = P_CO2
    P_i[c.species_names.index("CH4")] = P_CH4
    P_i *= 1e6
    c.RH = np.ones(len(c.species_names)) * 0.5
    c.P_top = 1e-5 * 1e6
    c.max_rc_iters = 30
    c.convective_newton_step_size = 0.05
    c.solve_for_T_trop = True
    c.T_trop = c.rad.skin_temperature(0.3)
    c.surface_temperature(P_i, T_guess=300)
    if not c.RCE(P_i, c.T_surf, c.T, c.convecting_with_below):
        raise RuntimeError("climate did not converge on re-run")

    atm = OUT_DIR / f"atmosphere_cloud_{mid_:04d}.txt"
    c.out2atmosphere_txt(str(atm), eddy=np.ones(c.z.shape[0]) * EDDY, overwrite=True)
    pc = EvoAtmosphere("zahnle_HNOC.yaml", "input/settings_photochemistry.yaml",
                       "TRAPPIST1e_stellar_flux.txt", str(atm))
    pc.var.verbose = 0
    pc.var.atol = 1e-23
    pc.var.upwind_molec_diff = True
    pc.set_lower_bc("N2", bc_type="press", press=P_N2 * 1e6)
    pc.set_lower_bc("CO2", bc_type="press", press=P_CO2 * 1e6)
    pc.set_lower_bc("CH4", bc_type="press", press=P_CH4 * 1e6)
    pc.update_vertical_grid(TOA_pressure=1e-7 * 1e6)
    pc.initialize_robust_stepper(pc.wrk.usol)
    give_up, converged, n = False, False, 0
    while n < MAX_STEPS:
        give_up, converged = pc.robust_step()
        n += 1
        if give_up or converged:
            break
    steady = bool(converged and not give_up)
    atm.unlink(missing_ok=True)

    # photochemistry -> pRT (same as pipeline)
    mf = pc.mole_fraction_dict()
    p_bar = np.asarray(mf["pressure"]) / 1e6
    T = np.asarray(mf["temp"])
    x = {sp: np.asarray(mf[sp]) for sp in SPECIES}
    mu = sum(x[sp] * MOLECULAR_WEIGHTS[sp] for sp in SPECIES)
    mass = {sp: x[sp] * MOLECULAR_WEIGHTS[sp] / mu for sp in SPECIES}
    rt = Radtrans(pressures=p_bar[::-1], line_species=["H2O", "CO2", "CH4", "CO", "O2"],
                  rayleigh_species=["N2", "H2"], gas_continuum_contributors=["N2-N2", "N2-H2"],
                  line_opacity_mode="c-k")

    rows = []
    for pcl in CLOUD_TOPS_BAR:
        kw = {} if pcl is None else {"opaque_cloud_top_pressure": pcl}
        wl_cm, r_cm, _ = rt.calculate_transit_radii(
            temperatures=T[::-1], mass_fractions={sp: mass[sp][::-1] for sp in SPECIES},
            mean_molar_masses=mu[::-1], reference_gravity=GRAVITY_CGS,
            reference_pressure=REFERENCE_PRESSURE_BAR, planet_radius=PLANET_RADIUS_CGS,
            variable_gravity=True, **kw)
        wl_um = wl_cm * 1e4
        depth = (r_cm / R_STAR_CGS) ** 2 * 1e6
        chi2, off, tab = bin_and_fit(*convolve(wl_um, depth))
        tab.to_csv(SPEC_DIR / f"model_{mid_:04d}_{cloud_label(pcl)}.csv", index=False)
        rows.append({"model_id": mid_, "group": row.group, "P_N2": P_N2, "P_CO2": P_CO2,
                     "P_CH4": P_CH4, "T_surf": float(c.T_surf), "steady_state": steady,
                     "cloud_top_bar": np.nan if pcl is None else pcl, "cloud": cloud_label(pcl),
                     "chi2": chi2, "dchi2_vs_flat": chi2 - CHI2_FLAT, "offset_ppm": off,
                     "binned_ptp_ppm": float(np.ptp(tab.model_depth_ppm)),
                     "chi2_master": float(row.chi2), "runtime_s": time.time() - t0})
    return rows


# ---------------- run ----------------
todo = sample[~sample.model_id.isin(done)]
t_start = time.time()
for k, row in enumerate(todo.itertuples(index=False), 1):
    try:
        rows = run_model(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", header=not OUT_CSV.exists(), index=False)
        clear = rows[0]
        msg = (f"clear chi2={clear['chi2']:.3f} (master {clear['chi2_master']:.3f}) | "
               + " ".join(f"{r['cloud']}:{r['dchi2_vs_flat']:+.1f}" for r in rows))
    except Exception as exc_:
        traceback.print_exc(limit=1)
        msg = f"FAILED: {type(exc_).__name__}: {exc_}"
    eta = (time.time() - t_start) / k * (len(todo) - k) / 60
    print(f"[{k:2d}/{len(todo)}] model {int(row.model_id):04d} ({row.group}) | {msg} | ETA {eta:.0f} min")

# ---------------- summary ----------------
r = pd.read_csv(OUT_CSV)
r = r[r.model_id.isin(sample.model_id)].drop_duplicates(["model_id", "cloud"], keep="last")
clear = r[r.cloud == "clear"]
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Clear-sky chi2 reproduces master CSV: max |difference| = "
      f"{(clear.chi2 - clear.chi2_master).abs().max():.4f}")
print(f"Photochemical steady state          : {int(as_bool(clear.steady_state).sum())} of {len(clear)}")
order = ["clear"] + [cloud_label(p) for p in CLOUD_TOPS_BAR[1:]]
tab = r.pivot(index="model_id", columns="cloud", values="dchi2_vs_flat")[order]
grp = clear.set_index("model_id")[["group", "P_CO2", "T_surf"]]
tab = grp.join(tab).sort_values(["group", "P_CO2"])
print()
print("Delta chi2 relative to the flat line (negative = fits better than flat):")
print(tab.round(2).to_string())
print()
ex = tab[tab.group == "excluded"]
for thr, name in [(11.83, "no longer excluded (< 11.83)"), (6.18, "compatible (< 6.18)")]:
    print(f"Excluded models {name}:")
    for c_ in order:
        print(f"   {c_:>8s}: {(ex[c_] < thr).sum()} of {len(ex)}")
print()
print("Highest cloud-top pressure (deepest cloud) that brings each excluded model below 6.18:")
for mid_, row in ex.iterrows():
    ok_ = [p for p in CLOUD_TOPS_BAR[1:] if row[cloud_label(p)] < 6.18]
    print(f"   model {mid_:04d} (CO2 {row.P_CO2:.3f} bar, {row.T_surf:.0f} K): "
          f"{(str(max(ok_)) + ' bar') if ok_ else 'none tested'}")
print("=" * 70)
